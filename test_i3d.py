import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
import sys
import argparse
import logging
import time
import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torch.distributed as dist

from datasets.build import make_dataloader
import numpy as np
from pytorch_i3d import InceptionI3d

# from charades_dataset import Charades as Dataset
from tqdm import tqdm
from model_serialization import load_state_dict
from detection.utils.logger import Logger
from detection.utils.metric_logger import MetricLogger


from mpi4py import MPI
import apex
from apex.parallel import DistributedDataParallel as DDP
from detection.utils.comm import get_world_size
from detection.utils.comm import is_main_process, all_gather, synchronize

from datasets.evaluation.evaluation import ActionClassificationEvaluator

import pdb

def _accumulate_from_multiple_gpus(item_per_gpu):
    # all_keys
    all_items = all_gather(item_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_items:
        predictions.update(p)
    return predictions

def do_test(model, dataloader, device, distributed=False,logger=None, output_dir='', train_iters=0, evaluator=None):
    model.eval()
    if logger is None:
        logger = logging.getLogger("I3D.trainer")

    if distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps

    tot_loss = 0.0
    tot_loc_loss = 0.0
    tot_cls_loss = 0.0
    
    # run on dataset
    results = {}
    for iters, data in enumerate(tqdm(dataloader)):
        # get the inputs
        inputs, labels, video_names, start, end = data
        # wrap them in Variable
        inputs = inputs.to(device)
        t = inputs.size(2)
        labels = labels.to(device)

        per_frame_logits = model(inputs)
        per_frame_logits = per_frame_logits.mean(dim=-1)

        cls_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
        cls_loss = cls_loss.item()
        tot_cls_loss += cls_loss

        loss = cls_loss #(0.5*loc_loss + 0.5*cls_loss)
        tot_loss += loss
        # collect results
        per_frame_logits = per_frame_logits.detach().cpu()
        for batch_id, vid in enumerate(video_names):
            frame_id = int((start[batch_id] + end[batch_id])/2)

            pred = per_frame_logits[batch_id]
            if vid not in results:
                results[vid] = {}
            assert frame_id not in results[vid]
            results[vid][frame_id] = pred
            # if vid in results:
            #     results[vid] = torch.cat([results[vid], pred], dim=0)
            # else:
            #     results[vid] = []
    if hasattr(logger, 'log_values'):
        logger.log_values({"loss_cls_val": tot_cls_loss/(iters+1)}, step=train_iters)   

    results = _accumulate_from_multiple_gpus(results)
    if not is_main_process():
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    torch.save(results, os.path.join(output_dir, 'predictions.pth'))

    # Run video-level evaluation
    eval_results = evaluator.evaluate(results)
    print(eval_results)
    for k, v in eval_results.items():
        logger.info('{}:{}'.format(k, v))
        if isinstance(v, (float, int)) and hasattr(logger, 'log_values'):
            logger.log_values(eval_results, step=train_iters)
    
    # logger.info("AC_top_1: {}   AC_top_3:{}".format(AC_top_1, AC_top_3))

def main(ckpt,
         mode='rgb', 
         root='/home/data/vision7/A3D_2.0/frames/', 
         split_file='A3D_2.0_val.json',
         split='val',
         with_normal=True,
         save_dir=''):
    device = torch.device('cuda')

    num_gpus = MPI.COMM_WORLD.Get_size()
    distributed = False
    if num_gpus > 1:
        distributed = True

    local_rank = MPI.COMM_WORLD.Get_rank() % torch.cuda.device_count()

    # logger must be initialized after distributed!
    cfg = {'PROJECT': 'i3d_a3d'}
    logger = Logger("I3D",
                    cfg,#convert_to_dict(cfg, []),
                    project = 'i3d_a3d',
                    viz_backend="wandb" if args.use_wandb else "tensorboardx"
                    )
    save_dir = os.path.join(save_dir, logger.run_id)

    logger.info("Using {} GPUs".format(num_gpus))

    if distributed:
        torch.cuda.set_device(local_rank)
        host = os.environ["MASTER_ADDR"] if "MASTER_ADDR" in os.environ else "127.0.0.1"
        torch.distributed.init_process_group(
            backend="nccl",
            init_method='tcp://{}:12345'.format(host),
            rank=MPI.COMM_WORLD.Get_rank(),
            world_size=MPI.COMM_WORLD.Get_size()
        )

        synchronize()

    dataloader = make_dataloader(root,
                                split_file, 
                                mode,
                                seq_len=16, #64, 
                                overlap=15, #32,
                                phase='val', 
                                max_iters=None, 
                                batch_per_gpu=16,
                                num_workers=4, 
                                shuffle=False, 
                                distributed=distributed,
                                with_normal=with_normal)
    evaluator = ActionClassificationEvaluator(cfg=None,
                                              dataset=dataloader.dataset,
                                              split='val',
                                              mode='accuracy',#'mAP',
                                              output_dir=save_dir,
                                              with_normal=with_normal)

    # setup the model
    # set  dropout_keep_prob=0.0 for overfit
    if mode == 'flow':
        i3d = InceptionI3d(dataloader.dataset.num_classes, in_channels=2, dropout_keep_prob=0.5)
    else:
        i3d = InceptionI3d(dataloader.dataset.num_classes, in_channels=3, dropout_keep_prob=0.5)

    load_state_dict(i3d, torch.load(ckpt))

    i3d.to(device)
    if distributed:
        i3d = apex.parallel.convert_syncbn_model(i3d)
        i3d = DDP(i3d.cuda(), delay_allreduce=True)

    do_test(i3d, dataloader, device, distributed=distributed,logger=logger, output_dir=save_dir, train_iters=0, evaluator=evaluator)

if __name__=='__main__':
    '''run test on untrimed video'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode','-m', type=str, help='rgb or flow')
    parser.add_argument('--checkpoint', '-c', type=str)
    parser.add_argument('--root', '-r', type=str)
    parser.add_argument('-split', type=str)
    parser.add_argument('--save_dir', '-s', type=str, help="path to save the prediction results")
    parser.add_argument('--use_wandb', '-u', type=str, help="use_wandb")
    parser.add_argument('--split_file', '-sf', type=str, help="split file")
    args = parser.parse_args()


    # need to add argparse
    main(ckpt=args.checkpoint,
         mode=args.mode, 
         root=args.root, #'/home/data/vision7/A3D_2.0/frames/', 
         split_file=args.split_file, 
         split=args.split, #'val',
         with_normal=False,
         save_dir=args.save_dir)