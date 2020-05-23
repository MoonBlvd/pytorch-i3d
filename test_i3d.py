import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]='0'
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
from torchvision.models.video import r3d_18, mc3_18, r2plus1d_18
from pytorch_c3d import C3D

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
from train_i3d import do_val
import logging
import pdb

# def _accumulate_from_multiple_gpus(item_per_gpu):
#     # all_keys
#     all_items = all_gather(item_per_gpu)
#     if not is_main_process():
#         return
#     # merge the list of dicts
#     predictions = {}
#     for p in all_items:
#         predictions.update(p)
#     return predictions

# def do_test(model, dataloader, device, distributed=False,logger=None, output_dir='', train_iters=0, evaluator=None):
#     model.eval()
#     if logger is None:
#         logger = logging.getLogger("I3D.trainer")

#     if distributed:
#         model = model.module
#     torch.cuda.empty_cache()  # TODO check if it helps

#     tot_loss = 0.0
#     tot_loc_loss = 0.0
#     tot_cls_loss = 0.0
    
#     # run on dataset
#     results = {}
#     for iters, data in enumerate(tqdm(dataloader)):
#         # get the inputs
#         inputs, labels, video_names, start, end = data
#         # wrap them in Variable
#         inputs = inputs.to(device)
#         t = inputs.size(2)
#         labels = labels.to(device)

#         per_frame_logits = model(inputs)
#         per_frame_logits = per_frame_logits.mean(dim=-1)

#         cls_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
#         cls_loss = cls_loss.item()
#         tot_cls_loss += cls_loss

#         loss = cls_loss #(0.5*loc_loss + 0.5*cls_loss)
#         tot_loss += loss
#         # collect results
#         per_frame_logits = per_frame_logits.detach().cpu()
#         for batch_id, vid in enumerate(video_names):
#             frame_id = int((start[batch_id] + end[batch_id])/2)

#             pred = per_frame_logits[batch_id]
#             if vid not in results:
#                 results[vid] = {}
#             assert frame_id not in results[vid]
#             results[vid][frame_id] = pred
#             # if vid in results:
#             #     results[vid] = torch.cat([results[vid], pred], dim=0)
#             # else:
#             #     results[vid] = []
#     if hasattr(logger, 'log_values'):
#         logger.log_values({"loss_cls_val": tot_cls_loss/(iters+1)}, step=train_iters)   

#     results = _accumulate_from_multiple_gpus(results)
#     if not is_main_process():
#         return

#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     torch.save(results, os.path.join(output_dir, 'predictions.pth'))

#     # Run video-level evaluation
#     eval_results = evaluator.evaluate(results)
#     print(eval_results)
#     for k, v in eval_results.items():
#         logger.info('{}:{}'.format(k, v))
#         if isinstance(v, (float, int)) and hasattr(logger, 'log_values'):
#             logger.log_values(eval_results, step=train_iters)
    
#     # logger.info("AC_top_1: {}   AC_top_3:{}".format(AC_top_1, AC_top_3))

def main(ckpt,
         model_name,
         mode='rgb', 
         root='/home/data/vision7/A3D_2.0/frames/', 
         split_file='A3D_2.0_val.json',
         split='val',
         with_normal=True,
         batch_per_gpu=16,
         save_dir=''):
    device = torch.device('cuda')

    num_gpus = MPI.COMM_WORLD.Get_size()
    distributed = False
    if num_gpus > 1:
        distributed = True

    local_rank = MPI.COMM_WORLD.Get_rank() % torch.cuda.device_count()

    # logger must be initialized after distributed!
    if args.use_wandb :
        cfg = {'PROJECT': 'i3d_a3d'}
        logger = Logger("I3D",
                        cfg,#convert_to_dict(cfg, []),
                        project = 'i3d_a3d',
                        viz_backend="wandb" 
                        )
        save_dir = os.path.join(save_dir, logger.run_id)
    else:
        logger = logging.Logger('test_VAR_final')
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
                                model_name=model_name,
                                seq_len=16, #64, 
                                overlap=15, #32,
                                phase='val', 
                                max_iters=None, 
                                batch_per_gpu=batch_per_gpu,
                                num_workers=16, 
                                shuffle=False, 
                                distributed=distributed,
                                with_normal=with_normal)
    # evaluator = ActionClassificationEvaluator(cfg=None,
    #                                           dataset=dataloader.dataset,
    #                                           split='val',
    #                                           mode='accuracy',#'mAP',
    #                                           output_dir=save_dir,
    #                                           with_normal=with_normal)

    # setup the model
    # set  dropout_keep_prob=0.0 for overfit
    if model_name == 'i3d':
        if mode == 'flow':
            model = InceptionI3d(dataloader.dataset.num_classes, in_channels=2, dropout_keep_prob=0.5)
        else:
            model = InceptionI3d(dataloader.dataset.num_classes, in_channels=3, dropout_keep_prob=0.5)
        model.replace_logits(dataloader.dataset.num_classes)
    elif model_name == 'r3d_18':
        model = r3d_18(pretrained=False, num_classes=dataloader.dataset.num_classes)
    elif model_name == 'mc3_18':
        model = mc3_18(pretrained=False, num_classes=dataloader.dataset.num_classes)
    elif model_name == 'r2plus1d_18':
        model = r2plus1d_18(pretrained=False, num_classes=dataloader.dataset.num_classes)
    elif model_name == 'c3d':
        model = C3D(pretrained=False, num_classes=dataloader.dataset.num_classes)
    else:
        raise NameError('unknown model name:{}'.format(model_name))

    model.load_state_dict(torch.load(ckpt))
    # do_test(i3d, dataloader, device, distributed=distributed,logger=logger, output_dir=save_dir, train_iters=0, evaluator=evaluator)
    model.to(device)
    model.eval()
    do_val(model_name,
            model, 
            dataloader, 
            device,
            distributed, 
            logger, 
            output_dir=os.path.join('test_output'),
            train_iters=0)

if __name__=='__main__':
    '''run test on untrimed video'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='name of the model to run')
    parser.add_argument('--mode', type=str, help='rgb or flow')
    parser.add_argument('--checkpoint', '-c', type=str)
    parser.add_argument('--root', '-r', type=str)
    parser.add_argument('--split', type=str)
    parser.add_argument('--save_dir', '-s', default='', type=str, help="path to save the prediction results")
    parser.add_argument('--batch_per_gpu', type=int)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--use_wandb', const=True, nargs='?')
    parser.add_argument('--split_file', '-sf', type=str, help="split file")
    args = parser.parse_args()


    # need to add argparse
    main(ckpt=args.checkpoint,
         model_name=args.model_name,
         mode=args.mode, 
         root=args.root, #'/home/data/vision7/A3D_2.0/frames/', 
         split_file=args.split_file, 
         split=args.split, #'val',
         with_normal=False,
         batch_per_gpu=args.batch_per_gpu,
         save_dir=args.save_dir)