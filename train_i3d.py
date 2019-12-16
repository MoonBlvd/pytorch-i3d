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

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        # print("all_losses:", all_losses)
        # print("\n")
        all_losses = torch.stack(all_losses, dim=0)
        if torch.isnan(torch.sum(all_losses)):
            pdb.set_trace()
        dist.reduce(all_losses, dst=0)
        if torch.isnan(torch.sum(all_losses)):
            pdb.set_trace()
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
            
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses

def do_train(model, 
             train_dataloader, 
             val_dataloader, 
             device,
             checkpoint_peroid=1000, 
             save_model='', 
             logger=None, 
             distributed=False,
             evaluator=None):

    lr = 0.003 #init_lr
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
    # NOTE: maybe the weight decay is too soon?
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [24000, 32000])#[2000,10000])#[300, 1000])

    num_steps_per_update = 1 #4 # accum gradient
    steps = 0
    if not os.path.isdir(save_model):
        os.makedirs(save_model)
    
    tot_loss = 0.0
    tot_loc_loss = 0.0
    tot_cls_loss = 0.0

    if logger is None:
        logger = logging.getLogger("I3D.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    end = time.time()

    max_iters = len(train_dataloader)
    model.train()
    optimizer.zero_grad()
    # label_counts = defaultdict(int)
    for iters, data in enumerate(tqdm(train_dataloader)):
        data_time = time.time() - end
        iters += 1

        # get the inputs
        inputs, labels, video_names, _, _ = data
        
        # wrap them in Variable
        inputs = Variable(inputs.to(device))
        t = inputs.size(2)
        labels = Variable(labels.to(device))

        per_frame_logits = model(inputs) # B X C X T X H X W
        per_frame_logits = per_frame_logits.mean(dim=-1) # B X C
        # # upsample to input size
        # per_frame_logits = F.interpolate(per_frame_logits, t, mode='linear', align_corners=True)
        
        # compute localization loss
        # loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)

        # compute classification loss (with max-pooling along time B x C x T)
        # cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0], torch.max(labels, dim=2)[0])
    
        # loss = (0.5*loc_loss + 0.5*cls_loss)/num_steps_per_update

        cls_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
        loss = cls_loss

        # track time
        batch_time = time.time() - end
        end = time.time()
        # reduce losses over all GPUs for logging purposes
        loss_dict = {"loss_cls": cls_loss} #{"loss_loc": loc_loss, "loss_cls": cls_loss}
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = loss_dict_reduced['loss_cls'] #0.5 * loss_dict_reduced['loss_loc'] + 0.5 * loss_dict_reduced['loss_cls']

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 50)

        meters.update(loss=losses_reduced, **loss_dict_reduced)
        meters.update(time=batch_time, data=data_time)

        # estimate the rest of the running time
        eta_seconds = meters.time.global_avg * (max_iters - iters)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iters % num_steps_per_update == 0 or iters == len(train_dataloader):
            steps += 1
            optimizer.step()
            optimizer.zero_grad()

            lr_sched.step()
            if steps % 20 == 0:
                # NOTE: Add log file 
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "iter: {iter}",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        iter=iters,
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                ) 

                for name, meter in meters.meters.items():
                    logger.log_values({name: meter.median}, step=iters)
                logger.log_values({"grad_norm": grad_norm}, step=iters)
            
            if steps % checkpoint_peroid == 0:
                del inputs, loss
                model.eval()
                do_val(model, 
                       val_dataloader, 
                       device,
                       distributed, 
                       logger, 
                       output_dir=os.path.join(save_model, 'inference'),
                       train_iters=iters,
                       evaluator=evaluator)
                model.train()
                
                save_dir = os.path.join(save_model, str(steps).zfill(6)+'.pt')
                if hasattr(model, 'module'):
                    torch.save(model.module.state_dict(), save_dir)
                else:
                    torch.save(model.state_dict(), save_dir)


def do_val(model, val_dataloader, device, distributed=False,logger=None, output_dir='', train_iters=0, evaluator=None):
    if logger is None:
        logger = logging.getLogger("I3D.trainer")

    # if distributed:
    #     model = model.module

    torch.cuda.empty_cache()  # TODO check if it helps

    tot_loss = 0.0
    tot_loc_loss = 0.0
    tot_cls_loss = 0.0
    
    # run on dataset
    results = {}
    for iters, data in enumerate(tqdm(val_dataloader)):
        # get the inputs
        inputs, labels, video_names, start, end = data
        # print("inputs shape")
        # wrap them in Variable
        inputs = inputs.to(device)
        t = inputs.size(2)
        labels = labels.to(device)

        per_frame_logits = model(inputs)
        per_frame_logits = per_frame_logits.mean(dim=-1)

        # # upsample to input size
        # per_frame_logits = F.interpolate(per_frame_logits, t, mode='linear', align_corners=True)

        # # compute localization loss
        # loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
        # loc_loss = loc_loss.item()
        # tot_loc_loss += loc_loss

        # compute classification loss (with max-pooling along time B x C x T)
        # cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0], torch.max(labels, dim=2)[0])

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
    for k, v in eval_results.items():
        logger.info('{}:{}'.format(k, v))
        if isinstance(v, (float, int)) and hasattr(logger, 'log_values'):
            logger.log_values(eval_results, step=train_iters)
    
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

def run(init_lr=0.1, 
        max_steps=64e3, 
        mode='rgb', 
        root='/home/data/vision7/A3D_2.0/frames/', #'/ssd/Charades_v1_rgb', 
        train_split='A3D_2.0_train.json', #'charades/charades.json', 
        val_split='A3D_2.0_val.json',
        checkpoint_peroid=1000,
        batch_size=8*5, 
        save_model='',
        with_normal=True):

    
    num_gpus = MPI.COMM_WORLD.Get_size()
    distributed = False
    if num_gpus > 1:
        distributed = True

    local_rank = MPI.COMM_WORLD.Get_rank() % torch.cuda.device_count()

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
    # logger must be initialized after distributed!
    cfg = {'PROJECT': 'i3d_a3d'}
    logger = Logger("I3D",
                    cfg,#convert_to_dict(cfg, []),
                    project = 'i3d_a3d',
                    viz_backend="wandb" if args.use_wandb else "tensorboardx"
                    )
    save_model = os.path.join(save_model, logger.run_id)

    logger.info("Using {} GPUs".format(num_gpus))
    # setup dataset
    train_dataloader = make_dataloader(root,
                                       train_split, 
                                       mode,
                                       seq_len=16, #64,
                                       overlap=15, #32,
                                       phase='train', 
                                       max_iters=40000, 
                                       batch_per_gpu=16,
                                       num_workers=4, 
                                       shuffle=True, 
                                       distributed=distributed,
                                       with_normal=with_normal)

    val_dataloader = make_dataloader(root,
                                     val_split, 
                                     mode,
                                     seq_len=16, #64, 
                                     overlap=15, #32,
                                     phase='val', 
                                     max_iters=None, 
                                     batch_per_gpu=16,#16,
                                     num_workers=4, 
                                     shuffle=False, 
                                     distributed=distributed,
                                     with_normal=with_normal)

    evaluator = ActionClassificationEvaluator(cfg=None,
                                              dataset=val_dataloader.dataset,
                                              split='val',
                                              mode='accuracy',#'mAP',
                                              output_dir=save_model,
                                              with_normal=with_normal)
    # setup the model
    # set  dropout_keep_prob=0.0 for overfit
    if mode == 'flow':
        i3d = InceptionI3d(train_dataloader.dataset.num_classes, in_channels=2, dropout_keep_prob=0.5)
        # load_state_dict(i3d, torch.load('models/flow_imagenet.pt'), ignored_prefix='logits')
    else:
        i3d = InceptionI3d(train_dataloader.dataset.num_classes, in_channels=3, dropout_keep_prob=0.5)
        # load_state_dict(i3d, torch.load('models/rgb_imagenet.pt'), ignored_prefix='logits')

    i3d.replace_logits(train_dataloader.dataset.num_classes)

    device = torch.device('cuda')
    i3d.to(device)
    if distributed:
        i3d = apex.parallel.convert_syncbn_model(i3d)
        i3d = DDP(i3d.cuda(), delay_allreduce=True)

    do_train(i3d, 
             train_dataloader, 
             val_dataloader, 
             device=device,
             checkpoint_peroid=checkpoint_peroid, 
             save_model=save_model, 
             logger=logger,
             evaluator=evaluator)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str, help='rgb or flow')
    parser.add_argument('-save_model', type=str)
    parser.add_argument('-root', type=str)
    parser.add_argument('-train_split', type=str)
    parser.add_argument('-val_split', type=str)
    parser.add_argument('-use_wandb', type=bool, default=True)
    args = parser.parse_args()
    
    # need to add argparse
    run(mode=args.mode, 
        train_split=args.train_split, 
        val_split=args.val_split, 
        root=args.root, 
        save_model=args.save_model, 
        checkpoint_peroid=2000,
        with_normal=False
        )
