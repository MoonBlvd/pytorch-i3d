import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]='2'
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
from detection.utils.comm import synchronize, get_rank

from mpi4py import MPI
import apex
from apex.parallel import DistributedDataParallel as DDP
from detection.utils.comm import get_world_size
from detection.utils.comm import is_main_process, all_gather, synchronize

from datasets.evaluation.evaluation import ActionClassificationEvaluator
from sklearn.metrics import ConfusionMatrixDisplay

import pdb
def inference(model, 
              val_dataloader, 
              device, 
              output_dir=''):
    model.eval()

    # run on dataset
    results = defaultdict(list)
    for iters, data in enumerate(tqdm(val_dataloader)):
        # get the inputs
        inputs, _, video_names, start, end = data
        inputs = inputs.to(device)
        t = inputs.size(2)

        # get feature
        # pdb.set_trace()
        per_frame_feature = model(inputs, extract_features=True)
        per_frame_feature = per_frame_feature.squeeze().detach().cpu()
        # collect features
        for batch_id, vid in enumerate(video_names):
            # frame_id = int((start[batch_id] + end[batch_id])/2)

            feature = per_frame_feature[batch_id]
            # if vid not in results:
            #     results[vid] = {}
            # assert frame_id not in results[vid]
            # results[vid][frame_id] = feature
            results[vid].append(feature)
        # if iters > 5:
        #     break
    # results = _accumulate_from_multiple_gpus(results)
    
    if not is_main_process():
        return
    # pdb.set_trace()
    for vid in results.keys():
        results[vid] = torch.stack(results[vid], dim=0)#.mean(dim=0)
    
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # pdb.set_trace()
    torch.save(results, output_dir)

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

def main(model_name, 
         mode,
         root,
         val_split,
         ckpt,
         batch_per_gpu):
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

    val_dataloader = make_dataloader(root,
                                        val_split, 
                                        mode,
                                        model_name,
                                        seq_len=16, #64, 
                                        overlap=15, #32,
                                        phase='val', 
                                        max_iters=None, 
                                        batch_per_gpu=batch_per_gpu,
                                        num_workers=16, 
                                        shuffle=False, 
                                        distributed=distributed,
                                        with_normal=False)

    if model_name == 'i3d':
        if mode == 'flow':
            model = InceptionI3d(val_dataloader.dataset.num_classes, in_channels=2, dropout_keep_prob=0.5)
        else:
            model = InceptionI3d(val_dataloader.dataset.num_classes, in_channels=3, dropout_keep_prob=0.5)
        model.replace_logits(val_dataloader.dataset.num_classes)
    elif model_name == 'r3d_18':
        model = r3d_18(pretrained=False, num_classes=val_dataloader.dataset.num_classes)
    elif model_name == 'mc3_18':
        model = mc3_18(pretrained=False, num_classes=val_dataloader.dataset.num_classes)
    elif model_name == 'r2plus1d_18':
        model = r2plus1d_18(pretrained=False, num_classes=val_dataloader.dataset.num_classes)
    elif model_name == 'c3d':
        model = C3D(pretrained=False, num_classes=val_dataloader.dataset.num_classes)
    else:
        raise NameError('unknown model name:{}'.format(model_name))

    # pdb.set_trace()
    for param in model.parameters():
        pass
    
    device = torch.device('cuda')
    model.to(device)
    # if distributed:
    #     model = apex.parallel.convert_syncbn_model(model)
    #     model = DDP(model.cuda(), delay_allreduce=True)
    # load_state_dict(model, torch.load(ckpt))
    
    model.load_state_dict(torch.load(ckpt, map_location=device))
    # pdb.set_trace()
    for param in model.parameters():
        pass
    output_dir = os.path.join('/home/data/vision7/A3D_2.0/vac_features/', model_name+'.pth')
    inference(model, 
              val_dataloader, 
              device, 
              output_dir=output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_name', type=str, help='name of the model to run')
    parser.add_argument('-mode', type=str, help='rgb or flow')
    parser.add_argument('-root', type=str)
    parser.add_argument('-val_split', type=str)
    parser.add_argument('-ckpt', type=str, help='path to the trained checkpoint')
    parser.add_argument('-batch_per_gpu', type=int)
    args = parser.parse_args()
    
    main(model_name=args.model_name, 
         mode=args.mode,
         root=args.root,
         val_split=args.val_split,
         ckpt=args.ckpt,
         batch_per_gpu=args.batch_per_gpu
         )