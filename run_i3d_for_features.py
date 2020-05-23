import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]='0, 1'
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
import json
import pdb
name_to_id = {
                'ego: start_stop_or_stationary': 1, 
                'ego: moving_ahead_or_waiting': 2, 
                'ego: lateral': 3, 
                'ego: oncoming': 4, 
                'ego: turning': 5, 
                'ego: pedestrian': 6, 
                'ego: obstacle': 7, 
                'ego: leave_to_right': 8, 
                'ego: leave_to_left': 8, # NOTE: Jan 18, merge leave right/left as out-of-control
                'other: start_stop_or_stationary': 9, 
                'other: moving_ahead_or_waiting': 10, 
                'other: lateral': 11, 
                'other: oncoming': 12, 
                'other: turning': 13, 
                'other: pedestrian': 14, 
                'other: obstacle': 15, 
                'other: leave_to_right': 16, 
                'other: leave_to_left': 16, }
def evaluate(all_pred):
    '''
    all_pred: a dict saving scores for each video
    '''
    confusion_matrix = np.zeros((17,17))
    top_3_acc = np.zeros(17)
    all_annos = json.load(open('/home/data/vision7/A3D_2.0/A3D_2.0_val.json', 'r'))
    for vid, scores in all_pred.items():
        scores = scores.mean(dim=0)
        anomaly_cls = all_annos[vid]['anomaly_class']
        label = name_to_id[anomaly_cls]
        sorted_pred = torch.argsort(scores, descending=True)
        top1_pred = sorted_pred[0]
        confusion_matrix[label][top1_pred] += 1
        # get top 3
        if label in sorted_pred[:3]:
            top_3_acc[label] += 1
    acc = confusion_matrix.diagonal() / (confusion_matrix.sum(axis=1) + 1e-5)
    top_3_acc = top_3_acc / (confusion_matrix.sum(axis=1) + 1e-5)
    np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
    print("Top 1 Accuracy: {} \t mean Top 1 Accuracy: {:.4f} \t \
           Top 3 Accuracy: {} \t mean Top 3 Accuracy: {:.4f}".format(acc[1:], 
                                                                    acc[1:].mean(),
                                                                    top_3_acc[1:],
                                                                    top_3_acc[1:].mean()))
def inference(model, 
              val_dataloader, 
              device, 
              output_dir=''):
    model.eval()
    extract_features = False
    # run on dataset
    results = defaultdict(list)
    for iters, data in enumerate(tqdm(val_dataloader)):
        # get the inputs
        inputs, _, video_names, start, end = data
        inputs = inputs.to(device)
        t = inputs.size(2)

        # get features or results
        # pdb.set_trace()
        if extract_features:
            per_frame_feature = model(inputs, extract_features=extract_features)
            per_frame_feature = per_frame_feature.squeeze().detach().cpu()
            # collect features
            for batch_id, vid in enumerate(video_names):
                feature = per_frame_feature[batch_id]
                results[vid].append(feature)
        else:
            logits = model(inputs, extract_features=extract_features)
            logits = logits.squeeze(-1).detach().cpu()
            softmax_scores = F.softmax(logits, dim=1)
            # collect scores
            for batch_id, vid in enumerate(video_names):
                scores = softmax_scores[batch_id]
                results[vid].append(scores)
    results = _accumulate_from_multiple_gpus(results)
    
    if not is_main_process():
        return
    # pdb.set_trace()
    for vid in results.keys():
        results[vid] = torch.stack(results[vid], dim=0)#.mean(dim=0)
    
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # pdb.set_trace()
    # torch.save(results, output_dir)
    if not extract_features:
        evaluate(results)

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
                                        overlap=8, #32,
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
    if distributed:
        model = apex.parallel.convert_syncbn_model(model)
        model = DDP(model.cuda(), delay_allreduce=True)
<<<<<<< HEAD
    load_state_dict(model, torch.load(ckpt))
=======
    # load_state_dict(model, torch.load(ckpt))
    model.load_state_dict(torch.load(ckpt, map_location=device))
>>>>>>> 5ab1ce69e37a0c040ab682148e1642bcf749f984
    
    # model.load_state_dict(torch.load(ckpt, map_location=device))
    # pdb.set_trace()
    for param in model.parameters():
        pass
    output_dir = os.path.join('/mnt/workspace/datasets/A3D_2.0/vac_features/', model_name+'.pth')
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
