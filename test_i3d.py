import sys
import argparse
import os 

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from datasets.build import make_dataloader

from pytorch_i3d import InceptionI3d

from tqdm import tqdm
from model_serialization import load_state_dict

import pdb


def do_test(mode='rgb', split=None, root='', checkpoint='', save_dir=''):
    
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
    for iters, data in enumerate(tqdm(val_dataloader)):
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

def main():

    dataloader = make_dataloader(root,
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

if __name__=='__main__':
    '''run test on untrimed video'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode','-m', type=str, help='rgb or flow')
    parser.add_argument('--checkpoint', '-c', type=str)
    parser.add_argument('--root', '-r', type=str)
    parser.add_argument('-split', type=str)
    parser.add_argument('--save_dir', '-s', type=str, help="path to save the prediction results")
    args = parser.parse_args()

    # need to add argparse
    main()
    do_test(mode=args.mode, split=args.split, root=args.root, checkpoint=args.checkpoint, save_dir=args.save_dir)