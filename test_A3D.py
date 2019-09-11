import sys
import argparse

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
# from build_sam
import pdb


def do_test(mode='rgb', split=None, root='', checkpoint=''):
    
    # setup dataloader
    test_dataloader = make_dataloader(root,
                                     split, 
                                     mode,
                                     phase='val', 
                                     max_iters=None, 
                                     batch_per_gpu=4,
                                     num_workers=8, 
                                     shuffle=False, 
                                     distributed=False,
                                     seq_len=32,
                                     overlap=31)

    # setup the model
    in_channels = 2 if mode == 'flow' else 3
    i3d = InceptionI3d(test_dataloader.dataset.num_classes, in_channels=in_channels)
    load_state_dict(i3d, torch.load(checkpoint))
    i3d.cuda()
    i3d = nn.DataParallel(i3d)
    i3d.eval()

    results = {}
    prev_vid = ''
    all_scores = {}
    for idx, data in enumerate(tqdm(test_dataloader)):
        # get the inputs
        inputs, labels, vids, starts, ends = data        
        
        # wrap them in Variable
        inputs = inputs.cuda()
        t = inputs.size(2)
        labels = labels.cuda()

        per_frame_logits = i3d(inputs)
        # upsample to input size
        per_frame_logits = F.upsample(per_frame_logits, t, mode='linear', align_corners=True)

        per_frame_logits = per_frame_logits.detach().cpu()

        del inputs, labels


        for batch_idx, vid in enumerate(vids):
            # input = inputs[batch_idx]
            # label = labels[batch_idx]
            start = starts[batch_idx]
            end = ends[batch_idx]

            # scores
            if start == 0 and vid != prev_vid:
                if all_scores:
                    # save the result of an untrimed video
                    # average score
                    avg_scores = []
                    for k in sorted(all_scores.keys()):
                        avg_scores.append(torch.mean(torch.cat(all_scores[k], dim=0), dim=0))
                    avg_scores = torch.stack(avg_scores, dim=0)
                    scores, classes = torch.max(avg_scores, dim=1)
                    prev_cls = -1
                    results[vid] = []
                    # pdb.set_trace()
                    for i, cls in enumerate(classes):
                        if cls != prev_cls:
                            if i > 0:
                                result['segment'].append(i) # append end of the segment
                                results[vid].append(result)
                            result = {}
                            result['segment'] = [i] # append start of the segment
                            result['label'] = int(cls)
                        elif i + 1 == len(classes):
                            result['segment'].append(i+1) # append end of the segment
                            results[vid].append(result)

                        prev_cls = cls   
                
                    all_scores = {}
                    # pdb.set_trace()

            seq_scores = F.sigmoid(per_frame_logits)
            for i in range(seq_scores.shape[-1]):
                frame_id = int(start.squeeze()) + i
                if frame_id in all_scores:
                    all_scores[frame_id].append(seq_scores[..., i])
                else:
                    all_scores[frame_id] = [seq_scores[..., i]]
            prev_vid = vid

    pdb.set_trace()

if __name__=='__main__':
    '''run test on untrimed video'''
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str, help='rgb or flow')
    parser.add_argument('-checkpoint', type=str)
    parser.add_argument('-root', type=str)
    parser.add_argument('-split', type=str)
    args = parser.parse_args()

    # need to add argparse
    do_test(mode=args.mode, split=args.split, root=args.root, checkpoint=args.checkpoint)