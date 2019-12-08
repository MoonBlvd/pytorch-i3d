# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   author: klaus
#   email: klaus.cheng@qq.com
#   created date: 2019/12/04
#   description:
#
#================================================================

import pickle as pkl
import json
import os
import argparse
import numpy as np
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--file_path', '-f', type=str, help='path to A3D 1.0 dir')
parser.add_argument('--save_name',
                    '-s',
                    type=str,
                    help='path and file name to save the generated json')
args = parser.parse_args()

f = open(os.path.join(args.file_path, 'A3D_labels.pkl'), 'rb')
a3d_anno = pkl.load(f)

anno_for_i3d = {}
num = 0
for vid in a3d_anno.keys():
    anno_for_i3d[vid] = {}
    start = a3d_anno[vid]['clip_start']
    start = int(start)
    end = a3d_anno[vid]['clip_end']
    end = int(end)
    anno_for_i3d[vid]['video_start'] = start
    anno_for_i3d[vid]['video_end'] = end
    targets = a3d_anno[vid]['target']
    targets = np.asarray(targets)
    if (len(np.argwhere(targets == 1)) == 0):
        del anno_for_i3d[vid]
        continue
    anno_for_i3d[vid]['anomaly_start'] = start + np.argwhere(targets == 1)[0][0].item() 
    anno_for_i3d[vid]['anomaly_end'] = start + np.argwhere(targets == 1)[-1][0].item() 

    anno_for_i3d[vid]['num_frames'] = end - start + 1

    num += 1
    anno_for_i3d[vid]['anomaly_class'] = 0  
    if num < 1578:
        anno_for_i3d[vid]['subset'] = 'train'
    else:
        anno_for_i3d[vid]['subset'] = 'val'
print(num) 
json.dump(anno_for_i3d, open(args.save_name, 'w'), indent=2)
