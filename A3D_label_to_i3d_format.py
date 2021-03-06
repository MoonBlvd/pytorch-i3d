'''
Convert A3D annotation to the format of pytorch-i3d 

{
    '124wqsdas3_000001':{
        'num_frames': 60,
        'subset': 'training',
        'video_start': 1020,
        'video_end': 1090,
        'anomaly_start': 1020,
        'anomaly_end': 1090
    }
}
'''
import os
import json
import argparse
import glob
import numpy as np

import pdb

parser = argparse.ArgumentParser()
# parser.add_argument('--root','-r', type=str, help='root directory of A3D')
parser.add_argument('--file_path', '-f', type=str, help='path to A3D labels')
parser.add_argument('--save_name', '-s', type=str,
                    help='path and file name to save the generated json')
parser.add_argument('--split', '-sp', type=str, default='', help='train or val')
args = parser.parse_args()

all_A3D_anno_files = glob.glob(os.path.join(args.file_path, '*.json'))
if args.split == 'train':
    split_file = '/home/data/vision7/A3D_2.0/train_split.txt'
elif args.split == 'val':
    split_file = '/home/data/vision7/A3D_2.0/val_split.txt'
elif args.split == 'all':
    split_file = '/home/data/vision7/A3D_2.0/all_split.txt'
else:
    raise NameError()
    
with open(split_file, 'r') as f:
    video_names_in_split = f.read().splitlines()

anno_for_i3d = {}
for anno_file in all_A3D_anno_files:
    anno = json.load(open(anno_file, 'r'))
    key = anno['video_name']
    if key not in video_names_in_split:
        continue
    anno_for_i3d[key] = {}
    anno_for_i3d[key]['video_start'] = anno['video_start']    # 1-index
    anno_for_i3d[key]['video_end'] = anno['video_end']    # 1-index
    anno_for_i3d[key]['anomaly_start'] = anno['anomaly_start']
    anno_for_i3d[key]['anomaly_end'] = anno['anomaly_end']
    
    if anno['ego_involve']:
        anno_for_i3d[key]['anomaly_class'] = 'ego: ' + anno['accident_name']
    else:
        anno_for_i3d[key]['anomaly_class'] = 'other: ' + anno['accident_name']
    anno_for_i3d[key]['num_frames'] = anno['num_frames']
    anno_for_i3d[key]['subset'] = args.split
print("number of videos in split:", len(anno_for_i3d))
json.dump(anno_for_i3d, open(args.save_name, 'w'), indent=2)
