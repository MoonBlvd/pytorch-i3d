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

import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--root','-r', type=str, help='root directory of A3D')
parser.add_argument('--file_path','-f', type=str, help='path to A3D labels')
parser.add_argument('--save_dir','-s', type=str, help='path to save the generated json')
args = parser.parse_args()


all_A3D_anno_files = glob.glob(os.path.join(args.file_path, '*.json'))

anno_for_i3d
for anno_file in all_A3D_anno_files:
    anno = json.load(open(anno_file, 'r'))
    anno_for_i3d['video_start'] = anno['video_start'] # 1-index
    anno_for_i3d['video_end'] = anno['video_end'] # 1-index
    anno_for_i3d['anomaly_start'] = anno['anomaly_start']
    anno_for_i3d['anomaly_end'] = anno['anomaly_end']
    anno_for_i3d['anomaly_class'] = anno['accident_id']
    anno_for_i3d['num_frames'] = anno['num_frames']
    anno_for_i3d['subset'] = 'training'
    pdb.set_trace()
    anno['']