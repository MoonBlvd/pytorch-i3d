import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate
from PIL import Image
import numpy as np
import json
import csv
import h5py

import os
import os.path

import cv2
import pdb

def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3,0,1,2]))

class A3D(data_utl.Dataset):
    '''
    A3D dataset for I3D
    '''
    def __init__(self, 
                 split_file, 
                 split, 
                 root, 
                 mode, 
                 transforms=None, 
                 horizontal_flip=None, 
                 save_dir='', 
                 seq_len=16, 
                 overlap=0, 
                 with_normal=True):
        
        self.split_file = split_file
        self.transforms = transforms
        self.mode = mode
        self.root = root
        self.save_dir = save_dir
        self.seq_len = seq_len
        self.overlap = overlap
        self.fps = 5 # original fps is 10
        assert self.fps <= 10
        self.down_sample_rate = int(10/self.fps)

        self.num_classes = 17 # 16 known anomay type plus a normal, 0 is normal
        self.with_normal = with_normal
        
        self.name_to_id = {'normal': 0,
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
                            'other: leave_to_left': 16, 
                            'ego: unknown': 17,
                            'other: unknown': 18}
        # self.name_shorts = ['Normal', 'Ego: ST', 'Ego: AH', 'Ego: LA', 'Ego: OC', 
        #                     'Ego: TC', 'Ego: VP', 'Ego: VO', 'Ego: LL/LR', 
        #                     'Other: ST', 'Other: AH', 'Other: LA', 'Other: OC', 
        #                     'Other: TC', 'Other: VP', 'Other: VO', 'Other: LL/LR']
        self.name_shorts = [str(int(i)) for i in range(1, 17)]
        assert len(self.name_shorts) == self.num_classes - 1
        self.id_to_name = {v:k for k, v in self.name_to_id.items()}
        self.id_to_name[8] = 'ego: out_of_control'
        self.id_to_name[16] = 'other: out_of_control'

        self.data = self.make_dataset(split_file, split, root, mode)
        print("Number of used video:", len(self.data))

    def make_dataset(self, split_file, split, root, mode):
        dataset = []
        with open(split_file, 'r') as f:
            data = json.load(f)

        sample_category_stats = {v:0 for v in self.name_to_id.values()}

        self.video_level_classes = {}
        for idx, vid in enumerate(data.keys()):
            if 'unknown' in data[vid]['anomaly_class']:
                # skip unknown
                continue
            num_frames = data[vid]['num_frames']
            class_id = self.name_to_id[data[vid]['anomaly_class']]
            self.video_level_classes[vid] = {'class_id': class_id}

            if self.with_normal:
                start = 0
                end = num_frames
            else:
                start = data[vid]['anomaly_start']
                end = data[vid]['anomaly_end']    

            for t in range(start, end, (self.seq_len-self.overlap) * self.down_sample_rate):
                seq_start = int(t - (self.seq_len/2) * self.down_sample_rate)
                seq_end = int(t + (self.seq_len/2) * self.down_sample_rate)
                                
                # NOTE: for original I3D, one clip has only one label
                label = np.zeros(self.num_classes, np.float32) 

                # NOTE: method 1, assign the label of the middle frame
                if t >= data[vid]['anomaly_start'] and t < data[vid]['anomaly_end']:
                    label[class_id] = 1 # abnormal, ego involve classes
                    sample_category_stats[class_id] += 1
                else:
                    label[0] = 1 # normal 
                    sample_category_stats[0] += 1
                
                dataset.append({"vid": vid, 
                                "label_id": class_id,
                                "label": np.array(class_id), #label, # NOTE: use label for BCEloss, use class_id for CEloss
                                "seq_start": seq_start, # NOTE: 0-index
                                "seq_end": seq_end,# NOTE: 0-index
                                "num_frames": num_frames
                                })

        print("======== Number of samples of all categories ========")
        [print('{}:{}'.format(self.id_to_name[k], v)) for k, v in sample_category_stats.items()]


        category_weights = {k:1/counts for k, counts in sample_category_stats.items() if counts>0}
        self.weights = []
        for sample in dataset:
            self.weights.append(category_weights[sample['label_id']])

        return dataset

    def load_rgb_frames(self, image_dir, vid, start, end, num_frames):
        frames = []
        pad_front = 0
        pad_end = 0
        for i in range(start, end, self.down_sample_rate):
            if i < 0 :
                pad_front += 1
                continue
            if i >= num_frames:
                pad_end += 1
                continue
            img = Image.open(os.path.join(image_dir, vid, 'images', str(i).zfill(6)+'.jpg'))
            frames.append(img)
        # Pad the sequence
        for i in range(pad_front):
            frames.insert(0, frames[0])
        for i in range(pad_end):
            frames.append(frames[-1])
        return frames 
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        data = self.data[index]
        vid = data["vid"]
        label = data["label"]
        start = data["seq_start"]
        end = data["seq_end"]

        if os.path.exists(os.path.join(self.save_dir, vid+'.npy')):
            return 0, 0, vid

        if self.mode == 'rgb':
            imgs = self.load_rgb_frames(self.root, vid, start, end, data["num_frames"])
        else:
            imgs = self.load_flow_frames(self.root, vid, start, end, data["num_frames"])
        imgs, label = self.transforms(imgs, label)
        return imgs, label, vid, start, end

    def __len__(self):
        return len(self.data)