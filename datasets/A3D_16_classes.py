from collections import defaultdict
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
                 with_normal=False):
        self.split = split
        if self.split == 'train':
            self.num_clips = 1
        else:
            self.num_clips = 10

        self.split_file = split_file
        self.transforms = transforms
        self.mode = mode
        self.root = root
        self.save_dir = save_dir
        self.seq_len = seq_len
        self.overlap = overlap
        self.fps = 10 # original fps is 10
        assert self.fps <= 10
        self.down_sample_rate = int(10/self.fps)

        self.num_classes = 16 # 16 known classes
        
        self.name_to_id = {
                            'ego: start_stop_or_stationary': 0, 
                            'ego: moving_ahead_or_waiting': 1, 
                            'ego: lateral': 2, 
                            'ego: oncoming': 3, 
                            'ego: turning': 4, 
                            'ego: pedestrian': 5, 
                            'ego: obstacle': 6, 
                            'ego: leave_to_right': 7, 
                            'ego: leave_to_left': 7, # NOTE: Jan 18, merge leave right/left as out-of-control
                            'other: start_stop_or_stationary': 8, 
                            'other: moving_ahead_or_waiting': 9, 
                            'other: lateral': 10, 
                            'other: oncoming': 11, 
                            'other: turning': 12, 
                            'other: pedestrian': 13, 
                            'other: obstacle': 14, 
                            'other: leave_to_right': 15, 
                            'other: leave_to_left': 15, 
                            'ego: unknown': 16,
                            'other: unknown': 17}
        # self.name_shorts = ['Normal', 'Ego: ST', 'Ego: AH', 'Ego: LA', 'Ego: OC', 
        #                     'Ego: TC', 'Ego: VP', 'Ego: VO', 'Ego: LL/LR', 
        #                     'Other: ST', 'Other: AH', 'Other: LA', 'Other: OC', 
        #                     'Other: TC', 'Other: VP', 'Other: VO', 'Other: LL/LR']
        self.name_shorts = [str(int(i)) for i in range(0, 16)]
        assert len(self.name_shorts) == self.num_classes
        self.id_to_name = {v:k for k, v in self.name_to_id.items()}
        self.id_to_name[7] = 'ego: out_of_control'
        self.id_to_name[15] = 'other: out_of_control'

        self.data, self.data_category_stats = self.make_dataset(split_file, split, root, mode)
        print("Number of used clips:", len(self.data))
        
    def make_dataset(self, split_file, split, root, mode):
        dataset = []
        with open(split_file, 'r') as f:
            data = json.load(f)

        sample_category_stats = defaultdict(int)

        self.video_level_classes = {}
        for idx, vid in enumerate(data.keys()):
            if 'unknown' in data[vid]['anomaly_class']:
                # skip unknown
                continue
            num_frames = data[vid]['num_frames']
            class_id = self.name_to_id[data[vid]['anomaly_class']]
            self.video_level_classes[vid] = {'class_id': class_id}
            
            start = data[vid]['anomaly_start']
            end = data[vid]['anomaly_end']    
            
            sample_category_stats[class_id] += 1

            if self.split == 'train':
                dataset.append([vid, class_id, start, end])
            else:
                video_len = end - start
                delta = max([video_len - self.seq_len * self.down_sample_rate, 0])
                
                for clip_idx in range(self.num_clips):
                    clip_start = start + int(delta * clip_idx / (self.num_clips-1))
                    clip_end = min([clip_start + self.seq_len * self.down_sample_rate, end])
                    dataset.append([vid, class_id, clip_start, clip_end])

        print("======== Number of samples of all categories ========")
        [print('{}:{}'.format(self.id_to_name[k], v)) for k, v in sample_category_stats.items()] # self.id_to_name[k]

        # category_weights = {k:1/counts for k, counts in sample_category_stats.items() if counts>0}
        # self.weights = []
        # for sample in dataset:
        #     self.weights.append(category_weights[sample['label_id']])

        return dataset, sample_category_stats

    def load_rgb_frames(self, image_dir, vid, indices):
        frames = []
        for idx in indices:
            img = Image.open(os.path.join(image_dir, vid, 'images', str(idx).zfill(6)+'.jpg'))
            # NOTE: try to crop image to remove Anan or Carcrash logos
            img = img.crop(box=[0, 80, 1280, 640])
            frames.append(img)
        return frames 

    def get_frame_index(self, start, end):
        '''
        start: anomaly start
        end: anomaly end
        '''
        if end - start >= self.seq_len * self.down_sample_rate:
            clip_start = start + np.random.randint(end - start - self.seq_len * self.down_sample_rate + 1)
            clip_end = clip_start + self.seq_len * self.down_sample_rate
        else:
            clip_start = start
            clip_end = end
        indices = np.arange(clip_start, clip_end, self.down_sample_rate)
        while len(indices) < self.seq_len:
            indices = np.insert(indices, 0, indices[0])
        
        return indices

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        vid, class_id, start, end = self.data[index]
        indices = self.get_frame_index(start,  end)
        imgs = self.load_rgb_frames(self.root, vid, indices)
        label = None
        imgs, label = self.transforms(imgs, label)
        return imgs, class_id, vid #imgs, label, vid, start, end

    def __len__(self):
        return len(self.data)