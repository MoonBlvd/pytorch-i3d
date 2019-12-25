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
                 mode, transforms=None, horizontal_flip=None, save_dir='', seq_len=16, overlap=0, with_normal=True):
        
        self.split_file = split_file
        self.transforms = transforms
        self.mode = mode
        self.root = root
        self.save_dir = save_dir
        self.seq_len = seq_len
        self.overlap = overlap
        self.fps = 10
        self.num_classes = 19 # 18 known anomay type plus a normal, 0 is normal
        self.with_normal = with_normal
        
        self.name_to_id = {'normal': 0,
                            'start_stop_or_stationary': 1, 
                            'moving_ahead_or_waiting': 2, 
                            'lateral': 3, 
                            'oncoming': 4, 
                            'turning': 5, 
                            'pedestrian': 6, 
                            'obstacle': 7, 
                            'leave_to_right': 8, 
                            'leave_to_left': 9, 
                            'unknown': 10}
        self.id_to_name = {v:k for k, v in self.name_to_id.items()}

        self.data = self.make_dataset(split_file, split, root, mode)
        print("Number of used video:", len(self.data))

    def make_dataset(self, split_file, split, root, mode):
        dataset = []
        with open(split_file, 'r') as f:
            data = json.load(f)
        self.valid_videos = []

        sample_category_stats = {v:0 for v in self.name_to_id.values()}

        self.video_level_classes = {}
        for idx, vid in enumerate(data.keys()):
            
            if data[vid]['video_start'] is None or \
               data[vid]['video_start'] is None or \
               data[vid]['anomaly_start'] is None or \
               data[vid]['anomaly_end'] is None:
                # NOTE: Sep 5, Some videos may have null video_start, meaning there is a bug and we skip the video for now
                continue
            # if data[vid]['subset'] != split:
            #     continue
            if not os.path.exists(os.path.join(root, vid)):
                continue
            if int(data[vid]['anomaly_class']) == 10:
                # skip unknown
                continue
            num_frames = data[vid]['num_frames']
            
            self.valid_videos.append(vid)
            
            assert int(data[vid]['anomaly_class']) > 0
            
            class_id = int(data[vid]['anomaly_class'])
            
            self.video_level_classes[vid] = {'class_id':class_id}

            n_skip = 0
            if self.with_normal:
                start = 0
                end = num_frames
            else:
                start = data[vid]['anomaly_start']
                end = data[vid]['anomaly_end']            
            for t in range(start, end, self.seq_len-self.overlap):
                seq_start = t - self.seq_len/2 
                seq_end = t + self.seq_len/2 
                                
                # NOTE: for original I3D, one clip has only one label
                label = np.zeros(self.num_classes, np.float32) 

                # NOTE: method 1, assign the label of the middle frame
                if t >= data[vid]['anomaly_start'] and t < data[vid]['anomaly_end']:
                    label[class_id] = 1 # abnormal, ego involve classes
                    sample_category_stats[class_id] += 1
                else:
                    # #NOTE: skip some normal
                    # if n_skip < 8:
                    #     n_skip += 1
                    #     continue
                    # n_skip = 0
                    label[0] = 1 # normal 
                    sample_category_stats[0] += 1
                
                dataset.append({"vid": vid, 
                                "label_id": class_id,
                                "label": label, 
                                "start": int(seq_start), # NOTE: 0-index
                                "end": int(seq_end),# NOTE: 0-index
                                "num_frames": num_frames
                                })
            
            # # NOTE: for over fitting on 10 videos
            # if idx >=1:
            #     break
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
        for i in range(start, end):
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
        start = data["start"]
        end = data["end"]

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