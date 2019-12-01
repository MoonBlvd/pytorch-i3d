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
    def __init__(self, split_file, split, root, mode, transforms=None, horizontal_flip=None, save_dir='', seq_len=16, overlap=0):
        
        self.split_file = split_file
        self.transforms = transforms
        self.mode = mode
        self.root = root
        self.save_dir = save_dir
        self.seq_len = seq_len
        self.overlap = overlap
        self.fps = 30
        self.num_classes = 11

        self.data = self.make_dataset(split_file, split, root, mode)
        

    def make_dataset(self, split_file, split, root, mode):
        dataset = []
        with open(split_file, 'r') as f:
            data = json.load(f)


        for vid in data.keys():
            if not data[vid]['video_start']:
                # NOTE: Sep 5, Some videos may have null video_start, meaning there is a bug and we skip the video for now
                continue
            if data[vid]['subset'] != split:
                continue
            if not os.path.exists(os.path.join(root, vid)):
                continue

            num_frames = data[vid]['num_frames']
            # init label
            labels = np.zeros([self.num_classes, num_frames], np.float32)        
            # normal label
            labels[0, :data[vid]['anomaly_start']] = 1
            # anomaly label
            labels[int(data[vid]['anomaly_class']), 
                   data[vid]['anomaly_start']:data[vid]['anomaly_end']] = 1 # binary classification
            # normal label
            labels[0, data[vid]['anomaly_end']:] = 1

            for t in range(0, num_frames, (self.seq_len - self.overlap)):
                if num_frames - t < self.seq_len:
                    seq_start = num_frames - self.seq_len
                    seq_end = num_frames
                else:
                    seq_start = t
                    seq_end = t + self.seq_len

                dataset.append({"vid": vid, 
                                "label": labels[:, seq_start: seq_end], 
                                "start": seq_start, # NOTE: 0-index
                                "end": seq_end# NOTE: 0-index
                                # "image_dir": 
                                })
            
            # if mode == 'flow':
            #     num_frames = num_frames//2
        print(len(dataset) ) 
        return dataset

    def load_rgb_frames(self, image_dir, vid, start, end):
        frames = []
        for i in range(start, end):
            # img = cv2.imread(os.path.join(image_dir, vid, 'images', str(i).zfill(6)+'.jpg'))[:, :, [2, 1, 0]]
            img = Image.open(os.path.join(image_dir, vid, 'images', str(i).zfill(6)+'.jpg'))
            w,h = img.size
            # if w < 226 or h < 226:
            #     d = 226.-min(w,h)
            #     sc = 1+d/min(w,h)
            #     img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc)
            # img = (img/255.)*2 - 1
            frames.append(img)
        return frames #torch.stack(frames, dim=1)

    # def load_flow_frames(image_dir, vid, start, num):
        #   frames = []
        #   for i in range(start, start+num):
        #     imgx = cv2.imread(os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'x.jpg'), cv2.IMREAD_GRAYSCALE)
        #     imgy = cv2.imread(os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'y.jpg'), cv2.IMREAD_GRAYSCALE)
            
        #     w,h = imgx.shape
        #     if w < 224 or h < 224:
        #         d = 224.-min(w,h)
        #         sc = 1+d/min(w,h)
        #         imgx = cv2.resize(imgx,dsize=(0,0),fx=sc,fy=sc)
        #         imgy = cv2.resize(imgy,dsize=(0,0),fx=sc,fy=sc)
                
        #     imgx = (imgx/255.)*2 - 1
        #     imgy = (imgy/255.)*2 - 1
        #     img = np.asarray([imgx, imgy]).transpose([1,2,0])
        #     frames.append(img)
        #   return np.asarray(frames, dtype=np.float32)

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
            imgs = self.load_rgb_frames(self.root, vid, start, end)
        else:
            imgs = self.load_flow_frames(self.root, vid, start, end)
        imgs, label = self.transforms(imgs, label)
        return imgs, label, vid, start, end

    def __len__(self):
        return len(self.data)
