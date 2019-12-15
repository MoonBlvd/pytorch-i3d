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
                 mode, transforms=None, horizontal_flip=None, save_dir='', seq_len=16, overlap=0):
        
        self.split_file = split_file
        self.transforms = transforms
        self.mode = mode
        self.root = root
        self.save_dir = save_dir
        self.seq_len = seq_len
        self.overlap = overlap
        self.fps = 10
        self.num_classes = 10 # 9 known anomay type plus a normal, 0 is normal

        
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

        if split == 'train':
            self.data = self.make_train_dataset(split_file, split, root, mode)
            print("Number of used video:", len(self.data))
        elif split in ['val', 'test']:
            self.data = self.make_test_dataset(split_file, split, root, mode)

    def make_train_dataset(self, split_file, split, root, mode):
        dataset = []
        with open(split_file, 'r') as f:
            data = json.load(f)
        self.valid_videos = []

        sample_category_stats = {v:0 for v in self.name_to_id.values()}
        for idx, vid in enumerate(data.keys()):
            
            if data[vid]['video_start'] is None or \
               data[vid]['video_start'] is None or \
               data[vid]['anomaly_start'] is None or \
               data[vid]['anomaly_end'] is None:
                # NOTE: Sep 5, Some videos may have null video_start, meaning there is a bug and we skip the video for now
                continue
            if data[vid]['subset'] != split:
                continue
            if not os.path.exists(os.path.join(root, vid)):
                continue
            if int(data[vid]['anomaly_class']) == 10:
                # skip unknown
                continue
            num_frames = data[vid]['num_frames']
            if num_frames < self.seq_len:
                continue
            print("Training videos:", vid)
            self.valid_videos.append(vid)
            
            # NOTE: this is for the temporal label
            # init label
            labels = np.zeros([self.num_classes, num_frames], np.float32)        
            # normal label
            labels[0, :data[vid]['anomaly_start']] = 1
            # anomaly label
            labels[int(data[vid]['anomaly_class']), 
                   data[vid]['anomaly_start']:data[vid]['anomaly_end']] = 1 # binary classification
            # normal label
            labels[0, data[vid]['anomaly_end']:] = 1
            
            assert int(data[vid]['anomaly_class']) > 0
                
            for t in range(0, num_frames, (self.seq_len - self.overlap)):
                if num_frames - t < self.seq_len:
                    seq_start = num_frames - self.seq_len
                    seq_end = num_frames
                else:
                    seq_start = t
                    seq_end = t + self.seq_len
                
                # label = labels[:, seq_start: seq_end]
                
                # NOTE: for original I3D, one clip has only one label
                label = np.zeros(self.num_classes, np.float32) 
                # NOTE: method 1, assign the label of the middle frame
                # middle_idx = int(seq_end-seq_start/2)
                # if middle_idx >= data[vid]['anomaly_start'] and middle_idx < data[vid]['anomaly_end']:
                #     label[int(data[vid]['anomaly_class'])] = 1 # abnormal
                #     sample_category_stats[int(data[vid]['anomaly_class'])] += 1
                # else:
                #     label[0] = 1 # normal 
                #     sample_category_stats[0] += 1

                # NOTE: method 2, assign the accident label if over 1/3 of the frames are abnormal
                if sum(labels[:, seq_start:seq_end].nonzero()[0] > 0) >= self.seq_len/3:
                    label[int(data[vid]['anomaly_class'])] = 1 # abnormal
                    sample_category_stats[int(data[vid]['anomaly_class'])] += 1
                else:
                    label[0] = 1 # normal 
                    sample_category_stats[0] += 1

                dataset.append({"vid": vid, 
                                "label": label, 
                                "start": seq_start, # NOTE: 0-index
                                "end": seq_end,# NOTE: 0-index
                                # "image_dir": 
                                })
            
            # if mode == 'flow':
            #     num_frames = num_frames//2
            
            # NOTE: for over fitting on 10 videos
            if idx >=9:
                break
        print("Number of samples of all categories:")
        [print('{}:{}'.format(self.id_to_name[k], v)) for k, v in sample_category_stats.items()]
        return dataset

    def make_test_dataset(self, split_file, split, root, mode):
        
        dataset = []
        with open(split_file, 'r') as f:
            data = json.load(f)
        self.valid_videos = []
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
            
            print("Validating videos:", vid)
            num_frames = data[vid]['num_frames']

            self.valid_videos.append(vid)
            
            # # init label
            # labels = np.zeros([self.num_classes, num_frames], np.float32)        
            # # normal label
            # labels[0, :data[vid]['anomaly_start']] = 1
            # # anomaly label
            # labels[int(data[vid]['anomaly_class']), 
            #        data[vid]['anomaly_start']:data[vid]['anomaly_end']] = 1 
            # # normal label
            # labels[0, data[vid]['anomaly_end']:] = 1

            # NOTE: for original I3D, one clip has only one label
            label = np.zeros([self.num_classes], np.float32) 
            label[int(data[vid]['anomaly_class'])] = 1
            dataset.append({"vid": vid, 
                            "label": label, 
                            "start": 0, # NOTE: 0-index
                            "end": num_frames# NOTE: 0-index
                            })
            
            # if mode == 'flow':
            #     num_frames = num_frames//2
            if idx >=9:
                break
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

class A3DBinary(data_utl.Dataset):
    '''
    A3D dataset for I3D binary classification
    '''
    def __init__(self, 
                 split_file, 
                 split, 
                 root, 
                 mode, transforms=None, horizontal_flip=None, save_dir='', seq_len=16, overlap=0):
        
        self.split_file = split_file
        self.transforms = transforms
        self.mode = mode
        self.root = root
        self.save_dir = save_dir
        self.seq_len = seq_len
        self.overlap = overlap
        self.fps = 10
        self.num_classes = 2 # binary

        if split == 'train':
            self.data = self.make_train_dataset(split_file, split, root, mode)
            print("Number of used video:", len(self.data))
        elif split in ['val', 'test']:
            self.data = self.make_test_dataset(split_file, split, root, mode)

    def make_train_dataset(self, split_file, split, root, mode):
        dataset = []
        with open(split_file, 'r') as f:
            data = json.load(f)
        self.valid_videos = []

        sample_category_stats = {'normal':0, 'abnormal': 0}
        for idx, vid in enumerate(data.keys()):
            
            if data[vid]['video_start'] is None or \
               data[vid]['video_start'] is None or \
               data[vid]['anomaly_start'] is None or \
               data[vid]['anomaly_end'] is None:
                # NOTE: Sep 5, Some videos may have null video_start, meaning there is a bug and we skip the video for now
                continue
            if data[vid]['subset'] != split:
                continue
            if not os.path.exists(os.path.join(root, vid)):
                continue

            num_frames = data[vid]['num_frames']
            if num_frames < self.seq_len:
                continue
            print("Training videos:", vid)
            self.valid_videos.append(vid)
            
            # NOTE: this is for the temporal label
            # init label
            labels = np.zeros([2, num_frames], np.float32)        
            # normal label
            labels[0, :data[vid]['anomaly_start']] = 1
            # anomaly label
            labels[1, data[vid]['anomaly_start']:data[vid]['anomaly_end']] = 1 # binary classification
            # normal label
            labels[0, data[vid]['anomaly_end']:] = 1
            
            assert int(data[vid]['anomaly_class']) > 0
                
            for t in range(0, num_frames, (self.seq_len - self.overlap)):
                if num_frames - t < self.seq_len:
                    seq_start = num_frames - self.seq_len
                    seq_end = num_frames
                else:
                    seq_start = t
                    seq_end = t + self.seq_len
                
                # label = labels[:, seq_start: seq_end]
                
                # NOTE: for original I3D, one clip has only one label
                label = np.zeros(2, np.float32) 
                # NOTE: method 1, assign the label of the middle frame
                # middle_idx = int(seq_end-seq_start/2)
                # if middle_idx >= data[vid]['anomaly_start'] and middle_idx < data[vid]['anomaly_end']:
                #     label[int(data[vid]['anomaly_class'])] = 1 # abnormal
                #     sample_category_stats[int(data[vid]['anomaly_class'])] += 1
                # else:
                #     label[0] = 1 # normal 
                #     sample_category_stats[0] += 1

                # NOTE: method 2, assign the accident label if over 1/3 of the frames are abnormal
                if sum(labels[:, seq_start:seq_end].nonzero()[0] > 0) >= self.seq_len/3:
                    label[1] = 1 # abnormal
                    sample_category_stats['abnormal'] += 1
                else:
                    label[0] = 1 # normal 
                    sample_category_stats['normal'] += 1

                dataset.append({"vid": vid, 
                                "label": label, 
                                "start": seq_start, # NOTE: 0-index
                                "end": seq_end,# NOTE: 0-index
                                # "image_dir": 
                                })
            
            # if mode == 'flow':
            #     num_frames = num_frames//2
            
            # NOTE: for over fitting on 10 videos
            if idx >=9:
                break
        print("Number of samples of all categories:")
        print(sample_category_stats)
        return dataset

    def make_test_dataset(self, split_file, split, root, mode):
        
        dataset = []
        with open(split_file, 'r') as f:
            data = json.load(f)
        self.valid_videos = []
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
            
            print("Validating videos:", vid)
            num_frames = data[vid]['num_frames']

            self.valid_videos.append(vid)

            # NOTE: for original I3D, one clip has only one label
            label = np.zeros(2, np.float32) 
            label[1] = 1
            dataset.append({"vid": vid, 
                            "label": label, 
                            "start": 0, # NOTE: 0-index
                            "end": num_frames# NOTE: 0-index
                            })
            
            # if mode == 'flow':
            #     num_frames = num_frames//2
            if idx >=9:
                break
        return dataset

    def load_rgb_frames(self, image_dir, vid, start, end):
        frames = []
        for i in range(start, end):
            # img = cv2.imread(os.path.join(image_dir, vid, 'images', str(i).zfill(6)+'.jpg'))[:, :, [2, 1, 0]]
            img = Image.open(os.path.join(image_dir, vid, 'images', str(i).zfill(6)+'.jpg'))
            frames.append(img)
        return frames #torch.stack(frames, dim=1)


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