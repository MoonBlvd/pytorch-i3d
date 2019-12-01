import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets
import videotransforms as T
from videotransforms import CenterCrop, Resize
import numpy as np

from pytorch_i3d import InceptionI3d
from torch.utils.data import DataLoader
# from charades_dataset_full import Charades as Dataset
from A3D import A3D as Dataset
from model_serialization import load_state_dict
import pdb

def run(max_steps=64e3, 
        mode='rgb', 
        root='/ssd2/charades/Charades_v1_rgb', 
        split='charades/charades.json', 
        batch_size=1, 
        load_model='', 
        save_dir=''):
    # setup dataset
    # test_transforms = T.Compose([videotransforms.CenterCrop(224)])
    test_transforms = T.Compose([T.Resize(min_size=(240,), max_size=320),
                                 T.ToTensor(),
                                 T.Normalize(mean=None, std=None, to_bgr255=False)])

    dataset = Dataset(split, 
                      'val', 
                      root, 
                      mode, 
                      test_transforms, 
                      save_dir=save_dir)

    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=0, 
                            pin_memory=True)

    dataloaders = {'val': dataloader}
    datasets = {'val': dataset}

    # val_dataset = Dataset(split, 
    #                       'testing', 
    #                       root, 
    #                       mode, 
    #                       test_transforms, 
    #                       save_dir=save_dir)
    # val_dataloader = DataLoader(val_dataset, 
    #                             batch_size=batch_size, 
    #                             shuffle=True, 
    #                             num_workers=8, 
    #                             pin_memory=True)    

    # dataloaders = {'train': dataloader, 'val': val_dataloader}
    # datasets = {'train': dataset, 'val': val_dataset}

    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(dataset.num_classes, in_channels=2)
    else:
        i3d = InceptionI3d(dataset.num_classes, in_channels=3)
    i3d.replace_logits(dataset.num_classes)
    load_state_dict(i3d, torch.load(load_model), ignored_prefix='logits')
    i3d.cuda()

    # for phase in ['train', 'val']:
    for phase in ['val']:
        i3d.eval()  # Set model to evaluate mode
                
        tot_loss = 0.0
        tot_loc_loss = 0.0
        tot_cls_loss = 0.0
        
        # Iterate over data.
        for data in dataloaders[phase]:
            # get the inputs
            inputs, labels, name, start, end = data
            feature_save_dir = os.path.join(save_dir, name[0])
            if not os.path.exists(feature_save_dir):
                os.makedirs(feature_save_dir)

            b,c,t,h,w = inputs.shape
            if t > 1600:
                features = []
                for start in range(1, t-56, 1600):
                    end = min(t-1, start+1600+56)
                    start = max(1, start-48)
                    ip = Variable(torch.from_numpy(inputs.numpy()[:,:,start:end]).cuda(), volatile=True)
                    features.append(i3d.extract_features(ip).squeeze(0).permute(1,2,3,0).data.cpu().numpy())
                np.save(os.path.join(save_dir, name[0]), np.concatenate(features, axis=0))
            else:
                # wrap them in Variable
                inputs = Variable(inputs.cuda(), volatile=True)
                features = i3d.extract_features(inputs)
                np.save(os.path.join(feature_save_dir, str(int(start)) + '_' + str(int(end)) + '.npy'), features.squeeze().data.cpu().numpy())
                        # features.squeeze(0).permute(1,2,3,0).data.cpu().numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str, help='rgb or flow')
    parser.add_argument('-load_model', type=str)
    parser.add_argument('-split', type=str)
    parser.add_argument('-root', type=str)
    parser.add_argument('-gpu', type=str)
    parser.add_argument('-save_dir', type=str)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    # need to add argparse
    run(mode=args.mode, split=args.split, root=args.root, load_model=args.load_model, save_dir=args.save_dir)
