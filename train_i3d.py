import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets, transforms
import videotransforms
import videotransforms as T

import numpy as np

from pytorch_i3d import InceptionI3d

# from charades_dataset import Charades as Dataset
from tqdm import tqdm
from A3D import A3D as Dataset
from model_serialization import load_state_dict
from build_samplers import make_data_sampler, make_batch_data_sampler
import pdb

def do_train(i3d, train_dataloader, checkpoint_peroid=1000, save_model=''):
    i3d.cuda()
    i3d = nn.DataParallel(i3d)

    lr = 0.001#init_lr
    optimizer = optim.SGD(i3d.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])


    num_steps_per_update = 4 # accum gradient
    steps = 0
    if not os.path.isdir(save_model):
        os.makedirs(save_model)
    
    tot_loss = 0.0
    tot_loc_loss = 0.0
    tot_cls_loss = 0.0

    i3d.train()
    for iters, data in enumerate(tqdm(train_dataloader)):
       
        optimizer.zero_grad()
        # get the inputs
        inputs, labels, _, _, _ = data

        # wrap them in Variable
        inputs = Variable(inputs.cuda())
        t = inputs.size(2)
        labels = Variable(labels.cuda())

        per_frame_logits = i3d(inputs)
        # pdb.set_trace()
        # upsample to input size
        per_frame_logits = F.upsample(per_frame_logits, t, mode='linear', align_corners=True)
        # pdb.set_trace()
        # compute localization loss
        loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
        tot_loc_loss += loc_loss.data#[0]

        # compute classification loss (with max-pooling along time B x C x T)
        cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0], torch.max(labels, dim=2)[0])
        tot_cls_loss += cls_loss.data#[0]

        loss = (0.5*loc_loss + 0.5*cls_loss)/num_steps_per_update
        tot_loss += loss.data#[0]
        loss.backward()

        if iters % num_steps_per_update==0:
            steps += 1
            optimizer.step()
            optimizer.zero_grad()
            lr_sched.step()
            if steps % 10 == 0:
                print('Training || iters: {} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(iters, tot_loc_loss/(10*num_steps_per_update), tot_cls_loss/(10*num_steps_per_update), tot_loss/10))
                # save model
                
                tot_loss = tot_loc_loss = tot_cls_loss = 0.
            
            if steps % checkpoint_peroid == 0:
                save_dir = os.path.join(save_model, str(steps).zfill(6)+'.pt')
                torch.save(i3d.module.state_dict(), save_dir)
    
def run(init_lr=0.1, max_steps=64e3, mode='rgb', root='/ssd/Charades_v1_rgb', train_split='charades/charades.json', batch_size=8*5, save_model=''):
    # setup dataset
    # train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
    #                                        videotransforms.RandomHorizontalFlip(),
    # ])
    # test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
    test_transforms = T.Compose([T.Resize(min_size=(240,), max_size=320),
                                 T.ToTensor(),
                                 T.Normalize(mean=None, std=None, to_bgr255=False)])

    
    # dataset = Dataset(train_split, 'training', root, mode, train_transforms)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=36, pin_memory=True)

    # val_dataset = Dataset(train_split, 'testing', root, mode, test_transforms)
    # val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=36, pin_memory=True)    

    # dataloaders = {'train': dataloader, 'val': val_dataloader}
    # datasets = {'train': dataset, 'val': val_dataset}
    dataset = Dataset(train_split, 
                      'training', 
                      root, 
                      mode, 
                      test_transforms, 
                      )
    max_iters = 10000
    sampler = make_data_sampler(dataset, shuffle=True, distributed=False)
    batch_sampler = make_batch_data_sampler(dataset, 
                                            sampler, 
                                            aspect_grouping=False, 
                                            segments_per_gpu=4,
                                            max_iters=max_iters, 
                                            start_iter=0, 
                                            dataset_name='A3D')

    # dataloader = DataLoader(dataset, 
    #                         batch_size=4,#batch_size, 
    #                         shuffle=True, 
    #                         num_workers=0, 
    #                         pin_memory=True)
    dataloader = DataLoader(dataset, 
                            num_workers=0, 
                            batch_sampler=batch_sampler)

    train_dataloader = dataloader
    train_dataset = dataset

    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(dataset.num_classes, in_channels=2)
        load_state_dict(i3d, torch.load('models/flow_imagenet.pt'), ignored_prefix='logits')
    else:
        i3d = InceptionI3d(dataset.num_classes, in_channels=3)
        load_state_dict(i3d, torch.load('models/rgb_imagenet.pt'), ignored_prefix='logits')
    i3d.replace_logits(dataset.num_classes)
    #i3d.load_state_dict(torch.load('/ssd/models/000920.pt'))

    do_train(i3d, train_dataloader, checkpoint_peroid=1000, save_model=save_model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str, help='rgb or flow')
    parser.add_argument('-save_model', type=str)
    parser.add_argument('-root', type=str)
    parser.add_argument('-split', type=str)
    args = parser.parse_args()

    # need to add argparse
    run(mode=args.mode, train_split=args.split, root=args.root, save_model=args.save_model)
