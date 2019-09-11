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
from datasets.build import make_dataloader
import numpy as np
from pytorch_i3d import InceptionI3d

# from charades_dataset import Charades as Dataset
from tqdm import tqdm
from model_serialization import load_state_dict

import pdb

def do_train(i3d, train_dataloader, val_dataloader, checkpoint_peroid=1000, save_model=''):
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
                print('Training || iters: {} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(
                    iters, 
                    tot_loc_loss/(10*num_steps_per_update), 
                    tot_cls_loss/(10*num_steps_per_update), 
                    tot_loss/10))
                tot_loss = tot_loc_loss = tot_cls_loss = 0.
            
            if steps % checkpoint_peroid == 0:
                del inputs, loss
                do_val(i3d, val_dataloader)
                i3d.train()
                save_dir = os.path.join(save_model, str(steps).zfill(6)+'.pt')
                torch.save(i3d.module.state_dict(), save_dir)


def do_val(i3d, val_dataloader):
    i3d.eval()

    tot_loss = 0.0
    tot_loc_loss = 0.0
    tot_cls_loss = 0.0

    for iters, data in enumerate(tqdm(val_dataloader)):
       
        # get the inputs
        inputs, labels, _, _, _ = data

        # wrap them in Variable
        inputs = inputs.cuda()
        t = inputs.size(2)
        labels = labels.cuda()

        per_frame_logits = i3d(inputs)
        # upsample to input size
        per_frame_logits = F.upsample(per_frame_logits, t, mode='linear', align_corners=True)
        # compute localization loss
        loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
        tot_loc_loss += loc_loss.data#[0]

        # compute classification loss (with max-pooling along time B x C x T)
        cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0], torch.max(labels, dim=2)[0])
        tot_cls_loss += cls_loss.data#[0]

        loss = (0.5*loc_loss + 0.5*cls_loss)
        tot_loss += loss.data#[0]
    print('Validation || Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(
        tot_loc_loss/(iters+1), 
        tot_cls_loss/(iters+1), 
        tot_loss/(iters+1)))                
            

def run(init_lr=0.1, max_steps=64e3, mode='rgb', root='/ssd/Charades_v1_rgb', train_split='charades/charades.json', batch_size=8*5, save_model=''):
    # setup dataset
    
    train_dataloader = make_dataloader(root,
                                       train_split, 
                                       mode,
                                       phase='train', 
                                       max_iters=40000, 
                                       batch_per_gpu=4,
                                       num_workers=4, 
                                       shuffle=True, 
                                       distributed=False)
    val_dataloader = make_dataloader(root,
                                     train_split, 
                                     mode,
                                     phase='val', 
                                     max_iters=None, 
                                     batch_per_gpu=4,
                                     num_workers=4, 
                                     shuffle=True, 
                                     distributed=False)
    
    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(train_dataloader.dataset.num_classes, in_channels=2)
        load_state_dict(i3d, torch.load('models/flow_imagenet.pt'), ignored_prefix='logits')
    else:
        i3d = InceptionI3d(train_dataloader.dataset.num_classes, in_channels=3)
        load_state_dict(i3d, torch.load('models/rgb_imagenet.pt'), ignored_prefix='logits')
    i3d.replace_logits(train_dataloader.dataset.num_classes)

    do_train(i3d, train_dataloader, val_dataloader, checkpoint_peroid=1000, save_model=save_model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str, help='rgb or flow')
    parser.add_argument('-save_model', type=str)
    parser.add_argument('-root', type=str)
    parser.add_argument('-split', type=str)
    args = parser.parse_args()

    # need to add argparse
    run(mode=args.mode, train_split=args.split, root=args.root, save_model=args.save_model)
