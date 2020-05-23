import os
import numpy as np 
import torch
import torch.nn.functional as F
import torch.optim as optim

from classifier import Classifier
from datasets.A3D_classifier_data import make_classifier_dataloader
import pdb
from tqdm import tqdm

def loss_func(pred, target):
    # softmax
    ce_loss = F.cross_entropy(pred, target)
    return ce_loss

def do_val(model, val_dataloader, iters=None):
    confusion_matrix = np.zeros((17,17))
    top_3_acc = np.zeros(17)
    
    for batch in tqdm(val_dataloader):
        feat, label = batch
        logits = model(feat)
        sorted_pred = torch.argsort(logits, dim=1, descending=True)
        pred = sorted_pred[:, 0]
    
        for l, p, sorted_p in zip(label, pred, sorted_pred):
            confusion_matrix[l][p] += 1
            # get top 3
            if l in sorted_p[:3]:
                top_3_acc[l] += 1
    
    acc = confusion_matrix.diagonal() / (confusion_matrix.sum(axis=1) + 1e-5)
    top_3_acc = top_3_acc / (confusion_matrix.sum(axis=1) + 1e-5)
    print("Iters:{} \t \
           Top 1 Accuracy: {} \t mean Top 1 Accuracy: {:.4f} \t \
           Top 3 Accuracy: {} \t mean Top 3 Accuracy: {:.4f}".format(iters, 
                                                                        acc[1:], 
                                                                        acc[1:].mean(),
                                                                        top_3_acc[1:],
                                                                        top_3_acc[1:].mean()))
    return acc, acc.mean()

def do_train():
    model_name = 'c3d'
    input_size = {'i3d':1024, 'c3d':4096, 'r3d_18':512, 'mc3_18':512, 'r2plus1d_18':512}
    shuffle = True
    distributed = False
    lr = 0.01 #init_lr
    batch_per_gpu = 64
    num_workers = 24 
    max_iters = 20000
    checkpoint_period = 500
    save_dir = os.path.join('/home/data/vision7/brianyao/DATA/classifier_ckpt', model_name)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = Classifier(input_size[model_name], num_classes=17)

    train_dataloader = make_classifier_dataloader(model_name,
                                                    shuffle=shuffle, 
                                                    distributed=distributed, 
                                                    is_train=True, 
                                                    batch_per_gpu=batch_per_gpu, 
                                                    num_workers=num_workers, 
                                                    max_iters=max_iters)
    val_dataloader = make_classifier_dataloader(model_name,
                                                    shuffle=False, 
                                                    distributed=distributed, 
                                                    is_train=False, 
                                                    batch_per_gpu=batch_per_gpu, 
                                                    num_workers=num_workers, 
                                                    max_iters=None)
    
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    for iters, batch in enumerate(train_dataloader):
        iters += 1
        feat, label = batch
        logits = model(feat)
        loss = loss_func(logits, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iters % 20 == 0:
            print("iter:{}, loss:{}".format(iters, loss.item()))

        if iters % checkpoint_period == 0:
            acc, mean_acc = do_val(model, val_dataloader, iters=iters)
            ckpt_name = '{}_acc_{}.pth'.format(str(iters).zfill(6), mean_acc)
            ckpt_name = os.path.join(save_dir, ckpt_name)
            torch.save(model.state_dict(), ckpt_name)
                
if __name__ == '__main__':
    do_train()