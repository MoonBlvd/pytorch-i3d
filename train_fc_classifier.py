import torch
from classifier import Classifier
from datasets.A3D_classifier import make_classifier_dataloader

def loss_func(pred, target):
    # sigmoid
    # bce_loss = F.binary_cross_entropy_with_logits(pred, target)
    # softmax
    ce_loss = F.cross_entropy(pred, target)
    return ce_loss
    
model_name = 'i3d'
shuffle = True
distributed = False
batch_per_gpu = 64
num_workers = 24 
max_iters = 5000

model = Classifier(input_size, num_classes=17)


train_dataloader = make_classifier_dataloader(model_name,
                                                shuffle=shuffle, 
                                                distributed=distributed, 
                                                is_train=True, 
                                                batch_per_gpu=batch_per_gpu, 
                                                num_workers=num_workers, 
                                                max_iters=max_iters)
val_dataloader = make_classifier_dataloader(model_name,
                                                shuffle=shuffle, 
                                                distributed=distributed, 
                                                is_train=False, 
                                                batch_per_gpu=batch_per_gpu, 
                                                num_workers=num_workers, 
                                                max_iters=max_iters)

for iters, batch in train_dataloader():
    feat, label = batch
    logits = model(feat)
    
    loss = loss_func(logits, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if iters %20:
        logger.info("iter:{}, loss:{}".format(iters, loss.item()))
            
