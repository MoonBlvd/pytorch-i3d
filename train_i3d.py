import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]= '0, 1, 2, 3'
import sys
import argparse
import logging
import time
import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torch.distributed as dist

from datasets.build import make_dataloader
import numpy as np
from pytorch_i3d import InceptionI3d
from torchvision.models.video import r3d_18, mc3_18, r2plus1d_18
from pytorch_c3d import C3D

# from charades_dataset import Charades as Dataset
from tqdm import tqdm
from model_serialization import load_state_dict
from detection.utils.logger import Logger
from detection.utils.metric_logger import MetricLogger
from detection.utils.comm import synchronize, get_rank

from mpi4py import MPI
import apex
from apex.parallel import DistributedDataParallel as DDP
from detection.utils.comm import get_world_size
from detection.utils.comm import is_main_process, all_gather, synchronize

from datasets.evaluation.evaluation import ActionClassificationEvaluator
from sklearn.metrics import ConfusionMatrixDisplay
import pickle as pkl
import pdb

np.set_printoptions(precision=3)

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        # print("all_losses:", all_losses)
        # print("\n")
        all_losses = torch.stack(all_losses, dim=0)
        if torch.isnan(torch.sum(all_losses)):
            pdb.set_trace()
        dist.reduce(all_losses, dst=0)
        if torch.isnan(torch.sum(all_losses)):
            pdb.set_trace()
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
            
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses

def loss_func(pred, target):
    # sigmoid
    # bce_loss = F.binary_cross_entropy_with_logits(pred, target)
    # softmax
    ce_loss = F.cross_entropy(pred, target)
    return ce_loss

def do_train(model_name,
             model, 
             train_dataloader, 
             val_dataloader, 
             device,
             checkpoint_peroid=1000, 
             save_model='', 
             logger=None, 
             distributed=False,
             evaluator=None):

    lr = 0.01 #0.003 #init_lr
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
    # NOTE: maybe the weight decay is too soon?
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [4000, 8000])

    num_steps_per_update = 1 #4 # accum gradient
    steps = 0

    tot_loc_loss = 0.0
    tot_cls_loss = 0.0

    if logger is None:
        logger = logging.getLogger("I3D.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    end = time.time()

    max_iters = len(train_dataloader)
    model.train()
    optimizer.zero_grad()
    # label_counts = defaultdict(int)
    for iters, data in enumerate(tqdm(train_dataloader)):
        data_time = time.time() - end
        iters += 1

        # get the inputs
        # inputs, labels, video_names, _, _ = data
        inputs, labels, video_names = data
        # wrap them in Variable
        inputs = Variable(inputs.to(device))
        t = inputs.size(2)
        labels = Variable(labels.to(device))
        per_frame_logits = model(inputs) # inputs: B X C X T X H X W
        # pdb.set_trace()
        if len(per_frame_logits.shape) == 3:
            per_frame_logits = per_frame_logits.mean(dim=-1) # B X C

        loss = loss_func(per_frame_logits, labels)
        # track time
        batch_time = time.time() - end
        
        # reduce losses over all GPUs for logging purposes
        loss_dict = {"loss_cls": loss} #{"loss_loc": loc_loss, "loss_cls": cls_loss}
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = loss_dict_reduced['loss_cls'] #0.5 * loss_dict_reduced['loss_loc'] + 0.5 * loss_dict_reduced['loss_cls']

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

        meters.update(loss=losses_reduced, **loss_dict_reduced)
        meters.update(time=batch_time, data=data_time)

        # estimate the rest of the running time
        eta_seconds = meters.time.global_avg * (max_iters - iters)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iters % num_steps_per_update == 0 or iters == len(train_dataloader):
            steps += 1
            optimizer.step()
            optimizer.zero_grad()

            lr_sched.step()
            if steps % 1 == 0:
                # NOTE: Add log file 
                info = meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "iter: {iter}",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        iter=iters,
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                
                if hasattr(logger, 'log_values'):
                    logger.info(info) 
                    for name, meter in meters.meters.items():
                        logger.log_values({name: meter.median}, step=iters)
                    logger.log_values({"grad_norm": grad_norm}, step=iters)
                else:
                    print(info)
            
            if steps % checkpoint_peroid == 0:
                del inputs, loss
                model.eval()
                do_val(model_name,
                       model, 
                       val_dataloader, 
                       device,
                       distributed, 
                       logger, 
                       output_dir=os.path.join(save_model, 'inference'),
                       train_iters=iters,
                       evaluator=evaluator)
                model.train()
                
                if get_rank() == 0: # only save model for rank0
                    if not os.path.isdir(save_model):
                        os.makedirs(save_model)
        
                    save_dir = os.path.join(save_model, str(steps).zfill(6)+'.pt')
                    if hasattr(model, 'module'):
                        torch.save(model.module.state_dict(), save_dir)
                    else:
                        torch.save(model.state_dict(), save_dir)

        end = time.time()
def do_val(model_name,
           model, 
           val_dataloader, 
           device, 
           distributed=False,
           logger=None, 
           output_dir='', 
           train_iters=0, 
           evaluator=None):
    if logger is None:
        logger = logging.getLogger("I3D.trainer")

    torch.cuda.empty_cache()  # TODO check if it helps

    tot_loc_loss = 0.0
    tot_cls_loss = 0.0
    
    # run on dataset
    results = defaultdict(list)
    target_labels = {}
    for iters, data in enumerate(tqdm(val_dataloader)):
        # get the inputs
        # inputs, labels, video_names, start, end = data
        inputs, labels, video_names= data
        # wrap them in Variable
        inputs = inputs.to(device)
        t = inputs.size(2)
        labels = labels.to(device)

        per_frame_logits = model(inputs)
        if len(per_frame_logits.shape) == 3:
            per_frame_logits = per_frame_logits.mean(dim=-1)

        loss = loss_func(per_frame_logits, labels)
        loss = loss.item() #(0.5*loc_loss + 0.5*cls_loss)
        tot_cls_loss += loss
        # collect results
        per_frame_logits = F.softmax(per_frame_logits, dim=1).detach().cpu()
        for batch_id, vid in enumerate(video_names):
            # frame_id = int((start[batch_id] + end[batch_id])/2)
            # pred = per_frame_logits[batch_id]
            # if vid not in results:
            #     results[vid] = {}
            # assert frame_id not in results[vid]
            # results[vid][frame_id] = pred
            pred = per_frame_logits[batch_id]
            results[vid].append(pred)
            target_labels[vid] = int(labels[batch_id].detach().cpu())

    if hasattr(logger, 'log_values'):
        logger.log_values({"loss_cls_val": tot_cls_loss/(iters+1)}, step=train_iters)   

    results = _accumulate_from_multiple_gpus(results)
    if not is_main_process():
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # torch.save(results, os.path.join(output_dir, 'predictions.pth'))
    # Run video-level evaluation
    per_vid_confusion_matrix = np.zeros((16, 16))
    per_clip_confusion_matrix = np.zeros((16, 16))
    per_vid_top3_acc = np.zeros(16)
    per_clip_top3_acc = np.zeros(16)
    per_class_clip_num = np.zeros(16)
    # Save per video prediction
    all_per_vid_results = {}
    for vid in results.keys():
        # ----------
        # NOTE: add clip-level accuracy
        clip_results = torch.stack(results[vid], dim=0)
        for per_clip_result in clip_results:
            _, sorted_pred = torch.sort(per_clip_result, descending=True)
            top1_pred = sorted_pred[0]
            top3_pred = sorted_pred[:3]
            y_true = target_labels[vid]
            per_class_clip_num[y_true] += 1
            per_clip_confusion_matrix[y_true, top1_pred] += 1
            if y_true in top3_pred:
                per_clip_top3_acc[y_true] += 1
        # ---------

        # ---------
        # per video results
        per_vid_results = torch.stack(results[vid], dim=0).mean(dim=0)
        all_per_vid_results[vid] = per_vid_results
        _, sorted_pred = torch.sort(per_vid_results, descending=True)
        top1_pred = sorted_pred[0]
        top3_pred = sorted_pred[:3]
        y_true = target_labels[vid]
        per_vid_confusion_matrix[y_true, top1_pred] += 1
        if y_true in top3_pred:
            per_vid_top3_acc[y_true] += 1
        # ---------
    
    # Save per video prediction
    pkl.dump(all_per_vid_results, open(os.path.join('/u/bryao/work/DATA/i3d_outputs', model_name, 'per_video_results.pkl'), 'wb'))

    # per clip results
    per_clip_top_1_acc = per_clip_confusion_matrix.diagonal() / (per_clip_confusion_matrix.sum(axis=1) + 1e-6)
    per_clip_top_3_acc = np.array([num/per_class_clip_num[i] for i, num in enumerate(per_clip_top3_acc)])
    logger.info("Clip-level evalutaion:")
    per_clip_result = "Top 1 acc: {}, Top 3 acc: {}, per_cls_acc: {}".format(np.around(per_clip_top_1_acc.mean(), 3), 
                                                                       np.around(per_clip_top_3_acc.mean(), 3), 
                                                                       np.around(per_clip_top_1_acc, 3))
    logger.info(per_clip_result)
    print(per_clip_result)
    print("confusion matrix:", np.around(per_clip_confusion_matrix, 3))
    # np.save('/u/bryao/work/DATA/i3d_outputs/r2plus1d_18/confusion_matrix_7000.npy', per_clip_confusion_matrix)
    if hasattr(logger, 'log_values'):
        logger.log_values({'Per clip Top 1 Acc': per_clip_top_1_acc.mean()}, step=train_iters)
        logger.log_values({'Per clip Top 3 Acc': per_clip_top_3_acc.mean()}, step=train_iters)

    # per video results
    data_category_stats = val_dataloader.dataset.data_category_stats
    top_1_acc = per_vid_confusion_matrix.diagonal() / (per_vid_confusion_matrix.sum(axis=1) + 1e-6)
    top_3_acc = np.array([num/data_category_stats[i] for i, num in enumerate(per_vid_top3_acc)])
    logger.info("Video-level evalutaion:")
    per_vid_result = "Top 1 acc: {}, Top 3 acc: {}, per_cls_acc: {}".format(np.around(top_1_acc.mean(), 3), 
                                                                       np.around(top_3_acc.mean(), 3), 
                                                                       np.around(top_1_acc, 3))
    logger.info(per_vid_result)
    print(per_vid_result)
    if hasattr(logger, 'log_values'):
        logger.log_values({'Per vid Top 1 Acc': top_1_acc.mean()}, step=train_iters)
        logger.log_values({'Per vid Top 3 Acc': top_3_acc.mean()}, step=train_iters)


    # # Run video-level evaluation
    # eval_results = evaluator.evaluate(results)
    # for k, v in eval_results.items():
    #     logger.info('{}:{}'.format(k, v))
    #     if isinstance(v, (float, int)) and hasattr(logger, 'log_values'):
    #         logger.log_values(eval_results, step=train_iters)

    # if 'confusion_matrix' in eval_results:
    #     cm_display = ConfusionMatrixDisplay(eval_results['confusion_matrix'], 
    #                         display_labels=val_dataloader.dataset.name_shorts)
    #     ret = cm_display.plot(fontsize=6)
    #     logger.log_plot(ret.figure_, label='Confusion Matrix',step=train_iters)

def _accumulate_from_multiple_gpus(item_per_gpu):
    # all_keys
    all_items = all_gather(item_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_items:
        predictions.update(p)
    return predictions

def run(model_name='i3d',
        init_lr=0.1, 
        gpu_id=0,
        max_steps=64e3, 
        batch_per_gpu=4,
        mode='rgb', 
        root='/home/data/vision7/A3D_2.0/frames/', #'/ssd/Charades_v1_rgb', 
        train_split='A3D_2.0_train.json', #'charades/charades.json', 
        val_split='A3D_2.0_val.json',
        checkpoint_peroid=1000,
        save_model='',
        with_normal=True):

    
    num_gpus = MPI.COMM_WORLD.Get_size()
    distributed = False
    if num_gpus > 1:
        distributed = True

    local_rank = MPI.COMM_WORLD.Get_rank() % torch.cuda.device_count()

    if distributed:
        torch.cuda.set_device(local_rank)
        host = os.environ["MASTER_ADDR"] if "MASTER_ADDR" in os.environ else "127.0.0.1"
        torch.distributed.init_process_group(
            backend="nccl",
            init_method='tcp://{}:12345'.format(host),
            rank=MPI.COMM_WORLD.Get_rank(),
            world_size=MPI.COMM_WORLD.Get_size()
        )

        synchronize()
    # logger must be initialized after distributed!
    cfg = {'PROJECT': 'i3d_a3d'}
    if args.use_wandb:
        logger = Logger("I3D",
                        cfg,#convert_to_dict(cfg, []),
                        project = 'i3d_a3d',
                        viz_backend="wandb" 
                        )
    else:
        logger = logging.Logger('I3D')

    logger.info("Using {} GPUs".format(num_gpus))
    # setup dataset

    train_dataloader = make_dataloader(root,
                                       train_split, 
                                       mode,
                                       model_name,
                                       seq_len=16, #64,
                                       overlap=15, #32,
                                       phase='train', 
                                       max_iters=10000, 
                                       batch_per_gpu=batch_per_gpu, #8,
                                       num_workers=16, 
                                       shuffle=True, 
                                       distributed=distributed,
                                       with_normal=with_normal)

    val_dataloader = make_dataloader(root,
                                     val_split, 
                                     mode,
                                     model_name,
                                     seq_len=16, #64, 
                                     overlap=15, #32,
                                     phase='val', 
                                     max_iters=None, 
                                     batch_per_gpu=batch_per_gpu, #8,
                                     num_workers=16, 
                                     shuffle=False, 
                                     distributed=distributed,
                                     with_normal=with_normal)
    # setup the model
    # set  dropout_keep_prob=0.0 for overfit
    logger.info("Running {} model".format(model_name))
    if model_name == 'i3d':
        if mode == 'flow':
            model = InceptionI3d(train_dataloader.dataset.num_classes, in_channels=2, dropout_keep_prob=0.5)
            load_state_dict(model, torch.load('models/flow_imagenet.pt'), ignored_prefix='logits')
            logger.info("Loaded pretrained I3D")
        else:
            model = InceptionI3d(train_dataloader.dataset.num_classes, in_channels=3, dropout_keep_prob=0.5)
            load_state_dict(model, torch.load('models/rgb_imagenet.pt'), ignored_prefix='logits')
            logger.info("Loaded pretrained I3D")
        model.replace_logits(train_dataloader.dataset.num_classes)
    elif model_name == 'r3d_18':
        model = r3d_18(pretrained=True, num_classes=train_dataloader.dataset.num_classes)
    elif model_name == 'mc3_18':
        model = mc3_18(pretrained=True, num_classes=train_dataloader.dataset.num_classes)
    elif model_name == 'r2plus1d_18':
        model = r2plus1d_18(pretrained=True, num_classes=train_dataloader.dataset.num_classes)
    elif model_name == 'c3d':
        model = C3D(pretrained=True, num_classes=train_dataloader.dataset.num_classes)
    else:
        raise NameError('unknown model name:{}'.format(model_name))
    if hasattr(logger, 'run_id'):
        run_id = logger.run_id
    else:
        run_id = 'no_wandb'
    
    save_model = os.path.join(save_model, model_name, run_id)

    # Create evaluator
    evaluator = ActionClassificationEvaluator(cfg=None,
                                              dataset=val_dataloader.dataset,
                                              split='val',
                                              mode='accuracy',#'mAP',
                                              output_dir=save_model,
                                              with_normal=with_normal)

    # device = torch.device('cuda')
    device = torch.device('cuda:{}'.format(gpu_id))
    model.to(device)
    if distributed:
        model = apex.parallel.convert_syncbn_model(model)
        model = DDP(model.to(device), delay_allreduce=True)
    # ckpt = '/home/data/vision7/brianyao/DATA/i3d_outputs/i3d/tg9s4ff5/004500.pt'
    # load_state_dict(model, torch.load(ckpt))

    # #######NOTE: only for testing checkpoint is correct or not ##
    # model.eval()
    # do_val(model, 
    #         val_dataloader, 
    #         device,
    #         distributed, 
    #         logger, 
    #         output_dir=os.path.join(save_model, 'inference'),
    #         train_iters=0,
    #         evaluator=evaluator)
    # pdb.set_trace()
    # #########
    do_train(model_name,
             model, 
             train_dataloader, 
             val_dataloader, 
             device=device,
             checkpoint_peroid=checkpoint_peroid, 
             save_model=save_model, 
             logger=logger,
             evaluator=evaluator)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_name', type=str, help='name of the model to run')
    parser.add_argument('-mode', type=str, help='rgb or flow')
    parser.add_argument('-save_model', type=str)
    parser.add_argument('-root', type=str)
    parser.add_argument('-train_split', type=str)
    parser.add_argument('-val_split', type=str)
    parser.add_argument('-batch_per_gpu', type=int)
    parser.add_argument('-gpu', type=int)
    parser.add_argument('-checkpoint_peroid', type=int)
    parser.add_argument('-use_wandb', const=True, nargs='?')
    args = parser.parse_args()
    
    # need to add argparse
    run(model_name=args.model_name,
        mode=args.mode, 
        gpu_id=args.gpu,
        batch_per_gpu=args.batch_per_gpu,
        train_split=args.train_split, 
        val_split=args.val_split, 
        root=args.root, 
        save_model=args.save_model, 
        checkpoint_peroid=args.checkpoint_peroid,
        with_normal=False
        )
