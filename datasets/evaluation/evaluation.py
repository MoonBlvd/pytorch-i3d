import os
import torch
import numpy as np
import glob
import json

from .activity_net_evaluator import ANETdetection
import pdb
from collections import defaultdict

class ActionClassificationEvaluator():
    '''evaluate action classification performance by'''
    def __init__(self, cfg, dataset, split='val', mode='accuracy', output_dir='', with_normal=True):
        '''anno_file is one json file contains all annotations of all video-level label'''
        self.cfg = cfg
        self.anno_file = dataset.split_file
        self.mode = mode
        self.with_normal = with_normal
        self.split = split

        # get per video GT
        self.gt = json.load(open(self.anno_file, 'r'))

        self.output_dir = output_dir

        # a list of video names that belong to a toy_dataset, otherwise None
        self.gt_video_classes = dataset.video_level_classes

        self.num_classes = dataset.num_classes

    def evaluate(self, pred):
        '''
        pred: dict of frame level prediction of all test or val videos
        '''
        eval_results = {}
        if self.mode == 'mAP':
            self.frame_level_mAP(pred)
            eval_results['AP'] = self.map_evaluator.ap
            eval_results['mAP'] = self.map_evaluator.mAP
            eval_results['Average_mAP'] = self.map_evaluator.average_mAP

        elif self.mode == 'accuracy':
            eval_results = self.video_level_Accuracy(pred)
        return eval_results
    
    def frame_level_mAP(self, predictions):
        # Prepare the prediction to the Activity Net format
        ANET_format_results = {'results':{}, 'version':'', 'external_data':None}
        for vid, pred in predictions.items():
            ANET_format_results['results'][vid] = []
            frame_ids = sorted(pred.keys())

            score, prev_label = pred[frame_ids[0]].sigmoid().max(dim=0)
            score = float(score)
            prev_label = int(prev_label)
            segment_result = {'label': prev_label, 'score':score, 'segment':[0]}
            for idx in frame_ids[1:]:
                score, label = pred[idx].sigmoid().max(dim=0)
                score = float(score)
                label = int(label)
                if label != prev_label or idx == frame_ids[-1]:
                    # print("idx:", idx)
                    # print("prev_label:", prev_label)
                    # print("label:", label)
                    
                    # save the old one
                    segment_result['segment'].append(idx)
                    length = segment_result['segment'][1] - segment_result['segment'][0]
                    segment_result['score'] /= length # NOTE: here the score is computed as the average score of the 
                    ANET_format_results['results'][vid].append(segment_result)

                    # start the new one
                    segment_result = {'label': label, 'score':score, 'segment':[idx]}
                else:
                    segment_result['score'] += score
                prev_label = label
        prediction_filename = os.path.join(self.output_dir, 'ANET_format_predictions.json')
        json.dump(ANET_format_results, open(prediction_filename,'w'))
        self.map_evaluator = ANETdetection(self.anno_file, 
                                            prediction_filename,
                                            subset=self.split,
                                            verbose=False,
                                            check_status=False,
                                            with_normal=self.with_normal)
        self.map_evaluator.evaluate()
        return

    def video_level_Accuracy(self, predictions):
        '''
        Compute accuracy: Assume no normal frames are used for evaluation
        '''
        TP_top_1 = 0
        TP_top_3 = 0
        num_predictions = 0
        per_class_correct_pred = {i:0 for i in range(self.num_classes) if i  > 0}
        per_class_num_gt = {i:0 for i in range(self.num_classes) if i  > 0}
        confusion_matrix = np.zeros((self.num_classes, self.num_classes))

        if self.with_normal:
            per_class_correct_pred[0] = 0
            per_class_num_gt[0] = 0
        for vid, prediction in predictions.items():
            gt_cls = int(self.gt_video_classes[vid]['class_id'])
            for frame_id, pred in prediction.items():
                sorted_score, sorted_cls_id = pred.softmax(dim=-1).sort(descending=True)
                top_1 = sorted_cls_id[:1].tolist()
                top_3 = sorted_cls_id[:3].tolist()
                if gt_cls in top_1:
                    TP_top_1 += 1
                    per_class_correct_pred[gt_cls] += 1
                if gt_cls in top_3:
                    TP_top_3 += 1
                num_predictions += 1
                per_class_num_gt[gt_cls] += 1

                confusion_matrix[gt_cls, sorted_cls_id[0]] += 1

        accuracy_top_1 = TP_top_1 / num_predictions
        accuracy_top_3 = TP_top_3 / num_predictions
        confusion_matrix = confusion_matrix / np.sum(confusion_matrix, axis=1, keepdims=True)
        confusion_matrix = np.around(confusion_matrix, decimals=2)

        per_class_accuracy = {}
        for class_id, num_correct_pred in per_class_correct_pred.items():
            if per_class_num_gt[class_id] > 0:
                per_class_accuracy['Class_{}_accuracy'.format(class_id)] = num_correct_pred/per_class_num_gt[class_id]
            else:
                per_class_accuracy['Class_{}_accuracy'.format(gt_cls)] = 0
                
        eval_result = {'Total accuracy top 1':accuracy_top_1,
                     'Total accuracy top 3':accuracy_top_3}
        eval_result.update(per_class_accuracy)
        eval_result['confusion_matrix'] = confusion_matrix[1:, 1:] # ignore Normal
        return eval_result
