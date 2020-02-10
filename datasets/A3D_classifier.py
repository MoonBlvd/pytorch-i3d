import os
import torch
import torch.utils.data as data_utl
import json

class A3DClassifier(data_utl.Dataset):
    def __init__(self, model='i3d', split='train'):
        self.name_to_id = {'normal': 0,
                            'ego: start_stop_or_stationary': 1, 
                            'ego: moving_ahead_or_waiting': 2, 
                            'ego: lateral': 3, 
                            'ego: oncoming': 4, 
                            'ego: turning': 5, 
                            'ego: pedestrian': 6, 
                            'ego: obstacle': 7, 
                            'ego: leave_to_right': 8, 
                            'ego: leave_to_left': 8, # NOTE: Jan 18, merge leave right/left as out-of-control
                            'other: start_stop_or_stationary': 9, 
                            'other: moving_ahead_or_waiting': 10, 
                            'other: lateral': 11, 
                            'other: oncoming': 12, 
                            'other: turning': 13, 
                            'other: pedestrian': 14, 
                            'other: obstacle': 15, 
                            'other: leave_to_right': 16, 
                            'other: leave_to_left': 16, 
                            'ego: unknown': 17,
                            'other: unknown': 18}

        anno_file = 'A3D_2.0_{}.json'.format(split)

        feature_root = '/home/data/vision7/A3D_2.0/vac_features/'
        feature_file_name = '{}_{}.pth'.format(model, split)
        feature_file_name = os.path.join(feature_root, feature_file_name)
        features = torch.load(feature_file_name)

        annos = json.load(open(anno_file, 'r'))
        self.data_list = []
        for vid, feat in tqdm(features):
            label = self.name_to_id[annos[vid]['anomaly_class']]
            self.data_list.append([vid, feat, label])
    def __getitem__(self, index):
        vid, feat, label = self.data_list[index]

        return feat, label

    def __len__(self):
        return len(self.data_list)

def make_classifier_dataloader(model,
                               shuffle, 
                               distributed, 
                               is_train, 
                               batch_per_gpu, 
                               num_workers, 
                               max_iters):
    if is_train:
        split = 'train'
    else:
        split = 'val'

    dataset = A3DClassifier(model=model, split=split)

    sampler = make_data_sampler(dataset, 
                                shuffle=shuffle, 
                                distributed=distributed, 
                                is_train=is_train)
    batch_sampler = make_batch_data_sampler(dataset, 
                                            sampler, 
                                            aspect_grouping=False, 
                                            batch_per_gpu=batch_per_gpu,
                                            max_iters=max_iters, 
                                            start_iter=0, 
                                            dataset_name='A3D_classifier')

    dataloader =  DataLoader(dataset, 
                            num_workers=num_workers, 
                            batch_sampler=batch_sampler)