import torch
from torch.utils.data import DataLoader

from .A3D import A3D as Dataset
from . import videotransforms as T
from .build_samplers import make_data_sampler, make_batch_data_sampler

def make_dataloader(root, split, mode='rgb', phase='train', max_iters=None, batch_per_gpu=1, num_workers=0, shuffle=True, distributed=False,seq_len=16, overlap=0):
    

    if phase == 'train':
        transforms = T.Compose([T.Resize(min_size=(240,), max_size=320),
                                T.RandomHorizontalFlip(p=0.5),
                                T.ToTensor(),
                                T.Normalize(mean=None, std=None, to_bgr255=False)])
    elif phase in ['val', 'test']:
        transforms = T.Compose([T.Resize(min_size=(240,), max_size=320),
                                T.ToTensor(),
                                T.Normalize(mean=None, std=None, to_bgr255=False)])
    else:
        raise NameError()
    dataset = Dataset(split, 
                      phase, 
                      root, 
                      mode, 
                      transforms,
                      seq_len=seq_len, 
                      overlap=overlap 
                      )

    sampler = make_data_sampler(dataset, shuffle=shuffle, distributed=distributed)
    batch_sampler = make_batch_data_sampler(dataset, 
                                            sampler, 
                                            aspect_grouping=False, 
                                            batch_per_gpu=batch_per_gpu,
                                            max_iters=max_iters, 
                                            start_iter=0, 
                                            dataset_name='A3D')
    dataloader =  DataLoader(dataset, 
                            num_workers=num_workers, 
                            batch_sampler=batch_sampler)

    return dataloader