# Thanks to TuZheng (LaneDet) https://github.com/Turoad/lanedet
from baseline.utils import Registry, build_from_cfg

import torch
from functools import partial
import numpy as np
import random

DATASETS = Registry('datasets')
PROCESS = Registry('process')

def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return torch.nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)

def build_dataset(split_cfg, cfg):
    return build(split_cfg, DATASETS, default_args=dict(cfg=cfg))

def worker_init_fn(worker_id, seed):
    worker_seed = worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def build_dataloader(split_cfg, cfg, is_train=True):
    if is_train:
        shuffle = True
        # batch_size = cfg.batch_size # uncomment when RuntimeError('each element in list of batch should be of equal size') happens
    else:
        shuffle = False
        # batch_size = 1 # uncomment when RuntimeError('each element in list of batch should be of equal size') happens

    dataset = build_dataset(split_cfg, cfg)

    init_fn = partial(
            worker_init_fn, seed=cfg.seed)
    
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size = cfg.batch_size, shuffle = shuffle,
        num_workers = cfg.workers, pin_memory = False, persistent_workers = True, drop_last = False,
        worker_init_fn=init_fn)
    
    # replace line 42-45 when RuntimeError('each element in list of batch should be of equal size') happens
    # data_loader = torch.utils.data.DataLoader(
    #     dataset, batch_size = batch_size, shuffle = shuffle,
    #     num_workers = cfg.workers, pin_memory = False, drop_last = False,
    #     worker_init_fn=init_fn)

    print(f"[DataLoader] Initialized with {cfg.workers} workers, batch_size={cfg.batch_size}")
    return data_loader
