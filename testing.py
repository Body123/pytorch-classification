import os
import sys
import random
#import builtins

import torch
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from munch import munchify
from torch.utils.tensorboard import SummaryWriter
from utils.func import *
from train import train, evaluate
from utils.metrics import Estimator
from data.builder import generate_dataset
from modules.builder import generate_model

def main():
    args = parse_config()
    cfg = load_config(args.config)
    if cfg['base']['HPO']:
        hyperparameter_tuning(cfg)
    cfg = munchify(cfg)
    #config_check(cfg)

    worker(0, 1, cfg)


def worker(gpu, n_gpus, cfg):
    if cfg.dist.distributed:
        torch.cuda.set_device(gpu)
        cfg.dist.gpu = gpu
        cfg.dist.rank = cfg.dist.rank * n_gpus + gpu
        dist.init_process_group(
            backend=cfg.dist.backend,
            init_method='env://',
            world_size=cfg.dist.world_size,
            rank=cfg.dist.rank
        )
        torch.distributed.barrier()

        cfg.train.batch_size = int(cfg.train.batch_size / cfg.dist.world_size)
        cfg.train.num_workers = int((cfg.train.num_workers + n_gpus - 1) / n_gpus)


    if cfg.base.random_seed != -1:
        seed = cfg.base.random_seed + cfg.dist.rank # different seed for different process if distributed
        set_random_seed(seed, cfg.base.cudnn_deterministic)

    _, test_dataset, val_dataset = generate_dataset(cfg)
    # thresholds on each class if bigger than 0.5 than 1, bigger than 1.5 then 2, used in case
    # the loss is mean squared/absolute error, smooth_l1
    estimator = Estimator(cfg.train.criterion, cfg.data.num_classes)

    # test
    print('This is the performance of the best validation model:')
    model = generate_model(cfg)
    evaluate(cfg, model, val_dataset, estimator)

    print('This is the performance on the test data:')
    model = generate_model(cfg)
    evaluate(cfg, model, test_dataset, estimator, test_file=True)


def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic


def hyperparameter_tuning(cfg):
    import nni
    params = nni.get_next_parameter()
    config_update(cfg, params)
    print_msg('Hyper-parameters optimization mode.', appendixs=params.keys())


if __name__ == '__main__':
    main()
