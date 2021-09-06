""" Module run.py (By: Charley Zhang, Nov 2020)
Is called by shell scripts to run experiments.
Main job: call appropriate experiment main.py file and pass on env args.
    (1) Get config file & validate settings
    (2) Parse GPU device info
    (3) Set experiment seed
    (4) Run experiment via the corresponding emain.py
"""

import sys, os
import signal
import math, random
import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)},
                    suppress=True,
                    precision=3,
                    linewidth=150)

import wandb
import click
import torch
import copy

import configs, experiments
from experiments.pgl import run_experiment as run_pgl
from experiments.finetune_bcv import run_experiment as run_finetune_bcv
# from experiments.finetune import run_experiment as run_finetune


USER_CHOICES = ("charzhar", "yzhang46")
EXPERIMENTS = {
    'pgl': run_pgl,
    'finetune_bcv': run_finetune_bcv,
}


@click.command()
@click.option('--config', required=True, type=click.Path(exists=True))
@click.option('-ddp', '--distributed', is_flag=True, 
              help='Flag to indicate whether to use distributed training.')
def run_cli(config, distributed):

    """ Get config + env setup. """
    cfg = configs.get_config(config)

    # GPU Device Parsing
    gpu_indices = []
    gpu_env = os.getenv('SGE_HGR_gpu_card')
    if gpu_env:
        gpu_indices = [int(i) for i in gpu_env.strip().split(' ')]
        gpu_str_indices = [str(i) for i in gpu_indices]
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_str_indices)
        print(f'[GPU(s)] Cuda Visible: {os.getenv("CUDA_VISIBLE_DEVICES")}')
    
    cfg.experiment.gpu_idxs = gpu_indices
    if distributed and len(gpu_indices) > 1:
        os.environ['MASTER_ADDR'] = 'localhost'              #
        os.environ['MASTER_PORT'] = '8888'
        cfg.experiment.distributed = True
    else:
        cfg.experiment.distributed = False
        cfg.experiment.rank = 0
        device = f'cuda' if torch.cuda.is_available() else 'cpu'
        cfg.experiment.device = device
        print(f' > Using device(s): {gpu_indices}.')

    parse_cfg(cfg)

    """ Additional experiment setup. """
    exp_args = []
    experiment_main = EXPERIMENTS[cfg['experiment']['name']]

    if 'finetune' in cfg.experiment.name:
        exp_args.append(torch.multiprocessing.Queue())

    try:
        if cfg.experiment.distributed:
            print(f' > Using distributed. Spawning {len(gpu_indices)} processes.')
            spawn_args = tuple([cfg] + exp_args)
            torch.multiprocessing.spawn(
                experiment_main.run, 
                args=spawn_args,
                nprocs=len(gpu_indices))
        else:
            experiment_main.run(0, cfg, *exp_args)  # rank 0 (N/A to run mode)
    except KeyboardInterrupt:
        import psutil

        print(f'Exiting! Killing the children.')
        child_processes = psutil.Process().children(recursive=True)
        for child in child_processes:
            print(f'Killing child process (PID={child.pid})')
            child.kill()
        if cfg.experiment.distributed:
            torch.distributed.destroy_process_group()


def parse_cfg(cfg):
    # Make sure 'best' is not in the experiment id or name
    if 'best' in cfg.experiment.id:
        raise ValueError(f'"best" cannot be in the experiment id!')
    if 'best' in cfg.experiment.name:
        raise ValueError(f'"best" cannot be in the experiment name!')

    # Adjust multi-gpu parameters
    N_gpus = len(cfg.experiment.gpu_idxs)
    if not cfg.experiment.distributed and N_gpus > 1:
        print(f'* {N_gpus} GPUs detected! Adjusting batch size and LR:')
        
        # Train and test (if exists) batch sizes
        tr_batch = cfg.train.batch_size
        cfg.train.batch_size = tr_batch * N_gpus
        print(f'  Train Batch: {tr_batch} -> {cfg.train.batch_size}')
        
        if 'test' in cfg and 'batch_size' in cfg.test:
            te_batch = cfg.test.batch_size
            cfg.test.batch_size = te_batch * N_gpus - 1
            print(f'  Test Batch: {te_batch} -> {cfg.test.batch_size}')
        
        # Data num_workers adjustment
        if cfg.train.num_workers > 0:
            inc = 2 if N_gpus == 2 else 3
            orig_workers = cfg.train.num_workers
            cfg.train.num_workers = orig_workers + inc
            print(f'  # Workers: {orig_workers} -> {cfg.train.num_workers}')

        # LR adjustment
        old_lr = cfg.train.optimizer.lr
        print(f'  LR: {old_lr} -> {N_gpus * old_lr}')
        cfg.train.optimizer.lr = old_lr * N_gpus

    # Adjust cfg if overfit minibatch
    if cfg.experiment.debug.overfitbatch:
        print(f'🚨 Overfitting a set of minibatches!')
        print(f'    Start-Epoch: {cfg.train.start_epoch} -> {0}')
        print(f'    Train-Epochs: {cfg.train.epochs} -> {50}')
        cfg.train.start_epoch = 0
        cfg.train.epochs = 50


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # fixed input/model: ~5-10% speedup


def setup_dist(rank, world_size):
    # Kill any lingering processing from previous runs
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
    )


if __name__ == '__main__':
    run_cli()