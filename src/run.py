""" Module run.py (By: Charley Zhang, Nov 2020)
Is called by shell scripts to run experiments.
Main job: call appropriate experiment main.py file and pass on env args.
    (1) Get config file & validate settings
    (2) Parse GPU device info
    (3) Set experiment seed
    (4) Run experiment via the corresponding emain.py
"""

import sys, os
import math, random
import numpy as np

import wandb
import click
import torch

import configs, experiments
from experiments.finetune import finetune as finetune_main
from experiments.pretrain import byol as byol_main


USER_CHOICES = ("charzhar", "yzhang46")
EXPERIMENTS = {
    'finetune': finetune_main,
    'byol': byol_main
}


@click.command()
@click.option("--user", type=click.Choice(USER_CHOICES, case_sensitive=True))
@click.option("--gpu")
@click.option("--config", type=click.Path(exists=True))
@click.option("--checkpoint", type=str)
def run_cli(user, gpu, config, checkpoint=None):
    if checkpoint:
        assert os.path.isfile(checkpoint), \
        f"Checkpoint doesn't exist ({checkpoint})"

    """ Get config + env setup. """
    cfg = configs.get_config(config)
    validate_cfg(cfg)
    gpu_indices = [int(i) for i in gpu.strip().split(',')]

    # experiment setup
    _set_seed(cfg['experiment']['seed'])

    device = f'cuda' if torch.cuda.is_available() else 'cpu'
    cfg['experiment']['device'] = device
    cfg['experiment']['gpu_idxs'] = gpu_indices
    print(f" > Using device(s): {gpu_indices}.")

    if len(gpu_indices) > 1:
        cfg['train']['batch_size'] = cfg['train']['batch_size'] * len(gpu_indices)
        cfg['test']['batch_size'] = cfg['test']['batch_size'] * len(gpu_indices)
        if 'batch_lab_size' in cfg['train']:
            cfg['train']['batch_lab_size'] *= len(gpu_indices)
        # cfg['train']['optimizer']['lr'] *= len(gpu_indices)
    
    experiment_main = EXPERIMENTS[cfg['experiment']['name']]
    experiment_main.run(cfg, checkpoint=checkpoint)


def validate_cfg(cfg):
    pass


def _set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # fixed input/model: ~5-10% speedup


if __name__ == '__main__':
    run_cli()