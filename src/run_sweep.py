""" Module run.py (By: Charley Zhang, Nov 2020)
Is called by shell scripts to run experiments.
Main job: call appropriate experiment main.py file and pass on env args.
    (1) Get config file & validate settings
    (2) Parse GPU device info
    (3) Set experiment seed
    (4) Run experiment via the corresponding emain.py

Updates
-------
(2020.11)
  - Added negative seeds for random seed sampling.
  - Added sun grid array job submissions. 
  - Improved exception catching in main experiment to encompass all exceptions.
"""

import sys, os
import pathlib
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

import lib
from lib.utils.train.configs import get_config

# Experiment code modules
import experiments

from run import parse_cfg, EXPERIMENTS


# ⭐ Sweep Setup ⭐
CFG_FILE = 'ftmmwhs_train.yaml'


def run_cli():
    config = CFG_FILE
    distributed = False
    
    # --- ##  Get experiment configuration  ## --- #
    given_cfg_path = pathlib.Path(config)
    if given_cfg_path.exists():
        print(f'[CFG] Given cfg file "{str(given_cfg_path)}" exists! Loading..')
        cfg = get_config(config, merge_default=False, search_dir='')
    else:
        print(f'[CFG] Given cfg file "{str(given_cfg_path)}" does not exist! '
              'Searching for matching name in experiment\'s config folder..')
        curr_path = pathlib.Path(__file__).parent.absolute()
        exp_cfg_path = None
        for exp in EXPERIMENTS:
            if exp in given_cfg_path.name:
                exp_cfg_path = str(curr_path / 'experiments' / exp / 'configs')
                print(f' ✔ Cfg experiment matched at {exp_cfg_path}.')
                break

        if not exp_cfg_path:
            msg = (f'Given config file "{config}" does not contain any of the '
                   f'experiment names in them: {list(EXPERIMENTS.keys())}')
            raise ValueError(msg)
        cfg = cfg = get_config(config, merge_default=False, 
                               search_dir=exp_cfg_path)
    
    ## Sweep-specific configuration: set sampled & sweep paramters
    sweep_run = wandb.init()
    sweep_cfg = sweep_run.config
    
    def set_config_item(cfg, k, v):
        split_str = k.split('.')
        if len(split_str) == 1:
            assert k in cfg, f'{k} not in cfg ({cfg})'
            cfg[k] = v 
        else:
            set_config_item(cfg[split_str[0]], '.'.join(split_str[1:]), v)

    for k, v in dict(sweep_cfg).items():
        split_str = k.split('.')
        assert split_str[0] == 'sweep', f'{split_str}'
        set_config_item(cfg, '.'.join(split_str[1:]), v)

    cfg.experiment.sweep = True
    cfg.experiment.sweep_id = sweep_run.sweep_id or "unknown"
    cfg.experiment.sweep_run_name = sweep_run.name or sweep_run.id or "unknown"
    
    
    # --- ##  Handle array job submissions '-t'  ## --- #
    env_vars = dict(os.environ)
    task_id = env_vars['SGE_TASK_ID']
    
    if task_id != 'undefined':  
        # Change run number if experiment name ends with 'r\d' (e.g. _r1, _r2)
        exp_id = cfg.experiment.id
        matched = re.findall(r'r\d+', exp_id)
        if matched:
            cfg.experiment.id = exp_id.replace(matched[-1], f'r{task_id}')
        else:
            cfg.experiment.id = exp_id + f'_r{task_id}'
        print(f'[QSUB -t] Array job detected! Changing exp_id from {exp_id} to '
              f'{cfg.experiment.id}')
    else:
        print(f'[QSUB -t] No array job detected! exp_id remains '
              f'{cfg.experiment.id}')
    
    # --- ##  GPU device parsing and distributed training init  ## --- #
    gpu_indices = []
    gpu_env = os.getenv('SGE_HGR_gpu_card')
    if gpu_env:
        gpu_indices = [int(i) for i in gpu_env.strip().split(' ')]
        gpu_str_indices = [str(i) for i in gpu_indices]
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_str_indices)
        print(f'[GPUs] Cuda Visible: {os.getenv("CUDA_VISIBLE_DEVICES")}')
    
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
        print(f'[GPUs] Using device(s) with id(s): {gpu_indices}.')

    # --- ##  Final setup  ## --- #
    
    # Config checking & automatic value adjustments based on training env
    parse_cfg(cfg)
    
    # Experiment-specific arguments to pass to run_experiment.py
    exp_run_args = []
    experiment_main = EXPERIMENTS[cfg['experiment']['name']]
    
    if cfg.experiment.name in ('ftbcv', 'ftmmwhs'):
        exp_run_args.append(torch.multiprocessing.Queue())

    # Run within exception wrapper so processes can end gracefully
    try:
        if cfg.experiment.distributed:
            print(f'[RUN] Using distributed. Spawning {len(gpu_indices)} '
                  'processes.')
            spawn_args = tuple([cfg] + exp_run_args)
            torch.multiprocessing.spawn(
                experiment_main.run, 
                args=spawn_args,
                nprocs=len(gpu_indices))
        else:
            experiment_main.run(0, cfg, *exp_run_args)  # rank 0
    except BaseException as err:
        print(f'\nException thrown:\n', '-' * 30, f'\n{err}', sep='')
        print(f'\nTraceback:\n', '-' * 30, sep='')
        import traceback
        traceback.print_exc()

        print('\n\n' + '*' * 80 + '\n[END] Program Exit Cleanup Initiated!\n')
        kill_children()
        if cfg.experiment.distributed:
            torch.distributed.destroy_process_group()
    finally:
        print('🛑 Ended 🛑\n')


def kill_children():
    print(f'[END] Kill the kids!')
    import psutil
    child_processes = psutil.Process().children(recursive=True)
    for child in child_processes:
        print(f'[END] > Killing child process (PID={child.pid})')
        child.kill()


def set_seed(seed):
    if seed >= 0:
        print(f'[SEED] Setting seed to {seed}.')
    else:
        seed = random.randrange(2 ** 20)
        print(f'[SEED] Random seed not give, set to: {seed}')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # fixed input/model: ~5-10% speedup


def setup_dist(rank, world_size):
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
    )


if __name__ == '__main__':
    run_cli()