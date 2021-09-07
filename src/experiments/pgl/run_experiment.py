""" Module run_experiment.py (Author: Charley Zhang, 2021)
Experiment for PGL baselines.

Used for:
  - Training a model via PGL or BYOL.
"""

import sys, os
import math, random
import pathlib
import time
import pprint
import warnings
import itertools
import inspect
import multiprocessing
import numpy as np
import pandas as pd
import collections
from collections import namedtuple

import torch, torchvision
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import SimpleITK as sitk

import experiments, lib
from experiments import setup
from experiments.ftbcv import data_setup
from lib.utils import devices, timers, statistics
from lib.utils.train import ramps
from lib.utils.io import output
from lib.nets import init as init_net
from lib.nets.ema import create_ema_model, update_ema_model


WATCH = timers.StopWatch()

SAVE_AFTER = 0.0  # Only save model when over this percentage of epochs done.
SAVE_MOST_RECENT_MODEL = True
SAVE_BEST_METRICS = {  # metric (str): max_better (bool)
    'train_ep_loss': False,
}
SUMMARIZE = {
    'triggers': ['train_ep_loss'],
    'saves': ['train_ep_loss']
}


def run(rank, cfg):
    # Experiment environment setup
    from run import set_seed, setup_dist
    set_seed(cfg['experiment']['seed'])

    if cfg.experiment.distributed:
        cfg.experiment.device = f'cuda:{rank}'
        cfg.experiment.rank = rank
        worldsize = len(cfg.experiment.gpu_idxs)
        cfg.experiment.worldsize = worldsize
        setup_dist(rank, worldsize)
        cfg.__dict__ = cfg

    # ------------------ ##  Experiment Setup  ## ------------------ #
    if rank == 0:
        output.header_one('I. BCV Finetune Training Components Setup')
    
    global gpu_indices
    global device

    gpu_indices = cfg.experiment.gpu_idxs
    device = cfg.experiment.device
    debug = cfg.experiment.debug

    if rank == 0:
        print(f"[Experiment Settings (@supervised/emain.py)]")
        print(f" > Prepping train config..")
        print(f"\t - experiment:  {cfg.experiment.project} - "
                f"{cfg.experiment.name}, id({cfg.experiment.id})")
        print(f"\t - batch_size {cfg.train.batch_size}, "
              f"\t - start epoch: {cfg.train.start_epoch}/"
                f"{cfg.train.epochs},")
        print(f"\t - Optimizer ({cfg.train.optimizer.name}): "
              f"\t - lr {cfg.train.optimizer.lr}, "
              f"\t - wt_decay {cfg.train.optimizer.wt_decay} ")
        print(f"\t - Scheduler ({cfg.train.scheduler.name}): "
              f"\t - rampup: {cfg.train.scheduler.rampup_rates}\n")

        wandb_settings = {}
        if cfg['experiment']['debug']['wandb']:
            wandb_settings = {
                'project': cfg['experiment']['project'],
                'name': cfg.experiment.id + '_' + cfg.experiment.name,
                'config': cfg,
                'notes': cfg['experiment']['description']
            }
        tracker = statistics.WandBTracker(wandb_settings=wandb_settings)
    source_code = {
        'run': inspect.getsource(inspect.getmodule(inspect.currentframe())),
        'data': inspect.getsource(inspect.getmodule(data_setup)),
        'setup': inspect.getsource(inspect.getmodule(setup)),
        'metrics': inspect.getsource(inspect.getmodule(lib.assess.seg_metrics))
    } 

    # Get Model 
    if rank == 0:
        output.header_three('Model Setup')
    model_d = get_model(cfg)
    model = model_d['model'].to(device)
    ema_model = model_d['ema_model'].to(device)
    if not cfg.experiment.distributed and len(gpu_indices) > 1:
        print(f'  * {len(gpu_indices)} GPUs, using nn.DataParallel.')
        model = torch.nn.DataParallel(model)
        ema_model = torch.nn.DataParallel(ema_model)
    elif cfg.experiment.distributed:
        model = DDP(model, device_ids=[rank], output_device=rank)
        print(f'  * Rank {rank} using device {device} via nn.DDP.')
        if cfg.model.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    # Get criterion, optimizer, scheduler
    if rank == 0:
        output.header_three('Criterion + Optimizer + Scheduler Setup')
    from lib.assess.losses3d import BYOL3d
    criterion = BYOL3d()
    
    optimizer = setup.get_optimizer(cfg, model.parameters())
    scheduler = setup.get_scheduler(cfg, optimizer)

    # Data Pipeline
    if rank == 0:
        output.header_three('Data Setup')
    from .data_setup import get_data_components
    data_d = get_data_components(cfg)
    
    train_df = data_d['train_df']
    train_set = data_d['train_set']
    train_loader = data_d['train_loader']
    
    # ------------------ ##  Training Action  ## ------------------ #
    if rank == 0:
        output.header_one('II. Training')
    
    if debug['overfitbatch']:
        print(f'🚨  Overfitting a set of minibatches! \n')
        batches = []
        for i, batch in enumerate(train_loader):
            batches.append(batch)
            if len(batches) >= 2: 
                break
        train_loader = list(itertools.islice(itertools.cycle(batches), 30))
    
    tot_epochs = cfg['train']['epochs'] - cfg['train']['start_epoch']
    global_iter = 0
    for epoch in range(cfg['train']['start_epoch'], cfg['train']['epochs']):
        if rank == 0:
            sec_header = f'Starting Epoch {epoch+1} (lr: {scheduler.lr:.7f})'
            output.subsection(sec_header)
        
        WATCH.tic('epoch')
        model.train()
        epmeter = statistics.EpochMeters()
        WATCH.tic('iter')

        for it, batch in enumerate(train_loader):
            samples = batch['samples']
            records = batch['records']
            crop_objs = batch['crops']
            
            import IPython; IPython.embed(); 
            
            X = batch['X'].to(device, non_blocking=True)
            
            with torch.no_grad():
                emaout = ema_model(X, enc_only=True)['out']
            out = model(X, enc_only=True)['out']
            
            loss_d = criterion(out, emaout)
            loss = loss_d['loss']
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Iteration & Epoch Training Metrics
            iter_time = WATCH.toc('iter', disp=False)
            loss_str = f'loss {loss.item():.3f}'
            if rank == 0:
                print(f"\n    Iter {it+1}/{len(train_loader)} "
                      f"({iter_time:.1f} sec, {mem():.1f} GB) - {loss_str} ")

            epmeter.update({'loss': loss.item()})
            
            global_iter += 1
            WATCH.tic('iter')
            if debug['break_train_iter'] and it >= 2: 
                break

        # -- Epoch Values -- #
        epoch_mets = statistics.EpochMetrics()
        train_mets = epmeter.avg()
        epoch_mets.update(train_mets, ['loss'], 'train_ep_')

        if rank == 0:
            epoch_mets.print(pre="\nEpoch Stats\n-----------\n")

            force_sum = True if epoch == cfg.train.epochs - 1 else False
            tracker.update(epoch_mets, log=True, summarize=SUMMARIZE,
                           after_epoch=int(SAVE_AFTER * tot_epochs), 
                           force_summarize=force_sum)
            
            if not debug['save']:
                print(f'🚨  Model-saving functionality is off! \n')
            if debug['save'] and epoch % 10 == 9:
                save_model(cfg, model.state_dict(), tracker, epoch, source_code)
        
        scheduler.step(epoch=epoch, value=0)
        WATCH.toc(name='epoch')
        # End of epoch #


def get_model(cfg):
    if cfg.model.name == 'denseunet3d':
        from lib.nets.volumetric.denseunet3d import get_model as get_dunet
        model = get_dunet(201, num_classes=14, deconv=False)
        ema_model = create_ema_model(get_dunet(201, num_classes=14, 
                                               deconv=False))
    elif cfg.model.name == 'genesis_unet3d':
        from .pgl_unet3d import UNet3D as genesis_unet3d
        model = genesis_unet3d(n_input=1, n_class=14, act='relu')
        ema_model = create_ema_model(genesis_unet3d(n_input=1, n_class=14, 
                                                    act='relu'))
    elif cfg.model.name == 'resmednet3d':  # already inited
        from lib.nets.volumetric.resnet3d_mednet import generate_resnet3d
        model = generate_resnet3d(in_channels=1, classes=14, model_depth=34)
    elif cfg.model.name == 'gn_unet3d':
        from lib.nets.volumetric.resunet3d import UNet3D
        model = UNet3D(1, 14, final_sigmoid=False, is_segmentation=False)
    else:
        raise ValueError(f'Model {cfg.model.name} is not supported.')

    # initalize model weights
    if cfg.model.init:
        # Initialize
        init_type = cfg.model.init
        init_net.init_weights(model, init_type=init_type)
        print(f'   (Model) Successfully initialized weights via {init_type}.')
    
    # Initialize parameters
    if cfg.experiment.checkpoint.file:  # Checkpoint handling
        filename = cfg.experiment.checkpoint.file
        curr_path = pathlib.Path(__file__).parent.absolute()
        if pathlib.Path(filename).exists():
            filepath = str(pathlib.Path(filename).absolute())
        elif (curr_path / filename).exists():
            filepath = str(curr_path / filename)
        elif (curr_path / 'artifacts' / filename).exists():
            filepath = str(curr_path / 'artifacts' / filename)
        else:
            filepath = None
            print(f'Give filename {filename} could not be found.')

        if filepath:
            checkpoint_d = torch.load(filepath, map_location='cpu')
            state_dict = checkpoint_d['state_dict']
            print(model.load_state_dict(state_dict))

    return {
        'model': model,
        'ema_model': ema_model
    }


def save_model(cfg, state_dict, tracker, epoch, code_d):
    """ Does not delete old model files. """
    end = 'last'
    curr_path = pathlib.Path(__file__).parent.absolute()
    fn_start = f"{cfg['experiment']['id']}_{cfg['experiment']['name']}_"
    
    filename = fn_start + f'ep{epoch}_' + end + '.pth'
    save_path = os.path.join(curr_path, 'artifacts', filename)
    print(f"Saving model -> {filename}")
    torch.save({
        'state_dict': state_dict,
        'code_dict': code_d,
        'tracker': tracker,
        'config': cfg,
        'epoch': epoch,
        },
        save_path
    )


def mem():
    """ Get primary GPU card memory usage. """
    if not torch.cuda.is_available():
        return -1.
    mem_map = devices.get_gpu_memory_map()
    prim_card_num = gpu_indices[0]
    return mem_map[prim_card_num]/1000






