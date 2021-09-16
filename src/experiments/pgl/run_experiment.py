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
import torch.nn.functional as F
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
    predictor = model_d['predictor'].to(device)
    if not cfg.experiment.distributed and len(gpu_indices) > 1:
        print(f'  * {len(gpu_indices)} GPUs, using nn.DataParallel.')
        model = torch.nn.DataParallel(model)
        ema_model = torch.nn.DataParallel(ema_model)
        predictor = torch.nn.DataParallel(predictor)
    elif cfg.experiment.distributed:
        model = DDP(model, device_ids=[rank], output_device=rank, 
                    find_unused_parameters=True)
        # ema_model = DDP(ema_model, device_ids=[rank], output_device=rank)
        predictor = DDP(predictor, device_ids=[rank], output_device=rank)
        print(f'  * Rank {rank} using device {device} via nn.DDP.')
        if cfg.model.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            predictor = torch.nn.SyncBatchNorm.convert_sync_batchnorm(predictor)
            print(f'  * Rank {rank} using syncBN.')
    
    # Get criterion, optimizer, scheduler
    if rank == 0:
        output.header_three('Criterion + Optimizer + Scheduler Setup')
    from lib.assess.losses3d import BYOL3d
    criterion = BYOL3d()
    
    params = list(model.parameters()) + list(predictor.parameters())
    optimizer = setup.get_optimizer(cfg, params)
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
        print(f'   (Training BYOL Mode: {cfg.train.train_byol})')
    
    if debug['overfitbatch']:
        print(f'🚨  Overfitting a set of minibatches! \n')
        batches = []
        for i, batch in enumerate(train_loader):
            batches.append(batch)
            if len(batches) >= 2: 
                break
        train_loader = list(itertools.islice(itertools.cycle(batches), 30))
    
    synchronize()
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
            
            X1 = batch['X1'].unsqueeze(1).to(device, non_blocking=True)
            X2 = batch['X2'].unsqueeze(1).to(device, non_blocking=True)

            if it <= 4 and False:  # visualize NN input
                for i in range(5):  
                    im1 = X1[i][0].detach().cpu()
                    im2 = X2[i][0].detach().cpu()

                    dataset = str(samples[i].dataset)
                    if dataset == 'msd':
                        dataset += f'|{samples[i].task}'

                    save_image(im1, f'it{it}_im{i}_view1_{dataset}', samples[i], 
                               history=records[i][0], is_mask=False)
                    save_image(im2, f'it{it}_im{i}_view2_{dataset}', samples[i], 
                               history=records[i][1], is_mask=False)

            proj1 = model(X1)['out']
            proj2 = model(X2)['out']
            pred1 = predictor(proj1)
            pred2 = predictor(proj2)

            optimizer.zero_grad()
            with torch.no_grad():
                targ1 = ema_model(X1)['out']
                targ2 = ema_model(X2)['out']

            loss1 = criterion(pred1, targ1.detach())['loss'] 
            loss2 = criterion(pred2, targ2.detach())['loss']
            loss = loss1 + loss2
            
            loss.backward()
            optimizer.step()
            update_ema_model(model, ema_model, cfg.model.ema_alpha)

            # Iteration & Epoch Training Metrics
            if rank == 0:
                iter_time = WATCH.toc('iter', disp=False)
                loss_str = f'loss {loss.item():.3f}'
                print_niter = 1 if not debug.overfitbatch else 20
                if it % print_niter == print_niter - 1:
                    print(f"    Iter {it+1}/{len(train_loader)} "
                        f"({iter_time:.1f} sec, {mem():.1f} GB) - {loss_str} ",
                        flush=True)

            epmeter.update({'loss': loss.item()})
            
            global_iter += 1
            WATCH.tic('iter')
            if debug['break_train_iter'] and it >= 10: 
                break
        
        scheduler.step(epoch=epoch, value=0)
        
        # -- Epoch Values -- #
        if rank == 0:
            epoch_mets = statistics.EpochMetrics()
            train_mets = epmeter.avg()
            epoch_mets.update(train_mets, ['loss'], 'train_ep_')
            
            epoch_mets.print(pre="\nEpoch Stats\n-----------\n")

            force_sum = True if epoch == cfg.train.epochs - 1 else False
            tracker.update(epoch_mets, log=True, summarize=SUMMARIZE,
                           after_epoch=int(SAVE_AFTER * tot_epochs), 
                           force_summarize=force_sum)
            
            if not debug['save']:
                print(f'🚨  Model-saving functionality is off! \n')
            if debug['save'] and epoch % 50 == 49:
                save_model(cfg, model.state_dict(), ema_model.state_dict(),
                           predictor.state_dict(),
                           tracker, epoch, source_code)
            WATCH.toc(name='epoch')
        # End of epoch #


def get_model(cfg):
    patch_size = cfg.train.patch_size
    num_proj_dims = cfg.model.feat_channels
    num_hidden_dims = cfg.model.latent_channels
    if cfg.model.name == 'nnunet3d':
        from experiments.pgl.nnunet3d import PGL_UNet3d, ProjectionHead
        model = PGL_UNet3d(in_channels=1, num_classes=14, 
                           is_byol=cfg.train.train_byol,
                           latent_channels=num_hidden_dims,
                           feat_channels=num_proj_dims)
        ema_model = PGL_UNet3d(in_channels=1, num_classes=14,
                               is_byol=cfg.train.train_byol,
                               latent_channels=num_hidden_dims,
                               feat_channels=num_proj_dims)
        ema_model = create_ema_model(ema_model)

        if not cfg.train.train_byol:
            predictor = ProjectionHead(num_proj_dims, 
                                       latent_channels=num_hidden_dims, 
                                       out_channels=num_proj_dims)
        else:
            from experiments.pgl.nnunet3d import GlobalProjectionHead
            predictor = GlobalProjectionHead(num_proj_dims, 
                                             latent_channels=num_hidden_dims,
                                             out_channels=num_proj_dims)
    # elif cfg.model.name == 'denseunet3d':
    #     from lib.nets.volumetric.denseunet3d import get_model as get_dunet
    #     model = get_dunet(201, num_classes=14, deconv=False)
    #     ema_model = create_ema_model(get_dunet(201, num_classes=14, 
    #                                            deconv=False))
    # elif cfg.model.name == 'genesis_unet3d':
    #     from .pgl_unet3d import UNet3D as genesis_unet3d
    #     model = genesis_unet3d(n_input=1, n_class=14, act='relu')
    #     ema_model = create_ema_model(genesis_unet3d(n_input=1, n_class=14, 
    #                                                 act='relu'))
    # elif cfg.model.name == 'resmednet3d':  # already inited
    #     from lib.nets.volumetric.resnet3d_mednet import generate_resnet3d
    #     model = generate_resnet3d(in_channels=1, classes=14, model_depth=34)
    # elif cfg.model.name == 'gn_unet3d':
    #     from lib.nets.volumetric.resunet3d import UNet3D
    #     model = UNet3D(1, 14, final_sigmoid=False, is_segmentation=False)
    else:
        raise ValueError(f'Model {cfg.model.name} is not supported.')

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
            ema_state_dict = checkpoint_d['ema_state_dict']
            print(ema_model.load_state_dict(ema_state_dict))
            predictor_state_dict = checkpoint_d['predictor_state_dict']
            print(predictor.load_state_dict(predictor_state_dict))
    else: 
        # Initialize
        init_type = cfg.model.init
        if init_type:
            init_net.init_weights(model, init_type=init_type)
            init_net.init_weights(ema_model, init_type=init_type)
            init_net.init_weights(predictor, init_type=init_type)
        print(f'   (Model) Successfully initialized weights via {init_type}.')

    return {
        'model': model,
        'ema_model': ema_model,
        'predictor': predictor
    }


def save_model(cfg, state_dict, ema_state_dict, pred_state_dict,
               tracker, epoch, code_d):
    """ Does not delete old model files. """
    curr_path = pathlib.Path(__file__).parent.absolute()
    fn_start = f"{cfg['experiment']['id']}_{cfg['experiment']['name']}_"
    
    filename = fn_start + f'ep{epoch}_' + '.pth'
    save_path = os.path.join(curr_path, 'artifacts', filename)
    print(f"Saving model -> {filename}")
    torch.save({
        'state_dict': state_dict,
        'ema_state_dict': ema_state_dict,
        'predictor_state_dict': pred_state_dict,
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


def synchronize():
    """
    Helper function to synchronize between multiple processes when
    using distributed training
    """
    if not torch.distributed.is_initialized():
        return
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    if world_size == 1:
        return
    torch.distributed.barrier()


def save_image(data, name, sample, history=None, is_mask=False):
    if is_mask:
        if data.ndim == 4 and data.shape[0] == 1:
            data = data.squeeze(0)
        elif data.ndim == 4 and data.shape[0] > 1:
            data = data.argmax(0)
        assert data.ndim == 3
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        data = data.astype(np.uint8)
    else:
        assert data.ndim == 3
        if history:
            from data.transforms.z_normalize import ZNormalize
            data = ZNormalize().invert(data, history['ZNormalize'])
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        data = data.astype(np.int16)

    sitk_crop = sitk.GetImageFromArray(data)
    sitk_crop.SetOrigin(sample.image.origin)
    sitk_crop.SetDirection(sample.image.direction)
    sitk_crop.SetSpacing(sample.image.spacing)

    if len(name) <= 7 or name[-7:] != '.nii.gz':
        name = name + '.nii.gz'
    sitk.WriteImage(sitk_crop, name)


