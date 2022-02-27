""" Module run_experiment.py (Author: Charley Zhang, 2022)
MICCAI'22 Spatial Pretraining Experiments

Used for:
  - Experiments for vector prediction. 
"""

import sys, os
import yaml
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
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import SimpleITK as sitk

import experiments, lib
from experiments import setup
from experiments.ftmmwhs import data_setup
from lib.utils import devices, timers, statistics
from lib.utils.train import ramps
from lib.utils.io import output
from lib.assess.seg_metrics import batch_cdj_metrics
from data.transforms.crops.inference import ChopBatchAggregate3d as CBA
from data.transforms.resize import resize_segmentation3d

from lib.utils.devices import ram
from experiments.train_utils import synchronize, save_image

WATCH = timers.StopWatch()

SAVE_AFTER = 0.0  # Only save model when over this percentage of epochs done.
SAVE_MOST_RECENT_MODEL = False  # Only save the best model
SAVE_BEST_METRICS = {  # metric (str): max_better (bool)
    'train_ep_loss': False,
}
SUMMARIZE = {
    'triggers': ['train_ep_loss'],
    'saves': ['train_ep_loss']
              # 'test_ep_dice', 'test_ep_jaccard']
}

CM = namedtuple('CM', ('tp', 'fp', 'fn', 'tn'))



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
        # inference_metrics_queue = torch.multiprocessing.Queue()  # replace

        torch.cuda.set_device(cfg.experiment.rank)

    # ------------------ ##  Experiment Setup  ## ------------------ #
    if rank == 0:
        output.header_one('I. MMWHS Finetune Training Components Setup')
    
    global gpu_indices
    global device

    gpu_indices = cfg.experiment.gpu_idxs
    device = cfg.experiment.device
    debug = cfg.experiment.debug
    task_config = cfg.tasks[cfg.tasks.name]

    if rank == 0:  # wandb sweeping or standard tracking
        if 'sweep' in cfg.experiment and cfg.experiment.sweep:
            wandb_settings = {
                'project': cfg['experiment']['project'],
                'name': cfg.experiment.id + '_' + cfg.experiment.name,
                'group': cfg.experiment.sweep_id,
                'job_type': cfg.experiment.sweep_run_name,
                'config': cfg
            }
        else:
            wandb_settings = {}
            if cfg['experiment']['debug']['wandb']:
                wandb_settings = {
                    'project': cfg['experiment']['project'],
                    'name': cfg.experiment.id + '_' + cfg.experiment.name,
                    'config': cfg,
                    'notes': cfg['experiment']['description']
                }
        tracker = statistics.WandBTracker(wandb_settings=wandb_settings)
        
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
    
    source_code = {
        'run': inspect.getsource(inspect.getmodule(inspect.currentframe())),
        'data': inspect.getsource(inspect.getmodule(data_setup)),
        'setup': inspect.getsource(inspect.getmodule(setup)),
        'metrics': inspect.getsource(inspect.getmodule(lib.assess.seg_metrics))
    } 

    # Training Components: model, criterion, optimizer, scheduler
    from experiments.rubik.model_setup import get_model_components
    if rank == 0:
        output.header_three('Model Setup')
    
    model_d = get_model_components(cfg)
    generator = model_d['generator'].to(device)
    discriminator = model_d['discriminator'].to(device)
    
    assert not cfg.experiment.distributed, 'Removed DDP model code'
    
    if rank == 0:
        output.header_three('Criterion + Optimizer + Scheduler Setup')
    
    criterion_gan = nn.MSELoss().to(device)
    criterion_recon = nn.L1Loss().to(device)
    
    optimizer_g = setup.get_optimizer(cfg, generator.parameters())
    optimizer_d = setup.get_optimizer(cfg, discriminator.parameters())
    
    scheduler = scheduler_g = setup.get_scheduler(cfg, optimizer_g)
    scheduler_d = setup.get_scheduler(cfg, optimizer_d)

    # Data Pipeline
    if rank == 0:
        output.header_three('Data Setup')
    if cfg.data.name == 'mmwhs':
        from experiments.rubik.data_setup import get_mmwhs_data_components
        data_d = get_mmwhs_data_components(cfg)
    elif cfg.data.name == 'pretrain':
        from experiments.rubik.data_setup import get_pretrain_data_components
        data_d = get_pretrain_data_components(cfg)
    else:
       assert False, 'Config data name invalid: {cfg.data.name}.'
    
    train_df = data_d['train_df']
    train_set = data_d['train_set']
    train_loader = data_d['train_loader']
    
    test_set = data_d['test_set']
    
    # ------------------ ##  Training Action  ## ------------------ #
    if rank == 0:
        output.header_one('II. Training')
    
    if debug['overfitbatch']:
        print(f'🚨  Overfitting a set of minibatches! \n')
        batches = []
        for i, batch in enumerate(train_loader):
            Y_id = batch['masks']
            vol = Y_id.shape[2] * Y_id.shape[3] * Y_id.shape[4]
            ids, fg_counts = Y_id.unique(return_counts=True)
            fg_vol = fg_counts[1:].sum()
            if len(ids > 4) and fg_vol > 0.25 * vol:
                print(f'Batch {i+1} successfully meets the criteria '
                      f'(volume: {fg_vol / vol}).')
                batches.append(batch)
            if len(batches) >= 2: 
                break
        train_loader = list(itertools.islice(itertools.cycle(batches), 50))
        test_set._samples = [test_set.samples[0]]
    
    synchronize()
    tot_epochs = cfg['train']['epochs'] - cfg['train']['start_epoch']
    global_iter = 0
    for epoch in range(cfg['train']['start_epoch'], cfg['train']['epochs']):
        if rank == 0:
            sec_header = f'Starting Epoch {epoch+1} (lr: {scheduler.lr:.7f})'
            output.subsection(sec_header)
            ram(disp=True)
        
        WATCH.tic('epoch')
        generator.train()
        discriminator.train()
        epmeter = statistics.EpochMeters()
        WATCH.tic('iter')

        for it, batch in enumerate(train_loader):
            
            samples = batch['samples']
            records = batch['records']
            
            X = batch['X'].float().to(device, non_blocking=True)
            Y = batch['Y'].float().to(device, non_blocking=True)
            B = X.shape[0]

            patch = (1, X.shape[2] // 2**4,  X.shape[3]// 2**4, X.shape[4]// 2**4)
            valid = torch.ones((X.shape[0], *patch), requires_grad=False).to(device)
            fake = torch.zeros((X.shape[0], *patch), requires_grad=False).to(device)
                        
            # Train Generator
            optimizer_g.zero_grad()
            out = g_out = generator(X)['out']
            pred_fake = discriminator(g_out.sigmoid(), X)
            loss_gan = criterion_gan(pred_fake, valid)
            loss_recon = criterion_recon(g_out.sigmoid(), Y)
            loss_recon *= task_config.recon_weight
            
            loss_g = loss_gan + loss_recon
            loss_g.backward()
            optimizer_g.step()
            
            # Train Discriminator
            optimizer_d.zero_grad()
            
            pred_real = discriminator(Y, X)
            loss_real = criterion_gan(pred_real, valid)
            
            pred_fake = discriminator(g_out.sigmoid().detach(), X)
            loss_fake = criterion_gan(pred_fake, fake)
            
            loss_d = 0.5 * (loss_real + loss_fake)
            loss_d.backward()
            optimizer_d.step()
            
            loss = loss_g + loss_d
            loss_str = (f'Loss: {loss.item():.4f} (gan {loss_gan.item():.4f}, '
                f'recon {loss_recon.item():.4f}, discrim {loss_d.item():.4f})')

            # Iteration & Epoch Training Metrics 
            print_every = max(1, len(train_loader) // 7)
            if rank == 0 and it % print_every == 0 or debug.mode:
                iter_metrics_d = {}
                iter_metrics_d['loss'] = loss.item()
                
                iter_time = WATCH.toc('iter', disp=False)
                print(
                    f"    Iter {it+1}/{len(train_loader)} "
                    f"({iter_time:.1f} sec, {mem():.1f} GB) - "
                    f"{loss_str}", flush=True
                )
                epmeter.update({'loss': loss.item() * B}, n=B)

            # Save images, masks, and predictions
            if it == 0 and epoch % 100 == 0:
                def save_volume(image_arr, ref_sitk, filename):
                    image_sitk = sitk.GetImageFromArray(image_arr)
                    image_sitk.SetSpacing((1, 1, 1))
                    image_sitk.SetOrigin(ref_sitk.GetOrigin())
                    image_sitk.SetDirection(ref_sitk.GetDirection())
                    sitk.WriteImage(image_sitk, filename)
                
                curr_path = pathlib.Path(__file__).parent.absolute()
                exp_save_path = curr_path / 'artifacts' / cfg.experiment.id
                os.makedirs(exp_save_path, exist_ok=True)
                
                print(f'💾  Saving Crops & Preds for epoch {epoch}!')
                for b in range(0, X.shape[0], 1):
                    image_arr = X[b][0].detach().cpu().numpy().astype(np.float32)
                    label_arr = Y[b][0].detach().cpu().numpy().astype(np.float32)
                    recon_arr = out[b][0].sigmoid().detach().cpu().numpy().astype(np.float32)
                    
                    ref_im = samples[b].image.sitk_image
                    save_volume(image_arr, ref_im, os.path.join(exp_save_path,
                        f'ep{epoch}_b{b}_input.nii.gz'))
                    save_volume(label_arr, ref_im, os.path.join(exp_save_path,
                        f'ep{epoch}_b{b}_orig.nii.gz'))
                    save_volume(recon_arr, ref_im, os.path.join(exp_save_path,
                        f'ep{epoch}_b{b}_recon.nii.gz'))

            global_iter += 1
            WATCH.tic('iter')
            if debug['break_train_iter'] and it >= 5: 
                break

        out = None; out_d = None  # save mem for inference
        del X; del loss  

        # -- Epoch Values -- #
        epoch_mets = statistics.EpochMetrics()

        # Training epoch values
        train_mets = epmeter.avg()
        tracked_epmets = ['loss']
        epoch_mets.update(train_mets, tracked_epmets, 'train_ep_')

        # Test + Epoch Metrics
        test_every_n = debug.test_every_n_epochs
        if epoch % test_every_n == test_every_n - 1 and task_config.test:
            raise NotImplementedError()
            if rank == 0:
                output.subsubsection('Test Metrics')
            N_test_exs = len(data_d['df'][data_d['df']['subset'] == 'test'])
            test_mets = test_metrics(cfg, model, test_set, epoch, 
                inference_metrics_queue, N_test_exs, name='test',
                overlap_perc=cfg.test.patch_overlap_perc,
                criterion=criterion)
            if rank == 0:
                epoch_mets.update(test_mets, tracked_epmets, 'test_ep_')
        
        scheduler_g.step(epoch=epoch, value=0)
        scheduler_d.step(epoch=epoch, value=0)

        if rank == 0:
            epoch_mets.print(pre="\nEpoch Stats\n-----------\n")

            force_sum = True if epoch == cfg.train.epochs - 1 else False
            tracker.update(epoch_mets, log=True, summarize=SUMMARIZE,
                           after_epoch=int(SAVE_AFTER * tot_epochs), 
                           force_summarize=force_sum)
            
            if not debug['save']:
                print(f'🚨  Model-saving functionality is off! \n')
            if debug['save'] and epoch >= SAVE_AFTER * tot_epochs:
                save_model(cfg, generator.state_dict(), tracker, epoch, 
                           source_code)
        
            WATCH.toc(name='epoch')
        synchronize()
        # --  End of epoch -- #
    
    return tracker


def save_model(cfg, state_dict, tracker, epoch, code_d):
    end = 'last'
    for met, max_gud in SAVE_BEST_METRICS.items():
        if tracker.is_best(met, max_better=max_gud):
            subset = met.split('_')[0]
            met_name = '_'.join(met.split('_')[2:]) 
            score = float(tracker.metrics_d[met][-1])
            score_str = f'{int(score):d}' if score.is_integer() else f'{score:.3f}'
            end = f"best-{subset}-{met_name}-{score_str}"
            print(f"(save_model) {end}")
            break
    if end == 'last' and not SAVE_MOST_RECENT_MODEL:
        return    

    # Create directories in exp/artifacts
    curr_path = pathlib.Path(__file__).parent.absolute()
    exp_save_path = curr_path / 'artifacts' / cfg.experiment.id
    os.makedirs(exp_save_path, exist_ok=True)
    
    fn_start = f"{cfg['experiment']['id']}_{cfg['experiment']['name']}_"
    if 'sweep' in cfg.experiment and cfg.experiment.sweep:
        fn_start += f'sweep-{cfg.experiment.sweep_id}_run-{wandb.run.id}'

    # Remove previous best-score checkpoints
    rm_files = [f for f in os.listdir(exp_save_path) if f[-3:] == 'pth' \
                and fn_start in f]
    if rm_files:
        match_str = end
        if end != 'last':
            match_str = f'best-{subset}-{met_name}'
        for f in rm_files:
            if match_str in f:
                rm_file = os.path.join(exp_save_path, f)
                print(f"Deleting file -x {f}")
                os.remove(rm_file)
    
    filename = fn_start + f'ep{epoch}_' + end + '.pth'
    save_path = os.path.join(exp_save_path, filename)
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