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



def run(rank, cfg, inference_metrics_queue):
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
        inference_metrics_queue = torch.multiprocessing.Queue()  # replace

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
    from experiments.prevec.model_setup import get_model_components
    if rank == 0:
        output.header_three('Model Setup')
    model_d = get_model_components(cfg)
    model = model_d['model'].to(device)
    if not cfg.experiment.distributed and len(gpu_indices) > 1:
        print(f'  * {len(gpu_indices)} GPUs, using nn.DataParallel.')
        model = torch.nn.DataParallel(model)
    elif cfg.experiment.distributed:
        model = DDP(model, device_ids=[rank], output_device=rank)
        print(f'  * Rank {rank} using device {device} via nn.DDP.')
        if cfg.model.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            print(f'  * Rank {rank} device using SyncBatchNorm.')
    
    if rank == 0:
        output.header_three('Criterion + Optimizer + Scheduler Setup')
    
    criterion = None
    if cfg.tasks.name == 'vec':
        if cfg.tasks.vec.loss == 'spherical':
            from experiments.prevec.prevec_losses import SphericalCriterion
            crit_prevec = SphericalCriterion(
                r_loss=cfg.tasks.vec.r_loss, 
                angle_loss=cfg.tasks.vec.ang_loss,
                r_weight=1, theta_weight=1, phi_weight=1
            )
            criterion = crit_prevec.to(device)
            # cfg.tasks.vec.pred_vectors
        else:
            assert False
            
        if task_config.num_classes > 0:
            print(f"????   Recon using loss: {task_config.loss_recon}.")
            if task_config.loss_recon == 'mse':
                criterion_recon = nn.MSELoss().to(device)
            else:
                criterion_recon = nn.L1Loss().to(device)
        if task_config.num_classes == 2:
            print(f"????   Recon using boundary loss: {task_config.loss_boundary}.")
            if task_config.loss_boundary == 'mse':
                criterion_boundary = nn.MSELoss().to(device)
            else:
                criterion_boundary = nn.L1Loss().to(device)
                
    elif cfg.tasks.name == 'mg':
        print(f"????   MG using recon loss: {task_config.loss_recon}.")
        if task_config.loss_recon == 'mse':
            criterion = nn.MSELoss().to(device)
        else:
            criterion = nn.L1Loss().to(device)
            
        if task_config.num_classes == 2:
            print(f"????   MG using boundary loss: {task_config.loss_boundary}.")
            if task_config.loss_boundary == 'mse':
                criterion_boundary = nn.MSELoss().to(device)
            else:
                criterion_boundary = nn.L1Loss().to(device)
    elif cfg.tasks.name == 'sar':
        criterion = model.loss
    
    optimizer = setup.get_optimizer(cfg, model.parameters())
    scheduler = setup.get_scheduler(cfg, optimizer)

    # Data Pipeline
    if rank == 0:
        output.header_three('Data Setup')
    if cfg.data.name == 'mmwhs':
        print(f'????  Using WHS to pretrain rather than collation set.')
        from experiments.prevec.data_setup import get_mmwhs_data_components
        data_d = get_mmwhs_data_components(cfg)
    elif cfg.data.name == 'pretrain':
        from experiments.prevec.data_setup import get_pretrain_data_components
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
        print(f'????  Overfitting a set of minibatches! \n')
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
        model.train()
        epmeter = statistics.EpochMeters()
        WATCH.tic('iter')

        for it, batch in enumerate(train_loader):
            
            samples = batch['samples']
            records = batch['records']
            
            X = batch['X'].float().to(device, non_blocking=True)
            # Y_id = batch['masks'].to(torch.uint8).to(device, non_blocking=True)

            out_d = model(X)
            out = out_d['out'] if isinstance(out_d, dict) else out_d
            
            # -- Loss Calculation -- #
            loss = 0
            loss_str = ''
            if cfg.tasks.name == 'vec':
                # create labels
                vectors = batch['vectors']
                Y_prevec = create_prevec_targ(vectors, records,
                    cfg).to(out.device)
                # print('in train before crit called'); import IPython; IPython.embed(); 
                loss_prevec_d = criterion(out, Y_prevec)
                loss += loss_prevec_d['loss']
                loss_str += f'PrevecLoss {loss.item():.4f} '
                
                if task_config.num_classes == 1:
                    Y = batch['Y'].to(device)
                    task_loss = criterion_recon(out_d['recon'].sigmoid(), Y)
                    loss += task_loss
                    loss_str += f'ReconLoss {loss.item():.4f} '
                elif task_config.num_classes == 2:
                    Y = batch['Y'].to(device)
                    recon_loss = criterion_recon(
                        out_d['recon'][:,0,...].sigmoid().unsqueeze(1), Y)
                    if not task_config.boundary_in_paint:
                        Y_boundary = batch['boundaries'].to(device)
                        bound_pred = out_d['recon'][:,1,...].sigmoid().unsqueeze(1)
                    else:
                        bound_mask = batch['boundary_masks'].to(device)
                        Y_boundary = batch['boundaries'].to(device) * bound_mask
                        bound_pred = out_d['recon'][:,1,...].sigmoid().unsqueeze(1) * bound_mask
                    bound_loss = criterion_boundary(bound_pred, Y_boundary)
                    
                    loss += recon_loss + bound_loss
                    loss_str += (f'ReconLoss {loss.item():.4f} (recon {recon_loss:.4f}, '
                                 f'bound {bound_loss:.4f}) ')
            elif cfg.tasks.name == 'mg':
                Y = batch['Y'].to(device)
                if task_config.num_classes == 2:
                    recon_loss = criterion(out[:,0,...].sigmoid().unsqueeze(1),
                                           Y)
                    if not task_config.boundary_in_paint:
                        Y_boundary = batch['boundaries'].to(device)
                        bound_pred = out[:,1,...].sigmoid().unsqueeze(1)
                    else:
                        bound_mask = batch['boundary_masks'].to(device)
                        Y_boundary = batch['boundaries'].to(device) * bound_mask
                        bound_pred = out[:,1,...].sigmoid().unsqueeze(1) * bound_mask
                    bound_loss = criterion_boundary(bound_pred, Y_boundary)
                    
                    loss += recon_loss + bound_loss
                    loss_str += (f'MGLoss {loss.item():.4f} (recon {recon_loss:.4f}, '
                                 f'bound {bound_loss:.4f}) ')
                else:
                    task_loss = criterion(out.sigmoid(), Y)
                    loss += task_loss
                    loss_str += f'MGLoss (recon-only) {loss.item():.4f} '
            elif cfg.tasks.name == 'sar':
                Y_recon = batch['Y_recon'].to(device)
                Y_scale = batch['Y_scale'].to(device)
                loss_d = model.loss(out_d, Y_recon, Y_scale)
                loss += loss_d['loss']
                loss_str += (f'SARLoss {loss.item():.4f} (recon: '
                             f'{loss_d["recon"].item():.4f}, scale: '
                             f'{loss_d["scale"].item():.4f})')
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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

                update_mets = ['loss']
                epmeter.update({k: iter_metrics_d[k] for k in update_mets})

            # Save images, masks, and predictions
            if (cfg.tasks.name == 'mg' or cfg.tasks.name == 'sar') or \
                (cfg.tasks.name == 'vec' and cfg.tasks.vec.num_classes > 0) and \
                    it == 0 and epoch % 100 == 0:
                def save_volume(image_arr, ref_sitk, filename):
                    image_sitk = sitk.GetImageFromArray(image_arr)
                    image_sitk.SetSpacing((1, 1, 1))
                    image_sitk.SetOrigin(ref_sitk.GetOrigin())
                    image_sitk.SetDirection(ref_sitk.GetDirection())
                    sitk.WriteImage(image_sitk, filename)
                
                curr_path = pathlib.Path(__file__).parent.absolute()
                exp_save_path = curr_path / 'artifacts' / cfg.experiment.id
                os.makedirs(exp_save_path, exist_ok=True)
                
                if cfg.tasks.name == 'sar':
                    Y = batch['Y_recon']
                
                print(f'????  Saving Crops & Preds for epoch {epoch}!')
                for b in range(0, X.shape[0], 1):
                    image_arr = X[b][0].detach().cpu().numpy().astype(np.float32)
                    label_arr = Y[b][0].detach().cpu().numpy().astype(np.float32)
                    if cfg.tasks.name == 'vec':
                        recon_arr = out_d['recon'][b][0].sigmoid().detach().cpu().numpy().astype(np.float32)
                    else:
                        recon_arr = out[b][0].sigmoid().detach().cpu().numpy().astype(np.float32)
                    
                    ref_im = samples[b].image.sitk_image
                    save_volume(image_arr, ref_im, os.path.join(exp_save_path,
                        f'ep{epoch}_b{b}_input.nii.gz'))
                    save_volume(label_arr, ref_im, os.path.join(exp_save_path,
                        f'ep{epoch}_b{b}_orig.nii.gz'))
                    save_volume(recon_arr, ref_im, os.path.join(exp_save_path,
                        f'ep{epoch}_b{b}_recon.nii.gz'))
                    
                    if out.shape[1] == 2:  # save boundary prediction as well
                        bound_targ_arr = Y_boundary[b][0].detach().cpu().numpy().astype(np.float32)
                        save_volume(bound_targ_arr, ref_im, os.path.join(exp_save_path,
                        f'ep{epoch}_b{b}_boundY.nii.gz'))
                        bound_arr = out[b][1].sigmoid().detach().cpu().numpy().astype(np.float32)
                        save_volume(bound_arr, ref_im, os.path.join(exp_save_path,
                        f'ep{epoch}_b{b}_bound.nii.gz'))

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
            if rank == 0:
                output.subsubsection('Test Metrics')
            N_test_exs = len(data_d['df'][data_d['df']['subset'] == 'test'])
            test_mets = test_metrics(cfg, model, test_set, epoch, 
                inference_metrics_queue, N_test_exs, name='test',
                overlap_perc=cfg.test.patch_overlap_perc,
                criterion=criterion)
            if rank == 0:
                epoch_mets.update(test_mets, tracked_epmets, 'test_ep_')
        
        scheduler.step(epoch=epoch, value=0)

        if rank == 0:
            epoch_mets.print(pre="\nEpoch Stats\n-----------\n")

            force_sum = True if epoch == cfg.train.epochs - 1 else False
            tracker.update(epoch_mets, log=True, summarize=SUMMARIZE,
                           after_epoch=int(SAVE_AFTER * tot_epochs), 
                           force_summarize=force_sum)
            
            if not debug['save']:
                print(f'????  Model-saving functionality is off! \n')
            if debug['save'] and epoch >= SAVE_AFTER * tot_epochs:
                save_model(cfg, model.state_dict(), tracker, epoch, source_code)
        
            WATCH.toc(name='epoch')
        synchronize()
        # --  End of epoch -- #



# ========================================================================== #
# * ### * ### * ### *      Training Functionality        * ### * ### * ### * #
# ========================================================================== #


def test_metrics(cfg, model, dataset, epoch, test_metrics_queue, 
                 num_examples, name='test', overlap_perc=0.2, criterion=None):
    """
    Args:
        num_examples: needed for DDP since rank 0 worker needs a total count
            of examples so it knows how many times to pull metrics from Q.
    """
    device = cfg.experiment.device
    if cfg.experiment.distributed:
        cba_device = cfg.experiment.device
        torch.cuda.set_device(cfg.experiment.rank)
    else:
        cba_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if len(cfg.experiment.gpu_idxs) > 1:
            cba_device = f'cuda:{len(cfg.experiment.gpu_idxs) - 1}'

    loss_accum, loss_norm = 0, 0
    WATCH.tic(name.title())
    epmeter = statistics.EpochMeters()
    with torch.no_grad():
        model.eval()

        WATCH.tic(f'{name}_iter')
        samples, processes = [], []
        for i in range(len(dataset)):
            if cfg.experiment.rank == 0:
                print(f' ???????  Inference for example {i+1}.')
            
            example_d = dataset[i]
            sample = example_d['sample']
            image = example_d['tensor']

            samples.append(sample)

            # Create Chop-Batch-Aggregate inference helper
            test_batch_size = cfg.test.batch_size
            num_classes = 1
            overlap = [int(overlap_perc * s) for s in cfg.train.patch_size]
            cba = CBA(image, cfg.train.patch_size, overlap, 
                      test_batch_size, num_classes, device='cpu')
            if cfg.experiment.rank == 0:
                print(f'     Getting predictions for {len(cba)} batches.',
                      flush=True)
            
            # pstart = time.time()
            for bidx, batch in enumerate(cba):
                crops, locations = batch
                B = len(locations)
                
                # create vectors & labels
                vectors, records = [], []
                for b in range(B):
                    crop_lower, crop_upper = locations[b][:3], locations[b][3:]
                    nine_points = [
                        (crop_upper + crop_lower) / 2,  # patch center
                        crop_lower,
                        crop_upper,
                        [crop_lower[0], crop_upper[1], crop_upper[2]],  # LR diag from lower
                        [crop_upper[0], crop_lower[1], crop_lower[2]],  # UR diag upper
                        [crop_lower[0], crop_lower[1], crop_upper[2]],
                        [crop_upper[0], crop_upper[1], crop_lower[2]],
                        [crop_lower[0], crop_upper[1], crop_lower[2]],
                        [crop_upper[0], crop_lower[1], crop_upper[2]],
                    ]
                    from experiments.prevec.vector import Vector3d
                    vol_patadj_center = [(crop_lower[0] + crop_upper[0]) / 2,
                                         image.shape[1] / 2,
                                         image.shape[2] / 2]
                    vectors.append([Vector3d(pt, vol_patadj_center) 
                                    for pt in nine_points])
                records = [{
                    'SpatialPretrainCropper3d': {
                        'input_shape': image.shape,
                        'input_volume_center': np.array(image.shape) / 2
                    }
                }] * len(locations)
                Y = create_prevec_targ(vectors, records, cfg).to(device)
                                
                crops = crops.to(device)
                out_d = model(crops)
                logits = out_d['out'] if isinstance(out_d, dict) else out_d
                
                if criterion:
                    loss_d = criterion(logits, Y)
                    loss = loss_d['loss'] if isinstance(loss_d, dict) else loss_d
                    loss_accum += loss
                    loss_norm += B
            del crops; del logits 

    # Accumulate metrics & print results
    final_metrics_d = {'loss': loss_accum.item()}
    if cfg.experiment.rank == 0:
        print(f'({name.title()}) Epoch {epoch} total loss: '
              f'{loss_accum.item():.4f}')

    if cfg.experiment.distributed:
        raise NotImplementedError()
        torch.cuda.empty_cache()
        sum_dice = sum([d['dice_mean'] for d in metric_results])
        sum_jaccard = sum([d['jaccard_mean'] for d in metric_results])
        tensor = torch.tensor([sum_dice, sum_jaccard]).float().cuda()
        
        torch.distributed.all_reduce(tensor)
        if cfg.experiment.rank == 0:
            dice_mean = tensor[0].item() / num_examples
            jaccard_mean = tensor[1].item() / num_examples
            print(f'??? {name.title()} DDP Results for {num_examples} images: \n'
                  f'       Avg Dice: {dice_mean:.2f} \n'
                  f'       Avg Jaccard: {jaccard_mean:.2f} \n', flush=True)

            WATCH.toc(name.title(), disp=True)
            return {
                'dice_mean': dice_mean,
                'jaccard_mean': jaccard_mean
            }
        return

    WATCH.toc(name.title(), disp=True)
    return final_metrics_d


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


def create_prevec_targ(vectors, records, cfg):
    prevec_cfg = cfg.tasks.vec
    vec_indices = prevec_cfg.pred_indices
    
    Y = torch.zeros(len(vectors), 3 * len(vec_indices))
    # Y[b] = [mag_v0, theta_v0, phi_v0, mag_v1, theta_v1, phi_v1, .... ]
    for b, (vecs, rec) in enumerate(zip(vectors, records)):
        vol_input_shape = rec['SpatialPretrainCropper3d']['input_shape']
        vol_center = rec['SpatialPretrainCropper3d']['input_volume_center']
        radius = math.sqrt(sum([n ** 2 for n in vol_center[1:]]))
                
        for i, vi in enumerate(vec_indices):  
            vec = vecs[vi]

            rho = vec.magnitude
            mag = rho / radius
            assert mag >= 0, f'Magnitude: {rho}, Radius {radius} on {rec} {vec}'
            if mag >= 1:
                # print(f'Batch{b}-I{i} | Mag: {rho}, Radius {radius} on {rec} {vec}')
                mag = 1
            Y[b, i * 3] = mag
            
            theta = vec.theta
            assert 0 <= theta <= math.pi, f'Theta: {theta} on {rec} {vec}'
            Y[b, i * 3 + 1] = theta / math.pi
            
            phi = vec.phi 
            assert -math.pi <= phi <= math.pi, f'Phi: {phi} on {rec} {vec}'
            Y[b, i * 3 + 2] = phi / math.pi
            # print(Y)
    return Y


def mem():
    """ Get primary GPU card memory usage. """
    if not torch.cuda.is_available():
        return -1.
    mem_map = devices.get_gpu_memory_map()
    prim_card_num = gpu_indices[0]
    return mem_map[prim_card_num]/1000