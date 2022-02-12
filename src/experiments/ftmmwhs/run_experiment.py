""" Module run_experiment.py (Author: Charley Zhang, 2021)
Experiment for fully-supervised classification.

Used for:
  - Testing the quality of features learned during pretraining.
  
Settings by Architecture:
    - NNUNet3d: BatchSize 4 (32x160x160)
    - DVN3d: bs4 32x160x160 9gb, bs4 32x192x192 14gb
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
from lib.nets import init as init_net
from data.transforms.crops.inference import ChopBatchAggregate3d as CBA
from data.transforms.resize import resize_segmentation3d

WATCH = timers.StopWatch()

SAVE_AFTER = 0.0  # Only save model when over this percentage of epochs done.
SAVE_MOST_RECENT_MODEL = False  # Only save the best model
SAVE_BEST_METRICS = {  # metric (str): max_better (bool)
    'test_ep_dice_mean': True,
}
SUMMARIZE = {
    'triggers': ['test_ep_dice_mean'],
    'saves': ['test_ep_dice_mean', 'test_ep_jaccard_mean']
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
    if rank == 0:
        output.header_three('Model Setup')
    model_d = get_model(cfg)
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
    from lib.assess.nnunet_loss import (DC_and_CE_loss, SoftDiceLoss,
                                        softmax_helper)
    from .data_setup import weights_d
    if cfg.train.criterion.name == 'soft_dice_nnunet':
        criterion = SoftDiceLoss(
            apply_nonlin=softmax_helper, 
            do_bg=cfg.train.criterion.soft_dice_nnunet.do_bg)
    elif cfg.train.criterion.name == 'dice_ce_nnunet':
        wt_key = cfg.train.criterion.dice_ce_nnunet.ce_kw.weights_key
        weight = None if not wt_key else data_setup.weights_d[wt_key]
        ce_kw = {'weight': weight}
        dc_kw = cfg.train.criterion.dice_ce_nnunet.dc_kw
        criterion = DC_and_CE_loss(dc_kw, ce_kw, ignore_label=None,
                                   weight_dice=0.5, weight_ce=0.5)
        criterion = criterion.to(device)
    
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
        model.train()
        epmeter = statistics.EpochMeters()
        WATCH.tic('iter')

        for it, batch in enumerate(train_loader):
            
            samples = batch['samples']
            records = batch['records']
            
            X = batch['images'].float().to(device, non_blocking=True)
            Y_id = batch['masks'].to(torch.uint8).to(device, non_blocking=True)

            out_d = model(X)
            out = out_d['out'] if isinstance(out_d, dict) else out_d
            loss_d = criterion(out, Y_id)
            loss = loss_d['loss']

            if cfg.train.deep_sup and isinstance(out_d, dict):
                num_classes = samples[0].mask.num_classes
                weights, losses = [1], [loss]
                
                if cfg.model.name == 'dvn3d':
                    weights.append(1/3)
                    logits = out_d['2x']
                    losses.append(criterion(logits, Y_id)['loss'])
                elif cfg.model.name in ('denseunet3d', 'unet3p3d', 'res2unet3d'):
                    resolutions = [2, 4, 8]
                    if cfg.model.name == 'unet3p3d':
                        resolutions.append(16)
                    for resolution in resolutions:
                        weights.append(1/resolution)
                        logits = out_d[f'{str(resolution)}x']
                        losses.append(criterion(logits, Y_id)['loss'])
                else:
                    for resolution in (2, 4, 8):
                        weights.append(1/resolution)
                        logits = out_d[f'{str(resolution)}x']

                        size = [s // resolution for s in Y_id.shape[2:]]
                        size[0] *= 2
                        targs = torch.zeros(list(Y_id.shape[:2]) + size)
                        for b in range(Y_id.shape[0]):
                            targs[b][0] = resize_segmentation3d(
                                            Y_id[b][0], size, 
                                            class_ids=list(range(1, num_classes)))
                            # save_image(targs[b][0], f'it{it}_res{resolution}_mask{b}', 
                            #             samples[b], history=records[b], is_mask=True)
                        losses.append(criterion(logits, targs.to(device))['loss'])
                weights = [w/sum(weights) for w in weights]
                # print(weights)
                # print(losses)
                loss = sum([w * l for w, l in zip(weights, losses)])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Iteration & Epoch Training Metrics 
            if rank == 0:
                targs = batch['masks_1h'].to(torch.uint8)
                with torch.no_grad():
                    pred_ids = out.argmax(1).unsqueeze(1)
                    del out; del out_d
                    preds = torch.zeros(targs.shape, dtype=torch.uint8,
                                        device=device)
                    preds.scatter_(1, pred_ids, 1)
                    preds = preds.cpu()
            
                iter_metrics_d = batch_metrics(preds, targs, 
                                               ignore_background=False)
                iter_metrics_d['loss'] = loss.item()

                iter_time = WATCH.toc('iter', disp=False)
                loss_str = criterion.get_loss_string(loss_d)
                print(
                    f"\n    Iter {it+1}/{len(train_loader)} "
                    f"({iter_time:.1f} sec, {mem():.1f} GB) - "
                    f"{loss_str} \n"
                    f"      jaccard {iter_metrics_d['jaccard_mean']:.3f}, "
                    f"dice {iter_metrics_d['dice_mean']:.3f}\n"
                    f"       {iter_metrics_d['dice_class']}", flush=True
                )

                update_mets = ['loss', 'dice_mean', 'jaccard_mean']
                epmeter.update({k: iter_metrics_d[k] for k in update_mets})

            # Save images, masks, and predictions
            if rank == 0 and cfg.experiment.debug.overfitbatch and it == 0:
                from data.transforms.z_normalize import ZNormalize
                for i in range(X.shape[0]):
                    img = ZNormalize().invert(X[i][0], records[i]['ZNormalize'])
                    img = img.detach().cpu().numpy().astype(np.int16)
                    mask = Y_id[i][0].detach().cpu().numpy().astype(np.uint8)
                    pred = pred_ids[i][0].detach().cpu().numpy().astype(np.uint8)

                    print(f'⭐ Iteration {it+1}, Example {i+1} Info..')
                    sim = samples[i].image.sitk_image
                    print(samples[i].image)
                    print(samples[i].mask)
                    print(records[i]['ScaledForegroundCropper3d'])

                    scrop = sitk.GetImageFromArray(img)
                    scrop.SetSpacing(sim.GetSpacing())
                    scrop.SetOrigin(sim.GetOrigin())
                    scrop.SetDirection(sim.GetDirection())
                    sitk.WriteImage(scrop, f'ofit_crop{i + 1}.nii.gz')

                    smask = sitk.GetImageFromArray(mask)
                    smask.SetSpacing(sim.GetSpacing())
                    smask.SetOrigin(sim.GetOrigin())
                    smask.SetDirection(sim.GetDirection())
                    sitk.WriteImage(smask, f'ofit_targ{i + 1}.nii.gz')
                    
                    spred = sitk.GetImageFromArray(pred)
                    spred.SetSpacing(sim.GetSpacing())
                    spred.SetOrigin(sim.GetOrigin())
                    spred.SetDirection(sim.GetDirection())
                    sitk.WriteImage(spred, f'ofit_pred{i + 1}.nii.gz')

            global_iter += 1
            WATCH.tic('iter')
            if debug['break_train_iter'] and it >= 5: 
                break

        out = None; out_d = None  # save mem for inference
        del X; del Y_id; del loss  

        # -- Epoch Values -- #
        epoch_mets = statistics.EpochMetrics()

        # Training epoch values
        train_mets = epmeter.avg()
        tracked_epmets = ['loss', 'dice_mean', 'jaccard_mean']
        epoch_mets.update(train_mets, tracked_epmets, 'train_ep_')

        # Test + Epoch Metrics
        test_every_n = debug.test_every_n_epochs
        if epoch % test_every_n == test_every_n - 1:
            if rank == 0:
                output.subsubsection('Test Metrics')
            N_test_exs = len(data_d['df'][data_d['df']['subset'] == 'test'])
            test_mets = test_metrics(cfg, model, test_set, epoch, 
                inference_metrics_queue, N_test_exs, name='test')
            if rank == 0:
                epoch_mets.update(test_mets, tracked_epmets, 'test_ep_')
            
            # output.subsubsection('Testing Metrics')
            # test_mets = test_metrics(cfg, model, test_set, epoch, name='test')
            # update_epoch_mets(test_mets, tracked_epmets, 'test_ep_')
        
        scheduler.step(epoch=epoch, value=0)

        if rank == 0:
            epoch_mets.print(pre="\nEpoch Stats\n-----------\n")

            force_sum = True if epoch == cfg.train.epochs - 1 else False
            tracker.update(epoch_mets, log=True, summarize=SUMMARIZE,
                           after_epoch=int(SAVE_AFTER * tot_epochs), 
                           force_summarize=force_sum)
            
            if not debug['save']:
                print(f'🚨  Model-saving functionality is off! \n')
            if debug['save'] and epoch >= SAVE_AFTER * tot_epochs:
                save_model(cfg, model.state_dict(), tracker, epoch, source_code)
        
            WATCH.toc(name='epoch')
        synchronize()
        # --  End of epoch -- #


def get_model(cfg, class_bal_bias=False):
    num_classes = cfg.data[cfg.data.name].num_classes
    in_channels, deep_sup = 1, cfg.train.deep_sup
    img_size = cfg.train.patch_size

    norm = cfg.model.norm
    act = cfg.model.act
    
    print(f'[NET] Model={cfg.model.name}, Class-Balanced Biases={class_bal_bias}')
    weights = data_setup.weights_d['mmwhs']
    percs = 1 / weights
    approx_logits = torch.log(percs)
    
    if 'cumednet' in cfg.model.name:
        from lib.nets.volumetric.cumednet3d import CUMedNet3d
        model = CUMedNet3d(in_channels, num_classes, 
                           stage_counts=[3, 4, 6, 3], init_channels=32,
                           stage_expansions=[2, 3, 6, 12], use_add=True,
                           concat_channels=16)
        if class_bal_bias:
            with torch.no_grad():
                model.final_conv2.bias = torch.nn.Parameter(approx_logits.clone())
    elif cfg.model.name == 'denseunet3d':
        from lib.nets.volumetric.denseunet3d import get_model as get_dunet
        model = get_dunet(cfg.model.denseunet3d.layers, num_classes=num_classes, 
                          deconv=True, deep_sup=deep_sup, norm=norm, act=act)
        if class_bal_bias:
            with torch.no_grad():
                final_biases = torch.nn.Parameter(approx_logits)
                model.conv2.bias = final_biases
    elif cfg.model.name == 'res2unet3d':
        from lib.nets.volumetric.res2unet3d import res2net50_v1b, res2net101_v1b
        if cfg.model.res2unet3d.layers == 50:
            model = res2net50_v1b(pretrained=False, 
                                  base_width=cfg.model.res2unet3d.base_width,
                                  act=act, norm=norm,
                                  in_channels=in_channels, 
                                  num_classes=num_classes)
        else:
            model = res2net101_v1b(pretrained=False, 
                                   base_width=cfg.model.res2unet3d.base_width,
                                   act=act, norm=norm,
                                   in_channels=in_channels, 
                                   num_classes=num_classes)
    elif cfg.model.name == 'hrnet3d':
        from lib.nets.volumetric.hrnet.seg_hrnet3d import get_model as get_hrnet
        hr_cfg_file = cfg.model.hrnet3d.cfg
        if not os.path.isfile(hr_cfg_file):
            src_path = pathlib.Path(__file__).parent.parent.parent
            hr_path = src_path / 'lib' / 'nets' / 'volumetric' / 'hrnet'
            hr_cfg_file = str(hr_path / hr_cfg_file)
        with open(hr_cfg_file, 'r') as f:
            hr_cfg = yaml.safe_load(f)
        model = get_hrnet(hr_cfg, in_channels, num_classes, pretrained=False)
    elif cfg.model.name == 'unetr3d':
        from monai.networks.nets.unetr import UNETR
        model = UNETR(in_channels, num_classes, img_size, 
                      feature_size=16, hidden_size=768, mlp_dim=3072,
                      num_heads=12, pos_embed='conv', norm_name='instance')
    elif cfg.model.name == 'unet3p3d':
        from lib.nets.volumetric.unet3p3d import UNet3plus
        model = UNet3plus(in_channels, num_classes, 24, deep_sup=deep_sup)
    elif cfg.model.name == 'nnunet3d':
        from experiments.ftbcv.nnunet3d import UNet3D
        model = UNet3D(n_input=1, n_class=num_classes, deep_sup=cfg.train.deep_sup)
        if class_bal_bias:
            with torch.no_grad():
                model.final_1x.bias = torch.nn.Parameter(approx_logits.clone())
                if cfg.train.deep_sup:
                    model.final_2x.bias = torch.nn.Parameter(approx_logits.clone())
                    model.final_4x.bias = torch.nn.Parameter(approx_logits.clone())
                    model.final_8x.bias = torch.nn.Parameter(approx_logits.clone())
    elif cfg.model.name == 'dvn3d':
        from lib.nets.volumetric.densevoxnet import DenseVoxNet
        model = DenseVoxNet(in_channels, num_classes, deep_sup=deep_sup)
        if class_bal_bias:
            raise NotImplementedError()
    elif 'cotr' in cfg.model.name:
        """ 
        Outputs: torch.Size([2, 14, 48, 192, 192])
                 torch.Size([2, 14, 48, 96, 96])
                 torch.Size([2, 14, 24, 48, 48])
                 torch.Size([2, 14, 12, 24, 24])
        """
        assert cfg.train.patch_size == [48, 192, 192]
        from lib.nets.volumetric.cotr.cotr3d import ResTranUnet
        model = ResTranUnet(norm_cfg='IN', img_size=cfg.train.patch_size,
                            num_classes=num_classes, 
                            deep_supervision=cfg.train.deep_sup)
    elif cfg.model.name == 'dod_unet3d':
        from experiments.ftbcv.dod_unet3d import UNet3D 
        model = UNet3D(num_classes=num_classes)
        with torch.no_grad():
            final_biases = torch.nn.Parameter(approx_logits)
            model.final_conv.conv1.bias = final_biases
    elif cfg.model.name == 'custom_denseunet3d':
        from lib.nets.volumetric.custom_denseunet3d import DenseUNet
        model = DenseUNet(name='201', out_channels=num_classes, deconv=True)
        with torch.no_grad():
            final_biases = torch.nn.Parameter(approx_logits)
            model.final_conv.bias = final_biases
    elif cfg.model.name == 'resmednet3d':  # already inited
        from lib.nets.volumetric.resnet3d_mednet import generate_resnet3d
        model = generate_resnet3d(in_channels=1, classes=num_classes, model_depth=34)
        with torch.no_grad():
            final_biases = torch.nn.Parameter(approx_logits)
            model.segm.conv_final.bias = final_biases
    elif cfg.model.name == 'genesis_unet3d':
        from experiments.ftbcv.ftbcv_unet3d import UNet3D as genesis_unet3d
        model = genesis_unet3d(n_input=1, n_class=num_classes, act='relu')
    elif cfg.model.name == 'gn_unet3d':
        from lib.nets.volumetric.resunet3d import UNet3D
        model = UNet3D(1, num_classes, final_sigmoid=False, is_segmentation=False)
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
    elif 'cotr' not in cfg.model.name and 'unetr' != cfg.model.name: 
        # Initialize
        init_type = cfg.model.init
        if init_type:
            init_net.init_weights(model, init_type=init_type)
        print(f'   (Model) Successfully initialized weights via {init_type}.')
        if cfg.model.name == 'genesis_unet3d':
            # Final layer init
            with torch.no_grad():
                final_biases = torch.nn.Parameter(approx_logits)
                model.out_tr.final_conv.bias = final_biases
            print(f'   (Model) Initialized last layer biases of Genesis-3DUnet to '
                  f'{model.out_tr.final_conv.bias}')

    return {
        'model': model
    }


def batch_metrics(preds, targs, ignore_background=True, naive_avg=False):
    """
    preds: BxCxDxHxW, targs: BxCxDxHxW
    Accum: https://github.com/Project-MONAI/MONAI/blob/67aa4cfba3a7e32786f22c5767bf6772c2a393d9/monai/metrics/utils.py#L108
    """
    def mean_with_nans(array):
        num_nans = np.count_nonzero(np.isnan(array))
        array = np.nan_to_num(array, nan=0)
        return array.sum() / (array.size - num_nans)

    cdj_tuple = batch_cdj_metrics(preds, targs,
                                  ignore_background=ignore_background)

    nt_conf = cdj_tuple.confusion  # named_tuple: 'tp', 'fp', 'fn', 'tn'
    nt_conf = CM(np.array(nt_conf.tp), np.array(nt_conf.fp), 
                 np.array(nt_conf.fn), np.array(nt_conf.tn))

    ec_dice = np.array(cdj_tuple.dice)  # ec = element class
    ec_jaccard = np.array(cdj_tuple.jaccard)
    ec_exists = np.array(cdj_tuple.exists)

    with np.errstate(divide='ignore', invalid='ignore'):
        if naive_avg:
            dice_class = ec_dice.mean(0)  # shape=(C,)
            dice_batch = ec_dice.mean(1)
            dice_batch_mean = dice_batch.mean()
            jaccard_class = ec_jaccard.mean(0)
            jaccard_batch = ec_jaccard.mean(1)
            jaccard_batch_mean = jaccard_batch.mean()
        else:
            dice_class = ec_dice.sum(0) / ec_exists.sum(0)
            dice_batch = ec_dice.sum(1) / ec_exists.sum(1)
            jaccard_class = ec_jaccard.sum(0) / ec_exists.sum(0)
            jaccard_batch = ec_jaccard.sum(1) / ec_exists.sum(1)
            
            dice_batch_mean = dice_batch.mean()
            jaccard_batch_mean = jaccard_batch.mean()
            # for a in (dice_class, dice_batch, jaccard_class, jaccard_batch):
            #     a[a == np.inf] = np.nan
            # batch_exist_count = np.sum(~np.isnan(dice_batch))
            # dice_mean = dice_batch.sum() / batch_exist_count
            # dice_mean = np.nan if dice_mean == np.inf else dice_mean
            # jaccard_mean = jaccard_batch.sum() / batch_exist_count
            # jaccard_mean = np.nan if jaccard_mean == np.inf else jaccard_mean

        ctp, cfp, cfn = nt_conf.tp.sum(0), nt_conf.fp.sum(0), nt_conf.fn.sum(0)
        batch_mean = mean_with_nans(2 * ctp / (2 * ctp + cfp + cfn))
        jaccard_mean = mean_with_nans(ctp / (ctp + cfp + cfn))

    return {
        'confusion_all': nt_conf,  # named tuple
        'dice_all': ec_dice,
        'dice_class': dice_class,  # shape=(C,)
        'dice_batch': dice_batch,  # shape=(B,)
        'dice_batch_mean': dice_batch_mean,
        'dice_mean': batch_mean,
        'jaccard_all': ec_jaccard,
        'jaccard_class': jaccard_class,  # shape=(C,)
        'jaccard_batch': jaccard_batch,  # shape=(B,)
        'jaccard_batch_mean': jaccard_batch_mean,
        'jaccard_mean': jaccard_mean
    }


def test_metrics(cfg, model, dataset, epoch, test_metrics_queue, 
                 num_examples, name='test', overlap_perc=0.2):
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

    WATCH.tic(name.title())
    epmeter = statistics.EpochMeters()
    with torch.no_grad():
        model.eval()

        WATCH.tic(f'{name}_iter')
        samples, processes = [], []
        for i in range(len(dataset)):
            if cfg.experiment.rank == 0:
                print(f' 🖼️  Inference for example {i+1}.')
            
            example_d = dataset[i]
            sample = example_d['sample']
            image = example_d['tensor']
            mask = example_d['mask_1h']  #  image: float32, mask: uint8

            samples.append(sample)

            # Create Chop-Batch-Aggregate inference helper
            test_batch_size = cfg.test.batch_size
            num_classes = sample.mask.num_classes
            overlap = [int(overlap_perc * s) for s in cfg.train.patch_size]
            cba = CBA(image, cfg.train.patch_size, overlap, 
                      test_batch_size, num_classes, device=cba_device)
            if cfg.experiment.rank == 0:
                print(f'     Getting predictions for {len(cba)} batches.',
                      flush=True)
            
            # pstart = time.time()
            for bidx, batch in enumerate(cba):
                crops, locations = batch
                crops = crops.to(device)
                out_d = model(crops)
                logits = out_d['out'] if isinstance(out_d, dict) else out_d
                cba.add_batch_predictions(logits, locations, act='none')
            # print(f'Predict {time.time() - pstart:.2f} sec ({mem():.1f} GB).')

            # Get final predictions, calculate metrics
            # pstart = time.time()
            del crops; del logits
            agg_predictions = cba.aggregate(ret='1_hot', cpu=True, numpy=False)
            # print(f'Agg {time.time() - pstart:.2f} sec')
            
            process = torch.multiprocessing.Process(target=test_metrics_worker,
                args=(test_metrics_queue, agg_predictions.unsqueeze(0),
                      mask.unsqueeze(0)))
            process.daemon = True
            process.start()
            processes.append(process)
            
            if cfg.experiment.rank == 0:
                elaps = WATCH.toc(f'{name}_iter', disp=False)
                print(f'Completed inference for vol {i+1} ({elaps:.2f} sec).\n',
                      flush=True)
            WATCH.tic(f'{name}_iter')

    # While last volume's metrics is computing, save sample
    if cfg.experiment.rank == 0 and epoch % 10 == 1:  # saves last volume in set  
        id_preds = agg_predictions.argmax(0).numpy().astype(np.uint16)
        sitk_pred = sitk.GetImageFromArray(id_preds, isVector=False)
        sitk_pred.SetOrigin(sample.mask.origin)
        sitk_pred.SetSpacing(sample.mask.spacing)
        sitk_pred.SetDirection(sample.mask.direction)

        curr_path = pathlib.Path(__file__).parent.absolute()
        filename = (f'{cfg.experiment.id}_ep{epoch}_lastex_'
                    f'prediction.nii.gz')
        save_path = os.path.join(curr_path, 'artifacts', filename)
        print(f'Saving prediction as "{filename}" | Success.', flush=True)
        sitk.WriteImage(sitk_pred, save_path)

    # Accumulate metrics & print results
    metric_results = []
    update_mets = ['dice_mean', 'jaccard_mean']
    data_cfg = cfg.data[cfg.data.name]
    for i in range(len(dataset)):
        mets = test_metrics_queue.get()
        metric_results.append(mets)
        # print(f'Rank {cfg.experiment.rank} Res{i+1} {mets["dice_mean"]}')

    for process in processes:
        process.join()

    for i, mets in enumerate(metric_results):
        samp_id, samp_idx = samples[i].id, samples[i].index
        epmeter.update({k: mets[k] for k in update_mets})
        if cfg.experiment.rank == 0:
            print(f'({name.title()}) Sample {samp_id}, idx={samp_idx} \n'
                  f'       Dice: {float(mets["dice_mean"]):.2f} \n'
                  f'        {mets["dice_class"]} \n'
                  f'       Jaccard: {float(mets["jaccard_mean"]):.2f} \n'
                  f'        {mets["jaccard_class"]}', flush=True) 

    if cfg.experiment.distributed:
        torch.cuda.empty_cache()
        sum_dice = sum([d['dice_mean'] for d in metric_results])
        sum_jaccard = sum([d['jaccard_mean'] for d in metric_results])
        tensor = torch.tensor([sum_dice, sum_jaccard]).float().cuda()
        
        torch.distributed.all_reduce(tensor)
        if cfg.experiment.rank == 0:
            dice_mean = tensor[0].item() / num_examples
            jaccard_mean = tensor[1].item() / num_examples
            print(f'⭐ {name.title()} DDP Results for {num_examples} images: \n'
                  f'       Avg Dice: {dice_mean:.2f} \n'
                  f'       Avg Jaccard: {jaccard_mean:.2f} \n', flush=True)

            WATCH.toc(name.title(), disp=True)
            return {
                'dice_mean': dice_mean,
                'jaccard_mean': jaccard_mean
            }
        return

    WATCH.toc(name.title(), disp=True)
    return epmeter.avg()


def test_metrics_worker(mp_metrics_queue, preds, targs):
    """
    Args:
        mp_queue: multiprocessing queue where metrics results are put
        preds: 1xCxDxHxW one-hot tensor 
        targs: 1xCxDxHxW one-hot tensor 
    """
    # start = time.time()
    # print(f'Worker got metrics! {preds.shape} {targs.shape}')
    mets = batch_metrics(preds, targs)
    # print(f'Metrics done! ({time.time() - start:.2f} sec)')
    # print(mets["dice_mean"], mets["dice_all"])
    # print(mp_metrics_queue, os.getpid())
    mp_metrics_queue.put(mets)
    # print(f'Successfully put in queue')


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


def ram(disp=False):
    """ Return (opt display) RAM usage of current process in megabytes. """
    import os, psutil
    process = psutil.Process(os.getpid())
    bytes = process.memory_info().rss
    mbytes = bytes // 1048576
    sys_mbytes = psutil.virtual_memory().total // 1048576
    if disp:
        print(f'🖥️  Current process (id={os.getpid()}) '
              f'RAM Usage: {mbytes:,} MBs / {sys_mbytes:,} Total MBs.')
    return mbytes


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





