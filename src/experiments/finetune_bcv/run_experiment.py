""" Module run_experiment.py (Author: Charley Zhang, 2021)
Experiment for fully-supervised classification.

Used for:
  - Testing the quality of features learned during pretraining.
"""

import sys, os
import math, random
import pathlib
import pprint
import warnings
import inspect
import numpy as np
import pandas as pd
from collections import namedtuple

import torch, torchvision
import torch.nn as nn
from torch.utils.data import DataLoader

import experiments, lib
from experiments import setup
from experiments.finetune_bcv import data_setup
from lib.utils import devices, timers, statistics
from lib.utils.train import ramps
from lib.utils.io import output
from lib.assess.seg_metrics import batch_cdj_metrics

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)},
                    suppress=True)

WATCH = timers.StopWatch()

SAVE_AFTER = 0.0  # Only save model when over this percentage of epochs done.
SAVE_LAST = False
SAVE_BEST_METRICS = {  # metric (str): max_better (bool)
    'val_ep_dice': True,
    'val_ep_jaccard': True,
    'val_ep_hausdorff': False,
}
SUMMARIZE = {
    'triggers': ['val_ep_dice'],
    'saves': ['val_ep_dice', 'val_ep_jaccard', 'val_ep_hausdorff',
              'test_ep_dice', 'test_ep_jaccard', 'test_ep_hausdorff']
}


def run(cfg, checkpoint=None):

    # ------------------ ##  Experiment Setup  ## ------------------ #
    output.header_one('I. BCV Finetune Training Components Setup')
    
    global gpu_indices
    global device
    gpu_indices = cfg['experiment']['gpu_idxs']
    device = cfg['experiment']['device']
    debug = cfg['experiment']['debug']

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
            'name': cfg['experiment']['id'] + '_' + cfg['experiment']['name'],
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

    # Training Components: model, criterion, optimizer, scheduler
    output.header_three('Model Setup')
    from .ftbcv_unet3d import UNet3D as genesis_unet3d
    from lib.nets.volumetric.resunet3d import UNet3D
    
    # model = genesis_unet3d(n_input=1, n_class=14, act='relu')
    model = UNet3D(1, 14, final_sigmoid=False, is_segmentation=False)
    model = model.to(device)
    if len(gpu_indices) > 1:
        print(f'  * {len(gpu_indices)} GPUs, using nn.DataParallel.')
        model = nn.DataParallel(model)
    
    output.header_three('Criterion + Optimizer + Scheduler Setup')
    from lib.assess.nnunet_loss import (DC_and_CE_loss, SoftDiceLoss,
                                        softmax_helper)
    criterion = SoftDiceLoss(apply_nonlin=softmax_helper, do_bg=False)
    criterion = DC_and_CE_loss({}, {}, ignore_label=None)
    
    optimizer = setup.get_optimizer(cfg, model.parameters())
    scheduler = setup.get_scheduler(cfg, optimizer)

    # Data Pipeline
    output.header_three('Data Setup')
    from .data_setup import get_data_components
    data_d = get_data_components(cfg)
    
    train_df = data_d['train_df']
    train_set = data_d['train_set']
    train_loader = data_d['train_loader']
    
    val_set = data_d['val_set']
    test_set = data_d['test_set']
    
    # ------------------ ##  Training Action  ## ------------------ #
    output.header_one('II. Training')
    
    if debug['overfitbatch']:
        batches = []
        for i, batch in enumerate(train_loader):
            if i in (1,2,3):
                batches.append(batch)
            elif i > 3: break
        train_loader = batches
    
    tot_epochs = cfg['train']['epochs'] - cfg['train']['start_epoch']
    global_iter = 0
    for epoch in range(cfg['train']['start_epoch'], cfg['train']['epochs']):
        output.subsection(f'Starting Epoch {epoch+1} (lr: {scheduler.lr:.7f})')
        
        WATCH.tic('epoch')
        model.train()
        epmeter = statistics.EpochMeters()
        WATCH.tic('iter')

        for it, batch in enumerate(train_loader):
            samples = batch['samples']
            records = batch['records']
            
            X = batch['images'].float().to(device, non_blocking=True)
            Y_id = batch['masks'].long().to(device, non_blocking=True)

            out_d = model(X)
            out = out_d['out']
            loss_d = criterion(out, Y_id)
            loss = loss_d['loss']
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Iteration & Epoch Training Metrics
            targs = Y = batch['masks_1h']
            pred_ids = out.detach().cpu().argmax(1).unsqueeze(1)
            preds = torch.zeros(Y.shape)
            preds.scatter_(1, pred_ids, 1)

            iter_metrics_d = batch_metrics(preds, targs, 
                                           ignore_background=False)
            iter_metrics_d['loss'] = loss.item()

            iter_time = WATCH.toc('iter', disp=False)
            loss_str = criterion.get_loss_string(loss_d)
            print(
                f"\n    Iter {it+1}/{len(train_loader)} ({iter_time:.1f} sec, "
                f"{mem():.1f} GB) - "
                f"{loss_str} \n"
                f"\t jaccard {iter_metrics_d['jaccard_mean']:.3f}\n"
                f"\t dice {iter_metrics_d['dice_mean']:.3f}\n"
                f"\t  {iter_metrics_d['dice_class']}\n"
            )

            update_mets = ['loss', 'dice_mean', 'jaccard_mean']
            epmeter.update({k: iter_metrics_d[k] for k in update_mets})
            
            global_iter += 1
            WATCH.tic('iter')
            if debug['break_train_iter'] and it >= 2: 
                break

        # -- Epoch Values -- #
        epoch_mets = statistics.EpochMetrics()

        # Training epoch values
        train_mets = epmeter.avg()
        tracked_epmets = ['loss', 'dice_mean', 'jaccard_mean']
        epoch_mets.update(train_mets, tracked_epmets, 'train_ep_')

        # Test + Epoch Metrics
        if False:# epoch % debug.test_every_n_epochs == debug.test_every_n_epochs - 1:
            output.subsubsection('Validation Metrics')
            val_mets = test_metrics(model, val_set, epoch, name='val')
            update_epoch_mets(val_mets, tracked_epmets, 'val_ep_')
            
            output.subsubsection('Testing Metrics')
            test_mets = test_metrics(model, test_set, epoch, name='test')
            update_epoch_mets(test_mets, tracked_epmets, 'test_ep_')
        epoch_mets.print(pre="\nEpoch Stats\n-----------\n")

        force_sum = True if epoch == cfg.train.epochs - 1 else False
        tracker.update(epoch_mets, log=True, summarize=SUMMARIZE,
            after_epoch=int(SAVE_AFTER * tot_epochs), force_summarize=force_sum)
        if debug['save'] and epoch >= SAVE_AFTER * tot_epochs:
            save_model(cfg, model.state_dict(), tracker, epoch, source_code)
        scheduler.step(epoch=epoch, value=0)
        WATCH.toc(name='epoch')
        # End of epoch #


def batch_metrics(preds, targs, ignore_background=False, naive_avg=False):
    """
    preds: BxCxDxHxW, targs: BxCxDxHxW
    Accum: https://github.com/Project-MONAI/MONAI/blob/67aa4cfba3a7e32786f22c5767bf6772c2a393d9/monai/metrics/utils.py#L108
    """
    cdj_tuple = batch_cdj_metrics(preds, targs,
                                  ignore_background=ignore_background)

    CM = namedtuple('ConfusionMatrix', ('tp', 'fp', 'fn', 'tn'))
    nt_conf = cdj_tuple.confusion  # named_tuple: tp, fp, tn, fn
    nt_conf = CM(np.array(nt_conf.tp), np.array(nt_conf.fp), 
                 np.array(nt_conf.fn), np.array(nt_conf.tn))

    ec_dice = np.array(cdj_tuple.dice)  # ec = element class
    ec_jaccard = np.array(cdj_tuple.jaccard)
    ec_exists = np.array(cdj_tuple.exists)

    with np.errstate(divide='ignore', invalid='ignore'):
        if naive_avg:
            dice_class = ec_dice.mean(0)  # shape=(C,)
            dice_batch = ec_dice.mean(1)
            dice_mean = dice_batch.mean()
            jaccard_class = ec_jaccard.mean(0)
            jaccard_batch = ec_jaccard.mean(1)
            jaccard_mean = jaccard_batch.mean()
        else:
            dice_class = ec_dice.sum(0) / ec_exists.sum(0)
            dice_batch = ec_dice.sum(1) / ec_exists.sum(1)
            jaccard_class = ec_jaccard.sum(0) / ec_exists.sum(0)
            jaccard_batch = ec_jaccard.sum(1) / ec_exists.sum(1)
            
            for a in (dice_class, dice_batch, jaccard_class, jaccard_batch):
                a[a == np.inf] = np.nan
            batch_exist_count = np.sum(~np.isnan(dice_batch))
            dice_mean = dice_batch.sum() / batch_exist_count
            dice_mean = np.nan if dice_mean == np.inf else dice_mean
            jaccard_mean = jaccard_batch.sum() / batch_exist_count
            jaccard_mean = np.nan if jaccard_mean == np.inf else jaccard_mean

    return {
        'confusion_all': nt_conf,  # named tuple
        'dice_all': ec_dice,
        'dice_class': dice_class,  # shape=(C,)
        'dice_batch': dice_batch,  # shape=(B,)
        'dice_mean': dice_mean,
        'jaccard_all': ec_jaccard,
        'jaccard_class': jaccard_class,  # shape=(C,)
        'jaccard_batch': jaccard_batch,  # shape=(B,)
        'jaccard_mean': jaccard_mean,
    }


def test_metrics(model, dataset, epoch, name='test'):
    
    return {}
    
    with torch.no_grad():
        model.eval()
        # testmeter = statistics.EpochMeters()
        WATCH.tic(name)
        accum_preds, accum_targs = None, None
        
        for it, batch in enumerate(loader):
            ids = batch[0]
            X = batch[1].to(device)
            Y = batch[2].long().to(device, non_blocking=True)

            preds = model(X).softmax(-1)
            targs = torch.zeros(preds.shape, device=device).scatter_(1, Y.view(-1,1), 1)
            
            # update df
            for b in range(X.shape[0]):
                df.loc[df['id'] == ids[b], f'epoch{epoch}_pred'] = int(preds[b].argmax())
                for ci, score in enumerate(preds[b]):
                    df.loc[df['id'] == ids[b], f'epoch{epoch}_c{ci}'] = float(score)
            
            if accum_preds is None:
                accum_preds = preds
                accum_targs = targs
            else:
                accum_preds = torch.cat((accum_preds, preds), dim=0)
                accum_targs = torch.cat((accum_targs, targs), dim=0)
            
        metrics_d = metrics.multiclass_metrics(accum_preds, accum_targs)

    test_time = WATCH.toc(name, disp=False)
    print(
        f"\n    {name} Results ({test_time:.1f} sec, {mem():.1f} GB) - \n"
        f"\t f1s {[f'{m:.2f}' for m in metrics_d['F1s']]} "
        f"(avg: {float(metrics_d['F1']):.3f}) \n"
        f"\t aucs {[f'{m:.2f}' for m in metrics_d['AUCs']]} (avg: {float(metrics_d['AUC']):.3f}) \n"
        f"\t accs {[f'{m:.2f}' for m in metrics_d['ACCs']]} (avg: {float(metrics_d['ACC']):.3f})"
    )
    
    return metrics_d


def save_model(cfg, state_dict, tracker, epoch, code_d):
    end = 'last'

    curr_path = pathlib.Path(__file__).parent.absolute()
    fn_start = f"{cfg['experiment']['id']}_{cfg['experiment']['name']}_"
    rm_files = [f for f in os.listdir(curr_path) if f[-3:] == 'pth' \
                and fn_start in f and end in f]
    assert len(rm_files) <= 1
    if rm_files:
        rm_file = os.path.join(curr_path, rm_files[0])
        print(f"Deleting file -x {rm_files[0]}")
        os.remove(rm_file)
    
    filename = fn_start + f"ep{epoch}_" + end + '.pth'
    print(f"Saving model -> {filename}")
    
    save_path = os.path.join(curr_path, filename)
    torch.save({
        'state_dict': state_dict,
        'code_dict': code_d,
        'tracker': tracker,
        'config': cfg,
        'epoch': epoch,
        },
        save_path
    )
    return

    # TODO metrics check
    end = 'last'
    for met, max_gud in SAVE_BEST_METRICS.items():
        if tracker.is_best(met, max_better=max_gud):
            end = f"best-{met.split('_')[0]}-{met.split('_')[-1]}"
            print(f"(emain/save_model) {end}: {tracker.best(met, max_better=max_gud):.2f}")
            break
    if end == 'last' and not SAVE_LAST:
        return
    

def mem():
    """ Get primary GPU card memory usage. """
    if not torch.cuda.is_available():
        return -1.
    mem_map = devices.get_gpu_memory_map()
    prim_card_num = gpu_indices[0]
    return mem_map[prim_card_num]/1000






