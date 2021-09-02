""" Module run_experiment.py (Author: Charley Zhang, 2021)
Experiment for fully-supervised classification.

Used for:
  - Testing the quality of features learned during pretraining.
"""

import sys, os
import math, random
import pathlib
import pprint
import numpy as np
import pandas as pd

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

    # Training Components: model, criterion, optimizer, scheduler
    output.header_three('Model Setup')
    from experiments.pgl.pgl_unet3d import UNet3D
    model = UNet3D(n_input=1, n_class=1, act='relu')
    model = model.to(device)
    if len(gpu_indices) > 1:
        print(f'  * {len(gpu_indices)} GPUs, using nn.DataParallel.')
        model = nn.DataParallel(model)
    
    output.header_three('Criterion + Optimizer + Scheduler Setup')
    
    criterion = setup.get_criterion(cfg)
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
            Y = batch['masks_1h'].long().to(device, non_blocking=True)
            
            out = model(X)
            optimizer.zero_grad()
            loss = criterion(out, Y)
            
            loss.backward()
            optimizer.step()

            # Iteration & Epoch Training Metrics
            preds = out.detach().cpu().softmax(-1)
            targs = Y.detach().cpu()

            iter_metrics_d = batch_metrics(preds, targs)
            iter_metrics_d['loss'] = loss.item()

            iter_time = WATCH.toc('iter', disp=False)
            missing_clsids = sorted(set(range(train_set.num_classes)) - 
                                    set([n.item() for n in Y.unique()]))
            ast = '*' if missing_clsids else ''
            print(
                f"\n    Iter {it+1}/{len(train_loader)} ({iter_time:.1f} sec, "
                f"{mem():.1f} GB) - "
                f"loss {loss.item():.3f} (missing classes: {missing_clsids})\n"
                f"\t\t dice{ast} {float(iter_metrics_d['dice_mean']):.3f}, "
                f"jaccard {float(iter_metrics_d['jaccard_mean']):.3f}"
            )

            update_mets = ['loss', 'dice_mean', 'jaccard_mean']
            epmeter.update({k: iter_metrics_d[k] for k in update_mets})
            
            global_iter += 1
            WATCH.tic('iter')
            if debug['break_train_iter'] and it >= 2: 
                break
        
        print(f'Epoch is done!')
        import sys; sys.exit(1)
        # Training epoch values
        epoch_metrics_d = {}
        train_mets = epmeter.avg(no_avg=['TPs', 'FPs', 'FNs', 'TNs'])
        for k, v in train_mets.items():
            if k in ['loss', 'AUC', 'ACC', 'SEN', 'SPE', 'F1']:
                epoch_metrics_d['train_ep_' + k.lower()] = v
            if k == 'TPs':
                fps = train_mets['FPs']
                fns = train_mets['FNs']
                f1 = ((2 * v) / (2 * v + fps + fns)).mean()
                epoch_metrics_d['train_ep_of1'] = f1

        # Test + Epoch Metrics
        if epoch % debug.test_every_n_epochs == debug.test_every_n_epochs - 1:
            output.subsubsection('Validation Metrics')
            val_mets = test_metrics(model, val_set, epoch, name='val')
            for k, v in val_mets.items():
                if k in ['dice', 'jaccard', 'hausdorff']:
                    epoch_metrics_d['val_ep_' + k.lower()] = v
            
            output.subsubsection('Testing Metrics')
            test_mets = test_metrics(model, test_set, epoch, name='test')
            for k, v in test_mets.items():
                if k in ['dice', 'jaccard', 'hausdorff']:
                    epoch_metrics_d['test_ep_' + k.lower()] = v
                    
        print("\nEpoch Stats\n-----------")
        for k, v in epoch_metrics_d.items():
            if isinstance(v, float):
                print(f"  {k: <21} {v:.4f}")
            else:
                print(f"  {k: <21} {v:d}")

        force_sum = True if epoch == cfg.train.epochs - 1 else False
        tracker.update(epoch_metrics_d, log=True, summarize=SUMMARIZE,
            after_epoch=int(SAVE_AFTER * tot_epochs), force_summarize=force_sum)
        if debug['save'] and epoch >= SAVE_AFTER * tot_epochs:
            save_model(cfg, model.state_dict(), lab_criterion, optimizer, 
                       tracker, epoch)
        scheduler.step(epoch=epoch, value=0)
        WATCH.toc(name='epoch')
        # End of epoch #
    
    # save csv for distribution analysis
    if debug.save:
        filename = f"{cfg['experiment']['id']}_{cfg['experiment']['name']}_last.csv"
        curr_path = pathlib.Path(__file__).parent.absolute()
        save_path = os.path.join(curr_path, filename)
        df.to_csv(save_path)


def batch_metrics(preds, targs):
    """
    preds: BxCxDxHxW, targs: BxCxDxHxW
    Accum: https://github.com/Project-MONAI/MONAI/blob/67aa4cfba3a7e32786f22c5767bf6772c2a393d9/monai/metrics/utils.py#L108
    """
    cdj_tuple = batch_cdj_metrics(preds, targs, ignore_background=True)
    
    ec_dice = cdj_tuple.dice
    ec_jaccard = cdj_tuple.jaccard
    
    return {
        'dice_all': ec_dice,
        'dice_class': ec_dice.mean(0),  # shape=(C,)
        'dice_batch': ec_dice.mean(1),  # shape=(B,)
        'dice_mean': ec_dice.mean(1).mean(),
        'jaccard_all': ec_jaccard,
        'jaccard_class': ec_jaccard.mean(0),  # shape=(C,)
        'jaccard_batch': ec_jaccard.mean(1),  # shape=(B,)
        'jaccard_mean': ec_jaccard.mean(1).mean(),
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


def save_model(cfg, state, crit, opt, tracker, epoch):
    # check
    end = 'last'
    for met, max_gud in SAVE_BEST_METRICS.items():
        if tracker.is_best(met, max_better=max_gud):
            end = f"best-{met.split('_')[0]}-{met.split('_')[-1]}"
            print(f"(emain/save_model) {end}: {tracker.best(met, max_better=max_gud):.2f}")
            break
    if end == 'last' and not SAVE_LAST:
        return
    
    curr_path = pathlib.Path(__file__).parent.absolute()
    fn_start = f"{cfg['experiment']['id']}_{cfg['experiment']['name']}_"
    rm_files = [f for f in os.listdir(curr_path) if f[-3:] == 'pth' \
                and fn_start in f and end in f]
    assert len(rm_files) <= 1
    if rm_files:
        rm_file = os.path.join(curr_path, rm_files[0])
        print(f"Deleting file -x {rm_files[0]}")
        os.remove(rm_file)
    
    filename = fn_start + f"ep{epoch}_" + end
    print(f"Saving model -> {filename}")
    
    save_path = os.path.join(curr_path, filename + '.pth')
    torch.save({
        'state_dict': state,
        'criterion': crit,
        'optimizer': opt,
        'tracker': tracker,
        'config': cfg
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






