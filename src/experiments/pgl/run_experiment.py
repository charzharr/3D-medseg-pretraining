""" Module pgl/run_experiment.py
Reimplementation of Prior-Guided Learning paper.
    https://arxiv.org/abs/2011.12640
    
PGL Training Details:
    - Volume [-1024, 325], znorm -> 16x96x96 patches
    - 128 batch size
    - Trained via LARS (cosine lr, 2 epoch warmup, 0.2 LR) for 500 epochs

Implementation Notes:
    - Baseline use of Adam with 
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

from experiments import setup
from experiments.pgl import data_setup
from lib.utils import devices, timers, statistics
from lib.utils.train import ramps
from lib.utils.io import output
from lib.nets.ema import create_ema_model, update_ema_model



WATCH = timers.StopWatch()
SAVE_LAST = False
SAVE_BEST_METRICS = {  # metric (str): max_better (bool)
    'val_ep_auc': True,
    'val_ep_f1': True
}
SUMMARIZE = {
    'triggers': ['val_ep_auc'],
    'saves': ['val_ep_f1', 'val_ep_auc',
              'test_ep_f1', 'test_ep_sen', 'test_ep_spe',
              'test_ep_acc', 'test_ep_auc']
}


def run(cfg, checkpoint=None):
    
    # ------------------ ##  Experiment Setup  ## ------------------ #
    
    output.header_one('I. PGL Training Components Setup')
    
    global gpu_indices
    global device
    gpu_indices = cfg['experiment']['gpu_idxs']
    device = cfg['experiment']['device']
    debug = cfg['experiment']['debug']

    print(f"[Experiment Settings (@pgl/run_experiment.py)]")
    print(f" > Prepping train config..")
    print(f"\t - experiment:  {cfg['experiment']['project']} - "
          f"{cfg['experiment']['name']}, id({cfg['experiment']['id']})")
    print(f"\t - batch_size {cfg['train']['batch_size']}, "
          f"\t - start epoch: {cfg['train']['start_epoch']}/"
          f"{cfg['train']['epochs']},")
    print(f"\t - Optimizer ({cfg['train']['optimizer']['name']}): "
          f"\t - lr {cfg['train']['optimizer']['lr']}, "
          f"\t - wt_decay {cfg['train']['optimizer']['wt_decay']}, "
          f"\t - mom {cfg['train']['optimizer']['momentum']}, ")
    print(f"\t - Scheduler ({cfg['train']['scheduler']['name']}): "
          f"\t - rampup: {cfg['train']['scheduler']['rampup_rates']}\n")

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
    from .pgl_unet3d import UNet3D
    model = UNet3D(n_input=1, n_class=1, act='relu')
    ema_model = create_ema_model(UNet3D(n_input=1, n_class=1, act='relu'))
    model, ema_model = model.to(device), ema_model.to(device)
    if len(gpu_indices) > 1:
        print(f'  * {len(gpu_indices)} GPUs, using nn.DataParallel.')
        model = nn.DataParallel(model)
        ema_model = nn.DataParallel(ema_model)

    output.header_three('Criterion + Optimizer + Scheduler Setup')
    from lib.assess.losses3d import BYOL3d
    criterion = BYOL3d()
    
    optimizer = setup.get_optimizer(cfg, model.parameters())
    scheduler = setup.get_scheduler(cfg, optimizer)
    
    # Data Pipeline
    output.header_three('Data Setup')
    from .data_setup import get_data_components
    data_d = get_data_components(cfg)  # PGL: train_df, train_set, train_loader
    
    train_df = data_d['train_df']
    train_set = data_d['train_set']
    train_loader = data_d['train_loader']

    # ------------------ ##  Training Action  ## ------------------ #
    
    output.header_one('II. Training')
    if debug['overfitbatch']:
        batches = []
        for i, batch in enumerate(train_loader):
            if i in (3, 4, 5):
                batches.append(batch)
            elif i > 5:
                break
        train_loader = batches

    tot_epochs = cfg['train']['epochs'] - cfg['train']['start_epoch']
    global_iter = 0
    for epoch in range(cfg['train']['start_epoch'], cfg['train']['epochs']):
        output.subsection(f'Starting Epoch {epoch+1} (lr: {scheduler.lr:.7f})')

        WATCH.tic('epoch')
        model.train()
        ema_model.train()
        epmeter = statistics.EpochMeters()
        WATCH.tic('iter')

        for it, batch in enumerate(train_loader):

            ids = batch[0]
            X = batch[1].to(device)
            Y = batch[2].long().to(device, non_blocking=True)

            if ema_model is not None:
                with torch.no_grad():
                    emaout = ema_model(X)

            for p in range(cfg.train.passes):
                out = model(X)
                optimizer.zero_grad()
                loss = lab_criterion(out, Y)
                loss.backward()
                optimizer.step()

                if ema_model is not None:
                    update_ema_model(model, ema_model, cfg.model.alpha)

            X, Y, preds = X.detach(), Y.detach(), out.detach().softmax(-1)
            targs = torch.zeros(preds.shape, device=device).scatter_(1, Y.view(-1, 1), 1)

            # update df
            for b in range(X.shape[0]):
                df.loc[df['id'] == ids[b], f'epoch{epoch}_pred'] = int(preds[b].argmax())
                for ci, score in enumerate(preds[b]):
                    df.loc[df['id'] == ids[b], f'epoch{epoch}_c{ci}'] = float(score)

            metrics_d = metrics.multiclass_metrics(preds, targs)
            metrics_d['loss'] = loss.item()

            iter_time = WATCH.toc('iter', disp=False)
            missing_clsids = sorted(set(range(out.shape[1])) - set([n.item() for n in Y.unique()]))
            ast = '*' if missing_clsids else ''
            print(
                f"\n    Iter {it + 1}/{len(train_loader)} ({iter_time:.1f} sec, {mem():.1f} GB) - "
                f"loss {loss.item():.3f} (missing classes: {missing_clsids})\n"
                f"\t\t f1{ast} {float(metrics_d['F1']):.3f}, "
                f"auc {float(metrics_d['AUC']):.3f}"
            )

            update_mets = ['loss', 'ACC', 'F1', 'SEN', 'SPE', 'TPs', 'FPs', 'FNs', 'TNs']
            epmeter.update({k: metrics_d[k] for k in update_mets})

            global_iter += 1
            WATCH.tic('iter')
            if debug['break_train_iter'] and it >= 2:
                break

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
        if epoch % debug['test_every_n_epochs'] == debug['test_every_n_epochs'] - 1:
            print(f"\nValidating..\n")
            val_mets = test_metrics(model, val_loader, df, epoch, name='val')
            for k, v in val_mets.items():
                if k in ['AUC', 'ACC', 'SEN', 'SPE', 'bACC', 'F1']:
                    epoch_metrics_d['val_ep_' + k.lower()] = v

            if ema_model is not None:
                print(f" \n[EMA-Model]")
                ema_val_mets = test_metrics(ema_model, val_loader, df, epoch, name='val')
                for k, v in ema_val_mets.items():
                    if k in ['AUC', 'ACC', 'SEN', 'SPE', 'bACC', 'F1']:
                        epoch_metrics_d['val_ep_ema_' + k.lower()] = v

            print(f"\nTesting..\n")
            test_mets = test_metrics(model, test_loader, df, epoch, name='test')
            for k, v in test_mets.items():
                if k in ['AUC', 'ACC', 'SEN', 'SPE', 'bACC', 'F1']:
                    epoch_metrics_d['test_ep_' + k.lower()] = v

            if ema_model is not None:
                print(f" \n[EMA-Model]")
                ema_test_mets = test_metrics(ema_model, test_loader, df, epoch, name='test')
                for k, v in ema_test_mets.items():
                    if k in ['AUC', 'ACC', 'SEN', 'SPE', 'bACC', 'F1']:
                        epoch_metrics_d['test_ep_ema_' + k.lower()] = v

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
    if debug['save']:
        filename = f"{cfg['experiment']['id']}_{cfg['experiment']['name']}_last.csv"
        curr_path = pathlib.Path(__file__).parent.absolute()
        save_path = os.path.join(curr_path, filename)
        df.to_csv(save_path)


def test_metrics(model, loader, df, epoch, name='test'):
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
            targs = torch.zeros(preds.shape, device=device).scatter_(1, Y.view(-1, 1), 1)

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
    return mem_map[prim_card_num] / 1000






