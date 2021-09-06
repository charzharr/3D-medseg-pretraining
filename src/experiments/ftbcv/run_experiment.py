""" Module run_experiment.py (Author: Charley Zhang, 2021)
Experiment for fully-supervised classification.

Used for:
  - Testing the quality of features learned during pretraining.


Inference Notes
(9/4 After mulitprocess, argmax optim, max gpu use)
    - For ~150x512x512 (2GPUs): 34s predict, 18s agg
    - After Optim (2GPUs): 25s predict, 3 sec agg!!  (tot 6 volume val: 2.05min)
    - After Optim (1GPU)
    - After fp16 final change:
        1GPU: 3.1 min
        2GPU: 2.08 min
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
from lib.assess.seg_metrics import batch_cdj_metrics
from lib.nets import init as init_net
from data.transforms.crops.inference import ChopBatchAggregate3d as CBA

WATCH = timers.StopWatch()

SAVE_AFTER = 0.0  # Only save model when over this percentage of epochs done.
SAVE_MOST_RECENT_MODEL = True
SAVE_BEST_METRICS = {  # metric (str): max_better (bool)
    'val_ep_dice_mean': True,
}
SUMMARIZE = {
    'triggers': ['val_ep_dice_mean'],
    'saves': ['val_ep_dice_mean', 'val_ep_jaccard_mean']
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
        criterion = DC_and_CE_loss(dc_kw, ce_kw, ignore_label=None).to(device)
    
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
    
    val_set = data_d['val_set']
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
        val_set._samples = [val_set.samples[0]]
    
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
            pred_ids = out.detach().argmax(1).unsqueeze(1).cpu()
            preds = torch.zeros(Y.shape, dtype=torch.uint8)
            preds.scatter_(1, pred_ids, 1)

            iter_metrics_d = batch_metrics(preds, targs, 
                                           ignore_background=False)
            iter_metrics_d['loss'] = loss.item()

            iter_time = WATCH.toc('iter', disp=False)
            loss_str = criterion.get_loss_string(loss_d)
            if rank == 0:
                print(
                    f"\n    Iter {it+1}/{len(train_loader)} "
                    f"({iter_time:.1f} sec, {mem():.1f} GB) - "
                    f"{loss_str} \n"
                    f"      jaccard {iter_metrics_d['jaccard_mean']:.3f}, "
                    f"dice {iter_metrics_d['dice_mean']:.3f}\n"
                    f"       {iter_metrics_d['dice_class']}"
                )

            update_mets = ['loss', 'dice_mean', 'jaccard_mean']
            epmeter.update({k: iter_metrics_d[k] for k in update_mets})
            
            global_iter += 1
            WATCH.tic('iter')
            if debug['break_train_iter'] and it >= 2: 
                break

        del X  # save some memory for inference
        del Y_id
        del out

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
                output.subsubsection('Validation Metrics')
            N_val_exs = len(data_d['df'][data_d['df']['subset'] == 'val'])
            val_mets = test_metrics(cfg, model, val_set, epoch, 
                inference_metrics_queue, N_val_exs, name='val')
            if rank == 0:
                epoch_mets.update(val_mets, tracked_epmets, 'val_ep_')
            
            # output.subsubsection('Testing Metrics')
            # test_mets = test_metrics(cfg, model, test_set, epoch, name='test')
            # update_epoch_mets(test_mets, tracked_epmets, 'test_ep_')
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
        
        scheduler.step(epoch=epoch, value=0)
        WATCH.toc(name='epoch')
        # End of epoch #


def get_model(cfg):

    weights = data_setup.weights_d['bcv_cbrt']
    percs = 1 / weights
    approx_logits = torch.log(percs)

    if cfg.model.name == 'denseunet3d':
        from lib.nets.volumetric.denseunet3d import get_model as get_dunet
        model = get_dunet(201, num_classes=14, deconv=False)
        with torch.no_grad():
            final_biases = torch.nn.Parameter(approx_logits)
            model.conv2.bias = final_biases
    elif cfg.model.name == 'resmednet3d':  # already inited
        from lib.nets.volumetric.resnet3d_mednet import generate_resnet3d
        model = generate_resnet3d(in_channels=1, classes=14, model_depth=34)
        with torch.no_grad():
            final_biases = torch.nn.Parameter(approx_logits)
            model.segm.conv_final.bias = final_biases
    elif cfg.model.name == 'genesis_unet3d':
        from .ftbcv_unet3d import UNet3D as genesis_unet3d
        model = genesis_unet3d(n_input=1, n_class=14, act='relu')
    elif cfg.model.name == 'gn_unet3d':
        from lib.nets.volumetric.resunet3d import UNet3D
        model = UNet3D(1, 14, final_sigmoid=False, is_segmentation=False)
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
    else: 
        if cfg.model.name == 'genesis_unet3d':
            # Initialize
            init_type = cfg.model.init
            if init_type:
                init_net.init_weights(model, init_type=init_type)
            print(f'   (Model) Successfully initialized weights via {init_type}.')

            # Final layer init
            with torch.no_grad():
                final_biases = torch.nn.Parameter(approx_logits)
                model.out_tr.final_conv.bias = final_biases
            print(f'   (Model) Initialized last layer biases of Genesis-3DUnet to '
                  f'{model.out_tr.final_conv.bias}')

    return {
        'model': model
    }


def batch_metrics(preds, targs, ignore_background=False, naive_avg=False):
    """
    preds: BxCxDxHxW, targs: BxCxDxHxW
    Accum: https://github.com/Project-MONAI/MONAI/blob/67aa4cfba3a7e32786f22c5767bf6772c2a393d9/monai/metrics/utils.py#L108
    """
    cdj_tuple = batch_cdj_metrics(preds, targs,
                                  ignore_background=ignore_background)

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


def test_metrics(cfg, model, dataset, epoch, test_metrics_queue, 
                 num_examples, name='test'):
    """
    Args:
        num_examples: needed for DDP since rank 0 worker needs a total count
            of examples so it knows how many times to pull metrics from Q.
    """
    if cfg.experiment.distributed:
        cba_device = cfg.experiment.device
    else:
        cba_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if len(cfg.experiment.gpu_idxs) > 1:
            cba_device = f'cuda:{len(cfg.experiment.gpu_idxs) - 1}'

    WATCH.tic(name.title())
    epmeter = statistics.EpochMeters()
    with torch.no_grad():
        model.eval()

        WATCH.tic(f'{name}_iter')
        processes = []
        for i in range(len(dataset)):
            if cfg.experiment.rank == 0:
                print(f' 🖼️  Inference for example {i+1}.')
            
            example_d = dataset[i]
            sample = example_d['sample']
            image = example_d['tensor']
            mask = example_d['mask_1h']  #  image: float32, mask: uint8

            # Create Chop-Batch-Aggregate inference helper
            test_batch_size = cfg.test.batch_size
            num_classes = sample.mask.num_classes
            cba = CBA(image, cfg.train.patch_size, (0, 0, 0), 
                      test_batch_size, num_classes, device=cba_device)
            if cfg.experiment.rank == 0:
                print(f'     Getting predictions for {len(cba)} batches.')
            
            # pstart = time.time()
            for bidx, batch in enumerate(cba):
                crops, locations = batch
                crops = crops.to(device)
                logits = model(crops)['out']
                cba.add_batch_predictions(logits, locations, act='none')
            # print(f'Predict {time.time() - pstart:.2f} sec ({mem():.1f} GB).')

            # Get final predictions, calculate metrics
            # pstart = time.time()
            del crops
            del logits
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
                print(f'Completed inference for vol {i+1} ({elaps:.2f} sec).\n')
            WATCH.tic(f'{name}_iter')

    # While last volume's metrics is computing, save sample
    if cfg.experiment.rank == 0 and epoch % 10 == 1:  # saves the last volume in set  
        id_preds = agg_predictions.argmax(0).numpy().astype(np.uint16)
        sitk_pred = sitk.GetImageFromArray(id_preds, isVector=False)
        sitk_pred.SetOrigin(sample.mask.origin)
        sitk_pred.SetSpacing(sample.mask.spacing)
        sitk_pred.SetDirection(sample.mask.direction)

        curr_path = pathlib.Path(__file__).parent.absolute()
        filename = (f'{cfg.experiment.id}_ep{epoch}_lastex_'
                    f'prediction.nii.gz')
        save_path = os.path.join(curr_path, 'artifacts', filename)
        print(f'Saving prediction as "{filename}" | Success.')
        sitk.WriteImage(sitk_pred, save_path)

    # Accumulate metrics & print results    
    if cfg.experiment.rank == 0:
        update_mets = ['dice_mean', 'jaccard_mean']
        data_cfg = cfg.data[cfg.data.name]
        if not cfg.experiment.distributed:
            num_examples = len(processes)
        for i in range(num_examples):
            mets = test_metrics_queue.get()
            epmeter.update({k: mets[k] for k in update_mets})
            print(f'({name.title()}) Example {i+1} \n'
                  f'       Dice: {float(mets["dice_mean"]):.2f} \n'
                  f'        {mets["dice_class"]} \n'
                  f'       Jaccard: {float(mets["jaccard_mean"]):.2f} \n'
                  f'        {mets["jaccard_class"]}')
    for process in processes:
        process.join() 
    
    if cfg.experiment.rank == 0:
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
            score_str = f'{score:d}' if score.is_integer() else f'{score:.3f}'
            end = f"best-{subset}-{met_name}-{score_str}"
            print(f"(save_model) {end}")
            break
    if end == 'last' and not SAVE_MOST_RECENT_MODEL:
        return

    curr_path = pathlib.Path(__file__).parent.absolute()
    fn_start = f"{cfg['experiment']['id']}_{cfg['experiment']['name']}_"
    rm_files = [f for f in os.listdir(curr_path) if f[-3:] == 'pth' \
                and fn_start in f]
    if rm_files:
        match_str = end
        if end != 'last':
            match_str = f'best-{subset}-{met_name}'
        for f in rm_files:
            if match_str in f:
                rm_file = os.path.join(curr_path, f)
                print(f"Deleting file -x {f}")
                os.remove(rm_file)
    
    filename = fn_start + f'ep{epoch}_' + end + '.pth'
    save_path = os.path.join(curr_path, filename)
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






