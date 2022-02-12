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
    from experiments.prevec.model_setup import get_model
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
    
    criterions = []
    if 'prevec' in cfg.tasks.names:
        if cfg.tasks.prevec.loss == 'spherical':
            from experiments.prevec.prevec_losses import SphericalCriterion
            crit_prevec = SphericalCriterion(
                r_loss='l1', angle_loss='l1',
                r_weight=1, theta_weight=1, phi_weight=1
            )
            crit_prevec = crit_prevec.to(device)
            criterions.append(crit_prevec)
            # cfg.tasks.prevec.pred_vectors
        else:
            assert False
    
    optimizer = setup.get_optimizer(cfg, model.parameters())
    scheduler = setup.get_scheduler(cfg, optimizer)

    # Data Pipeline
    if rank == 0:
        output.header_three('Data Setup')
    from experiments.prevec.data_setup import get_mmwhs_data, get_bcv_data
    
    if cfg.data.name == 'mmwhs':
        data_d = get_mmwhs_data(cfg)
    else:
        assert False
    
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
            vectors = batch['vectors']
            
            X = batch['images'].float().to(device, non_blocking=True)
            # Y_id = batch['masks'].to(torch.uint8).to(device, non_blocking=True)

            out_d = model(X)
            out = out_d['out'] if isinstance(out_d, dict) else out_d
            
            loss = 0
            loss_str = ''
            if 'prevec' in cfg.tasks.names:
                # create labels
                Y_prevec = create_prevec_targ(vectors, records,
                    cfg.tasks.prevec.pred_indices).to(out.device)
                loss_prevec_d = crit_prevec(out, Y_prevec)
                loss += loss_prevec_d['loss']
                loss_str += f'PrevecLoss {loss.item():.4f} '
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Iteration & Epoch Training Metrics 
            if rank == 0:
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
        tracked_epmets = ['loss']
        epoch_mets.update(train_mets, tracked_epmets, 'train_ep_')

        # Test + Epoch Metrics
        test_every_n = debug.test_every_n_epochs
        if False and epoch % test_every_n == test_every_n - 1:
            if rank == 0:
                output.subsubsection('Test Metrics')
            N_test_exs = len(data_d['df'][data_d['df']['subset'] == 'test'])
            test_mets = test_metrics(cfg, model, test_set, epoch, 
                inference_metrics_queue, N_test_exs, name='test',
                overlap_perc=cfg.test.patch_overlap_perc,
                criterions=criterions)
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



# ========================================================================== #
# * ### * ### * ### *      Training Functionality        * ### * ### * ### * #
# ========================================================================== #



def test_metrics(cfg, model, dataset, epoch, test_metrics_queue, 
                 num_examples, name='test', overlap_perc=0.2, criterions=None):
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


def create_prevec_targ(vectors, records, indices):
    Y = torch.zeros(len(vectors), 3 * len(indices))
    for b, (vecs, rec) in enumerate(zip(vectors, records)):
        vol_input_shape = rec['SpatialPretrainCropper3d']['input_shape']
        vol_center = rec['SpatialPretrainCropper3d']['input_volume_center']
        radius = math.sqrt(sum([n ** 2 for n in vol_center]))
        
        for i in range(0, len(indices)):
            vec = vecs[i]
            Y[b, i * 3] = vec.magnitude / radius
            
            theta = vec.theta
            if theta > math.pi:
                theta -= math.pi 
            elif theta < 0:
                theta += math.pi
            Y[b, i * 3 +1] = theta / math.pi
            
            phi = vec.phi
            if phi > 2 * math.pi:
                phi -= 2 * math.pi 
            elif phi < 0:
                phi += 2 * math.pi
            Y[b, i * 3 +2] = phi / (2 * math.pi)
            # print(Y)
    return Y


def mem():
    """ Get primary GPU card memory usage. """
    if not torch.cuda.is_available():
        return -1.
    mem_map = devices.get_gpu_memory_map()
    prim_card_num = gpu_indices[0]
    return mem_map[prim_card_num]/1000