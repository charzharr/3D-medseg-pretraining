
import time
import copy
import SimpleITK as sitk
import torch
import collections

from experiments.ftmmwhs import data_setup as mmwhs_setup
from experiments.ftbcv import data_setup as bcv_setup
from experiments.ftspleen import data_setup as spleen_setup

import data.transforms as myT
from experiments.prevec.data_vec import PretrainDataset

# Pretraining modules
from .cropper import SpatialPretrainCropper3d
from .vector import Vector3d
from .sampler import ValueSampler
from .reorient import flip3d, rotate3d



def get_pretrain_data_components(config):
    print('🖼️  Getting MMWHS, BCV, MSD Liver, MSD Lung samples.')
    from experiments.pretrain_data_setup import get_df_samples
    collect_d = get_df_samples(config)

    train_df = df = collect_d['df']
    samples = collect_d['samples']
    
    debug = config.experiment.debug
    shuffle = False if debug.overfitbatch or debug.mode else True
    num_train_workers = 0 if debug.mode or debug.overfitbatch else \
                        config.train.num_workers
    
    task_config = config.tasks[config.tasks.name]
    if config.tasks.name == 'vec':
        raise NotImplementedError('Vec only works for individual datasets.')
        print(f'🖼️  Task: {config.tasks.name}, Config: {task_config}.')
        
        print(f'\nTrain Data Components:')
        from .data_vec import VecSampleSet
        train_set = VecSampleSet(config, samples)
        train_loader = torch.utils.data.DataLoader(
            train_set, 
            shuffle=shuffle,
            batch_size=config.train.batch_size,
            collate_fn=VecSampleSet._collate,
            num_workers=num_train_workers
        )
    
    elif config.tasks.name == 'mg':
        print(f'🖼️  Task: {config.tasks.name}, Config: {task_config}.')
        
        print(f'\nTrain Data Components:')
        from .data_mg import MGSampleSet
        train_set = MGSampleSet(config, samples)
        train_loader = torch.utils.data.DataLoader(
            train_set, 
            shuffle=shuffle,
            batch_size=config.train.batch_size,
            collate_fn=MGSampleSet._collate_mg,
            num_workers=num_train_workers
        )
    
    elif config.tasks.name == 'sar':
        print(f'🖼️  Task: {config.tasks.name}, Config: {task_config}.')
        
        print(f'\nTrain Data Components:')
        from .data_sar import SARSampleSet
        train_set = SARSampleSet(config, samples)
        train_loader = torch.utils.data.DataLoader(
            train_set, 
            shuffle=shuffle,
            batch_size=config.train.batch_size,
            collate_fn=SARSampleSet._collate,
            num_workers=num_train_workers
        )
        
    else:
        assert False, f'{config.tasks.name} is not supported for all pretrain.'
    
    return {
        'df': df,
        'samples': samples,
        'train_df': train_df,
        'train_set': train_set,
        'train_loader': train_loader,
        'test_df': None,
        'test_set': None,
        'val_df': None,
        'val_set': None,
    }
    

def get_mmwhs_data_components(config):
    
    # Get samples and create datasets / dataloaders
    samples_d = mmwhs_setup.get_data_components(config, get_samples_only=True)
    
    print(f'\nTrain Data Components:')
    train_samples = samples_d['train_samples']
    train_set = PretrainDataset(config, train_samples, is_train=True)
    debug = config.experiment.debug
    shuffle = False if debug.overfitbatch or debug.mode else True
    num_workers = 0 if debug.mode or debug.overfitbatch else config.train.num_workers
    train_loader = torch.utils.data.DataLoader(
        train_set,
        shuffle=shuffle,
        batch_size=config.train.batch_size,
        collate_fn=PretrainDataset._collate,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else 2
    )
    print(f'💠 Torch Dataloader initialized with {num_workers} workers!\n'
          f'   Batch-size={config.train.batch_size}, '
            f'Patches-per-batch={config.train.batch_crops_per_volume}, '
            f'Shuffle={shuffle}. \n')
        
    print(f'\nTest Data Components:')
    test_samples = samples_d['test_samples']
    test_set = PretrainDataset(config, test_samples, is_train=False)
    
    return {
        'df': samples_d['df'],
        'samples': samples_d['samples'],
        'train_df': samples_d['train_df'],
        'train_set': train_set,
        'train_loader': train_loader,
        'test_df': samples_d['test_df'],
        'test_set': test_set
    }


def get_bcv_data(config):  # TODO when necessary.
    pass


# ============================================================================ #
# * ### * ### * ### *  New 2/18/22 Code: General Pretrain  * ### * ### * ### * #
# ============================================================================ #





