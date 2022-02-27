
import sys, os
import math, random
import pathlib
import copy
from threading import local
import time
import collections
import multiprocessing
from collections import OrderedDict

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import SimpleITK as sitk

import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as tf
import albumentations as A
from skimage.transform import resize as nd_resize
from scipy.ndimage import rotate as nd_rotate

# Experiment-specific Data Things
import data.transforms as myT
from data.transforms.crops import (
    ScaledUniformNoAirCropper3d,
    ScaledOverlapCropper3d
)
from data.transforms.models_genesis import (
    flip,
    local_pixel_shuffle,
    nonlinear_intensity_map,
    in_paint,
    out_paint
)
from experiments.rubik.sampler import ValueSampler

from experiments.ftmmwhs import data_setup as mmwhs_setup
from experiments.ftbcv import data_setup as bcv_setup
from experiments.ftspleen import data_setup as spleen_setup

import data.transforms as myT

# Pretraining modules
from .cropper import SpatialPretrainCropper3d
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
    if config.tasks.name == 'rubikpp':
        from .data_rubik import RubikSampleSet
        print(f'🖼️  Task: {config.tasks.name}, Config: {task_config}.')
        
        print(f'\nTrain Data Components:')
        train_set = RubikSampleSet(config, samples)
        train_loader = torch.utils.data.DataLoader(
            train_set, 
            shuffle=shuffle,
            batch_size=config.train.batch_size,
            collate_fn=RubikSampleSet._collate,
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
