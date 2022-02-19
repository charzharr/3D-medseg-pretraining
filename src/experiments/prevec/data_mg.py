""" Module data_mg.py (By: Charley Zhang, 2022)

Definitions of non-preloaded data containers and loaders for unsupervised
Models Genesis pretraining. 
    - Implements efficient multi-patch per crop.
    - Skips mostly air crops via a mean patch threshold. 
"""

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
from experiments.prevec.sampler import ValueSampler


# Constants or Settings for Data
VISUALIZE = False
PRELOAD_DATA = False



# ========================================================================== #
# * ### * ### * ### *              Loading               * ### * ### * ### * #
# ========================================================================== #


class MGSampleSet(torch.utils.data.Dataset):
    """ Designed for Models Genesis pretraining. """
    clamp_low = -1000 
    clamp_high = 1000
    resample_interp = 'bspline'  # bspline, linear, nearest
    
    def __init__(self, config, samples):
        self.config = config 
        if 'examples_per_volume' in config.train:
            self.size_multiplier = config.train.dataset_len_multiplier
        
        self.crops_per_volume = config.train.crops_per_volume
        self.cropper = ScaledUniformNoAirCropper3d()
        self.scale_sampler = ValueSampler(False, config.train.scale_range)
        print(f'🖼️  Sampling from scales: {config.train.scale_range}')
        
        self.samples = samples
        
        self._preprocess_time = None  # EMA of preprocess time
    
    def __len__(self):
        return len(self.samples) * self.size_multiplier  # hackey
    
    def __getitem__(self, idx):
        assert 0 <= idx < len(self), f'Index {idx} is out of bounds.'
        start = time.time()
        
        orig_idx = idx
        idx = idx // self.size_multiplier  # hackey way of extending epochs
        
        sample = self.samples[idx]
        image_sitk = sample.image.sitk_image
        image_arr = self._preprocess(image_sitk)
        
        # --- Get Valid Crops & Apply Augmentations --- #
        crop_meta_list = self.cropper(
            image_arr, 
            n_times=self.crops_per_volume, 
            final_shape=self.config.train.patch_size,
            scale_sampler=self.scale_sampler,
            interpolation='trilinear',
            min_mean_thresh=self.config.train.min_mean_crop_thresh
        )
        crops, records = zip(*crop_meta_list)
        
        fin_crops = []
        labs = []
        boundaries = []   # for custom mg usage
        for i, crop in enumerate(crops):
            y = copy.deepcopy(crop)
            record = records[i]
            
            import IPython; IPython.embed(); 
            
            crop, y, flip_flags = flip(crop, y, p=self.config.tasks.mg.t_mg_flip)
            record['flip'] = flip_flags
            
            crop = local_pixel_shuffle(crop, 
                                       p=self.config.tasks.mg.t_mg_shuffle)
            crop = nonlinear_intensity_map(crop, 
                                           p=self.config.tasks.mg.t_mg_nonlinear)

            if random.random() < self.config.tasks.mg.t_mg_paint:
                if random.random() < self.config.tasks.mg.t_mg_inpaint:
                    crop = in_paint(crop, p=1.0)
                    record['in_paint'] = True
                else:
                    crop = out_paint(crop, p=1.0)
                    record['out_paint'] = True
        
            fin_crops.append(crop)
            labs.append(y)
            records.append(record)
            
        print('🏗️  Preprocess time', self._preprocess_time)
        return {
            'sample': sample,
            'records': records,
            'image_tensors': crops,
            'label_tensors': labs,
            'boundary_tensors': boundaries
        }
        
    @staticmethod
    def _collate_mg(batch):
        """ 
        Args:
            batch (list of size B): each element is the dict returned by __getitem__
        """
        def _combine(sequence, new_elem):
            if new_elem is None:
                return 
            if isinstance(new_elem, collections.abc.Sequence):
                sequence.extend(list(new_elem))
            else:
                sequence.append(new_elem)
        
        samples = []
        records = []
        image_tensors = []
        label_tensors = []
        
        for b, example in enumerate(batch):
            _combine(samples, example['sample'])
            _combine(records, example['records'])
            _combine(image_tensors, example['image_tensors'])
            _combine(label_tensors, example['label_tensors'])
        
        return {
            'X': torch.stack(image_tensors, dim=0), 
            'Y': torch.stack(label_tensors, dim=0),
            'samples': samples,
            'records': records
        }
        
    def _preprocess(self, image_sitk):
        """
        1. Clamps between -1000 and 1000
        2. Resamples to 1mm spacing each side
        3. Normalizes to intensities between 0 and 1
        4. Converts to numpy array and returns
        """
        start = time.time()
        clip_min, clip_max = self.clamp_low, self.clamp_high
        image_sitk = sitk.Clamp(image_sitk, sitk.sitkInt16, clip_min, clip_max)
        
        # resample
        orig_spacing = np.array(image_sitk.GetSpacing())
        new_spacing = [1,1,1]
        if orig_spacing.tolist() != new_spacing:
            resample = sitk.ResampleImageFilter()
            if self.resample_interp == 'bspline':
                resample.SetInterpolator = sitk.sitkBSpline 
            elif self.resample_interp == 'linear':
                resample.SetInterpolator = sitk.sitkLinear
            else:
                resample.SetInterpolator = sitk.sitkNearestNeighbor
            resample.SetOutputDirection(image_sitk.GetDirection())
            resample.SetOutputOrigin(image_sitk.GetOrigin())
            resample.SetOutputSpacing(new_spacing)

            orig_size = np.array(image_sitk.GetSize())
            new_size = orig_size * (orig_spacing / new_spacing)
            new_size = np.ceil(new_size).astype(np.int32).tolist()
            resample.SetSize(new_size)
            image_sitk = resample.Execute(image_sitk)
        
        # normalize & get array
        image_sitk = 1.0 * (image_sitk - clip_min) / (clip_max - clip_min)
        image_arr = sitk.GetArrayFromImage(image_sitk)
        
        # time tracking
        tot_time = time.time() - start
        if self._preprocess_time is None:
            self._preprocess_time = tot_time 
        else:
            self._preprocess_time = 0.9 * self._preprocess_time + 0.1 * tot_time
        
        return image_arr