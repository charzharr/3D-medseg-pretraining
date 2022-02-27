
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
from experiments.rubik.reorient import rotate3d



class RubikSampleSet(torch.utils.data.Dataset):
    """ Designed for Models Genesis pretraining. """
    clamp_low = -1000 
    clamp_high = 1000
    resample_interp = 'bspline'  # bspline, linear, nearest
    
    def __init__(self, config, samples):
        self.config = config 
        self.task_config = config.tasks[config.tasks.name]
        self.size_multiplier = config.train.dataset_sample_multiplier
        
        self.crops_per_volume = config.train.batch_crops_per_volume
        self.cropper = ScaledUniformNoAirCropper3d()
        self.scale_sampler = ValueSampler(False, config.train.scale_range)
        print(f'🖼️  Sampling from scales: {config.train.scale_range}')
        
        self.samples = samples
        
        self.m = self.task_config.m    # number of cube levels to rotate per axes
        self.n = self.task_config.n    # number of sub-cubes per dim
        
        self._preprocess_time = None  # EMA of preprocess time
        self._getitem_time = None
        
        print(f'💠 MGDataset created with {len(self.samples)} samples. \n'
              f'   Crops/Vol={self.crops_per_volume}, '
              f'Virtual-Size={len(self)}.')
    
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
        
        # --- Get Valid Non-Air Crops --- #
        min_mean_thresh = self.config.train.min_mean_crop_thresh
        if sample.dataset == 'mmwhs':
            min_mean_thresh = 0
        
        crop_meta_list = self.cropper(
            image_arr, 
            n_times=self.crops_per_volume, 
            final_shape=self.config.train.patch_size,
            scale_sampler=self.scale_sampler,
            interpolation='trilinear',
            min_mean_thresh=min_mean_thresh
        )
        crops, records = zip(*crop_meta_list)
        
        # --- Augment Crops & Prepare Labels --- #
        def get_partitions(length, n_partitions):
            """ Returns list of start and stop indices [start, stop). """
            partitions = []
            step_size = length // n_partitions
            assert step_size > 0
            for l in range(n_partitions):
                partitions.append([step_size * l, step_size * (l + 1)])
            partitions[-1][-1] = length
            return partitions
        
        samples = []
        fin_crops = []
        fin_labs = []
        for i, crop in enumerate(crops):
            # labels are untouched
            y = copy.deepcopy(crop)
            record = records[i]
            
            # rubik-rotate m layers
            rubik_ops = []
            for axis in range(3):
                cube_layers = np.random.choice(list(range(self.n)), 
                                               size=self.m, replace=False)
                axis_partitions = get_partitions(crop.shape[axis], self.n)
                for layer_idx in cube_layers:
                    start, stop = axis_partitions[layer_idx]
                    num_rot90s = int(np.random.choice([1, 2, 3], size=1)[0])
                    
                    if axis == 0:
                        rot_crop = np.rot90(copy.deepcopy(crop),
                                            k=num_rot90s,
                                            axes=(1, 2))
                        crop[start:stop,:,:] = rot_crop[start:stop,:,:]
                    elif axis == 1:
                        rot_crop = np.rot90(copy.deepcopy(crop),
                                            k=num_rot90s,
                                            axes=(2, 0))
                        crop[:,start:stop,:] = rot_crop[:,start:stop,:]
                    else:
                        rot_crop = np.rot90(copy.deepcopy(crop),
                                            k=num_rot90s,
                                            axes=(1, 0))
                        crop[:,:,start:stop] = rot_crop[:,:,start:stop]
                    
                    rubik_ops.append({
                        'axis': axis, 'layer': layer_idx, 'start': start,
                        'stop': stop, 'num_rot90s': num_rot90s
                    })
            record['rubik'] = rubik_ops
                                    
            # crop, y, flip_flags = flip(crop, y, p=self.task_config.t_mg_flip)
            # record['flip'] = flip_flags
            
            fin_crops.append(torch.tensor(crop.astype(np.float32).copy()))
            fin_labs.append(torch.tensor(y.astype(np.float32).copy()))
            samples.append(sample)
                    
        tot_time = time.time() - start
        if self._getitem_time is None:
            self._getitem_time = tot_time 
        else:
            self._getitem_time = 0.9 * self._getitem_time + 0.1 * tot_time
        
        # print('🏗️  Preprocess time', self._preprocess_time)
        # print('🏗️  GetItem time', self._preprocess_time)
        return {
            'samples': samples,
            'records': records,
            'image_tensors': fin_crops,
            'label_tensors': fin_labs,
        }
        
    @staticmethod
    def _collate(batch):
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
            _combine(samples, example['samples'])
            _combine(records, example['records'])
            _combine(image_tensors, example['image_tensors'])
            _combine(label_tensors, example['label_tensors'])
        
        return {
            'X': torch.stack(image_tensors, dim=0).unsqueeze(1), 
            'Y': torch.stack(label_tensors, dim=0).unsqueeze(1),
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
        image_sitk = sitk.Cast(image_sitk, sitk.sitkFloat32)
        image_arr = sitk.GetArrayFromImage(image_sitk)
        
        # time tracking
        tot_time = time.time() - start
        if self._preprocess_time is None:
            self._preprocess_time = tot_time 
        else:
            self._preprocess_time = 0.9 * self._preprocess_time + 0.1 * tot_time
        
        return image_arr

