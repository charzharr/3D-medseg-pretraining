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
from skimage import filters
from scipy import ndimage as ndi

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
from .cropper import SpatialPretrainCropper3d
from .vector import Vector3d
from .sampler import ValueSampler
from .reorient import flip3d, rotate3d

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




# ============================================================================ #
# * ### * ### * ### *     V1 Code: Only MMWHS Pretrain     * ### * ### * ### * #
# ============================================================================ #


class PretrainDataset(torch.utils.data.Dataset):
    """ Generalized dataset object (based on sample abstraction). 
    
    Notes
      - When is_train is off, returns the entire volumes and skips augmentation.
    """
    
    def __init__(self, config, samples, is_train=True):
        super().__init__()
        
        self.config = config 
        self.task_config = config.tasks[config.tasks.name]
        self.is_train = is_train
        self.samples = samples
                
        self.num_classes = config.data[config.data.name].num_classes
        
        self.crops_per_volume = config.train.batch_crops_per_volume
        self.batch_crops_per_volume = config.train.batch_crops_per_volume
        if not is_train:
            self.crops_per_volume = 1
        
        ## Training Transforms
        if self.is_train:
            if self.config.experiment.debug.overfitbatch:
                self.config.train.scale_range = 1.
            
            # Transforms: Cropper
            cropscale_sampler = ValueSampler(False,  # continuous 
                                             self.config.train.scale_range)
            print(f'🖼️  Sampling from scales: {config.train.scale_range}')
            self.T_crop_scale = SpatialPretrainCropper3d(
                final_shape=self.config.train.patch_size,
                scale_sampler=cropscale_sampler,
                cubic_crop=self.config.train.cubic_crop
            )
            # self.T_crop_scale = ScaledUniformNoAirCropper3d(
            #     final_shape=self.config.train.patch_size,
            #     scale_sampler=cropscale_sampler,
            #     cubic_crop=self.config.train.cubic_crop
            # )
            self.T_orient = None 
            
            self.T_augment = [
                myT.GaussianNoise(p=config.train.t_gn, mean=0., var=(0, 0.05)),
                myT.GaussianBlur(p=config.train.t_gb, spacing=1,
                                 sigma=(0.5, 1)),
                myT.ScaleIntensity(p=config.train.t_intensity_scale, 
                                   scale=(0.75, 1.25)),
                myT.Gamma(p=config.train.t_gamma, gamma=(0.7, 1.5))
            ]
            print(f'(PretrainDataset) Training Transforms: \n'
                  f'  Crop: {self.T_crop_scale}, \n   Aug: {self.T_augment}')
        print(f'💠 PretrainDataset created with {len(self.samples)} samples. \n'
              f'   Train={is_train}, Crops/Vol={self.crops_per_volume}, '
              f'Virtual-Size={len(self)}, Indices: {[s.index for s in self.samples]}.')

    def __len__(self):
        return len(self.samples) * self.crops_per_volume  # hackey
    
    def __getitem__(self, idx):
        """
        Returns:
            A dictionary containing the original sample object, a preprocessed
            full tensor volume. 
        """
        start = time.time()
        
        # Sample retrieval
        orig_idx = idx  
        idx = idx // self.crops_per_volume   # hackey way of inc epoch samples
        assert 0 <= idx < len(self), f'Index {idx} is out of bounds.'
        
        sample = self.samples[idx]
        
        sitk_image = sample.image.sitk_image  # loads path to sitk obj
        tensor = torch.tensor(sitk.GetArrayFromImage(sitk_image)).float()
        assert 'float32' in str(tensor.dtype)
        
        mini, maxi = tensor.min(), tensor.max()
        tensor = (tensor - mini) / (maxi - mini)  # [0, 1] compat with augs
        
        sitk_mask = sample.mask.sitk_image
        mask_tens = torch.from_numpy(sitk.GetArrayFromImage(sitk_mask))
        assert mask_tens.shape == tensor.shape
        
        record_dicts = [copy.deepcopy(sample.records)   # OrderedDict()
                      for _ in range(self.config.train.batch_crops_per_volume)]  
        
        # Test-case (return the entire clean image)
        if not self.is_train:
            mask_1h = sample.mask.get_one_hot(crop=mask_tens, 
                channel_first=True, tensor=True)    
            return {
                'sample': sample,
                'tensor': tensor, 
                'mask': mask_tens, 
                'mask_1h': mask_1h,
                'record': record_dicts
            }
         
        # Cropping
        min_mean_thresh = self.config.train.min_mean_crop_thresh
        if self.config.data.name == 'mmwhs':
            min_mean_thresh = 0.01
        image_tup, mask_tup = self.T_crop_scale(tensor, mask=mask_tens,
            n_times=self.batch_crops_per_volume,
            min_mean_thresh=min_mean_thresh)
        
        image_crops = [e[0] for e in image_tup]
        image_records = [e[1] for e in image_tup]
        mask_crops = [e[0] for e in mask_tup]
        mask_records = [e[1] for e in mask_tup]
        if not isinstance(image_records, collections.abc.Sequence):
            image_crops, image_records = [image_crops], [image_records]
            mask_crops, mask_records = [mask_crops], [mask_records]
        for i, record in enumerate(image_records):
            record_dicts[i][self.T_crop_scale.__class__.__name__] = record
        
        # Training Augmentations
        final_crops, final_masks, final_masks_oh, final_vecs = [], [], [], []
        fin_labs, boundaries, boundary_masks = [], [], []
        for i, (ic, mc) in enumerate(zip(image_crops, mask_crops)):
            crop = {'image_crop': ic, 'mask_crop': mc}
            record = record_dicts[i]
            
            # Flip
            out_crop, out_mask = crop['image_crop'], crop['mask_crop']
            crop_meta = record_dicts[i]['SpatialPretrainCropper3d']
            origin_coords = crop_meta['input_volume_center']
            flip_d = flip3d(out_crop, origin_coords, mask=out_mask, 
                            vectors=crop_meta['final_crop_vectors'],
                            p_fx=self.task_config.t_flip_x,
                            p_fy=self.task_config.t_flip_y,
                            p_fz=self.task_config.t_flip_z)
            
            out_crop, out_mask = flip_d['image'], flip_d['mask']
            out_vectors = flip_d['vectors']
            record_dicts[i]['flip3d'] = flip_d['meta']
            
            # Rotate
            x_deg_sampler = None
            if self.task_config.t_rot_x_vals:
                x_deg_sampler = ValueSampler(True, self.task_config.t_rot_x_vals)
            y_deg_sampler = None
            if self.task_config.t_rot_y_vals:
                y_deg_sampler = ValueSampler(True, self.task_config.t_rot_y_vals)
            z_deg_sampler = None
            if self.task_config.t_rot_z_vals:
                z_deg_sampler = ValueSampler(True, self.task_config.t_rot_z_vals)
            rot_d = rotate3d(out_crop, origin_coords, mask=out_mask, 
                             vectors=out_vectors,
                             x_deg_sampler=x_deg_sampler,
                             y_deg_sampler=y_deg_sampler,
                             z_deg_sampler=z_deg_sampler)
            out_crop, out_mask = rot_d['image'], rot_d['mask']
            out_vectors = rot_d['vectors']
            record_dicts[i]['rotate3d'] = rot_d['meta']
            
            y = copy.deepcopy(out_crop.numpy())
            
            # for transform in self.T_augment:
            #     name = transform.name
            #     out_crop, receipt = transform(out_crop)
            #     record_dicts[i][name] = receipt['image_crop'] if receipt else None
            
            # Intensity Augment
            out_crop = out_crop.numpy()
            
            out_crop = local_pixel_shuffle(
                out_crop, 
                p=self.task_config.t_mg_shuffle,
                n_shuffle_windows=self.task_config.t_mg_shuffle_times
            )
            out_crop = nonlinear_intensity_map(
                out_crop, 
                p=self.task_config.t_mg_nonlinear
            )
            
            # Paint
            if random.random() < self.task_config.t_mg_paint:
                if random.random() < self.task_config.t_mg_inpaint:
                    out_crop, paint_mask = in_paint(out_crop, p=1.0,
                        uniform_paint=self.task_config.t_mg_inpaint_uniform,
                        n_paints=self.task_config.t_mg_inpaint_times,
                        get_paint_mask=True)
                    record['in_paint'] = True
                else:
                    out_crop, paint_mask = out_paint(out_crop, p=1.0,
                        uniform_paint=self.task_config.t_mg_outpaint_uniform,
                        n_paints=self.task_config.t_mg_outpaint_times,
                        get_paint_mask=True)
                    record['out_paint'] = True
            else:
                paint_mask = np.ones(out_crop.shape).astype('uint8')
                
            out_crop = torch.tensor(out_crop)
            final_crops.append(out_crop)
            final_masks.append(out_mask)
            final_masks_oh.append(sample.mask.get_one_hot(
                crop=out_mask, channel_first=True, tensor=True))
            final_vecs.append(out_vectors)
                
            fin_labs.append(torch.tensor(y.astype(np.float32).copy()))
            filtered_crop = ndi.median_filter(y, size=5)
            schar_edges = filters.scharr(filtered_crop)
            boundary = np.clip((schar_edges - 0.01) * 30, 0, 1)
            boundaries.append(torch.tensor(boundary))
            boundary_masks.append(torch.tensor(paint_mask, dtype=torch.uint8))
            
        
        return {
            'sample': [sample] * len(final_crops),
            'tensor': final_crops,
            'mask': final_masks,
            'mask_one_hot': final_masks_oh,
            'record': record_dicts,
            
            # experiment-dependent
            'vectors': final_vecs,
            
            # MG things
            'label_tensors': fin_labs,
            'boundary_tensors': boundaries,
            'boundary_mask_tensors': boundary_masks,
        }
        
    @staticmethod
    def _collate(batch):
        """ Handles cases where each volume can output multiple crops. 
        Assumptions:
        - 'sample' is a list of samples
        - 'tensor', 'mask', 'mask_one_hot' are lists of lists of tensors
        - 'record' are list of lists of OrderedDicts
        """
        def _combine(sequence, new_elem):
            if new_elem is None:
                return 
            if isinstance(new_elem, collections.abc.Sequence):
                sequence.extend(list(new_elem))
            else:
                sequence.append(new_elem)
                
        images, masks, masks_1h, samples, records = [], [], [], [], []
        vecs = []
        
        label_tensors = []
        boundary_arrs = []
        boundary_mask_arrs = []
        for example in batch:
            _combine(samples, example['sample'])
            images += example['tensor']
            masks += example['mask']
            masks_1h += example['mask_one_hot']
            records += example['record']
            
            vecs += example['vectors']
            
            _combine(label_tensors, example['label_tensors'])
            _combine(boundary_arrs, example['boundary_tensors'])
            _combine(boundary_mask_arrs, example['boundary_mask_tensors'])
        
        return {
            'X': torch.stack(images, dim=0).unsqueeze(1),
            'masks': torch.stack(masks, dim=0).unsqueeze(1),
            'masks_1h': torch.stack(masks_1h, dim=0),
            'samples': samples,
            'records': records,
            'vectors': vecs,
            
            'Y': torch.stack(label_tensors, dim=0).unsqueeze(1),
            'boundaries': torch.stack(boundary_arrs, dim=0).unsqueeze(1),
            'boundary_masks': torch.stack(boundary_mask_arrs, dim=0).unsqueeze(1),
        }
    





# ========================================================================== #
# * ### * ### * ### *                             * ### * ### * ### * #
# ========================================================================== #


# Copied over from data_mg.py

# class VecSampleSet(torch.utils.data.Dataset):
#     """ Designed for Models Genesis pretraining. """
#     clamp_low = -1000 
#     clamp_high = 1000
#     resample_interp = 'bspline'  # bspline, linear, nearest
    
#     def __init__(self, config, samples):
#         self.config = config 
#         self.task_config = config.tasks[config.tasks.name]
#         self.size_multiplier = config.train.dataset_sample_multiplier
        
#         self.crops_per_volume = config.train.batch_crops_per_volume
#         self.cropper = ScaledUniformNoAirCropper3d()
#         self.scale_sampler = ValueSampler(False, config.train.scale_range)
#         print(f'🖼️  Sampling from scales: {config.train.scale_range}')
        
#         self.samples = samples
        
#         self._preprocess_time = None  # EMA of preprocess time
#         self._getitem_time = None
        
#         print(f'💠 VecDataset created with {len(self.samples)} samples. \n'
#               f'   Crops/Vol={self.crops_per_volume}, '
#               f'Virtual-Size={len(self)}.')
    
#     def __len__(self):
#         return len(self.samples) * self.size_multiplier  # hackey
    
#     def __getitem__(self, idx):
#         assert 0 <= idx < len(self), f'Index {idx} is out of bounds.'
#         start = time.time()
        
#         orig_idx = idx
#         idx = idx // self.size_multiplier  # hackey way of extending epochs
        
#         sample = self.samples[idx]
#         image_sitk = sample.image.sitk_image
#         image_arr = self._preprocess(image_sitk)
        
#         # --- Get Valid Crops & Apply Augmentations --- #
#         min_mean_thresh = self.config.train.min_mean_crop_thresh
#         if sample.dataset == 'mmwhs':
#             min_mean_thresh = 0
        
#         crop_meta_list = self.cropper(
#             image_arr, 
#             n_times=self.crops_per_volume, 
#             final_shape=self.config.train.patch_size,
#             scale_sampler=self.scale_sampler,
#             interpolation='trilinear',
#             min_mean_thresh=min_mean_thresh
#         )
#         crops, records = zip(*crop_meta_list)
        
#         samples = []
#         fin_crops = []
#         fin_labs = []
#         boundaries, boundary_masks = [], []   # for custom mg usage
#         for i, crop in enumerate(crops):
#             y = copy.deepcopy(crop)
#             record = records[i]
                                    
#             crop, y, flip_flags = flip(crop, y, p=self.task_config.t_mg_flip)
#             record['flip'] = flip_flags
                        
#             crop = local_pixel_shuffle(
#                 crop, 
#                 p=self.task_config.t_mg_shuffle,
#                 n_shuffle_windows=self.task_config.t_mg_shuffle_times
#             )
#             crop = nonlinear_intensity_map(
#                 crop, 
#                 p=self.task_config.t_mg_nonlinear
#             )

#             if random.random() < self.task_config.t_mg_paint:
#                 if random.random() < self.task_config.t_mg_inpaint:
#                     crop, paint_mask = in_paint(crop, p=1.0,
#                         uniform_paint=self.task_config.t_mg_inpaint_uniform,
#                         n_paints=self.task_config.t_mg_inpaint_times,
#                         get_paint_mask=True)
#                     record['in_paint'] = True
#                 else:
#                     crop, paint_mask = out_paint(crop, p=1.0,
#                         uniform_paint=self.task_config.t_mg_outpaint_uniform,
#                         n_paints=self.task_config.t_mg_outpaint_times,
#                         get_paint_mask=True)
#                     record['out_paint'] = True
#             else:
#                 paint_mask = np.ones(crop.shape).astype('uint8')
            
            
#             fin_crops.append(torch.tensor(crop.astype(np.float32).copy()))
#             fin_labs.append(torch.tensor(y.astype(np.float32).copy()))
#             samples.append(sample)
            
#             filtered_crop = ndi.median_filter(y, size=5)
#             schar_edges = filters.scharr(filtered_crop)
#             boundary = np.clip((schar_edges - 0.01) * 30, 0, 1)
#             boundaries.append(torch.tensor(boundary))
#             boundary_masks.append(torch.tensor(paint_mask, dtype=torch.uint8))
                    
#         tot_time = time.time() - start
#         if self._getitem_time is None:
#             self._getitem_time = tot_time 
#         else:
#             self._getitem_time = 0.9 * self._getitem_time + 0.1 * tot_time
        
#         # print('🏗️  Preprocess time', self._preprocess_time)
#         # print('🏗️  GetItem time', self._preprocess_time)
#         return {
#             'samples': samples,
#             'records': records,
#             'image_tensors': fin_crops,
#             'label_tensors': fin_labs,
#             'boundary_tensors': boundaries,
#             'boundary_mask_tensors': boundary_masks,
#         }
        
#     @staticmethod
#     def _collate_mg(batch):
#         """ 
#         Args:
#             batch (list of size B): each element is the dict returned by __getitem__
#         """
#         def _combine(sequence, new_elem):
#             if new_elem is None:
#                 return 
#             if isinstance(new_elem, collections.abc.Sequence):
#                 sequence.extend(list(new_elem))
#             else:
#                 sequence.append(new_elem)
        
#         samples = []
#         records = []
#         image_tensors = []
#         label_tensors = []
#         boundary_arrs = []
#         boundary_mask_arrs = []
        
#         for b, example in enumerate(batch):
#             _combine(samples, example['samples'])
#             _combine(records, example['records'])
#             _combine(image_tensors, example['image_tensors'])
#             _combine(label_tensors, example['label_tensors'])
            
#             _combine(boundary_arrs, example['boundary_tensors'])
#             _combine(boundary_mask_arrs, example['boundary_mask_tensors'])
        
#         return {
#             'X': torch.stack(image_tensors, dim=0).unsqueeze(1), 
#             'Y': torch.stack(label_tensors, dim=0).unsqueeze(1),
#             'boundaries': torch.stack(boundary_arrs, dim=0).unsqueeze(1),
#             'boundary_masks': torch.stack(boundary_mask_arrs, dim=0).unsqueeze(1),
#             'samples': samples,
#             'records': records,
#         }
        
#     def _preprocess(self, image_sitk):
#         """
#         1. Clamps between -1000 and 1000
#         2. Resamples to 1mm spacing each side
#         3. Normalizes to intensities between 0 and 1
#         4. Converts to numpy array and returns
#         """
#         start = time.time()
#         clip_min, clip_max = self.clamp_low, self.clamp_high
#         image_sitk = sitk.Clamp(image_sitk, sitk.sitkInt16, clip_min, clip_max)
                
#         # resample
#         orig_spacing = np.array(image_sitk.GetSpacing())
#         new_spacing = [1,1,1]
#         if orig_spacing.tolist() != new_spacing:
#             resample = sitk.ResampleImageFilter()
#             if self.resample_interp == 'bspline':
#                 resample.SetInterpolator = sitk.sitkBSpline 
#             elif self.resample_interp == 'linear':
#                 resample.SetInterpolator = sitk.sitkLinear
#             else:
#                 resample.SetInterpolator = sitk.sitkNearestNeighbor
#             resample.SetOutputDirection(image_sitk.GetDirection())
#             resample.SetOutputOrigin(image_sitk.GetOrigin())
#             resample.SetOutputSpacing(new_spacing)

#             orig_size = np.array(image_sitk.GetSize())
#             new_size = orig_size * (orig_spacing / new_spacing)
#             new_size = np.ceil(new_size).astype(np.int32).tolist()
#             resample.SetSize(new_size)
#             image_sitk = resample.Execute(image_sitk)
                    
#         # normalize & get array
#         image_sitk = 1.0 * (image_sitk - clip_min) / (clip_max - clip_min)
#         image_sitk = sitk.Cast(image_sitk, sitk.sitkFloat32)
#         image_arr = sitk.GetArrayFromImage(image_sitk)
        
#         # time tracking
#         tot_time = time.time() - start
#         if self._preprocess_time is None:
#             self._preprocess_time = tot_time 
#         else:
#             self._preprocess_time = 0.9 * self._preprocess_time + 0.1 * tot_time
        
#         return image_arr