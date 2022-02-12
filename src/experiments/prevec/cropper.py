""" cropper.py  (Author: Charley Zhang, 2022)

Made specifically for 3D spatial pretraining tasks.
ScaledUniformCrop from crop transforms does not support the following
functionality, so I added them:
  - Non-square crops & retain cubic crops
  - Discrete scales to select from
  - Exact edge point & volume information for vector prediction pretext task.
"""

import warnings
import math
import numpy as np
import torch
from collections import Sequence

if __name__ == '__main__':
    import sys, pathlib
    curr_path = pathlib.Path().absolute()
    sys.path.append(str(curr_path.parent.parent.parent))

from lib.utils.parse import parse_bool, parse_range
from data.transforms.transform_base import Transform
from data.transforms.crops.utils import sample_crop, crop_resize_3d
from experiments.prevec.sampler import ValueSampler
        

class SpatialPretrainCropper3d:
    
    def __init__(
            self, 
            final_shape=None, 
            scale_sampler=None, 
            cubic_crop=True,
            interpolation='trilinear',
            return_record=True
            ):
        """
        Args:
            final_shape: shape of fine crop after scaled sample & 
        
        """
        if scale_sampler:
            assert isinstance(scale_sampler, ValueSampler)
        assert len(final_shape) == 3, 'Shape must be 3D'
        
        self.final_shape = final_shape
        self.scale_sampler = scale_sampler
        self.cubic_crop = cubic_crop
        self.interpolation = interpolation
        self.return_record = parse_bool(return_record, 'return_record')
    
    def __call__(
            self,
            volume_tensor,       # data in (DxHxW)
            mask=None,           # opt data in (DxHxW)
            n_times=1,           # opt #crops to extract
            final_shape=None,    # override init
            scale_sampler=None,  # override init
            cubic_crop=None,     # override init
            interpolation=None   # override init
            ):
        
        # Get crop parameters
        if final_shape is None:
            final_shape = self.final_shape
        assert len(final_shape) == 3
        
        if scale_sampler is None:
            scale_sampler = self.scale_sampler
        assert isinstance(scale_sampler, ValueSampler)
        
        if cubic_crop is None:
            cubic_crop = self.cubic_crop
        cubic_crop = parse_bool(cubic_crop, 'cubic_crop')
            
        if interpolation is None:
            interpolation = self.interpolation
        
        volume_shape = volume_tensor.shape
        D, H, W = volume_shape
        ndim = len(volume_shape)
        
        # Apply transform to inputs
        image_crop_record_list = []
        if mask is not None:
            mask_crop_record_list = []
        
        for n in range(n_times):
            
            if cubic_crop:  # same scale for all dims
                dim_scale = scale_sampler.sample()
                _d = int(dim_scale * final_shape[0])
                _h = int(dim_scale * final_shape[1])
                _w = int(dim_scale * final_shape[2])
                
                dim_ratios = [_d/D, _h/H, _w/W]
                max_idx = np.array(dim_ratios).argmax()
                if dim_ratios[max_idx] > 1.0:
                    msg = (f'Tried to extract cubic patch of final shape '
                           f'{final_shape} out of a {volume_shape} volume.'
                           f'Resulted in an oversized patch of {[_d,_h,_w]} '
                           f'(scale {dim_scale:.2f}).')
                    warnings.warn(msg)
                    _d = int(_d / dim_ratios[max_idx])
                    _h = int(_h / dim_ratios[max_idx])
                    _w = int(_w / dim_ratios[max_idx])            
            else:  # random scale calculated independently for all dims
                dim_scale = scale_sampler.sample()
                _d = min(D, int(dim_scale * final_shape[0]))
                dim_scale = scale_sampler.sample()
                _h = min(H, int(dim_scale * final_shape[1]))
                dim_scale = scale_sampler.sample()
                _w = min(W, int(dim_scale * final_shape[2]))
            crop_shape = [_d, _h, _w]
            assert 3 == sum([c <= s for c, s in zip(crop_shape, volume_shape)])
            
            crop_lower, crop_upper = sample_crop(volume_shape, crop_shape)
            resized_crop = crop_resize_3d(volume_tensor, crop_lower, crop_upper, 
                                          final_shape, 
                                          interpolation=interpolation)
            if self.return_record:
                # info for vector prediction
                inpvol_center = [s / 2 for s in volume_tensor.shape]
                outpat_center = [(p1 + p2) / 2 for p1, p2 in 
                                 zip(crop_lower, crop_upper)]
                vol_patadj_center = [outpat_center[0], inpvol_center[1],
                                     inpvol_center[2]]
                nine_points = [
                    outpat_center,  # patch center
                    crop_lower,
                    crop_upper,
                    [crop_lower[0], crop_upper[1], crop_upper[2]],  # LR diag from lower
                    [crop_upper[0], crop_lower[1], crop_lower[2]],  # UR diag upper
                    [crop_lower[0], crop_lower[1], crop_upper[2]],
                    [crop_upper[0], crop_upper[1], crop_lower[2]],
                    [crop_lower[0], crop_upper[1], crop_lower[2]],
                    [crop_upper[0], crop_lower[1], crop_upper[2]],
                ]
                from experiments.prevec.vector import Vector3d
                nine_vectors = [Vector3d(pt, vol_patadj_center) 
                                for pt in nine_points]
                
                record = {
                    'input_shape': tuple(volume_tensor.shape),
                    'output_shape': tuple(resized_crop.shape),
                    'interpolation': interpolation,
                    'final_crop_lower': tuple(crop_lower),
                    'final_crop_upper': tuple(crop_upper),
                    'final_crop_shape': tuple(crop_shape),
                    'input_volume_center': inpvol_center,
                    'input_volume_patchadjusted_center': vol_patadj_center,
                    'final_crop_center': outpat_center,
                    'final_crop_points': nine_points,
                    'final_crop_vectors': nine_vectors
                }
                image_crop_record_list.append((resized_crop, record))
            else:
                image_crop_record_list.append(resized_crop)
            
            if mask is not None:
                resized_mask = crop_resize_3d(mask, crop_lower, crop_upper, 
                                final_shape, interpolation='nearest')
                if self.return_record:
                    mask_record = {
                        'input_shape': tuple(volume_tensor.shape),
                        'output_shape': tuple(resized_crop.shape),
                        'interpolation': 'nearest',
                        'final_crop_lower': tuple(crop_lower),
                        'final_crop_upper': tuple(crop_upper),
                        'final_crop_shape': tuple(crop_shape)
                    }
                    mask_crop_record_list.append((resized_mask, mask_record))
                else:
                    mask_crop_record_list.append(resized_mask)
        
        # Return results in desired format
        if n_times == 1:
            if mask is not None:
                return image_crop_record_list[0], mask_crop_record_list[0]
            return image_crop_record_list[0]
        
        if mask is not None:
            return image_crop_record_list, mask_crop_record_list
        return image_crop_record_list
    
    
    def reapply(self, image, record, interpolation=None):
        crop_lower = record['final_crop_lower']
        crop_upper = record['final_crop_upper']
        final_shape = record['output_shape']
        
        if interpolation is None:
            interpolation = record['interpolation']
        resized_crop = crop_resize_3d(image, crop_lower, crop_upper, 
                        final_shape, interpolation=interpolation)
        return resized_crop
    
    
    def _parse_data_input(self, data):
        if isinstance(data, dict):
            return data
        elif isinstance(data, torch.Tensor) or isinstance(data, np.ndarray):
            return data
        else:
            msg = '"data" input must be a dict, array, or tensor.'
            raise ValueError(msg)
        
    
    def _parse_scale_range(self, value):
        return Transform.parse_dimensional_ranges(value, 3, 'scale_range',
                                                  min_constraint=0)



if __name__ == '__main__':
    cropper = SpatialPretrainCropper3d((16, 96, 96), scale_range=(0.5, 1.5))
    
    D, H, W = 256, 256, 256
    image = torch.randn((D, H, W))
    
    scale_sampler = ValueSampler(is_discrete=True, scales=[1.])
    crops_y_records = cropper(image, scale_sampler=scale_sampler, 
                              cubic=True, n_times=1)
    
    import IPython; IPython.embed(); 

    
    