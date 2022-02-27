""" transforms/crops/scaled_uniform_noair_crop.py  (Author: Charley Zhang, 2022)

Practically the same as ScaledUniformCrop3d except there is a lower bound 
threshold for the mean of a crop to pass to be valid (avoid air in scans).

UniformCrop from torchio is way too slow and has very stringent input 
requirements. This class works directly with tensors or array and is much 
faster & easily configurable. 

Changes:
    (Feb 2022)
    - Added scale_sampler like all modern croppers (made instance check general)
    - Added numpy support in addition to tensor.
    - Added direct shape inputs to bypass size sampling.
        (used for SAR self-supervised pretraining)
"""

import math
import warnings

import numpy as np
import torch

if __name__ == '__main__':
    import sys, pathlib
    curr_path = pathlib.Path().absolute()
    sys.path.append(str(curr_path.parent.parent.parent))

from lib.utils.parse import parse_bool, parse_range
from data.transforms.transform_base import Transform
from data.transforms.crops.utils import sample_crop, crop_resize_3d


DEBUG = False  # save rejected crops
WARN = False   # warnings on crops being larger than volume


class ScaledUniformNoAirCropper3d:
    
    def __init__(
            self, 
            final_shape=None, 
            scale_sampler=None, 
            cubic_crop=True,
            min_mean_thresh=None,
            interpolation='trilinear',
            return_record=True
            ):
        """
        Args:
            final_shape: shape of fine crop after scaled window is resized.
            scale_sampler: object with a sample() function that returns a scale.
            min_mean_thresh: value threshold of air so that if the average value
                of a crop is below this, then it is rejected.
        """
        if final_shape is not None:
            assert len(final_shape) == 3, 'Shape must be 3D'
        self.final_shape = final_shape
        self.scale_sampler = scale_sampler
        self.cubic_crop = cubic_crop
        self.min_mean_thresh = min_mean_thresh
        self.interpolation = interpolation
        self.return_record = parse_bool(return_record, 'return_record')
    
    
    def __call__(
            self,
            volume,
            mask=None,
            n_times=1,
            final_shape=None,     # override init
            scale_sampler=None,   # override init
            cubic_crop=None,      # override init
            min_mean_thresh=None, # override init
            interpolation=None,   # override init
            patch_size=None       # override random crop size sampling
            ):
        assert isinstance(volume, (torch.Tensor, np.ndarray))
        
        # Get crop parameters
        if final_shape is None:
            final_shape = self.final_shape
        else:
            assert len(final_shape) == 3, 'final_shape must be 3D'
        
        if scale_sampler is None:
            scale_sampler = self.scale_sampler
        if patch_size is None:
            assert hasattr(scale_sampler, 'sample') and \
                   callable(scale_sampler.sample)
        
        if cubic_crop is None:
            cubic_crop = self.cubic_crop
        cubic_crop = parse_bool(cubic_crop, 'cubic_crop')
        
        if min_mean_thresh is None:
            min_mean_thresh = self.min_mean_thresh
        
        if interpolation is None:
            interpolation = self.interpolation
        
        volume_shape = volume.shape
        D, H, W = volume_shape
        ndim = len(volume_shape)
        
        image_crop_record_list = []
        if mask is not None:
            assert isinstance(mask, (torch.Tensor, np.ndarray))
            mask_crop_record_list = []
        
        # Apply transform to inputs
        n_valid_crops = 0
        while n_valid_crops < n_times:
            
            # Get crop upper & lower bounding coordinates
            if patch_size is not None:
                assert len(patch_size) == 3
                assert min(tuple(patch_size)) <= min(tuple(volume.shape))
                _d = int(patch_size[0])
                _h = int(patch_size[1])
                _w = int(patch_size[2])
            elif cubic_crop:  # same scale for all dims
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
                    if WARN:
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
            
            # Get resized crop & check validity
            resized_crop = crop_resize_3d(volume, crop_lower, crop_upper, 
                                          final_shape, 
                                          interpolation=interpolation)
            if min_mean_thresh is not None:
                if resized_crop.mean() < min_mean_thresh:
                    if DEBUG:
                        print(f'🏗️  Crop rejected,mean: {resized_crop.mean()}')
                        print(f'    Lower: {crop_lower}, Upper: {crop_upper}')
                    continue  # if not valid, sample another one

            n_valid_crops += 1
            
            # Record creation
            if self.return_record:
                record = {
                    'input_shape': tuple(volume.shape),
                    'output_shape': tuple(resized_crop.shape),
                    'interpolation': interpolation,
                    'init_crop_lower': tuple(crop_lower),
                    'init_crop_upper': tuple(crop_upper),
                    'init_crop_shape': tuple(crop_shape)
                }
                image_crop_record_list.append((resized_crop, record))
            else:
                image_crop_record_list.append(resized_crop)
            
            if mask is not None:
                resized_mask = crop_resize_3d(mask, crop_lower, crop_upper, 
                                final_shape, interpolation='nearest')
                if self.return_record:
                    mask_record = {
                        'input_shape': tuple(volume.shape),
                        'output_shape': tuple(resized_crop.shape),
                        'interpolation': 'nearest',
                        'init_crop_lower': tuple(crop_lower),
                        'init_crop_upper': tuple(crop_upper),
                        'init_crop_shape': tuple(crop_shape)
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
        crop_lower = record['init_crop_lower']
        crop_upper = record['init_crop_upper']
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

    
    