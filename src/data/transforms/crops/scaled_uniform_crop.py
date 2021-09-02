""" transforms/crops/scaled_uniform_crop.py  (Author: Charley Zhang, 2021)

UniformCrop from torchio is way too slow and has very stringent input 
requirements. This class works directly with tensors or array and is much 
faster & easily configurable. 
"""

import math

import numpy as np
import torch

if __name__ == '__main__':
    import sys, pathlib
    curr_path = pathlib.Path().absolute()
    sys.path.append(str(curr_path.parent.parent.parent))

from lib.utils.parse import parse_bool, parse_range
from data.transforms.transform_base import Transform
from data.transforms.crops.utils import sample_crop, crop_resize_3d



class ScaledUniformCropper3d:
    
    def __init__(
            self, 
            final_shape, 
            scale_range=(0.8, 1.2), 
            interpolation='trilinear',
            return_record=True
            ):
        """
        Args:
            final_shape: shape of fine crop after scaled sample & 
        
        """
        assert len(final_shape) == 3, 'Shape must be 3D'
        self.final_shape = final_shape
        self.scale_range = self._parse_scale_range(scale_range)
        self.interpolation = interpolation
        self.return_record = parse_bool(return_record, 'return_record')
    
    
    def __call__(
            self,
            volume_tensor,
            mask=None,
            n_times=1,
            final_shape=None,
            scale_range=None,
            interpolation=None
            ):
        
        # Get crop parameters
        if final_shape is None:
            final_shape = self.final_shape
        else:
            assert len(final_shape) == 3
        
        if scale_range is None:
            scale_range = self.scale_range
        else:
            scale_range = self._parse_scale_range(scale_range)
            
        if interpolation is None:
            interpolation = self.interpolation
        
        volume_shape = volume_tensor.shape
        ndim = len(volume_shape)
        
        image_crop_record_list = []
        if mask is not None:
            mask_crop_record_list = []
        
        # Apply transform to inputs
        for n in range(n_times):
            crop_shape = [
                torch.randint(math.ceil(scale_range[i][0] * final_shape[i]), 
                int(scale_range[i][1] * final_shape[i]), (1,)).item()
                for i in range(0, 3)
            ]
            crop_lower, crop_upper = sample_crop(volume_shape, crop_shape)
            resized_crop = crop_resize_3d(volume_tensor, crop_lower, crop_upper, 
                                final_shape, interpolation=interpolation)
            if self.return_record:
                record = {
                    'input_shape': tuple(volume_tensor.shape),
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
                        'input_shape': tuple(volume_tensor.shape),
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
        

if __name__ == '__main__':
    
    cropper = ScaledUniformCropper3d((16, 96, 96), scale_range=(0.5, 1.5))
    
    image = torch.randn((100, 512, 512))
    crops_y_records = cropper(image, n_times=1)
    
    import IPython; IPython.embed(); 

    
    