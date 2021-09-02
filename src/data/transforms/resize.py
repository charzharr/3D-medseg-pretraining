""" transforms/resize.py  (Author: Charley Zhang, 2021)

Tips of implementations from MONAI:
https://github.com/Project-MONAI/MONAI/blob/dev/monai/transforms/spatial/array.py
"""

import numbers
import collections

import numpy as np
import torch

from lib.utils.parse import parse_probability, parse_bool
from data.transforms.transform_base import Transform


class Resize3d(Transform):
    """ Resize a tensor or array via torch's interpolation function. 
    Properties:
        ✔ Tensor  ✘ Numpy Array (TODO later if necessary)
        ✔ 3D
        ✔ Differentiable
        ✔* Invertible (✔ differentiable; *info loss is possible) 
    """
    interpolations = ('nearest', 'linear', 'bilinear', 'bicubic', 'trilinear',
                      'area')
    
    def __init__(self, size, p=1.0, default_interpolation='trilinear',
                 return_record=True):
        msg = (f'Given "default_interpolation," {default_interpolation}, '
               f'must be one of {self.interpolations}')
        self.p = parse_probability(p, 'p')
        self.size = self._parse_size(size)
        self.return_record = parse_bool(return_record, 'return_record')
        
        assert default_interpolation in self.interpolations, msg
        self.default_interpolation = default_interpolation
        
    
    def apply_transform(self, data, size=None, interpolation=None):
        data = self._parse_data_input(data)
        if torch.rand((1,)).item() > self.p:
            return data, None
        
        # Get transform parameters
        if size is not None:
            size = self._parse_size(size)
        else:
            size = self.size
        
        if interpolation is not None:
            msg = (f'Given "interpolation," {interpolation}, '
                   f'must be one of {self.interpolations}')
            assert interpolation in self.interpolations, msg
            interpolation = interpolation
        else:
            interpolation = self.default_interpolation
        
        # Apply transform to inputs
        is_single_image = False
        if isinstance(data, np.ndarray) or isinstance(data, torch.Tensor):
            is_single_image = True
            data = {'image': data}
        
        ret_data, ret_records = {}, {}
        for k, v in data.items():
            is_image = 'image' in k
            is_mask = False if is_image else 'mask' in k
            if is_image or is_mask:
                image = v
                interp_mode = interpolation if is_image else 'nearest'
                t_image = Resize3d.resize(image, size, interp_mode)
                
                if tuple(t_image.shape) != tuple(size):
                    import IPython; IPython.embed(); 
                
                ret_data[k] = t_image
                ret_records[k] = {
                    'input_shape': tuple(image.shape),
                    'output_shape': tuple(t_image.shape),
                    'interpolation': interp_mode,
                }
            else:
                ret_data[k] = v
                
        # Return in correct format
        if is_single_image:
            if self.return_record:
                return ret_data['image'], ret_records['image']
            return ret_data['image']
        if self.return_record:
            return ret_data, ret_records
        return ret_data
    
    
    def reapply(self, image, record):
        final_size = record['output_shape']
        interpolation = record['interpolation']
        return Resize3d.resize(image, final_size, interpolation)
    
    
    def invert(self, image, record):
        final_size = record['input_shape']
        interpolation = record['interpolation']
        return Resize3d.resize(image, final_size, interpolation)
    
         
    @staticmethod
    def resize(image, size, interpolation):
        assert isinstance(image, torch.Tensor), 'Only tensors supported.'
        
        shape = image.shape
        if image.ndim == 4:
            msg = ('If you give Resize3D a 4 dim tensor, then dim 1 must be '
                   f'1 (shape=1xDxHxW, instead of {shape}).')
            assert shape[0] == 1, msg
        else:
            image = image.unsqueeze(0).unsqueeze(0)
        
        dtype = image.dtype
        if 'float' not in str(dtype):
            image = image.float()
        
        align_applies = 'linear' in interpolation or interpolation == 'bicubic'
        ret_image = torch.nn.functional.interpolate(  
            image,
            size=size,
            mode=interpolation,
            align_corners=False if align_applies else None   # damn warning.
        )
        
        if ret_image.dtype != dtype:
            ret_image = ret_image.to(dtype)
        
        if len(shape) == 4:
            return ret_image
        return ret_image.squeeze(0).squeeze(0)


    def _parse_size(self, size):
        if isinstance(size, numbers.Number):
            assert isinstance(size, int), '"Size" must be an int'
            return [size for _ in range(3)]
        elif isinstance(size, collections.Sequence):
            assert len(size) == 3, '"Size" must be 3 dimensions.'
            return list(size)
        else:
            msg = '"size" must be a number or a sequence of 3 nums.'
            raise ValueError(msg)