""" transforms/flip.py (Author: Charley Zhang, 2021)
A transform that works on 2D/3D tensors & np arrays.

----------------------------------------------------------------------
PGL Transformations (2 spatial, 4 intensity): 
- Crop & Scale (110% to 140% of final crop size), at least 10% overlap
- 50% flip along x, y, z axis
- Gaussian noise (uniform from 0 to 0.1 variance), p=0.1
- Gaussian blur (sigma=[0.5, 1]) p=0.2
- Brightness / contrast (1st mult by [0.75, 1.25] then clipped), p=0.5
- Gamma transform (λ=[0.7,1.5], then scaled to [0,1]), p=0.5

Reversal for MSE calculation:
- First reverse axis flipping
- Then reverse the crop/scaling
"""

import collections
import numbers

import numpy as np
import torch

from data.transforms.transform_base import Transform
from lib.utils.parse import parse_probability, parse_bool


class Flip3d(Transform):
    """
    Properties:
        ✔ Tensor or Numpy Array
        ✔ 3D  ✘ 2D
        ✔ Differentiable
        ✔ Invertible  ✔ Differentiable
    History: list of size ndim with True value at every axis that was flipped.
        e.g. [True, False, True] means axis 0 & 2 have been flipped.
    """
    
    def __init__(self, p=1.0, flipx=True, flipy=True, flipz=True,
                 return_record=True):
        self.p = parse_probability(p, 'p')
        self.return_record = parse_bool(return_record, 'return_record')
        
        self.flipx = parse_bool(flipx, 'flipx')
        self.flipy = parse_bool(flipy, 'flipz')
        self.flipz = parse_bool(flipz, 'flipz')
    
    
    def apply_transform(self, data):
        data = self._parse_data_input(data)
        
        flip_flags = None
        is_single_image = False
        if isinstance(data, np.ndarray) or isinstance(data, torch.Tensor):
            is_single_image = True
            data = {'image': data}
        
        ret_data, ret_records = {}, {}
        for k, v in data.items():
            if 'image' in k or 'mask' in k:
                image = v
                if flip_flags is None:
                    flip_flags = [False] * image.ndim
                    for i in range(image.ndim):
                        if torch.rand((1,)).item() <= self.p:
                            flip_flags[i] = True
                
                t_image = Flip3d.flip(image, flip_flags)
                ret_data[k] = t_image
                ret_records[k] = {
                    'input_shape': tuple(image.shape),
                    'output_shape': tuple(t_image.shape),
                    'flipped_flags': flip_flags
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
        flipped_flags = record['flipped_flags']
        return Flip3d.flip(image, flipped_flags)
        
    
    def invert(self, image, record):
        flipped_flags = record['flipped_flags']
        return Flip3d.flip(image, flipped_flags)
    
    
    @staticmethod
    def flip(image, flip_flags):
        assert isinstance(image, np.ndarray) or isinstance(image, torch.Tensor)
        
        if isinstance(image, torch.Tensor):
            dims = [i for i, flag in enumerate(flip_flags) if flag]
            image = torch.flip(image, dims)
            return image
        
        for dim, flag in enumerate(flip_flags):
            if not flag:
                continue
            image = np.flip(image, axis=dim)
        return image
