""" transforms/intensity_scale.py  (Author: Charley Zhang, 2021) 
Brightness & contrast augmentation via scaling an image's intensity values.
"""

import collections
import numbers

import numpy as np
import torch

from data.transforms.transform_base import Transform
from lib.utils.parse import parse_probability, parse_bool


class ScaleIntensity(Transform):
    """ Samples a uniform value & multiplies image with it followed by clipping
    of tranformed image to the intensity bounds of the original image.
    
    (This is called brightness & contrast transformation in PGL paper)
    Properties:
        ✔ Tensor or Numpy Array
        ✔ 3D  ✔ 2D
        ✔ Differentiable
        ✔* Invertible (*information loss from clipping)
    """
    
    def __init__(self, p=1.0, scale=(0.75, 1.25), return_record=True):
        """
        Args:
            p: probability to apply this transform
            mean: can be a value or a tupled range of values to sample from
            var: can be a value or a tupled range of values to sample from
        """
        self.p = parse_probability(p, 'p')
        self.return_record = parse_bool(return_record, 'return_record')
        self.scale = self._parse_scale(scale)
        
    
    def apply_transform(self, data, scale=None):
        """
        Args:
            image: dict with key 'image' that is a np.array or torch.Tensor
                "image" can also be an array or tensor itself.
            gamma: if a value is given for gamma, it will override the any
                preset gamma or range of gammas to sample from.
        """
        data = self._parse_data_input(data)
        if torch.rand((1,)).item() > self.p:
            if self.return_record:
                return data, None
            return data
        
        # Sample scale
        scale = self._parse_scale(scale) if scale else self.scale
        if isinstance(scale, collections.Sequence):
            scale = torch.rand((1,)).item() * (scale[1] - scale[0]) + scale[0]
        
        # Apply intenisty scale transform
        is_single_image = False
        if isinstance(data, np.ndarray) or isinstance(data, torch.Tensor):
            is_single_image = True
            data = {'image': data}
        
        ret_data, ret_records = {}, {}
        for k, v in data.items():
            if 'image' in k:
                image = v
                image_min = image.min()
                image_max = image.max()
                
                t_image = ScaleIntensity.scale_intensity(
                            image, scale, image_min, image_max)
                ret_data[k] = t_image
                ret_records[k] = {
                    'input_shape': tuple(image.shape),
                    'output_shape': tuple(t_image.shape),
                    'old_image_min': self.to_record_value(image_min),
                    'old_image_max': self.to_record_value(image_max),
                    'intensity_scale': scale
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
        scale = record['intensity_scale']
        im_min = image.min()
        im_max = image.max() 
        return ScaleIntensity.scale_intensity(image, scale, im_min, im_max)
    
    
    def invert(self, image, record):
        scale = record['intensity_scale']
        ret_image = image / scale
        return ret_image
    
    
    @staticmethod
    def scale_intensity(image, scale, clip_min, clip_max):
        assert isinstance(image, np.ndarray) or isinstance(image, torch.Tensor)
        
        ret_image = image * scale
        ret_image = ret_image.clip(clip_min, clip_max)
        return ret_image

    
    
    
    
    def _parse_scale(self, value):
        if isinstance(value, numbers.Number):
            return float(value)
        elif isinstance(value, collections.Sequence):
            msg = (f'If you give a sequence for "scale", it must be '
                   f'length 2, not {len(value)}.')
            value = tuple(value)
            assert len(value) == 2, msg
            
            msg = f'1st value {value[0]} must be smaller than 2nd {value[1]}.'
            assert value[0] <= value[1]
            
            return value
        else:
            msg = '"scale" must be a number or a sequence of 2 nums'
            raise ValueError(msg)


