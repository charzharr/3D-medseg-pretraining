""" transforms/gaussian_noise.py  (Author: Charley Zhang, 2021) """

import warnings
import collections
import numbers

import numpy as np
import torch

from data.transforms.transform_base import Transform
from lib.utils.parse import parse_probability, parse_bool


class GaussianNoise(Transform):
    """
    Properties:
        ✔ Tensor or Numpy Array
        ✔ 3D  ✔ 2D
        ✔ Differentiable
        ✘ Invertible
    """
    
    def __init__(self, p=1.0, mean=0., var=1., return_record=True):
        """
        Args:
            p: probability to apply this transform
            mean: can be a value or a tupled range of values to sample from
            var: can be a value or a tupled range of values to sample from
        """
        self.p = parse_probability(p, 'p')
        self.return_record = parse_bool(return_record, 'return_record')
        self.mean = self._parse_mean_var(mean)
        self.var = self._parse_mean_var(var)
        
    
    def apply_transform(self, data):
        data = self._parse_data_input(data)
        if torch.rand((1,)).item() > self.p:
            return data, None
        
        # Sample mean or variance if a tuple is given
        mean = self.mean
        if isinstance(mean, collections.Sequence):
            mean = torch.rand((1,)).item() * (mean[1] - mean[0]) + mean[0]
        
        var = self.var
        if isinstance(var, collections.Sequence):
            var = torch.rand((1,)).item() * (var[1] - var[0]) + var[0]
        
        # Process data inputs
        is_single_image = False
        if isinstance(data, np.ndarray) or isinstance(data, torch.Tensor):
            is_single_image = True
            data = {'image': data}
        
        ret_data, ret_records = {}, {}
        for k, v in data.items():
            if 'image' in k:
                image = v
                t_image = GaussianNoise.gaussian_noise(image, mean, var)
                ret_data[k] = t_image
                ret_records[k] = {
                    'input_shape': tuple(image.shape),
                    'output_shape': tuple(t_image.shape),
                    'mean': mean, 'variance': var
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
        warnings.warn('Gaussian Noise reapply will not produce the same '
                      'noise maps!')
        mean = record['mean']
        variance = record['variance']
        return GaussianNoise.gaussian_noise(image, mean, variance)
    
    
    @staticmethod
    def gaussian_noise(image, mean, var):
        assert isinstance(image, np.ndarray) or isinstance(image, torch.Tensor)

        shape = image.shape
        if isinstance(image, np.ndarray):
            ret_image = image + np.random.normal(mean, var, shape)
        else:
            ret_image = image + torch.normal(mean, var, size=shape)
        return ret_image
            
    
    def _parse_mean_var(self, value):
        if isinstance(value, numbers.Number):
            return float(value)
        elif isinstance(value, collections.Sequence):
            msg = (f'If you give a sequence for "mean" or "var", it must be '
                   f'length 2, not {len(value)}.')
            value = tuple(value)
            assert len(value) == 2, msg
            
            msg = f'1st value {value[0]} must be smaller than 2nd {value[1]}.'
            assert value[0] <= value[1]
            
            return value
        else:
            msg = '"mean" or "var" must be a number or a sequence of 2 nums'
            raise ValueError(msg)
