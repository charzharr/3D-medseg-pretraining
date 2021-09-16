""" transforms/gamma.py  (Author: Charley Zhang, 2021) """

import numbers
import collections

import numpy as np
import torch

from data.transforms.transform_base import Transform
from lib.utils.parse import parse_probability, parse_bool


class Gamma(Transform):
    """ Divide tensor or np-array by its mean & then divide by its std. 
    Properties:
        ✔ Tensor or Numpy Array
        ✔ 3D  ✔ 2D
        ✔ Differentiable
        ✔ Invertible (✔ differentiable)
    """
    
    def __init__(self, p=1.0, gamma=(0.7, 1.5), return_record=True):
        self.p = parse_probability(p, 'p')
        self.return_record = parse_bool(return_record, 'return_record')
        self.gamma = self._parse_gamma(gamma)
    
    
    def apply_transform(self, data, gamma=None):
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
        
        # Sample gamma
        gamma = self._parse_gamma(gamma) if gamma else self.gamma
        if isinstance(gamma, collections.Sequence):
            diff = gamma[1] - gamma[0]
            gamma = torch.rand((1,)).item() * diff + gamma[0]
        
        # Apply transform to inputs
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
                
                t_image = Gamma.gamma(image, gamma, image_min, image_max)
                
                ret_data[k] = t_image
                ret_records[k] = {
                    'input_shape': tuple(image.shape),
                    'output_shape': tuple(t_image.shape),
                    'gamma': gamma,
                    'old_image_min': self.to_record_value(image_min), 
                    'old_image_max': self.to_record_value(image_max)
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
        gamma = record['gamma']
        image_min = image.min()
        image_max = image.max()
        return Gamma.gamma(image, gamma, image_min, image_max)

        
    def invert(self, image, record):
        raise NotImplementedError()
        gamma = record['gamma']
        old_min = record['old_image_min']
        old_max = record['old_image_max']
        old_range = old_max - old_min
        
        ret_image = image * old_range + old_min
        ret_image = ret_image ** (1 / gamma)
        return ret_image
    
    
    @staticmethod
    def gamma(image, gamma, image_min, image_max):
        """
        When image values are close to 0 or negative, can cause NaN values.
        
        Read up more here:
        https://discuss.pytorch.org/t/problem-training-gamma-correction/60800/4
        """
        assert isinstance(image, np.ndarray) or isinstance(image, torch.Tensor)

        # Apply gamma transform
        image = (image - image_min) / (image_max - image_min)
        image = image ** gamma 
        image = image * (image_max - image_min) + image_min
        # print(image.shape, gamma, 'old', image_min, image_max, 'new', 
        #       image.min(), image.max())
        return image

        # print('before', bool(torch.isnan(image).any()))
        # if image_min < 0:
        #     if isinstance(image, np.ndarray):
        #         ret_image = np.sign(image) * np.abs(image) ** gamma
        #     else:
        #         ret_image = image.sign() * image.abs() ** gamma
        # else:
        #     ret_image = image ** gamma
        # print('after', bool(torch.isnan(ret_image).any()))
        
        # Normalize intensities to [0, 1]
        # image_range = image_max - image_min + 1e-7  # avoid divide by 0
        # ret_image = (ret_image - image_min) / image_range
        # return ret_image

        
    def _parse_gamma(self, gamma):
        if isinstance(gamma, numbers.Number):
            return gamma
        elif isinstance(gamma, collections.Sequence):
            msg = (f'If you give a sequence for "gamma", it must be '
                   f'length 2, not {len(gamma)}.')
            gamma = tuple(gamma)
            assert len(gamma) == 2, msg
            
            msg = f'1st value {gamma[0]} must be smaller than 2nd {gamma[1]}.'
            assert gamma[0] <= gamma[1]
            
            return gamma
        else:
            msg = '"gamma" must be a number or a range as a tuple of size 2.'
            raise ValueError(msg)
