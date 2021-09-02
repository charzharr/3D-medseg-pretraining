""" transforms/z_normalize.py  (Author: Charley Zhang, 2021) """

import numpy as np
import torch

from lib.utils.parse import parse_probability, parse_bool
from data.transforms.transform_base import Transform


class ZNormalize(Transform):
    """ Divide tensor or np-array by its mean & then divide by its std. 
    Properties:
        ✔ Tensor or Numpy Array
        ✔ 3D  ✔ 2D  (scalar images only, doesn't handle channel norm)
        ✔ Differentiable
        ✔ Invertible  ✔ Differentiable
    """
    
    def __init__(self, p=1.0, return_record=True):
        self.p = parse_probability(p, 'p')
        self.return_record = parse_bool(return_record, 'return_record')
    
    
    def apply_transform(self, data):
        data = self._parse_data_input(data)
        if torch.rand((1,)).item() > self.p:
            return data, None
        
        is_single_image = False
        if isinstance(data, np.ndarray) or isinstance(data, torch.Tensor):
            is_single_image = True
            data = {'image': data}
        
        ret_data, ret_records = {}, {}
        for k, v in data.items():
            if 'image' in k:
                image = v
                if isinstance(image, torch.Tensor):
                    dtype = image.dtype
                    if 'float' not in str(dtype):
                        image = image.double()
                
                mean = image.mean()
                std = image.std()
                t_image = ZNormalize.z_normalize(image, mean, std)
                
                if isinstance(t_image, torch.Tensor):
                    t_image = t_image.float()
                ret_data[k] = t_image
                
                ret_records[k] = {
                    'input_shape': tuple(image.shape),
                    'output_shape': tuple(t_image.shape),
                    'mean': self.to_record_value(mean), 
                    'std': self.to_record_value(std)
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
        assert isinstance(image, torch.Tensor) or isinstance(image, np.ndarray)
        mean = record['mean']
        std = record['std']
        return ZNormalize.z_normalize(image, mean, std)
        
    
    def invert(self, image, record):
        assert isinstance(image, torch.Tensor) or isinstance(image, np.ndarray)
        mean = record['mean']
        std = record['std']
        
        ret_image = image * std + mean
        return ret_image
    
    @staticmethod
    def z_normalize(image, mean, std):
        ret_image = (image - mean) / std
        return ret_image


