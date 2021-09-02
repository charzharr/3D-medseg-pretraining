""" transforms/transform_base.py  (Author: Charley Zhang, 2021) 
Base class for all 2D & 3D image transforms. 
"""

from multiprocessing import Value
import time
import logging
import numbers
import collections
from abc import ABC, abstractmethod

import numpy as np
import torch


class Transform(ABC):
    """ Base class for image transforms. """
    
    @property
    def is_invertible(self):
        return hasattr(self, 'invert')
    
    @property
    def name(self):
        return type(self).__name__
    
    def __call__(self, data, *args, **kwargs):
        """
        Input: 
            Case 1 - data is a dict: transforms are only applied when
                the dict key contains 'image' or 'mask', rest are ignored.
            Case 2 - data is a list: every list element is assumed to be
                an image (type torch.Tensor or np.ndarray)
            Case 3 - data is a an image: obviously a regular transform
                is performed. If you want to only transform a mask, 
                then you have input the dict {'mask': mask_image}.
        """
        start = time.time()
        ret = self.apply_transform(data, *args, **kwargs)
        logging.info(f'🤖 {self.name} took {time.time() - start:.2f} sec.')
        return ret
        
    @abstractmethod
    def apply_transform(self, *args, **kwargs):
        pass
    
    def to_record_value(self, value):
        """ Sometimes transform paramters are tensors & we only want
        python elementary types in the transform records. This handles that. """
        if isinstance(value, torch.Tensor):
            return value.item()
        return value
    
    def _parse_data_input(self, data):
        if isinstance(data, dict):
            for k, v in data.items():
                if 'image' in k or 'mask' in k:
                    assert v.ndim <= 3
            return data
        elif isinstance(data, torch.Tensor) or isinstance(data, np.ndarray):
            assert data.ndim <= 3
            return data
        else:
            msg = '"data" input must be a dict, array, or tensor.'
            raise ValueError(msg)
    
    @staticmethod
    def parse_dimensional_ranges(range_input, ndim, name, 
                                  min_constraint=None, max_constraint=None):
        """
        Args:
            range_input (number, range, sequence of ranges): range input to be
                parsed.
            ndim (int): dimensionality of an image.
            name (str): name of parameter for useful error messages.
            min_constraint (opt. number): absolute minimum sampling value
            max_constraint (opt. number): absolute maximum sampling value
        Returns:
            A range (tuple of 2 inclusive values) for each image dimension.
                e.g. for 3D images: [(1, 1), (0.5, 1), (0.5, 1)]
        """
        assert ndim in (2, 3), f'Only 2D & 3D images are supported, not {ndim}.'
        if isinstance(range_input, numbers.Number):
            if min_constraint is not None and range_input < min_constraint:
                msg = (f'Given range number {range_input} for {name} is less '
                       f'than the min_constraint {min_constraint}')
                raise ValueError(msg)
            if max_constraint is not None and range_input > max_constraint:
                msg = (f'Given range number {range_input} for {name} is more '
                       f'than the max_constraint {max_constraint}')
                raise ValueError(msg)
            out_range = [(range_input, range_input) for _ in range(ndim)]
        elif isinstance(range_input, collections.Sequence):
            if isinstance(range_input[0], collections.Sequence):
                if len(range_input) != ndim:
                    msg = (f'If given multiple ranges, then you must give '
                           f'exactly ndim={ndim} ranges.')
                    raise ValueError(msg)
                for r in range_input:
                    if len(r) != 2:
                        msg = (f'If given multiple ranges for {name}, then '
                                'they must all be valid with 2 numbers.')
                        raise ValueError(msg)
                    if r[0] > r[1]:
                        msg = (f'Invalid range for {name}: first value {r[0]} '
                               f'is larger than second {r[1]}.')
                        raise ValueError(msg)
                out_range = [tuple(rang) for rang in range_input]
            else:  # given just 1 range for all dims 
                if len(range_input) != 2:
                    msg = (f'If given a range for {name}, then '
                            'it must be valid range with 2 numbers.')
                    raise ValueError(msg)
                r = range_input
                if r[0] > r[1]: 
                    msg = (f'Invalid range for {name}: first value {r[0]} '
                            f'is larger than second {r[1]}.')
                    raise ValueError(msg)
                out_range = [tuple(range_input) for _ in range(ndim)]
        else:
            msg = (f'Range for "{name}" must be a number, range, or '
                   f'{ndim} ranges')
            raise ValueError(msg)
        
        # Check range min, max is within sepcificiations
        for r in out_range:
            if min_constraint is not None and r[0] < min_constraint:
                msg = (f'Processed {range_input} to {out_range}, but min range '
                       f'{r[0]} for {name} is less than the '
                       f'minimum constraint {min_constraint}')
                raise ValueError(msg)
            if max_constraint is not None and r[1] > max_constraint:
                msg = (f'Processed {range_input} to {out_range}, but max range '
                       f'{r[1]} for {name} is more than the '
                       f'maximum constraint {max_constraint}')
                raise ValueError(msg)
        
        return out_range
    
    @staticmethod
    def _sample_ranges(dim_ranges):
        sampled_out = []
        for r in dim_ranges:
            sampled_out.append(Transform._sample_range(r[0], r[1]))
        return sampled_out
    
    @staticmethod
    def _sample_range(min_val, max_val):
        if min_val == max_val:
            return min_val
        return torch.rand((1,)).item() * (max_val - min_val) + min_val
            
    
