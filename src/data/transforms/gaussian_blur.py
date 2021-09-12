""" transforms/gaussian_blur.py  (Author: Charley Zhang, 2021) """

import numbers
import collections
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.ndimage as ndi

from data.transforms.transform_base import Transform
from lib.utils.parse import parse_probability, parse_bool


class GaussianBlur(Transform):
    """ 
    Properties:
        ✔ Tensor  ✘ Numpy Array (TODO but not necessary for now)
        ✔ 3D  ✔ 2D
        ✘ Differentiable
        ✘ Invertible
    """
    
    def __init__(self, p=1.0, spacing=1, sigma=(0.5, 1.), ndim=3, 
                 return_record=True):
        """
        Args:
            p (float): probability of applying transform
            sigma (dim_range): can be a single number (applied to all dims), 
                a single range (each dim sampled independently), or ndim
                number of ranges.
            ndim (int): number of dimensions 
        """
        self.p = parse_probability(p, 'p')
        self.return_record = parse_bool(return_record, 'return_record')
        
        self.ndim = ndim
        assert ndim in (2, 3), f'Only 2D/3D supported for "ndim."'
        
        self.sigma = self._parse_sigma(sigma)
        self.spacing = self._parse_spacing(spacing)
    

    def apply_transform(self, data, spacing=None, sigma=None):
        data = self._parse_data_input(data)
        if torch.rand((1,)).item() > self.p:
            if self.return_record:
                return data, None
            return data
        
        # Sample sigma and spacing for each image dim
        if sigma is not None:
            sigma = self._parse_sigma(sigma)
        else:
            sigma = self.sigma
        sigma = self._sample_ranges(sigma)  # ndim list of sigmas per dim
        
        if spacing is not None:
            spacing = self._parse_sigma(spacing)
        else:
            spacing = self.spacing
        
        # Process data inputs
        is_single_image = False
        if isinstance(data, np.ndarray) or isinstance(data, torch.Tensor):
            is_single_image = True
            data = {'image': data}
            
        # Apply blur transform
        ret_data, ret_records = {}, {}
        for k, v in data.items():
            if 'image' in k:
                image = v
                t_image = GaussianBlur.gaussian_blur(image, spacing, sigma)
                ret_data[k] = t_image
                ret_records[k] = {
                    'input_shape': tuple(image.shape),
                    'output_shape': tuple(t_image.shape),
                    'std': sigma, 
                    'spacing': spacing
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
        sigma = record['std']
        spacing = record['spacing']
        return GaussianBlur.gaussian_blur(image, spacing, sigma)
    
    
    @staticmethod
    def gaussian_blur(image, spacing, sigma):
        conv_image = image
        if isinstance(image, torch.Tensor):
            conv_image = image.detach().cpu().numpy()
        
        std_physical = np.array(sigma) / np.array(spacing)
        ret_image = ndi.gaussian_filter(conv_image, std_physical)
        
        if isinstance(image, torch.Tensor):
            return torch.as_tensor(ret_image)
        return ret_image
        
    
    def _parse_kernel_size(self, kernel_size, ndim):
        if isinstance(kernel_size, int):
            return [kernel_size] * ndim
        elif isinstance(kernel_size, collections.Sequence):
            msg = (f'If you give a sequence for "kernel_size", it must be '
                   f'the same length as ndim ({ndim}), not {len(kernel_size)}.')
            kernel_size = list(kernel_size)
            assert len(kernel_size) == ndim, msg
            
            msg = ('All sizes in kernel sizes must be a positive int, ' 
                   'got: {}')
            for size in kernel_size:
                assert isinstance(size, int) and size > 0, msg.format(size)
            
            return kernel_size
        else:
            msg = '"kernel_size" must be an integer or a sequence of ints.'
            raise ValueError(msg)
        
    
    def _parse_sigma(self, value):
        return Transform.parse_dimensional_ranges(value, self.ndim, 'sigma', 
                                              min_constraint=0)

    def _parse_spacing(self, spacing):
        if isinstance(spacing, numbers.Number):
            spacing = [spacing for _ in range(self.ndim)]
        else:
            assert len(spacing) == self.ndim, 'Spacing length must match ndim.'
        return spacing

        
class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. 
            Output will have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                f'Only 1, 2 and 3 dimensions are supported. Received {dim}.'
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)
