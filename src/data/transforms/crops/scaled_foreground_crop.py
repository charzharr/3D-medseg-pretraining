""" transforms/crops/scaled_foreground_crop.py  (Author: Charley Zhang, 2021)

UniformCrop trains painfully slowly especially on verse sparse datasets like
BCV where class imbalance could be 10000:1. To alleviate this problem, this
class randomly samples a certain class, and then gets a random scaled crop
from the class's mask.
"""

import warnings
import math

import numpy as np
import torch

if __name__ == '__main__':
    import sys, pathlib
    curr_path = pathlib.Path().absolute()
    sys.path.append(str(curr_path.parent.parent.parent))

from lib.utils.parse import parse_bool, parse_range, parse_probability
from data.transforms.transform_base import Transform
from data.transforms.crops.utils import sample_crop, crop_resize_3d



class ScaledForegroundCropper3d:
    
    def __init__(
            self, 
            final_shape, 
            scale_range=(0.8, 1.2),
            foreground_p=0.5,
            interpolation='trilinear',
            return_record=True
            ):
        """
        Args:
            final_shape: shape of fine crop after scaled sample & 
            foreground_p: probability to sample a foreground crop
        """
        assert len(final_shape) == 3, 'Shape must be 3D'
        self.final_shape = final_shape
        self.scale_range = self._parse_scale_range(scale_range)
        self.foreground_p = parse_probability(foreground_p, 'foreground_p')
        self.interpolation = interpolation
        self.return_record = parse_bool(return_record, 'return_record')

        fp = self.foreground_p
        print(f'💠 ScaledForegroundCropper3d initiated (fg_p={fp}). \n'
              f'   final_shape={final_shape}, scale_range={self.scale_range}\n'
              f'   default_interpolation={self.interpolation}, '
              f'ret_record={self.return_record}')
    
    
    def __call__(
            self,
            volume_tensor,
            mask_tensor,
            n_times=1,
            final_shape=None,
            scale_range=None,
            foreground_p=None,
            interpolation=None
            ):
        """
        Args:
            volume_tensor: DxHxW
            mask_tensor: DxHxW (volume of class IDs)
        """

        assert volume_tensor.shape == mask_tensor.shape, 'Shape mismatch!'

        # Get crop parameters
        if final_shape is None:
            final_shape = self.final_shape
        else:
            assert len(final_shape) == 3
        
        if scale_range is None:
            scale_range = self.scale_range
        else:
            scale_range = self._parse_scale_range(scale_range)

        if foreground_p is None:
            foreground_p = self.foreground_p
        else:
            foreground_p = parse_probability(foreground_p, 'foreground_p')
            
        if interpolation is None:
            interpolation = self.interpolation
        
        volume_shape = volume_tensor.shape
        ndim = len(volume_shape)
        
        image_crop_record_list, mask_crop_record_list = [], []
        ids = None
        # Apply transform to inputs
        for n in range(n_times):

            sample_fg = True
            if torch.rand((1,)).item() > foreground_p:
                sample_fg = False

            crop_shape = []
            for i in range(0, 3):
                if scale_range[i][0] == scale_range[i][1]:
                    crop_shape.append(int(scale_range[i][0]) * final_shape[i])
                else:
                    min_shp = math.ceil(scale_range[i][0] * final_shape[i])
                    max_shp = int(scale_range[i][1] * final_shape[i])
                    samp_shape = torch.randint(min_shp, max_shp, (1,)).item()
                    crop_shape.append(samp_shape)
            crop_shape = [min(c, s) for c, s in zip(crop_shape, volume_shape)]

            if sample_fg:
                vol_shape = mask_tensor.shape
                tot_vol = vol_shape[0] * vol_shape[1] * vol_shape[2]
                if ids is None:
                    ids, cnts = mask_tensor.unique(return_counts=True, 
                                                   sorted=True)
                # print(f'[Took {time.time() - start:.2f}s to call unique().]')
                # print(f'Mask is of type: {mask_tensor.dtype}')

                # Sample a class id & index
                if len(ids) == 1:  # all background, just randomly sample
                    warnings.warn(f'You inputted a mask with only background.')
                    crop_lower, crop_upper = sample_crop(volume_shape, 
                                                         crop_shape)
                else:
                    num_fg_classes = len(ids[1:])
                    fg_index = torch.randint(1, num_fg_classes + 1, (1,)).item()
                    fg_class_id = ids[fg_index].item()

                    indices = torch.nonzero(mask_tensor == fg_class_id)
                    vol_class = len(indices)

                    # Sample crop
                    center_index = torch.randint(0, len(indices), (1,)).item()
                    center_index = indices[center_index].tolist()
                    # print(f'⭐ Index value: {mask_tensor[center_index[0], center_index[1], center_index[2]]}')

                    crop_lower, crop_upper = [], []
                    for dim, idx in enumerate(center_index):
                        dim_length = crop_shape[dim]
                        dim_max = vol_shape[dim]

                        half_dim_length = dim_length // 2
                        upper_idx = min(dim_max, idx + half_dim_length)

                        lower_idx = upper_idx - dim_length
                        if lower_idx < 0:
                            upper_idx -= lower_idx
                            lower_idx = 0
                        assert upper_idx - lower_idx == dim_length
                        assert 0 <= lower_idx < upper_idx <= dim_max

                        crop_lower.append(lower_idx)
                        crop_upper.append(upper_idx)
            else:  # random uniform sample
                # print(f'😄 Random Crop!')
                crop_lower, crop_upper = sample_crop(volume_shape, crop_shape)

            # print(crop_lower, crop_upper, np.array(crop_upper) - np.array(crop_lower))

            resized_crop = crop_resize_3d(volume_tensor, crop_lower, 
                crop_upper, final_shape, interpolation=interpolation)
            resized_mask = crop_resize_3d(mask_tensor, crop_lower, crop_upper, 
                                final_shape, interpolation='nearest')

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

                crop_shp = resized_mask.shape
                crop_vol = crop_shp[0] * crop_shp[1] * crop_shp[2]
                fg_vol = torch.sum(resized_mask > 0).item()
                mask_record = {
                        'input_shape': tuple(volume_tensor.shape),
                        'output_shape': tuple(resized_crop.shape),
                        'interpolation': 'nearest',
                        'init_crop_lower': tuple(crop_lower),
                        'init_crop_upper': tuple(crop_upper),
                        'init_crop_shape': tuple(crop_shape),
                        'foreground_vol': fg_vol,
                        'foreground_ratio': fg_vol / crop_vol
                    }
                if sample_fg:
                    class_vol = torch.sum(resized_mask == fg_class_id).item()
                    mask_record['foreground_id'] = fg_class_id
                    mask_record['class_vol'] = class_vol
                    mask_record['class_ratio'] = class_vol / crop_vol
                    # print(mask_record)

                mask_crop_record_list.append((resized_mask, mask_record))
            else:
                image_crop_record_list.append(resized_crop)
                mask_crop_record_list.append(resized_mask)    
        
        # Return results in desired format
        if n_times == 1:
            return image_crop_record_list[0], mask_crop_record_list[0]
        return image_crop_record_list, mask_crop_record_list
    
    
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
    mask_f = '/afs/crc.nd.edu/user/y/yzhang46/datasets/BCV-2015/train/label_nii/label0001.nii.gz'
    
    
    
    import SimpleITK as sitk

    mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_f))
    mask_tensor = torch.tensor(mask)
    image = torch.randn(mask_tensor.shape)

    cropper = ScaledForegroundCropper3d((16, 96, 96), scale_range=(1., 1.),
                                        foreground_p=.66)
    crops_y_records = cropper(image, mask_tensor, n_times=20)
    

    import IPython; IPython.embed(); 

    
    