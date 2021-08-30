""" Module transforms/crops/utils.py

Utility functions for cropping and related functionality.
"""

from cv2 import resize
import numpy as np
import torch

if __name__ == '__main__':
    import sys, pathlib
    curr_path = pathlib.Path().absolute()
    sys.path.append(str(curr_path.parent.parent.parent))

from data.transforms.resize import Resize3d


def crop_3d(image, lower, upper):
    shape = image.shape
    assert len(shape) == 3, f'Image must have 3 dims: DxHxW, gave: {shape}'
    
    if isinstance(image, torch.Tensor):
        return image[lower[0]:upper[0], 
                     lower[1]:upper[1], 
                     lower[2]:upper[2]].clone()
    elif isinstance(image, np.ndarray):
        return np.copy(image[lower[0]:upper[0], 
                             lower[1]:upper[1], 
                             lower[2]:upper[2]])
    else:
        raise ValueError(f'Image must be a np-array or torch-tensor.')

    
def crop_resize_3d(image, lower, upper, final_shape, 
                   resize_transform=None, interpolation=None):
    assert isinstance(image, torch.Tensor), 'For now, only tensors supported.'
    if resize_transform is not None:
        assert isinstance(resize_transform, Resize3d)
    
    crop = crop_3d(image, lower, upper)
    
    if resize_transform is None:
        resize_transform = Resize3d(final_shape, return_record=False)
    
    if interpolation is not None:
        return resize_transform(crop, interpolation=interpolation)
    return resize_transform(crop)
    

def sample_crop(image_shape, crop_shape, return_array=False):
    """ Sample crop indices for an arbitrary-dimensional image. """
    assert len(image_shape) == len(crop_shape)
    ndim = len(image_shape)
    valid_range = [image_shape[i] - crop_shape[i] for i in range(ndim)]
    for v in valid_range:
        assert v >= 0
    
    lower_indices = [torch.randint(n, (1,)).item() for n in valid_range]
    upper_indices = [lower_indices[i] + crop_shape[i] for i in range(ndim)]
    if return_array:
        return np.array(lower_indices), np.array(upper_indices)
    return lower_indices, upper_indices


def get_grid_locations_3d(image_size, patch_size, patch_overlap, sort=False):
    """ From torchio:
    https://github.com/fepegar/torchio/blob/master/torchio/data/sampler/grid.py
    
    Args:
        image_size: sequence or array of same dimension as image
        patch_size: patch lengths among all dims (same shape as image_size)
        patch_overlap: (same shape as image_size)
    """
    indices = []
    zipped = zip(image_size, patch_size, patch_overlap)
    for im_size_dim, patch_size_dim, patch_overlap_dim in zipped:
        end = im_size_dim + 1 - patch_size_dim
        step = patch_size_dim - patch_overlap_dim
        indices_dim = list(range(0, end, step))
        if indices_dim[-1] != im_size_dim - patch_size_dim:
            indices_dim.append(im_size_dim - patch_size_dim)
        indices.append(indices_dim)
    indices_ini = np.array(np.meshgrid(*indices)).reshape(3, -1).T
    indices_ini = np.unique(indices_ini, axis=0)
    indices_fin = indices_ini + np.array(patch_size)
    locations = np.hstack((indices_ini, indices_fin))
    
    if sort:
        return np.array(sorted(locations.tolist()))
    return locations


if __name__ == '__main__':
    
    get_grid_locations_3d((20, 20, 20), (5, 5, 5), (1, 1, 1))

