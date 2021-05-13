""" Module utils/io/images3d.py 
Contains common utilities for 3D images.
"""

import sys, os
import SimpleITK as sitk
import numpy as np

from . import files as file_utils

### Constants ###
IMAGE_EXTS = ['.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff', '.gif',
              '.nii', '.nii.gz', '.dcm']


### ======================================================================== ###
### * ### * ### * ### *          API Definitions         * ### * ### * ### * ### 
### ======================================================================== ###

__all__ = ['read_sitk3d',
           'write_sitk_gray3d', 'write_np_gray3d',
           'to_np_channel_last', 'to_np_channel_first'
          ]

### ---- ### ---- \\    Image Read/Write     // ---- ### ---- ###

def read_sitk3d(im_path, pixtype=sitk.sitkInt16):
    image = sitk.ReadImage(im_path, pixtype)
    dims = image.GetNumberOfComponentsPerPixel()
    return image


def write_sitk_gray3d(sitk_im, path, compress=False):
    r""" Save 3D sitk grayscale image. 
    Args:
        sitk_im: SimpleITK.Image
        path: str or pathlib.Path
        compress: bool
            If save-type is .nii or .nii.gz, doesn't affect output.
    """
    channels = sitk_im.GetNumberOfComponentsPerPixel()
    assert channels == 1, f"Only 3D gray sitk images are supported."
    _write_image(sitk_im, path, compress=compress)


def write_np_gray3d(np_im, path, extra_channel=False, compress=False):
    r""" Save 3D grayscale image of shape 1xHxWxD, HxWxD, or HxWxDx1 """
    if extra_channel:
        np_im = np_im.squeeze(0).squeeze(-1)
    assert np_im.ndim == 3, f"Only 3D gray images are supported."
    img = sitk.GetImageFromArray(np_im, isVector=False, compress=False)
    _write_image(img, path, compress=compress)


def _write_image(img, path, compress=False):
    r""" Creates directory structure and writes using sitk's writer. """ 
    file_utils.create_dirs_from_file(path)
    try:
        writer = sitk.ImageFileWriter()
        writer.SetFileName(path)
        writer.SetUseCompression(compress)
        writer.Execute(img)
        return True
    except:
        print(f"Error writing image to {path}!")
        file_utils.delete_file(path)
        return False
        

### ---- ### ---- \\    Image Structural Transforms     // ---- ### ---- ###


def sitk_to_np(sitk_im):
    return sitk.GetArrayFromImage(sitk_im)


def np_to_sitk(np_im, has_channels=False):
    r"""
    Args:
        has_channels: bool
            If image is rgb, has_channels=True
    """
    return sitk.GetImageFromArray(np_im, isVector=has_channels)
    

def to_np_channel_last(np_im):
    if np_im.shape[0] not in range(1, 6):
        print(f"(to_np_channel_last) WARNING: Weird image {np_im.shape}")
    return np.moveaxis(np_im, 0, -1)


def to_np_channel_first(np_im):
    if np_im.shape[-1] not in range(1, 6):
        print(f"(to_np_channel_first) WARNING: Weird image {np_im.shape}")
    return np.moveaxis(np_im, -1, 0)
        