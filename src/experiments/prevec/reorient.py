""" Module reorient.py
Flipping and rotating of 3D images along with their corresponding vectors and
masks. 

3D Rotate Code Resources:
 - https://github.com/HealthML/self-supervised-3d-tasks/blob/master/self_supervised_3d_tasks/preprocessing/preprocess_rotation.py
 - 
"""

import math
import copy
from collections import OrderedDict
import torch
from data.transforms.flip import Flip3d
from monai.transforms.spatial.array import Rotate


# def rotate_and_flip3d(volume_tensor, mask=None, vectors=[], 
#                       p_fx=0.5, p_fy=0.5, p_fz=0.5,
#                       p_rx=0.5, p_ry=0.5, p_rz=0.5, 
#                       rot_angles=(-10, 10)):
#     """ Called inside data loading infrastructure to jointly flip and rotate
#     3D crops.
#     Args:
#         volume_tensor (DxHxW): crop or image
#         mask (DxHxW): crop or whole mask
#     """
#     # Flip 
#     pass
    
#     # Rotate
    

def rotate3d(volume_tensor, pivot_coords, mask=None, vectors=[], 
             x_deg_sampler=None, y_deg_sampler=None, z_deg_sampler=None,
             interpolation='bilinear'):
    """ Rotates around plane cut through center of tensor. """
    meta = OrderedDict([('dim_rot_degrees', [0, 0, 0])])
    out_tens = volume_tensor
    out_mask = mask
    out_vecs = copy.deepcopy(vectors)
    
    # Sample rotation angles (wrt dimension 1)
    if x_deg_sampler is not None:
        rot_angle = x_deg_sampler.sample()
        meta['dim_rot_degrees'][0] = rot_angle
    if y_deg_sampler is not None:
        rot_angle = y_deg_sampler.sample()
        meta['dim_rot_degrees'][1] = rot_angle
    if z_deg_sampler is not None:
        rot_angle = z_deg_sampler.sample()
        meta['dim_rot_degrees'][2] = rot_angle
        
    rot_angles = meta['dim_rot_degrees']
    rot_radians = [a * math.pi / 180 for a in rot_angles]
    
    dim_2_axes = {
        0: (1, 2),
        1: (0, 2),
        2: (0, 1)
    }
    
    for dim, degree in enumerate(rot_angles):
        if degree == 0:
            continue
        axes = dim_2_axes[dim]
        out_tens = _rotate_image3d(out_tens, degree, axes=axes,
                                   interpolation='bilinear')
        if mask is not None:
            out_mask = _rotate_image3d(out_mask, degree, axes=axes,
                                       interpolation='nearest')
    if vectors:
        for i, vec in enumerate(out_vecs):
            for dim, ang in enumerate(rot_angles):
                if ang != 0:
                    if dim == 0:
                        vec = vec.rotate_dim1(ang, pivot_coords=pivot_coords)
                    elif dim == 1:
                        vec = vec.rotate_dim2(ang, pivot_coords=pivot_coords)
                    else:
                        vec = vec.rotate_dim3(ang, pivot_coords=pivot_coords)
            out_vecs[i] = vec
    return {
        'image': out_tens,
        'mask': out_mask,
        'vectors': out_vecs,
        'meta': meta
    }
    

def _rotate_image3d(volume_tensor, degree, axes=(0,1), 
                    interpolation='bilinear',
                    padding_mode='constant'):
    from scipy import ndimage
    
    rot_tens = ndimage.rotate(volume_tensor, degree, axes=axes,
                              order=0 if interpolation == 'nearest' else 3,
                              mode=padding_mode)
    return rot_tens


def flip3d(volume_tensor, origin_coords, 
           mask=None, vectors=[], p_fx=0.5, p_fy=0.5, p_fz=0.5):
    """ Flips (i.e. mirrors) a 3D image, mask, or vectors along 3 axes. 
    Args:
        volume_tensor: DxHxW image can be crop or whole volume
        origin_coords: (dim1, dim2, dim3) central coordinate from which 
            flip axes are located. 
    """
    meta = OrderedDict([('flipped_flags', [False, False, False])])
    out_tens = volume_tensor
    out_mask = mask
    out_vecs = copy.deepcopy(vectors)
    
    # Flip x
    if torch.rand((1,)).item() <= p_fx:
        out_tens = Flip3d.flip(out_tens, [True, False, False])
        meta['flipped_flags'][0] = True
        if mask is not None:
            out_mask = Flip3d.flip(out_mask, [True, False, False])
        if vectors:
            out_vecs = [vec.mirror_dim1(origin_coords[0]) for vec in out_vecs]
    
    # Flip y
    if torch.rand((1,)).item() <= p_fy:
        out_tens = Flip3d.flip(out_tens, [False, True, False])
        meta['flipped_flags'][1] = True 
        if mask is not None:
            out_mask = Flip3d.flip(out_mask, [False, True, False])
        if vectors:
            out_vecs = [vec.mirror_dim2(origin_coords[1]) for vec in out_vecs]
    
    # Flip z
    if torch.rand((1,)).item() <= p_fz:
        out_tens = Flip3d.flip(out_tens, [False, False, True])
        meta['flipped_flags'][2] = True 
        if mask is not None:
            out_mask = Flip3d.flip(out_mask, [False, False, True]) 
        if vectors:
            out_vecs = [vec.mirror_dim3(origin_coords[2]) for vec in out_vecs]
    return {
        'image': out_tens,
        'mask': out_mask,
        'vectors': out_vecs,
        'meta': meta
    }



    