
import SimpleITK as sitk
import numpy as np
import torch

from data.transforms.z_normalize import ZNormalize



def save_image(data, name, sample, history=None, is_mask=False):
    if is_mask:
        if data.ndim == 4 and data.shape[0] == 1:
            data = data.squeeze(0)
        elif data.ndim == 4 and data.shape[0] > 1:
            data = data.argmax(0)
        assert data.ndim == 3
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        data = data.astype(np.uint8)
    else:
        assert data.ndim == 3
        if history:
            from data.transforms.z_normalize import ZNormalize
            data = ZNormalize().invert(data, history['ZNormalize'])
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        data = data.astype(np.int16)

    sitk_crop = sitk.GetImageFromArray(data)
    sitk_crop.SetOrigin(sample.image.origin)
    sitk_crop.SetDirection(sample.image.direction)
    sitk_crop.SetSpacing(sample.image.spacing)

    if len(name) <= 7 or name[-7:] != '.nii.gz':
        name = name + '.nii.gz'
    sitk.WriteImage(sitk_crop, name)


def synchronize():
    """
    Helper function to synchronize between multiple processes when
    using distributed training.
    """
    if not torch.distributed.is_initialized():
        return
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    if world_size == 1:
        return
    torch.distributed.barrier()