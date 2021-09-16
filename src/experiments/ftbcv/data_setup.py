""" Module bcv/data_setup.py

Main Jobs:
- Collect the default dfs from the datasets we plan on using.
- Curate the data samples from each data set into a pretraining set
    - Downstream fine-tuning and evaluation will be done via the finetune exp.
- Create the samples, datasets, and patch loaders for fast training.

Fine-Tuning on BCV (settings take from PGL):
    - Net: custom 58M param, Res U-Net like, decoder-light
    - 20% validation, terminate when overfitting.
    - Patch size 48x160x160
    - SGD 0.01 for 100 epochs, batch 8
"""

# import logging
import time
import pathlib
import collections
import multiprocessing
from collections import OrderedDict
import pandas as pd

import numpy as np
import torch
import torchio
import dmt
from dmt.data import ScalarImage3D, ScalarMask3D, Sample, SampleSet
from dmt.utils.io.images3d import resample_sitk_isotropic
from dmt.data import OneToManyLoader

import data.transforms as myT
from data.transforms.resize import resize_segmentation3d
from data.bcv.dataset import get_df as get_bcv_df, CLASSES as bcv_classes
from data.utils import split

import SimpleITK as sitk


# Data Setup & Loading Constants
weights_d = {
    'bcv': torch.tensor([0.09024509, 0.60796585, 0.69857068, 0.70220383, 
       1.30277806, 1.52977076, 0.30930432, 0.45375822, 0.83759115, 0.87116222,
       1.15747258, 0.83470601, 2.41945138, 2.18501984]),
    'bcv_cbrt': torch.tensor([ 1.01478046,  6.83640367,  7.85522932,  
       7.89608309, 14.64936999, 17.20183864,  3.47803937,  5.10238259,  
       9.41847506,  9.79597224, 13.01545106,  9.38603252, 27.20604501, 
       24.56992877])
}


# ------ ##   Main API from run_experiment()  ## ------ #


def get_data_components(cfg):
    start = time.time()

    # 1. Collect the dfs & process them into 1
    if not cfg.data.bcv.split:
        # Random 60-20-20 split
        df = get_bcv_df()
        train_i, val_i, test_i = split(range(len(df)))
        print(f'Split created with these train-val-test splits: '
              f'{len(train_i)}-{len(val_i)}-{len(test_i)}')
        df.loc[val_i, 'subset'] = 'val'
        df.loc[test_i, 'subset'] = 'test'
    else:
        split_path = pathlib.Path(cfg.data.bcv.split)
        if split_path.exists():
            df = pd.read_csv(str(split_path.absolute()))
        else:
            curr_path = pathlib.Path(__file__).parent
            split_dir = curr_path.parent.parent / 'data' / 'bcv' / 'splits'
            df_path = split_dir / split_path
            if not df_path.exists():
                raise RuntimeError(f'Given {str(split_path)} does not exist.')
            df = pd.read_csv(df_path)
    from data.bcv.dataset import DATASET_DIR as bcv_dataset_path
    from data.utils import correct_df_directories
    df = correct_df_directories(df, bcv_dataset_path)
    if 'Unnamed: 0' in df:
        df = df.drop(labels='Unnamed: 0', axis=1)
        
    train_df = df[df['subset'] == 'train']
    val_df = df[df['subset'] == 'val']
    test_df = df[df['subset'] == 'test']

    if cfg.experiment.distributed:
        rank = cfg.experiment.rank
        worldsize = cfg.experiment.worldsize

        def get_indices_slice(rank, N, worldsize):
            if rank == worldsize - 1:
                return slice(rank * (N // worldsize), None, 1)
            return slice(rank * (N // worldsize), (rank + 1) * (N // worldsize))

        train_df = train_df[get_indices_slice(rank, len(train_df), worldsize)]
        val_df = val_df[get_indices_slice(rank, len(val_df), worldsize)]
        test_df = test_df[get_indices_slice(rank, len(test_df), worldsize)]

    # 2. Create master list of samples (threading?) & SampleSet
    start = time.time()
    sample_args = []
    for i, S in pd.concat([train_df, val_df, test_df]).iterrows():
        sample_args.append((i, S['id'], S['image'], S['mask'], S['imgsize'],
                            S['subset']))
    print(f'Collecting data samples:')
    with multiprocessing.Pool() as pool:  # took 5 sec to load BCV
        samples = pool.map(_get_sample, sample_args)
    # samples = map(_get_sample, sample_args)
    print(f'[Took {time.time() - start:.2f}s to get samples!]')
    
    train_samples, val_samples, test_samples = [], [], []
    for sample in samples:
        # print(sample.image.spacing)
        if sample.subset == 'train':
            train_samples.append(sample)
        elif sample.subset == 'val':
            val_samples.append(sample)
        else:
            assert sample.subset == 'test'
            test_samples.append(sample)
    
    print(f'\nTrain Data Components:')
    train_set = BCVSampleSet(cfg, train_samples, is_train=True)
    debug = cfg.experiment.debug
    shuffle = False if debug.overfitbatch or debug.mode else True
    num_workers = 0 if debug.mode or debug.overfitbatch else cfg.train.num_workers
    train_loader = torch.utils.data.DataLoader(
        train_set,
        shuffle=shuffle,
        batch_size=cfg.train.batch_size,
        collate_fn=collate,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else 2
    )
    print(f'💠 Torch Dataloader initialized with {num_workers} workers!\n'
          f'   Batch-size={cfg.train.batch_size}, Shuffle={shuffle}. \n')

    print(f'\nValidation Data Components:')
    val_set = BCVSampleSet(cfg, val_samples)

    print(f'\nTest Data Components:')
    test_set = BCVSampleSet(cfg, test_samples)
    
    print(f'[Took {time.time() - start:.2f} sec to load all data.]')

    return {
        'df': df,
        'samples': samples,
        'train_df': train_df,
        'train_set': train_set,
        'train_loader': train_loader,
        'val_df': val_df,
        'val_set': val_set,
        'test_df': test_df,
        'test_set': test_set
    }


def _get_sample(args):
    """ Called by get_data_components() to load samples in a parallel manner."""
    start = time.time()
    index, id, image, mask, size, subset = args
    class_names = _get_class_names('bcv')
    class_vals = list(range(len(class_names)))
    
    sitk_image = sitk.ReadImage(image, sitk.sitkInt16)
    sitk_image = sitk.Clamp(sitk_image, sitk.sitkInt16, -958, 325)

    mean, std = 82.92, 136.97
    sitk_image = (sitk_image - mean) / std
    # sitk_image = sitk.NormalizeImageFilter().Execute(sitk_image)
    record = OrderedDict([('ZNormalize', {'mean': mean, 'std': std})])

    new_mask = mask = sitk.ReadImage(mask, sitk.sitkUInt8)

    # resample
    orig_spacing = np.array(sitk_image.GetSpacing())
    new_spacing = list(orig_spacing[:2]) + [3]
    if orig_spacing.tolist() == new_spacing:
        resample = sitk.ResampleImageFilter()
        resample.SetInterpolator = sitk.sitkNearestNeighbor
        resample.SetOutputDirection(sitk_image.GetDirection())
        resample.SetOutputOrigin(sitk_image.GetOrigin())
        resample.SetOutputSpacing(new_spacing)

        orig_size = np.array(sitk_image.GetSize())
        new_size = orig_size * (orig_spacing / new_spacing)
        new_size = np.ceil(new_size).astype(np.int32).tolist()
        resample.SetSize(new_size)
        sitk_image = resample.Execute(sitk_image)
        
        mask_arr = sitk.GetArrayFromImage(mask)
        mask_arr = resize_segmentation3d(mask_arr, new_size[::-1], 
                                         class_ids=list(range(1, 14)))
        new_mask = sitk.GetImageFromArray(mask_arr)
        new_mask.SetOrigin(mask.GetOrigin())
        new_mask.SetDirection(mask.GetDirection())
        new_mask.SetSpacing(new_spacing)

        # if index > 26:
        #     sitk.WriteImage(sitk_image, 'aniso_orig_image.nii.gz')
        #     sitk.WriteImage(mask, 'aniso_orig_mask.nii.gz')
        #     sitk.WriteImage(new_sitk_image, 'aniso_resamp_image.nii.gz')
        #     sitk.WriteImage(new_mask, 'aniso_resamp_mask.nii.gz')

    

    # print(f'Setting spacing from {orig_spacing} to {new_spacing} \n'
    #       f'  Hypo new size from {orig_size} to {new_size} \n'
    #       f'  Size from {sitk_image.GetSize()} to {sitk.GetArrayFromImage(sitk_image).shape} \n'
    #       f'  New Mask: {mask.GetSize()}')

    image = ScalarImage3D(sitk_image, container_type=sitk.sitkFloat32)
    mask = ScalarMask3D(new_mask, class_names=class_names, 
                        class_values=class_vals)
    
    sample_dict = {
        'index': index, 
        'id': id, 
        'image': image, 
        'mask': mask,
        'size': size, 
        'subset': subset,
        'records': record
    }
    sample = Sample(sample_dict)
    print(f'    Took {time.time() - start:.2f}s for sample creation.')
    return sample


def _get_class_names(dataset, task=None):
    if dataset == 'kits':
        from data.kits19.dataset import CLASSES
        return CLASSES
    elif dataset == 'bcv':
        from data.bcv.dataset import CLASSES
        return ['background'] + CLASSES
    elif dataset == 'msd':
        from data.decathlon.dataset import TASKS, CLASSES_D
        assert task is not None and task in TASKS
        return CLASSES_D[task]
    
    raise ValueError(f'Given dataset "{dataset}" is invalid.')


# ------ ##   Data Structures for Storage and Loading  ## ------ #


def collate(batch):
    images, masks, masks_1h, samples, records = [], [], [], [], []
    for example in batch:
        images.append(example['tensor'])
        masks.append(example['mask'])
        masks_1h.append(example['mask_one_hot'])
        samples.append(example['sample'])
        records.append(example['record'])
        
        if len(images) >= 2 and example['tensor'].shape != images[-2].shape:
            import IPython; IPython.embed(); 
    
    return {
        'images': torch.stack(images, dim=0).unsqueeze(1),
        'masks': torch.stack(masks, dim=0).unsqueeze(1),
        'masks_1h': torch.stack(masks_1h, dim=0),
        'samples': samples,
        'records': records
    }


class BCVSampleSet(SampleSet):
    """
    Timings
        - 600x512x512 took 4.8s load, 1s resample, 0.5s clamp, 4 sec znorm
            (all pytorch) 5s load, 1s resample, 0.75s clamp, 1 sec znorm
                ~3 sec sleepup, tio transforms are slow, avoid on lg volumes
    """
    
    def __init__(self, cfg, samples, is_train=False):
        self.cfg = cfg
        self.crops_per_volume = cfg.train.examples_per_volume if is_train else 1
        self.is_train = is_train

        self.class_names = ['background'] + bcv_classes
        self.num_classes = len(self.class_names)

        super().__init__(samples)
        
        # Preprocessing transforms
        self.transforms = []
        if self.is_train:
            # self.crop = myT.ScaledUniformCropper3d(cfg.train.patch_size, 
            #                                        scale_range=(0.8, 1.2))
            if cfg.experiment.debug.overfitbatch:
                cfg.train.scale_range = 1.
            self.crop = myT.ScaledForegroundCropper3d(
                cfg.train.patch_size,
                scale_range=cfg.train.scale_range, 
                foreground_p=cfg.train.fg_bias)
            if not cfg.experiment.debug.overfitbatch:
                self.transforms = [
                    myT.Flip3d(p=0.5),
                    myT.GaussianNoise(p=0.1, mean=0., var=(0, 0.1)),
                    myT.GaussianBlur(p=0.2, spacing=1,   # ignore space for now
                                     sigma=(0.5, 1)),
                    myT.ScaleIntensity(p=0.25, scale=(0.75, 1.25)),
                    myT.Gamma(p=0.25, gamma=(0.7, 1.5))
                ]

        print(f'💠 BCVSampleSet created with {len(self.samples)} samples. \n'
              f'   Train={is_train}, Crops/Vol={self.crops_per_volume}, '
              f'Virtual-Size={len(self)}, #Transforms={len(self.transforms)}.\n'
              f'   Indices: {[s.index for s in self.samples]}')
            
    def __len__(self):
        return len(self.samples) * self.crops_per_volume  # hackey
    
    def __getitem__(self, idx):
        """
        Returns:
            A dictionary containing the original sample object, a preprocessed
            full tensor volume. 
        """
        orig_idx = idx  
        idx = idx // self.crops_per_volume   # hackey way of multiple crops
        
        start = time.time()
        assert 0 <= idx < len(self), f'Index {idx} is out of bounds.'
        sample = self.samples[idx]
        
        sitk_image = sample.image.sitk_image  # loads path to sitk obj
        tensor = torch.tensor(sitk.GetArrayFromImage(sitk_image)).float()
        assert 'float32' in str(tensor.dtype)
        
        sitk_mask = sample.mask.sitk_image
        mask_tens = torch.from_numpy(sitk.GetArrayFromImage(sitk_mask))
        assert mask_tens.shape == tensor.shape
        assert mask_tens.shape[1:] == (512, 512)
        
        record_dict = sample.records  # OrderedDict()
        
        if not self.is_train:
            mask_1h = sample.mask.get_one_hot(crop=mask_tens, 
                channel_first=True, tensor=True)    
            return {
                'sample': sample,
                'tensor': tensor, 
                'mask': mask_tens, 
                'mask_1h': mask_1h,
                'record': record_dict
            }
        
        # 1. Crop
        image_tup, mask_tup = self.crop(tensor, mask_tens)
        image_crop, image_record = image_tup
        mask_crop, mask_record = mask_tup
        record_dict[self.crop.__class__.__name__] = image_record
        
        # 2. Apply transforms to crop
        crop = {'image_crop': image_crop, 'mask_crop': mask_crop}
        for transform in self.transforms:
            name = transform.name
            crop, receipt = transform(crop)
            # if torch.isnan(crop['image_crop']).any():
            #     print(name, receipt)
            record_dict[name] = receipt['image_crop'] if receipt else None
        
        crop_1h = sample.mask.get_one_hot(crop=crop['mask_crop'], 
                                          channel_first=True, tensor=True)    
        # print(f'Transforms took {time.time() - start:.2f} sec.')
            
        return {
            'sample': sample,
            'tensor': crop['image_crop'],
            'mask': crop['mask_crop'],
            'mask_one_hot': crop_1h,
            'record': record_dict
        }
        
        
        




