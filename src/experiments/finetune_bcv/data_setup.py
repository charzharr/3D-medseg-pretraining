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

import logging
import time
import pathlib
import collections
import multiprocessing
from collections import OrderedDict
import pandas as pd

import torch
import torchio
import dmt
from dmt.data import ScalarImage3D, ScalarMask3D, Sample, SampleSet
from dmt.utils.io.images3d import resample_sitk_isotropic
from dmt.data import OneToManyLoader

import data.transforms as myT
from data.bcv.dataset import get_df as get_bcv_df, CLASSES as bcv_classes
from data.utils import split

import SimpleITK as sitk


# Data Setup & Loading Constants
NUM_WORKERS = 0


# ------ ##   Main API from run_experiment()  ## ------ #


def get_data_components(cfg):
    start = time.time()
    logging.getLogger().setLevel(logging.CRITICAL)

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

    # 2. Create master list of samples (threading?) & SampleSet
    start = time.time()
    sample_args = []
    for i, S in df.iterrows():
        sample_args.append((i, S['id'], S['image'], S['mask'], S['imgsize'],
                            S['subset']))
    with multiprocessing.pool.ThreadPool(12) as pool:  # took 5 sec to load BCV
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
    
    train_set = BCVSampleSet(cfg, train_samples, is_train=True)
    val_set = BCVSampleSet(cfg, val_samples)
    test_set = BCVSampleSet(cfg, test_samples)
    
    print(f'[Took {time.time() - start:.2f} sec to load all data.]')
    
    debug = cfg.experiment.debug
    shuffle = False if debug.overfitbatch or debug.mode else True
    train_loader = torch.utils.data.DataLoader(
        train_set,
        shuffle=shuffle,
        batch_size=cfg.train.batch_size,
        collate_fn=collate,
        num_workers=NUM_WORKERS,
        prefetch_factor=2 if NUM_WORKERS > 0 else 2
    )
    print(f'Torch Dataloader initialized with {NUM_WORKERS} workers!')
    
    return {
        'df': df,
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
    
    mask = ScalarMask3D(mask, class_names=class_names, class_values=class_vals)
    
    sitk_image = sitk.ReadImage(image, sitk.sitkInt16)
    sitk_image = sitk.Clamp(sitk_image, sitk.sitkInt16, -1024, 325)
    sitk_image = sitk.NormalizeImageFilter().Execute(sitk_image)
    image = ScalarImage3D(sitk_image, container_type=sitk.sitkFloat32)
    
    sample_dict = {
        'index': index, 
        'id': id, 
        'image': image, 
        'mask': mask,
        'size': size, 
        'subset': subset,
    }
    sample = Sample(sample_dict)
    print(f'[Took {time.time() - start:.2f}s for sample creation.]')
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
        if self.is_train:
            self.crop = myT.ScaledUniformCropper3d((48, 128, 128), 
                                                   scale_range=(0.8, 1.2))
            self.transforms = [
                myT.Flip3d(p=0.5),
                myT.GaussianNoise(p=0.1, mean=0., var=(0, 0.1)),
                myT.GaussianBlur(p=0.2, spacing=1,   # ignore space for now
                                 sigma=(0.5, 1)),
                myT.ScaleIntensity(p=0.5, scale=(0.75, 1.25)),
                myT.Gamma(p=0.5, gamma=(0.7, 1.5))
            ]
        print(f'💠 BCVSampleSet created with {len(self.samples)} samples. \n'
              f'   Train={is_train}, Crops/Vol={self.crops_per_volume}, '
              f'Virtual-Size={len(self)}')
            
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
        assert 'float32' in tensor.dtype
        
        sitk_mask = sample.mask.sitk_image
        mask_tens = torch.tensor(sitk.GetArrayFromImage(sitk_mask))
        assert mask_tens.shape == tensor.shape
        
        record_dict = OrderedDict()
        
        if not self.is_train:
            return {'tensor': tensor, 'mask': mask_tens, 'record': record_dict}
        
        # 1. Crop
        image_tup, mask_tup = self.crop(tensor, mask=mask_tens)
        image_crop, image_record = image_tup
        mask_crop, image_record = mask_tup
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
        
        
        




