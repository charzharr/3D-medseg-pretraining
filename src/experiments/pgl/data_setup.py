""" Module pgl/data_setup.py

Main Jobs:
- Collect the default dfs from the datasets we plan on using.
- Curate the data samples from each data set into a pretraining set
    - Downstream fine-tuning and evaluation will be done via the finetune exp.
- Create the samples, datasets, and patch loaders for fast training.

Implementation Details:
- Original PGL used 1808 CT scans from 5 public datasets
    660 from RibFrac + 1148 from subsets of MSD Hepatic Vessel, Colon Tumor,
        Pancreas, & Lung Tumor
- Preprocessing:
    (1) [-1024, +325]HU clip
    (2) Subtract mean, divide by std.
    (3) Patch crops of size 16 × 96 × 96

Our Pretraining (~1400 tot volumes, 1029 training):
    MSD Hepatic  Vessel, Pancreas, Lung Tumor | Liver, Spleen
    KiTS Kidney
Our Fine-Tuning
    BCV (30 labeled volumes)
    MSD Colon Tumor (126 labeled volumes)
    MM-WHS Cardiac Segmentation (20 labeled)
"""

import logging
import time
import collections
from multiprocessing.pool import ThreadPool
from collections import OrderedDict
import pandas as pd
import SimpleITK as sitk

import torch
import torchio
import dmt
from dmt.data import ScalarImage3D, ScalarMask3D, Sample, SampleSet
from dmt.utils.io.images3d import resample_sitk_isotropic
from dmt.data import OneToManyLoader

import data.transforms as myT
from data.kits19.dataset import get_df as get_kits_df
from data.decathlon.dataset import get_dfs as get_msd_dfs


# Data Setup & Loading Constants
NUM_WORKERS = 4


# ------ ##   Main API from run_experiment()  ## ------ #

def get_data_components(cfg):
    import time; start = time.time();
    logging.getLogger().setLevel(logging.CRITICAL)

    # 1. Collect the dfs & process them into 1
    kits_df = get_kits_df()
    kits_df['dataset'] = 'kits'

    msd_include_tasks = ['HepaticVessel', 'Pancreas', 'Lung', 'Liver', 'Spleen']
    msd_dfs = get_msd_dfs()  # dict of dfs
    msd_df_list = [msd_dfs[k] for k in msd_dfs if k in msd_include_tasks]
    for df in msd_df_list:
        df['dataset'] = 'msd'

    pretrain_df = pd.concat([kits_df] + msd_df_list)
    pretrain_df = pretrain_df[pretrain_df['subset'] == 'train']
    if 'Unnamed: 0' in pretrain_df:
        pretrain_df = pretrain_df.drop(labels='Unnamed: 0', axis=1)

    # 2. Create master list of samples (threading?) & SampleSet
    sample_args = []
    for i, S in pretrain_df.iterrows():
        sample_args.append((i, S['id'], S['image'], S['mask'], S['imgsize'],
                            S['subset'], S['dataset'], S['task']))
    with ThreadPool() as pool:
        samples = pool.map(_get_sample, sample_args)
        
    train_set = PreprocessSampleSet(cfg, samples)
    print(f'[Took {time.time() - start:.2f} sec.]')
    
    #### ⭐ TEMPORARY ⭐  test transforms here
    
    
    
    # Instantiate Transforms
    overlap_cropper = myT.ScaledOverlapCropper3d(num_overlap_crops=3)
    pgl_transforms = [
        myT.Flip3d(p=0.5),
        myT.GaussianNoise(p=0.1, mean=0., var=(0, 0.1)),
        myT.GaussianBlur(p=0.2, kernel_size=(3, 10, 10), sigma=(0.5, 1)),
        myT.ScaleIntensity(p=0.5, scale=(0.75, 1.25)),
        myT.Gamma(p=0.5, gamma=(0.7, 1.5))
    ]
    pgl_transforms = [
        myT.Flip3d(p=0.5),
        myT.GaussianNoise(p=1, mean=0., var=(0, 0.1)),
        myT.GaussianBlur(p=1, kernel_size=(3, 10, 10), sigma=(0.5, 1)),
        myT.ScaleIntensity(p=1, scale=(0.75, 1.25)),
        myT.Gamma(p=1, gamma=(0.7, 1.5))
    ]
    
    start = time.time()
    volume_d = train_set[0]  # sample, tensor, record_dict
    volume = volume_d['tensor']
    print(f'[Took {time.time() - start:.2f}s to get train_set volume.]')
    
    start = time.time()
    crops = overlap_cropper(volume, (16, 96, 96), scale_range=(1.1, 1.4),
                            min_overlap=0.1, n_times=3)
    print(f'[Took {time.time() - start:.2f}s to get get crops.]')
    
    # Example run on a pair of crops
    crop1, crop2 = crops[0][0]['tensor'], crops[0][1]['tensor']
    
    
    crop = {'crop_image': crop1, 'other_shit': 1}
    history = collections.OrderedDict()
    history['ScaledOverlapCropper3d'] = {k: v for k, v in crop.items() 
                                      if k != 'tensor'}
    for transform in pgl_transforms:
        name = transform.name
        crop, receipt = transform(crop)
        history[name] = receipt
        print(name, receipt)
        
    # crop_obj = Crop(crop['crop_image'], sample, history, mask=crop['crop_mask'])
    
                    
    
    
    
    
    
    #### ⭐ TRANSFORM TEST END ⭐  
    
    
    import IPython; IPython.embed(); 

    # 4. Create example loader
    shuffle = False if cfg.debug.overfitbatch or cfg.debug.mode else True
    train_loader = OneToManyLoader(
        train_set, 
        sample_processing_fn=patch_creator,
        examples_per_sample=8,
        example_collate_fn=example_collate_fn, 
        batch_size=cfg.train.batch_size,
        shuffle_samples=shuffle,
        shuffle_patches=shuffle, 
        num_workers=NUM_WORKERS, 
        headstart=True, 
        drop_last=True
    )
    
    
    import IPython; IPython.embed(); 
    
    for i, batch in enumerate(train_loader):
        print(i, batch)
    
    
    import IPython; IPython.embed(); 
                                   

    return {
        'train_df': pretrain_df,
        'train_set': train_set,
        'train_loader': train_loader,
    }


def _get_sample(args):
    """ Called by get_data_components() to load samples in a parallel manner."""
    index, id, image, mask, size, subset, dataset, task = args
    class_names = _get_class_names(dataset, task)
    class_vals = list(range(len(class_names)))
    # mask = ScalarMask3D(mask, class_names=class_names, class_values=class_vals)
    image = ScalarImage3D(image)
    
    sample_dict = {
        'index': index, 
        'id': id, 
        'image': image, 
        'size': size, 
        'subset': subset,
        'dataset': dataset
    }
    if pd.notna(task):
        sample_dict['task'] = task
    sample = Sample(sample_dict)
    return sample


def _get_class_names(dataset, task=None):
    if dataset == 'kits':
        from data.kits19.dataset import CLASSES
        return CLASSES
    elif dataset == 'bcv':
        from data.bcv.dataset import CLASSES
        return CLASSES
    elif dataset == 'msd':
        from data.decathlon.dataset import TASKS, CLASSES_D
        assert task is not None and task in TASKS
        return CLASSES_D[task]
    
    raise ValueError(f'Given dataset "{dataset}" is invalid.')


# ------ ##   Data Structures for Storage and Loading  ## ------ #

def patch_creator(sample_volume_dict):
    """ 'sample_processing_fn' for the sample dataloader in OTM.
    Receives a dict of a single sample & its preprocessed entire volume. 
    Args:
        sample_volume_dict: keys include 'sample' & 'tensor'
    Returns:
        Each patch counts as an example.
    """
    patches_per_sample = 8
    patch_size = 64
    
    sample = sample_volume_dict[0]['sample']
    volume_tensor = sample_volume_dict[0]['tensor']
    
    examples = []
    
    
    
    return examples


def example_collate_fn(batch_examples):
    print(type(batch_examples))
    batch = {
        'X': [],
        'samples': []
    }
    batch = batch_examples
    return batch


class Crop:
    def __init__(self, tensor, sample, transform_history):
        """
        Args:
            tensor (torch.Tensor): tensor data (X) to be inputted to models
            sample (dmt.Sample): a reference to the original sample object
                from which the crop came.
            transform_history (OrderedDict): OrderedDict showing 
                transforms from preprocessing to cropping to other augs.
        """
        assert isinstance(transform_history, collections.OrderedDict)
        self.transform_history = transform_history
        
        self._sample = sample
        self._tensor = tensor
        
    @property
    def tensor(self):
        tensor = self._tensor
        if tensor.dtype != torch.float32:
            tensor = tensor.to(torch.float32)
        return tensor
    
    @property
    def flip_history(self):
        if 'Flip' not in self.transform_history:
            return {'flipped_flags': [False, False, False]}
        return self.transform_history['Flip']
    
    @property
    def norm_history(self):
        if 'ZNorm' not in self.transform_history:
            return {'mean': 0, 'std': 1}
        return self.transform_history['ZNorm']
    
    

class PreprocessSampleSet(SampleSet):
    """
    Timings
        - 600x512x512 took 4.8s load, 1s resample, 0.5s clamp, 4 sec znorm
            (all pytorch) 5s load, 1s resample, 0.75s clamp, 1 sec znorm
                ~3 sec sleepup, tio transforms are slow, avoid on lg volumes
    """
    
    def __init__(self, cfg, samples):
        self.cfg = cfg
        super().__init__(samples)
        
        # Preprocessing transforms
        self.znorm = myT.ZNormalize()
    
    def __getitem__(self, idx):
        """
        Returns:
            A dictionary containing the original sample object, a preprocessed
            full tensor volume. 
        """
        
        start = time.time()
        assert 0 <= idx < len(self), f'Index {idx} is out of bounds.'
        sample = self.samples[idx]
        sitk_image = sample.image.sitk_image  # loads path to sitk obj
        print(f'[Took {time.time() - start:.2f}s to get sitk.] Raw: {sample.image.shape}')
        
        record_dict = OrderedDict()
        
        # 1. resample
        start = time.time()
        sitk_image = resample_sitk_isotropic(sitk_image, interpolation='linear')
        print(f'[Took {time.time() - start:.2f}s to resample.]')
        
        # 2. clamp + normalize
        start = time.time()
        tensor = torch.tensor(sitk.GetArrayFromImage(sitk_image))
        tensor = tensor.clamp(-1024, 325).float()
        print('   Tensor type:', tensor.dtype)
        print(f'[Took {time.time() - start:.2f}s to get array from image.]')
        
        start = time.time()
        mean, std = tensor.mean(), tensor.std()
        tensor1 = (tensor - mean) / std
        print(f'[Took {time.time() - start:.2f} sec (x1).] Raw: {tensor.shape}')
        
        start = time.time()
        tensor, norm_op = self.znorm(tensor)
        record_dict[self.znorm.name] = norm_op
        print(f'[Took {time.time() - start:.2f} sec (x2).] Raw: {tensor.shape}')
        
        return {
            'sample': sample,
            'tensor': tensor,
            'record_dict': record_dict
        }
        
        
        




