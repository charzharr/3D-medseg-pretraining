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
import numpy as np

import torch
import torchio
import dmt
from dmt.data import ScalarImage3D, ScalarMask3D, Sample, SampleSet
from dmt.utils.io.images3d import resample_sitk_isotropic
from dmt.data import OneToManyLoader

import data.transforms as myT
from data.kits19.dataset import get_df as get_kits_df
from data.decathlon.dataset import get_dfs as get_msd_dfs
from data.transforms.crops.scaled_overlap_crop import ScaledOverlapCropper3d
from data.transforms.crops.scaled_uniform_crop import ScaledUniformCropper3d


dataset_normalize = {
    'kits': {'spacing': 0.78, 'clip': [-79, 304], 'norm': [100.93, 76.9]},
    'msd': {
        'Liver': {'spacing': 1, 'clip': [-17, 201], 
                  'norm': [99.4, 39.36]},
        'Lung': {'spacing': 1.24, 'clip': [-1024, 325], 
                 'norm': [-158.58, 324.7]},
        'Pancreas': {'spacing': 2.5, 'clip': [-96, 215], 
                     'norm': [77.99, 75.4]},
        'HepaticVessel': {'spacing': 1.5, 'clip': [-3, 243], 
                     'norm': [104.37, 52.62]},
        'Spleen': {'spacing': 1.6, 'clip': [-41, 176], 
                     'norm': [99.29, 39.47]},
        'Colon': {'spacing': 3, 'clip': [-30, 165.82], 
                     'norm': [62.18, 32.65]},
    }
}



# ------ ##   Main API from run_experiment()  ## ------ #

def get_data_components(cfg):
    import time; start = time.time();

    # 1. Collect the dfs & process them into 1
    kits_df = get_kits_df()
    kits_df['dataset'] = 'kits'

    msd_include_tasks = ['HepaticVessel', 'Pancreas', 'Lung', 'Liver']
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
    
    # 4. Create example loader
    debug = cfg.experiment.debug
    shuffle = False if debug.overfitbatch or debug.mode else True
    train_loader = OneToManyLoader(
        train_set, 
        sample_processing_fn=patch_creator,
        examples_per_sample=cfg.train.examples_per_volume,
        example_collate_fn=example_collate_fn, 
        batch_size=cfg.train.batch_size,
        shuffle_samples=shuffle,
        shuffle_patches=shuffle, 
        num_workers=cfg.train.num_workers, 
        headstart=True, 
        drop_last=True
    )
    print(f'💠 OTMLoader created w/ {len(train_set)} train samples. \n'
              f'   BatchSize={cfg.train.batch_size}, '
              f'CropsPerVolume={cfg.train.examples_per_volume}, '
              f'ShuffleSamples/Patches={shuffle}, '
              f'NumWorkers={cfg.train.num_workers}.')
    
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


# ------ ##   Data Loading Components   ## ------ #

def patch_creator(sampset_return_dict):
    """ 'sample_processing_fn' for the sample dataloader in OTM.
    Receives a dict of a single sample & its preprocessed entire volume. 
    Args:
        sampset_return_dict: keys include 'sample' & 'tensor'
    Returns:
        List of examples. An example = a patch pair & their info.
    """
    sampset_return_dict = sampset_return_dict[0]
    sample = sampset_return_dict['sample']
    volume_tensor = sampset_return_dict['tensor']
    transform_history = sampset_return_dict['record_dict']
    
    cfg = sampset_return_dict['cfg']
    examples_per_sample = cfg.train.examples_per_volume
    patch_size = cfg.train.patch_size
    if cfg.train.train_byol:
        cropper = ScaledUniformCropper3d(patch_size, scale_range=(1.1, 1.4))
    else:
        cropper = ScaledOverlapCropper3d()
    
    transforms = [
        myT.Flip3d(p=0.5),
        myT.GaussianNoise(p=0.1, mean=0., var=(0, 0.1)),
        myT.GaussianBlur(p=0.2, spacing=1,   # ignore space for now
                            sigma=(0.5, 1)),
        myT.ScaleIntensity(p=0.5, scale=(0.75, 1.25)),
        myT.Gamma(p=0.5, gamma=(0.7, 1.5))
    ]
    
    if cfg.train.train_byol:
        crop_pairs = cropper(volume_tensor, n_times=examples_per_sample)
    else:
        crop_pairs = cropper(volume_tensor, patch_size, min_overlap=0.2,
                             scale_range=cfg.train.scale_range,
                             n_times=examples_per_sample)
    
    examples = []
    for i, crop_pair in enumerate(crop_pairs):
        if cfg.train.train_byol:
            crop1_t, crop2_t = crop_pair[0], crop_pair[0].clone()
            
            hist1 = collections.OrderedDict(transform_history)
            hist1['ScaledUniformCrop3d'] = dict(crop_pair[1])
            hist2 = collections.OrderedDict(transform_history)
            hist2['ScaledUniformCrop3d'] = dict(crop_pair[1])
        else:
            crop1_d, crop2_d = crop_pair
            crop1_t, crop2_t = crop1_d['final_tensor'], crop2_d['final_tensor']
            
            hist1 = collections.OrderedDict(transform_history)
            hist1['ScaledOverlapCrop3d'] = {k: v for k, v in crop1_d.items() 
                                            if 'tensor' not in k}
            hist2 = collections.OrderedDict(transform_history)
            hist2['ScaledOverlapCrop3d'] = {k: v for k, v in crop2_d.items() 
                                            if 'tensor' not in k}
        
        for transform in transforms:
            name = transform.name
            crop1_t, receipt = transform(crop1_t)
            hist1[name] = receipt if receipt else None
        for transform in transforms:
            name = transform.name
            crop2_t, receipt = transform(crop2_t)
            hist2[name] = receipt if receipt else None
        
        Crop1 = Crop(crop1_t, sample, hist1)
        Crop2 = Crop(crop2_t, sample, hist2)
        examples.append((Crop1, Crop2))
    
    return examples


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
        self.sample = sample
        self.tensor = tensor.float()
    
    @property
    def crop_history(self):
        return self.transform_history['ScaledOverlapCrop3d']
    
    @property
    def flip_history(self):
        if 'Flip3d' not in self.transform_history:
            return {'flipped_flags': [False, False, False]}
        return self.transform_history['Flip3d']
    
    @property
    def norm_history(self):
        if 'ZNormalize' not in self.transform_history:
            return {'mean': self.tensor.mean(), 'std': self.tensor.std()}
        return self.transform_history['ZNormalize']


def example_collate_fn(batch_examples):
    tensors1, tensors2 = [], []
    samples, hists, crop_objs = [], [], []
    for example in batch_examples:
        tensors1.append(example[0].tensor)
        tensors2.append(example[1].tensor)
        samples.append(example[0].sample)
        hists.append([example[0].transform_history, 
                      example[1].transform_history])
        crop_objs.append(example)
    
    return {
        'X': torch.cat([torch.stack(tensors1, dim=0), 
                        torch.stack(tensors2, dim=0)], dim=0).unsqueeze(1),
        'samples': samples,
        'crops': crop_objs,
        'records': hists
    }
    

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
        print(f'💠 PreprocessSampleSet created w/ {len(self.samples)} samples.')
    
    def __getitem__(self, idx):
        """
        Returns:
            A dictionary containing the original sample object, a preprocessed
            full tensor volume. 
        """
        start = time.time()
        assert 0 <= idx < len(self), f'Index {idx} is out of bounds.'
        
        # Get Data
        sample = self.samples[idx]
        dataset = sample.dataset
        sitk_image = sample.image.sitk_image  # loads path to sitk obj
        record_dict = OrderedDict()
                
        # Dataset-specific resample & normalize
        start = time.time()
        if dataset == 'msd':
            sub_task = sample.task
            norm_d = dataset_normalize['msd'][sub_task]
            ispac = norm_d['spacing']
            clip_min, clip_max = norm_d['clip']
            norm_sub, norm_div = norm_d['norm']
        else:
            assert dataset == 'kits'
            norm_d = dataset_normalize['kits']
            ispac = norm_d['spacing']
            clip_min, clip_max = norm_d['clip']
            norm_sub, norm_div = norm_d['norm']
        
        sitk_image = sitk.Clamp(sitk_image, sitk.sitkInt16, clip_min, clip_max)
        record_dict['Clamp'] = {'min': clip_min, 'max': clip_max}
        
        sitk_image = (sitk_image - norm_sub) / norm_div
        record_dict['ZNormalize'] = {'mean': norm_sub, 'std': norm_div}
        
        if dataset == 'kits':  # transpose axis
            transpose = sitk.PermuteAxesImageFilter()
            transpose.SetOrder([2, 1, 0])
            print('kits old', sitk_image.GetSpacing(), sitk_image.GetSize())
            sitk_image = transpose.Execute(sitk_image)
            print('kits new', sitk_image.GetSpacing(), sitk_image.GetSize())
        
        orig_spacing = np.array(sitk_image.GetSpacing())
        print(dataset, orig_spacing, ispac)
        if orig_spacing[-1] != ispac:
            resample = sitk.ResampleImageFilter()
            resample.SetInterpolator = sitk.sitkNearestNeighbor
            resample.SetOutputDirection(sitk_image.GetDirection())
            resample.SetOutputOrigin(sitk_image.GetOrigin())

            new_spacing = list(orig_spacing[:2]) + [ispac]
            resample.SetOutputSpacing(new_spacing)

            orig_size = np.array(sitk_image.GetSize())
            new_size = orig_size * (orig_spacing / new_spacing)
            new_size = np.ceil(new_size).astype(np.int32).tolist()
            resample.SetSize(new_size)
            sitk_image = resample.Execute(sitk_image)
            record_dict['Resample'] = {'interpolation': 'nearest',
                'old_size': orig_size, 'new_size': new_size,
                'old_spacing': orig_spacing, 'new_spacing': new_spacing}
        else:
            record_dict['Resample'] = None
        print(f'[Took {time.time() - start:.2f}s to norm + resample.]')
        
        # Get Tensor
        tensor = torch.from_numpy(sitk.GetArrayFromImage(sitk_image))
        record_dict['ReadImage'] = {'shape': tensor.shape}
        assert tensor.shape[1:] == (512, 512), f'{tensor.shape}{sample.dataset}'
        
        return {
            'cfg': self.cfg,
            'sample': sample,
            'tensor': tensor,
            'record_dict': record_dict
        }
        
        
        




