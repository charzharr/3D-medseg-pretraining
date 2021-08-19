""" Module dataset.py for BCV'15 Dataset

Overview
 - Total 50 abdomen CTs (30 labeled train, 20 unlabeled test) from ongoing 
     colorectal cancer chemotherapy trial & retrospective ventral hernia study
 - Abdominal CT segmentation for 13 classes:
    (1) spleen
    (2) right kidney
    (3) left kidney
    (4) gallbladder
    (5) esophagus
    (6) liver
    (7) stomach
    (8) aorta
    (9) inferior vena cava
    (10) portal vein and splenic vein
    (11) pancreas
    (12) right adrenal gland
    (13) left adrenal gland

Resources
 - https://www.synapse.org/#!Synapse:syn3193805/wiki/217789
"""

import os
import logging
import time
import random
import math
import multiprocessing
import pandas as pd
from pathlib import Path
from collections import OrderedDict
import nibabel as nib

import dmt
from dmt.dmt.data.images import ScalarImage3D
from dmt.dmt.data.label_masks import ScalarMask3D
from dmt.dmt.data.samples.sample import Sample
from dmt.dmt.data.samples.sampleset import SampleSet


__all__ = ['get_df']


# Dataset Constants
DATASET_DIR = '/afs/crc.nd.edu/user/y/yzhang46/datasets/BCV-2015/'
assert os.path.isdir(DATASET_DIR)
CLASSES = ['spleen', 'right_kidney', 'left_kidney', 'gallbladder', 'esophagus',
           'liver', 'stomach', 'aorta', 'inferior_vena_cava', 
           'portal_vein_and_splenic_vein', 'pancreas', 'right_adrenal_gland',
           'left_adrenal_gland']


### ======================================================================== ###
### * ### * ### * ### *           Custom Loading         * ### * ### * ### * ###
### ======================================================================== ###


class SampleSet(dmt.dmt.data.samples.sampleset.SampleSet):
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = sample.image.array
        mask = sample.mask.array
        return image, mask, sample

    
def get_sample(args, load_to_ram=True):
    i, R = args
    classnames = ['background'] + CLASSES
    image = ScalarImage3D(R['image'], permanent_load=load_to_ram)
    mask = ScalarMask3D(R['mask'], classnames)
    new_sample = Sample({
        'image': image,
        'mask': mask,
        'id': i,
        'shape': R['imgsize']
    })
    return i, new_sample

def get_datasets(df, load_to_ram=True):
    start = time.time()
    classnames = ['background'] + CLASSES
    
    results = []
    with multiprocessing.Pool() as pool:
        args = []
        for i, R in df.iterrows():
            args.append((i, R))
        results = pool.map(get_sample, args)
    # import IPython; IPython.embed(); 
    # results = [r.get() for r in results]
    
    train_ind, val_ind, test_ind = split_indices(range(len(df)))
    train_samples, val_samples, test_samples = [], [], []
    for i, new_sample in results:
        if i in train_ind:
            train_samples.append(new_sample)
        elif i in val_ind:
            val_samples.append(new_sample)
        else:
            test_samples.append(new_sample)
    print(f'DONE! {time.time() - start:.2f} sec')
    return {
        'train': SampleSet(train_samples),
        'val': SampleSet(val_samples),
        'test': SampleSet(test_samples)
    }


### ======================================================================== ###
### * ### * ### * ### *         Dataset Collection       * ### * ### * ### * ###
### ======================================================================== ###


def get_df(df_file=None, dataset_path=DATASET_DIR):
    """ Collect df for BCV dataset (for now, only labeled training samples). """
    if df_file:
        return pd.read_csv(df_file)
    
    ds_path = Path(dataset_path)
    default_df = ds_path / 'default_df.csv'
    if default_df.exists():
        logging.info(f"Loading default BCV df ({default_df.absolute()}).")
        return pd.read_csv(str(default_df))
    
    logging.info(f"Collecting BCV df.")
    start_time = time.time()
    
    train_path = ds_path / 'train' / 'img_nii'
    train_images = natural_sort([str(f) for f in train_path.iterdir() if f.suffix == '.gz'])
    label_path = ds_path / 'train' / 'label_nii'
    label_images = natural_sort([str(f) for f in label_path.iterdir() if f.suffix == '.gz'])
    assert len(train_images) == len(label_images)
    
    # trains, vals, tests = split(range(len(train_iamges)))
    
    df_d = OrderedDict([ 
        ('id', []),
        ('image', []),
        ('mask', []),
        ('imgsize', []),
        ('subset', []),
    ])
    for i, img in enumerate(train_images):
        img_path = Path(img)
        mask_path = Path(label_images[i])
        assert mask_path.name[5:9] == img_path.name[3:7]  # check idx
        
        vol = nib.load(img)
#         subset = 'train' if i in trains else 'test'
#         if i in vals:
#             subset = 'val'
        
        df_d['id'].append(img_path.name[3:7])
        df_d['image'].append(img)
        df_d['mask'].append(str(mask_path))
        df_d['subset'].append('train')
        df_d['imgsize'].append(vol.shape)
    
    df = pd.DataFrame(df_d)
    
    elapsed_time = time.time() - start_time 
    logging.info(f"Done collecting BCV ({elapsed_time:.1f} sec).")
    return df



### ======================================================================== ###
### * ### * ### * ### *         Additional Helpers       * ### * ### * ### * ###
### ======================================================================== ###


def split_indices(indices, train=0.6, val=0.2, test=0.2):
    N_indices = len(indices)
    indices = set(indices)
    train = set(random.sample(indices, k=math.ceil(0.6 * N_indices)))
    indices = indices - train
    test = set(random.sample(indices, k=math.ceil(0.2 * N_indices)))
    val = indices - test
    return train, val, test

def natural_sort(l): 
    import re
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)
    

if __name__ == '__main__':
    # logging.basicConfig(level=logging.DEBUG)
    logging.info('Print everything!!!')
    df = get_df()
    datasets_d = get_datasets(df)
    import IPython; IPython.embed(); 
