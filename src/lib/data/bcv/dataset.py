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
import pandas as pd
from pathlib import Path
from collections import OrderedDict
import nibabel as nib


__all__ = ['get_df']


# Dataset Constants
DATASET_DIR = '/afs/crc.nd.edu/user/y/yzhang46/datasets/BCV-2015/'
assert os.path.isdir(DATASET_DIR)
CLASSES = ['spleen', 'right_kidney', 'left_kidney', 'gallbladder', 'esophagus',
           'liver', 'stomach', 'aorta', 'inferior_vena_cava', 
           'portal_vein_and_splenic_vein', 'pancreas', 'right_adrenal_gland',
           'left_adrenal_gland']


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


def split(indices, train=0.6, val=0.2, test=0.2):
    indices = set(indices)
    train = random.sample(indices, k=math.ceil(0.6 * len(indices)))
    indices = indices - train
    test = random.sample(indices, k=math.ceil(0.2 * len(indices)))
    val = indices = test
    return train, val, test

def natural_sort(l): 
    import re
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)
    

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logging.info('Print everything!!!')
    df = get_df()
    import IPython; IPython.embed(); 
