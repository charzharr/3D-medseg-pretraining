""" Module dataset.py for KiTS-2019 Dataset (CT Kidney Tumor Segmentation)

Overview
 - Semantic Segmentation w/3 classes: background, kidney, kidney tumor
 - 210 training/val, 90 test (~29 GB)
    axial, superior to inferior, .nii files, 1mm to 5mm slices

Resources
 - https://kits19.grand-challenge.org/
 - https://github.com/neheller/kits19
"""

import os
import csv
import pandas as pd
import numpy as np
import nibabel as nib

from PIL import Image
from collections import OrderedDict
import torch, torchvision


__all__ = ['get_df', 'collect_df']


CLASSES = ['background', 'kidney', 'tumor']   # values 0, 1, 2
DATASET_DIR = '/afs/crc.nd.edu/user/y/yzhang46/datasets/KiTS-2019/'
assert os.path.isdir(DATASET_DIR)


def get_df(df_file=None, dataset_path=DATASET_DIR):
    if df_file:
        return pd.read_csv(df_file)
    
    ds_path = Path(dataset_path)
    default_df = ds_path / 'default_df.csv'
    if default_df.exists():
        logging.info(f"Loading default KiTS df ({default_df.absolute()}).")
        return pd.read_csv(str(default_df))
    
    df = collect_df(dataset_path)
    return df


def collect_df(path, save=None):
    data_path = os.path.join(path, 'data')
    case_dirs = [d for d in os.listdir(data_path) if d[:4] == 'case' and \
                 os.path.isdir(os.path.join(data_path, d))]
    case_dirs = natural_sort(case_dirs)
    assert len(case_dirs) == 300, ''
    
    df_d = OrderedDict([ 
        ('id', []),
        ('image', []),
        ('mask', []),
        ('imgsize', []),
        ('subset', []),
    ])
    for i, case in enumerate(case_dirs):
        case_path = os.path.join(data_path, case)
        case_imf = os.path.join(case_path, 'imaging.nii.gz')
        assert os.path.isfile(case_imf)
        
        case_lab = os.path.join(case_path, 'segmentation.nii.gz')
        test = True if not os.path.isfile(case_lab) else False
        
        vol = nib.load(case_imf)
        
        df_d['id'].append(case)
        df_d['image'].append(case_imf)
        df_d['mask'].append('' if test else case_lab)
        df_d['subset'].append('test' if test else 'train')
        df_d['imgsize'].append(vol.shape)
    
    df = pd.DataFrame(df_d)
    assert len(df[df['subset'] == 'train']) == 210
    
    if save:
        df.to_csv(save)
    return df


### ======================================================================== ###
### * ### * ### * ### *         Additional Helpers       * ### * ### * ### * ###
### ======================================================================== ###


def natural_sort(l): 
    import re
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


if __name__ == '__main__':
    df = collect_df(DATASET_DIR)
    print(df.head(), '\n', df.tail())

