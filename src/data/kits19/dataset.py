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
import pathlib
import csv
import pandas as pd
import numpy as np
import nibabel as nib

from PIL import Image
from collections import OrderedDict
import torch, torchvision

if __name__ == '__main__':  # add src to sys path & main namespace
    import sys
    curr_path = pathlib.Path().absolute()
    sys.path.append(str(curr_path.parent.parent))
    from data.utils import natural_sort, correct_df_directories
else:
    from ..utils import natural_sort, correct_df_directories

__all__ = ['get_df', 'collect_df']


# Constants
if sys.platform == "darwin":
    DATASET_DIR = '/Users/charzhar/Desktop/_Datasets/KiTS-2019'
else:
    DATASET_DIR = '/afs/crc.nd.edu/user/y/yzhang46/datasets/KiTS-2019/'
assert os.path.isdir(DATASET_DIR)
CLASSES = ['background', 'kidney', 'tumor']   # values 0, 1, 2


# Fundamental Collection Functions

def get_df(df_file=None, dataset_path=DATASET_DIR):
    """ Returns the dataframe describing the default dataset structure.
    In order of precedence, the following is returned:
        1. if df_file is given & exists, then that csv file is turned into a df
        2. if df_file is None, then we look for a default_df.csv file in the
            main dataset directory & load it
        3. if both of above comes up empty, then collect_df() is called to
            painstakingly collect all images & their info into a new df.
    """
    if df_file:
        df = pd.read_csv(df_file)
        df = correct_df_directories(df, dataset_path)
        return df
    
    ds_path = pathlib.Path(dataset_path)
    default_df = ds_path / 'default_df.csv'
    if default_df.exists():
        print(f"Loading default KiTS df ({default_df.absolute()}).")
        df = pd.read_csv(str(default_df))
    else:
        df = collect_df(dataset_path)
    df = correct_df_directories(df, dataset_path)
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


if __name__ == '__main__':
    import time; _start = time.time()
    # df = get_df(DATASET_DIR)
    df = get_df()
    print(df.head(), '\n', df.tail())
    print(f'[Took {time.time() - _start:.2f} secs.]')
    import IPython; IPython.embed();

