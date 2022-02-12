""" Module dataset.py for MMWHS'17 Dataset

Overview
 - Total 20 labeled CTs
 - Abdominal CT segmentation for 8 classes:
    (0) background
    (1) left ventricle blood cavity (LV)
    (2) right ventricle blood cavity (RV)
    (3) left atrium blood cavity (LA)
    (4) right atrium blood cavity (RA)
    (5) myocardium of the left ventricle (LV-myo)
    (6) ascending aorta (AO)
    (7) pulmonary artery (PA)

"""

import os, sys
import pathlib
import logging
import time
import pandas as pd
from pathlib import Path
from collections import OrderedDict
import nibabel as nib

if __name__ == '__main__':  # add src to sys path & main namespace
    curr_path = pathlib.Path().absolute()
    sys.path.append(str(curr_path.parent.parent))
    from data.utils import natural_sort, correct_df_directories
else:
    from ..utils import natural_sort, correct_df_directories

__all__ = ['get_df', 'collect_df']


# Dataset Constants
if sys.platform == "darwin":
    DATASET_DIR = '/Users/charzhar/Desktop/_Datasets/MMWHS-2017'
else:
    DATASET_DIR = '/afs/crc.nd.edu/user/y/yzhang46/datasets/MMWHS-2017'
assert os.path.isdir(DATASET_DIR)

CLASSES = ['background', 'lv', 'rv', 'la', 'ra', 'lvm', 'ao', 'pa']


# ======================================================================== #
# * ### * ### * ### *         Dataset Collection       * ### * ### * ### * #
# ======================================================================== #

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
        print(f"Loading default MMWHS df ({default_df.absolute()}).")
        df = pd.read_csv(str(default_df))
    else:
        df = collect_df(dataset_path)
    df = correct_df_directories(df, dataset_path)
    
    if 'Unnamed: 0' in df:
        df = df.drop(labels='Unnamed: 0', axis=1)
    return df


def collect_df(dataset_path, save=None):
    """ Collect df for MMWHS dataset (for now, only labeled training samples). 
    Note:
        - Only returns 'train' subset. You have to manually change. 
    """
    ds_path = Path(dataset_path)

    logging.info(f"Collecting MMWHS df.")
    start_time = time.time()

    train_path = ds_path / 'ct_train_images'
    train_images = natural_sort([str(f) for f in train_path.iterdir()
                                 if f.suffix == '.gz'])
    label_path = ds_path / 'ct_train_labels'
    label_images = natural_sort([str(f) for f in label_path.iterdir()
                                 if f.suffix == '.gz'])
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
        assert img_path.name == mask_path.name

        vol = nib.load(img)

        df_d['id'].append(i + 1)
        df_d['image'].append(img)
        df_d['mask'].append(str(mask_path))
        df_d['subset'].append('train')
        df_d['imgsize'].append(vol.shape)

    df = pd.DataFrame(df_d)

    elapsed_time = time.time() - start_time
    logging.info(f"Done collecting MMWHS ({elapsed_time:.1f} sec).")
    if save:
        df.to_csv(save)
    return df


if __name__ == '__main__':
    import time; _start = time.time()
    logging.basicConfig(level=logging.DEBUG)
    logging.info('Print everything!!!')

    # df = collect_df(DATASET_DIR)
    df = get_df()
    print(df.head(), '\n', df.tail())
    print(f'[Took {time.time() - _start:.2f} secs.]')
    import IPython; IPython.embed();
