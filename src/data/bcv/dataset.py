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
import pathlib
import logging
import time
import pandas as pd
from pathlib import Path
from collections import OrderedDict
import nibabel as nib

if __name__ == '__main__':  # add src to sys path & main namespace
    import sys
    curr_path = pathlib.Path().absolute()
    sys.path.append(str(curr_path.parent.parent))
    from data.utils import natural_sort, correct_df_directories
else:
    from ..utils import natural_sort, correct_df_directories

__all__ = ['get_df', 'collect_df']


# Dataset Constants
if sys.platform == "darwin":
    DATASET_DIR = '/Users/charzhar/Desktop/_Datasets/BCV-2015'
else:
    DATASET_DIR = '/afs/crc.nd.edu/user/y/yzhang46/datasets/BCV-2015/'
assert os.path.isdir(DATASET_DIR)

CLASSES = ['spleen', 'right_kidney', 'left_kidney', 'gallbladder', 'esophagus',
           'liver', 'stomach', 'aorta', 'inferior_vena_cava',
           'portal_vein_and_splenic_vein', 'pancreas', 'right_adrenal_gland',
           'left_adrenal_gland']


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
        print(f"Loading default BCV df ({default_df.absolute()}).")
        df = pd.read_csv(str(default_df))
    else:
        df = collect_df(dataset_path)
    df = correct_df_directories(df, dataset_path)
    return df


def collect_df(dataset_path, save=None):
    """ Collect df for BCV dataset (for now, only labeled training samples). """
    ds_path = Path(dataset_path)
    default_df = ds_path / 'default_df.csv'
    if default_df.exists():
        logging.info(f"Loading default BCV df ({default_df.absolute()}).")
        return pd.read_csv(str(default_df))

    logging.info(f"Collecting BCV df.")
    start_time = time.time()

    train_path = ds_path / 'train' / 'img_nii'
    train_images = natural_sort([str(f) for f in train_path.iterdir()
                                 if f.suffix == '.gz'])
    label_path = ds_path / 'train' / 'label_nii'
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
