""" Module dataset.py for Decathlon Dataset
Consists of 10 3D Semantic Segmentation Tasks:
3.  CT Liver          (131 Train, 70  Test, ~29GB)
6.  CT Lung           (63  Train, 32  Test, ~9GB)
7.  CT Pancreas       (281 Train, 139 Test, 12.3GB)
8.  CT Hepatic Vessel (303 Train, 140 Test, 12.3 GB)
9.  CT Spleen         (41  Train, 20  Test, 1.6 GB)
10. CT Colon          (126 Train, 64  Test, 6.2 GB)
  NOTE: Only training data have labels, so must split into train/val/test.

Resources
 - http://medicaldecathlon.com/
"""

import os, sys
import pathlib
import logging
import time
import pandas as pd
from collections import OrderedDict
import nibabel as nib

if __name__ == '__main__':  # add src to sys path & main namespace
    curr_path = pathlib.Path().absolute()
    sys.path.append(str(curr_path.parent.parent))
    from data.utils import natural_sort, correct_df_directories
else:
    from ..utils import natural_sort, correct_df_directories

__all__ = ['get_dfs', 'get_task_df', 'collect_task_df']


# Dataset Constants
if sys.platform == "darwin":
    DATASET_DIR = '/Users/charzhar/Desktop/_Datasets/MedicalSegmentationDecathlon/'
else:
    DATASET_DIR = '/afs/crc.nd.edu/user/y/yzhang46/datasets/MedicalSegmentationDecathlon/'
assert os.path.isdir(DATASET_DIR)

TASKS = [None, None, 'Liver', None, None, 
         'Lung', 'Pancreas', 'HepaticVessel', 'Spleen', 'Colon']
CLASSES_D = {
    'Liver': ['background', 'liver', 'cancer'],
    'Lung': ['background', 'cancer'],
    'Pancreas': ['background', 'pancreas', 'cancer'],
    'HepaticVessel': ['background', 'vessel', 'cancer'],
    'Spleen': ['background', 'spleen'],
    'Colon': ['background', 'cancer']
}


# ======================================================================== #
# * ### * ### * ### *         Dataset Collection       * ### * ### * ### * #
# ======================================================================== #


def get_dfs(dataset_path=DATASET_DIR):
    """Collect default dfs (train/test samples) from each Decathlon task."""
    ds_path = pathlib.Path(dataset_path)
    dataframes_d = {}
    for i, task in enumerate(TASKS):
        logging.info(f'Getting task {i+1}: {task}')
        if not task:
            continue
        
        task_path = ds_path / f'Task{i+1:02d}_{task}'
        msg = f"{task} path doesn't exist {task_path.absolute()}"
        assert task_path.exists(), msg
        
        # task_df = collect_task_df(task, task_path)
        task_df = get_task_df(task, task_path)
        dataframes_d[task] = task_df
    return dataframes_d


def get_task_df(task_name, task_path, df_file=None):
    """ Returns the dataframe describing the default dataset structure
    of a task from the Medical Segmentation Decathlon dataset.
    In order of precedence, the following is returned:
        1. if df_file is given, then that csv file is turned into a df
        2. if df_file is None, then we look for a default_df.csv file & load it
        3. if both of above comes up empty, then we call collect_task_df() to
            painstakingly collect all images & their info into a new df.
    """
    if df_file:
        df = pd.read_csv(df_file)
        df = correct_df_directories(df, task_path)
        return df

    assert task_name in TASKS, f"Invalid task {task_name}."
    task_path = pathlib.Path(task_path)
    default_df = task_path / 'default_df.csv'
    if default_df.exists():
        msg = f"Loading default MSD-{task_name} df ({default_df.absolute()})."
        logging.info(msg)
        df = pd.read_csv(str(default_df))
        df = correct_df_directories(df, task_path)
        if 'Unnamed: 0' in df:
            df = df.drop(labels='Unnamed: 0', axis=1)
        return df

    df = collect_task_df(task_name, task_path)
    if 'Unnamed: 0' in df:
        df = df.drop(labels='Unnamed: 0', axis=1)
    return df


def collect_task_df(task_name, task_path):
    """Collect default df (train/test samples) for specified task."""
    assert task_name in TASKS, f"Invalid task {task_name}."
    
    logging.info(f"Collecting df ({task_path}) for task {task_name}.")
    start_time = time.time()
    
    train_path = task_path / 'imagesTr'
    train_images = natural_sort([str(f) for f in train_path.iterdir() if f.suffix == '.gz'])
    label_path = task_path / 'labelsTr'
    label_images = natural_sort([str(f) for f in label_path.iterdir() if f.suffix == '.gz'])
    test_path = task_path / 'imagesTs'
    test_images = natural_sort([str(f) for f in test_path.iterdir() if f.suffix == '.gz'])
    assert len(train_images) == len(label_images), f"# labs & imgs don't match."
    
    df_d = OrderedDict([ 
        ('id', []),
        ('task', []),
        ('image', []),
        ('mask', []),
        ('imgsize', []),
        ('subset', []),
    ])
    for i, img in enumerate(train_images + test_images):
        img_path = pathlib.Path(img)
        mask = '' if i >= len(label_images) else label_images[i]
        if mask:
            assert img_path.name == pathlib.Path(mask).name
        vol = nib.load(img)
        
        df_d['id'].append(img_path.name.split('.')[0])
        df_d['task'].append(task_name)
        df_d['image'].append(img)
        df_d['mask'].append(mask)
        df_d['subset'].append('train' if mask else 'test')
        df_d['imgsize'].append(vol.shape)
    
    df = pd.DataFrame(df_d)
    
    elapsed_time = time.time() - start_time 
    logging.info(f"Done collecting Decathlon {task_name} ({elapsed_time:.1f} sec).")
    return df


if __name__ == '__main__':
    import time; _start = time.time()
    lformat = '%(levelname)s | %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=lformat)
    logging.info('Print everything!!!')
    dfs_d = get_dfs()
    print(f'[Took {time.time() - _start:.2f} secs.]')
    import IPython; IPython.embed();
