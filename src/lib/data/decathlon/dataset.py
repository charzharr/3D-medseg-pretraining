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

import os
import logging
import time
import pandas as pd
from pathlib import Path
from collections import OrderedDict
import nibabel as nib


__all__ = ['get_dfs', 'collect_task_df']


# Dataset Constants
TASKS = [None, None, 'Liver', None, None, 
         'Lung', 'Pancreas', 'HepaticVessel', 'Spleen', 'Colon']
DATASET_DIR = '/afs/crc.nd.edu/user/y/yzhang46/datasets/MedicalSegmentationDecathlon/'
assert os.path.isdir(DATASET_DIR)
CLASSES_D = {
    'Liver': ['background', 'liver', 'cancer'],
    'Lung': ['background', 'cancer'],
    'Pancreas': ['background', 'pancreas', 'cancer'],
    'HepaticVessel': ['background', 'vessel', 'cancer'],
    'Spleen': ['background', 'spleen'],
    'Colon': ['background', 'cancer']
}




### ======================================================================== ###
### * ### * ### * ### *         Dataset Collection       * ### * ### * ### * ###
### ======================================================================== ###


def get_dfs(dataset_path=DATASET_DIR):
    """Collect default dfs (train/test samples) from each Decathlon task."""
    ds_path = Path(dataset_path)
    dataframes_d = {}
    for i, task in enumerate(TASKS):
        logging.info(f'Getting task {i+1}: {task}')
        if not task:
            dataframes_d[task] = None
            continue
        
        task_path = ds_path / f'Task{i+1:02d}_{task}'
        assert task_path.exists(), f"{task} path doesn't exist {task_path.absolute()}"
        
        task_df = collect_task_df(task, task_path)
        dataframes_d[task] = task_df
    return dataframes_d
            

def collect_task_df(task_name, task_path):
    """Collect default df (train/test samples) for specified task."""
    assert task_name in TASKS, f"Invalid task {task_name}."
    
    dfp = task_path / f"default_df.csv"
    if dfp.exists():
        logging.info(f"Loading default df ({dfp.absolute()}) for task {task_name}.")
        return pd.read_csv(dfp.absolute())
    
    logging.info(f"Collecting df ({dfp.absolute()}) for task {task_name}.")
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
        img_path = Path(img)
        mask = '' if i >= len(label_images) else label_images[i]
        if mask:
            assert img_path.name == Path(mask).name
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



### ======================================================================== ###
### * ### * ### * ### *         Additional Helpers       * ### * ### * ### * ###
### ======================================================================== ###


def natural_sort(l): 
    import re
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)
    

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logging.info('Print everything!!!')
    dfs_d = get_dfs()
