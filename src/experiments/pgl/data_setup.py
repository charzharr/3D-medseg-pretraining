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

Our Pretraining (~1400 volumes):
    MSD Hepatic  Vessel, Pancreas, Lung Tumor | Liver, Spleen
    KiTS Kidney
Our Fine-Tuning
    BCV (30 labeled volumes)
    MSD Colon Tumor (126 labeled volumes)
    MM-WHS Cardiac Segmentation (20 labeled)
"""

import dmt
from dmt.data import ScalarImage3D

from data.kits19.dataset import get_df as get_kits_df
from data.decathlon.dataset import get_dfs as get_msd_dfs



# ------ ##   Main API from run_experiment()  ## ------ #

def get_data_components(cfg):

    # 1. Collect the dfs & process them into 1
    kits_df = get_kits_df()

    msd_include_tasks = ['HepaticVessel', 'Pancreas', 'Lung', 'Liver', 'Spleen']
    msd_dfs = get_msd_dfs()  # dict of dfs
    msd_df_list = [msd_dfs[k] for k in msd_dfs if k in msd_include_tasks]

    pretrain_df = None

    # 2. Create master list of samples (threading?)

    # 3. Create sampleset, patch sampler

    # 4. Create example loader
    train_loader = None

    return {
        'train_dfs': dfs_d,
        'train_loader': train_loader,

    }


# ------ ##   Data Structures for Storage and Loading  ## ------ #







