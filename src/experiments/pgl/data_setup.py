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









