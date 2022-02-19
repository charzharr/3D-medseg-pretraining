
import time
import copy
import SimpleITK as sitk
import torch
import collections

from experiments.ftmmwhs import data_setup as mmwhs_setup
from experiments.ftbcv import data_setup as bcv_setup
from experiments.ftspleen import data_setup as spleen_setup

import data.transforms as myT

# Pretraining modules
from .cropper import SpatialPretrainCropper3d
from .vector import Vector3d
from .sampler import ValueSampler
from .reorient import flip3d, rotate3d



def get_pretrain_data_components(config):
    print('🖼️  Getting MMWHS, BCV, MSD Liver, MSD Lung samples.')
    from experiments.pretrain_data_setup import get_df_samples
    collect_d = get_df_samples(config)

    train_df = df = collect_d['df']
    samples = collect_d['samples']
    
    debug = config.experiment.debug
    shuffle = False if debug.overfitbatch or debug.mode else True
    num_train_workers = 0 if debug.mode or debug.overfitbatch else \
                        config.train.num_workers
    
    if config.tasks.name == 'mg':
        task_config = config.tasks[config.tasks.name]
        print(f'🖼️  Task: {config.tasks.name}, Config: {task_config}.')
        
        print(f'\nTrain Data Components:')
        from .data_mg import MGSampleSet
        train_set = MGSampleSet(config, samples)
        train_loader = torch.utils.data.DataLoader(
            train_set, 
            shuffle=shuffle,
            batch_size=config.train.batch_size,
            collate_fn=MGSampleSet._collate_mg,
            num_workers=num_train_workers
        )
    
    elif config.tasks.name == 'universal':
        task_config = config.tasks[config.tasks.name]
        print(f'🖼️  Task: {config.tasks.name}, Config: {task_config}.')
    
    elif config.task.name == 'rot':
        task_config = config.tasks[config.tasks.name]
        print(f'🖼️  Task: {config.tasks.name}, Config: {task_config}.')
        
    else:
        assert False, f'{config.tasks.name} is not supported for all pretrain.'
    
    return {
        'df': df,
        'samples': samples,
        'train_df': train_df,
        'train_set': train_set,
        'train_loader': train_loader,
        'test_df': None,
        'test_set': None,
        'val_df': None,
        'val_set': None,
    }
    

def get_mmwhs_data_components(config):
    
    # Get samples and create datasets / dataloaders
    samples_d = mmwhs_setup.get_data_components(config, get_samples_only=True)
    
    print(f'\nTrain Data Components:')
    train_samples = samples_d['train_samples']
    train_set = PretrainDataset(config, train_samples, is_train=True)
    debug = config.experiment.debug
    shuffle = False if debug.overfitbatch or debug.mode else True
    num_workers = 0 if debug.mode or debug.overfitbatch else config.train.num_workers
    train_loader = torch.utils.data.DataLoader(
        train_set,
        shuffle=shuffle,
        batch_size=config.train.batch_size,
        collate_fn=_collate,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else 2
    )
    print(f'💠 Torch Dataloader initialized with {num_workers} workers!\n'
          f'   Batch-size={config.train.batch_size}, '
            f'Patches-per-batch={config.train.batch_crops_per_volume}, '
            f'Shuffle={shuffle}. \n')
        
    print(f'\nTest Data Components:')
    test_samples = samples_d['test_samples']
    test_set = PretrainDataset(config, test_samples, is_train=False)
    
    return {
        'df': samples_d['df'],
        'samples': samples_d['samples'],
        'train_df': samples_d['train_df'],
        'train_set': train_set,
        'train_loader': train_loader,
        'test_df': samples_d['test_df'],
        'test_set': test_set
    }


def get_bcv_data(config):  # TODO when necessary.
    pass


# ============================================================================ #
# * ### * ### * ### *  New 2/18/22 Code: General Pretrain  * ### * ### * ### * #
# ============================================================================ #






# ============================================================================ #
# * ### * ### * ### *     Old Code: Only MMWHS Pretrain    * ### * ### * ### * #
# ============================================================================ #


class PretrainDataset(torch.utils.data.Dataset):
    """ Generalized dataset object (based on sample abstraction). 
    
    Notes
      - When is_train is off, returns the entire volumes and skips augmentation.
    """
    
    def __init__(self, config, samples, is_train=True):
        super().__init__()
        
        self.config = config 
        self.is_train = is_train
        self.samples = samples
                
        self.num_classes = config.data[config.data.name].num_classes
        
        self.crops_per_volume = config.train.examples_per_volume
        self.batch_crops_per_volume = config.train.batch_crops_per_volume
        if not is_train:
            self.crops_per_volume = 1
        
        ## Training Transforms
        if self.is_train:
            if self.config.experiment.debug.overfitbatch:
                self.config.train.scale_range = 1.
            
            # Transforms: Cropper
            cropscale_sampler = ValueSampler(False,  # continuous 
                                             self.config.train.scale_range)
            self.T_crop_scale = SpatialPretrainCropper3d(
                final_shape=self.config.train.patch_size,
                scale_sampler=cropscale_sampler,
                cubic_crop=self.config.train.cubic_crop
            )
            self.T_orient = None 
            
            self.T_augment = [
                myT.GaussianNoise(p=config.train.t_gn, mean=0., var=(0, 0.05)),
                myT.GaussianBlur(p=config.train.t_gb, spacing=1,
                                 sigma=(0.5, 1)),
                myT.ScaleIntensity(p=config.train.t_intensity_scale, 
                                   scale=(0.75, 1.25)),
                myT.Gamma(p=config.train.t_gamma, gamma=(0.7, 1.5))
            ]
            print(f'(PretrainDataset) Training Transforms: \n'
                  f'  Crop: {self.T_crop_scale}, \n   Aug: {self.T_augment}')
        print(f'💠 PretrainDataset created with {len(self.samples)} samples. \n'
              f'   Train={is_train}, Crops/Vol={self.crops_per_volume}, '
              f'Virtual-Size={len(self)}, Indices: {[s.index for s in self.samples]}.')

    def __len__(self):
        return len(self.samples) * self.crops_per_volume  # hackey
    
    def __getitem__(self, idx):
        """
        Returns:
            A dictionary containing the original sample object, a preprocessed
            full tensor volume. 
        """
        start = time.time()
        
        # Sample retrieval
        orig_idx = idx  
        idx = idx // self.crops_per_volume   # hackey way of inc epoch samples
        assert 0 <= idx < len(self), f'Index {idx} is out of bounds.'
        
        sample = self.samples[idx]
        
        sitk_image = sample.image.sitk_image  # loads path to sitk obj
        tensor = torch.tensor(sitk.GetArrayFromImage(sitk_image)).float()
        assert 'float32' in str(tensor.dtype)
        
        sitk_mask = sample.mask.sitk_image
        mask_tens = torch.from_numpy(sitk.GetArrayFromImage(sitk_mask))
        assert mask_tens.shape == tensor.shape
        
        record_dicts = [copy.deepcopy(sample.records)   # OrderedDict()
                      for _ in range(self.config.train.batch_crops_per_volume)]  
        
        # Test-case (return the entire clean image)
        if not self.is_train:
            mask_1h = sample.mask.get_one_hot(crop=mask_tens, 
                channel_first=True, tensor=True)    
            return {
                'sample': sample,
                'tensor': tensor, 
                'mask': mask_tens, 
                'mask_1h': mask_1h,
                'record': record_dicts
            }
         
        # Cropping
        image_tup, mask_tup = self.T_crop_scale(tensor, mask=mask_tens,
            n_times=self.batch_crops_per_volume)
        
        image_crops = [e[0] for e in image_tup]
        image_records = [e[1] for e in image_tup]
        mask_crops = [e[0] for e in mask_tup]
        mask_records = [e[1] for e in mask_tup]
        if not isinstance(image_records, collections.abc.Sequence):
            image_crops, image_records = [image_crops], [image_records]
            mask_crops, mask_records = [mask_crops], [mask_records]
        for i, record in enumerate(image_records):
            record_dicts[i][self.T_crop_scale.__class__.__name__] = record
        
        # Training Augmentations
        final_crops, final_masks, final_masks_oh, final_vecs = [], [], [], []
        for i, (ic, mc) in enumerate(zip(image_crops, mask_crops)):
            crop = {'image_crop': ic, 'mask_crop': mc}
            for transform in self.T_augment:
                name = transform.name
                crop, receipt = transform(crop)
                record_dicts[i][name] = receipt['image_crop'] if receipt else None
            
            # Flip
            out_crop, out_mask = crop['image_crop'], crop['mask_crop']
            crop_meta = record_dicts[i]['SpatialPretrainCropper3d']
            origin_coords = crop_meta['input_volume_center']
            flip_d = flip3d(out_crop, origin_coords, mask=out_mask, 
                            vectors=crop_meta['final_crop_vectors'],
                            p_fx=self.config.train.t_flip_x,
                            p_fy=self.config.train.t_flip_y,
                            p_fz=self.config.train.t_flip_z)
            
            out_crop, out_mask = flip_d['image'], flip_d['mask']
            out_vectors = flip_d['vectors']
            record_dicts[i]['flip3d'] = flip_d['meta']
            
            # Rotate
            x_deg_sampler = None
            if self.config.train.t_rot_x_vals:
                x_deg_sampler = ValueSampler(True, self.config.train.t_rot_x_vals)
            y_deg_sampler = None
            if self.config.train.t_rot_y_vals:
                y_deg_sampler = ValueSampler(True, self.config.train.t_rot_y_vals)
            z_deg_sampler = None
            if self.config.train.t_rot_z_vals:
                z_deg_sampler = ValueSampler(True, self.config.train.t_rot_z_vals)
            rot_d = rotate3d(out_crop, origin_coords, mask=out_mask, 
                             vectors=out_vectors,
                             x_deg_sampler=x_deg_sampler,
                             y_deg_sampler=y_deg_sampler,
                             z_deg_sampler=z_deg_sampler)
            out_crop, out_mask = rot_d['image'], rot_d['mask']
            out_vectors = rot_d['vectors']
            record_dicts[i]['rotate3d'] = rot_d['meta']
            
            final_crops.append(out_crop)
            final_masks.append(out_mask)
            final_masks_oh.append(sample.mask.get_one_hot(
                crop=out_mask, channel_first=True, tensor=True))
            final_vecs.append(out_vectors)
        
        return {
            'sample': sample,
            'tensor': final_crops,
            'mask': final_masks,
            'mask_one_hot': final_masks_oh,
            'record': record_dicts,
            
            # experiment-dependent
            'vectors': final_vecs
        }


def _collate(batch):
    """ Handles cases where each volume can output multiple crops. 
    Assumptions:
      - 'sample' is a list of samples
      - 'tensor', 'mask', 'mask_one_hot' are lists of lists of tensors
      - 'record' are list of lists of OrderedDicts
    """
    images, masks, masks_1h, samples, records = [], [], [], [], []
    vecs = []
    for example in batch:
        samples.append(example['sample'])
        images += example['tensor']
        masks += example['mask']
        masks_1h += example['mask_one_hot']
        records += example['record']
        
        vecs += example['vectors']
    
    return {
        'images': torch.stack(images, dim=0).unsqueeze(1),
        'masks': torch.stack(masks, dim=0).unsqueeze(1),
        'masks_1h': torch.stack(masks_1h, dim=0),
        'samples': samples,
        'records': records,
        'vectors': vecs
    }
    

