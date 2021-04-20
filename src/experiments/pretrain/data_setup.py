

import pandas as pd
from lib.data.transform import TransformTwice
from lib.data.randaugment import RandAugmentMC, RandAugmentBM
from torchvision import transforms as transforms


def get_data_d(cfg):
    data_d = {}
    if cfg['data']['name'] == 'isic17':
        data_d = get_isic17(cfg)
    elif cfg['data']['name'] == 'isic18':
        data_d = get_isic18(cfg)
    else:
        raise NotImplementedError()
    
    return data_d


def get_isic17(cfg):
    from lib.data.skin import isic17 as dataset
    
    transforms_cfg = cfg['data']['transforms']
    normalize = transforms.Normalize(dataset.NORM_MEANS, dataset.NORM_STDS)
    size = transforms_cfg['size']
    # RA_transforms = RandAugmentMC(2, 2)
    if cfg['data']['transforms']['RA']['use']:
        RAcfg = transforms_cfg['RA']
        train_transforms = transforms.Compose([
            transforms.Resize((size, size)),
            # transforms.RandomHorizontalFlip(),
            RandAugmentBM(RAcfg['RA_n'], RAcfg['RA_m']),
            transforms.ToTensor(),
            normalize
        ])
    else:
        print(f'\t (emain-transforms) Using stdaug.')
        tcfg = transforms_cfg['default']
        train_transforms = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomAffine(
                degrees=tcfg['degree'], 
                translate=(tcfg['translate_x'], tcfg['translate_y'])),
            transforms.RandomHorizontalFlip(p=tcfg['hflip']),
            transforms.RandomVerticalFlip(p=tcfg['vflip']),
            transforms.ColorJitter(
                brightness=tcfg['brightness'], 
                contrast=tcfg['contrast'], 
                hue=tcfg['hue']),
            transforms.ToTensor(),
            normalize,
        ])
    test_transforms = transforms.Compose([  # validation & test
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        normalize,
    ])
    
    if cfg.data.isic17.split.use:
        split_cfg = cfg.data.isic17.split
        df = pd.read_csv(split_cfg.saved_split)
        if not split_cfg.unlabeled:
            train_df = df[(df['subset'] == 'train') & (df['unlabeled'] == False)]
            train_df = train_df.reset_index(drop=True)
        else:
            train_df = df[df['subset'] == 'train'].reset_index(drop=True)
            train_transforms = TransformTwice(train_transforms)
    else:
        df = dataset.get_df()
        train_df = df[df['subset'] == 'train'].reset_index(drop=True)
    val_df = df[df['subset'] == 'validation'].reset_index(drop=True)
    test_df = df[df['subset'] == 'test'].reset_index(drop=True)

    train_dataset = dataset.ISIC17_Dataset(train_df, transforms=train_transforms)
    val_dataset = dataset.ISIC17_Dataset(val_df, 
                               transforms=test_transforms)
    test_dataset = dataset.ISIC17_Dataset(test_df, 
                               transforms=test_transforms)
    
    return {
        'df': df,
        'dataset_train': train_dataset,
        'dataset_val': val_dataset,
        'dataset_test': test_dataset,
        'classes': dataset.CLASSES,
    }
    
    
def get_isic18(cfg):
    from lib.data.skin import dataset
    isic_cfg = cfg['data']['isic']
    transforms_cfg = cfg['data']['transforms']

    normalize = transforms.Normalize(dataset.NORM_MEANS, dataset.NORM_STDS)
    size = transforms_cfg['size']
    # RA_transforms = RandAugmentMC(2, 2)
    if cfg['data']['transforms']['RA']['use']:
        RAcfg = transforms_cfg['RA']
        train_transforms = transforms.Compose([
            transforms.Resize((size, size)),
            # transforms.RandomHorizontalFlip(),
            RandAugmentBM(RAcfg['RA_n'], RAcfg['RA_m']),
            transforms.ToTensor(),
            normalize
        ])
    else:
        print(f'\t (emain-transforms) Using stdaug.')
        tcfg = transforms_cfg['default']
        train_transforms = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomAffine(
                degrees=tcfg['degree'], 
                translate=(tcfg['translate_x'], tcfg['translate_y'])),
            transforms.RandomHorizontalFlip(p=tcfg['hflip']),
            transforms.RandomVerticalFlip(p=tcfg['vflip']),
            transforms.ColorJitter(
                brightness=tcfg['brightness'], 
                contrast=tcfg['contrast'], 
                hue=tcfg['hue']),
            transforms.ToTensor(),
            normalize,
        ])
    test_transforms = transforms.Compose([  # validation & test
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        normalize,
    ])

    split_df = pd.read_csv(isic_cfg['saved_split'])
    train_df = split_df[split_df['subset'] == 'train']
    if not isic_cfg['unlabeled']['use_ham10k']:
        train_df = train_df[train_df['unlabeled'] == False]  # only lab
    train_df = train_df.reset_index(drop=True)

    train_dataset = dataset.ISIC(train_df, transforms=train_transforms)
    val_dataset = dataset.ISIC(split_df[split_df['subset'] == 'validation'], 
                               transforms=test_transforms)
    test_dataset = dataset.ISIC(split_df[split_df['subset'] == 'test'], 
                               transforms=test_transforms)
    
    return {
        'df': split_df,
        'dataset_train': train_dataset,
        'dataset_val': val_dataset,
        'dataset_test': test_dataset,
        'classes': dataset.CLASSES
    }
    
