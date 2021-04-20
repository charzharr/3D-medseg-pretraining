""" setup.py (By: Charley Zhang)
Boilerplate for setting up common experiment components:
  - Gathers the necessary resources, modules, and utilities.
  - Configures all essential training components 
    (model architecture + params, criterion, optimizer, schedulers)
  - Initializes stat trackers and experiment trackers
"""

import sys, os
import torch
import random

import pathlib
from PIL import Image
import numpy as np
import albumentations as A
import torch, torchvision
import cv2

import lib
from lib.utils import schedulers, statistics
from lib.modules import nets


CURR_PATH = pathlib.Path(__file__).parent
SAMPLES_PER_CLASS = {
    'isic17': [374, 254, 1372],
    'isic18': [1113, 6705, 514, 327, 1099, 115, 142]
}
WEIGHTS_D = {
    'isic17': 666 / np.array(SAMPLES_PER_CLASS['isic17']),
    'isic18': 1430 / np.array(SAMPLES_PER_CLASS['isic18'])
#     'isic': 1430/np.array([1113, 6705, 514, 327, 1099, 115, 142])
}


### ======================================================================== ###
### * ### * ### * ### *       Training Components        * ### * ### * ### * ###
### ======================================================================== ###


def get_model(cfg):
    model_name = cfg['model']['name']
    num_classes = int(cfg['model']['num_classes'])
    if num_classes <= 0:
        classes = cfg['data'][cfg['data']['name']]['classes']
        num_classes = len(classes)

    def create_model(model_name, drop=True):
        
        if model_name == 'densenet':
            from lib.modules.nets import densenet
            if drop:
                fin_drop = cfg['model'][model_name]['final_drop_rate']
                lay_drop = cfg['model'][model_name]['layer_drop_rate']
            else:
                fin_drop = lay_drop = False
            
            model = densenet.get_model(
                cfg['model'][model_name]['name'], 
                num_classes=num_classes,
                pretrained=cfg['model'][model_name]['pretrained'], 
                only_encoder=cfg['model'][model_name]['only_encoder'], 
                layer_drop_rate=lay_drop,
                final_drop_rate=fin_drop
            )
        elif model_name == 'resnet':
            from lib.modules.nets import resnet
            model = resnet.get_model(
                cfg['model'][model_name]['name'], 
                num_classes=num_classes,
                pretrained=cfg['model'][model_name]['pretrained'], 
                final_drop_rate=cfg['model'][model_name]['final_drop_rate']
            )
        elif model_name == 'efficientnet':
            from lib.modules.nets import efficientnet
            model = efficientnet.get_model(
                cfg['model'][model_name]['name'], 
                num_classes=num_classes,
                pretrained=cfg['model'][model_name]['pretrained'], 
            )
        else:
            raise ValueError(f"Model name '{model_name}' is not supported.")
        return model
    
    model = create_model(model_name)

    ema_model = None
    if cfg['model']['ema']:
        from lib.modules.nets import ema
        ema_model = create_model(model_name, drop=cfg['model']['ema_drop'])
        for p in ema_model.parameters():
            p.requires_grad_(False)
            p.detach_()
    
    N_params = sum([p.numel() for p in model.parameters()])
    print(f"\t{model_name} (ema:{cfg['model']['ema']}) initialized "
          f"({N_params:,} params).")
    
    return {
        'model': model,
        'ema_model': ema_model
    }


def get_criterion(crit_cfg, dataset='isic'):
    crit_name = crit_cfg['name']

    from lib.modules.losses import classification as classif_losses
    if crit_name == 'hard_ce':
        weights = None
        if crit_cfg[crit_name]['weights']:
            taper = crit_cfg['hard_ce']['taper']
            weights = 1 + (WEIGHTS_D[dataset] - 1) / taper
        reduction = crit_cfg[crit_name]['reduction']
        criterion = classif_losses.HardCrossEntropy(weights=weights,
            reduction=reduction)
    elif crit_name == 'focal':
        criterion = classif_losses.FocalLoss(
            alpha=crit_cfg[crit_name]['alpha'],
            gamma=crit_cfg[crit_name]['gamma']
        )
    elif crit_name == 'class_balanced':
        samples_per_class = SAMPLES_PER_CLASS['isic']
        criterion = classif_losses.ClassBalancedLoss(
            samples_per_class,
            lossname=crit_cfg[crit_name]['lossname'],
            beta=crit_cfg[crit_name]['beta'],
            gamma=crit_cfg[crit_name]['gamma']
        )
    elif crit_name == 'soft_ce':
        weights = WEIGHTS_D[dataset] if crit_cfg[crit_name]['weights'] else None
        criterion = classif_losses.SoftCrossEntropy(weights=weights)
    elif crit_name == 'mse':
        criterion = classif_losses.MSE()
    elif crit_name == 'softmax_mse':
        weights = None
        if crit_cfg[crit_name]['weights']:
            taper = crit_cfg['taper']
            weights = 1 + (WEIGHTS_D[dataset] - 1) / taper
        criterion = classif_losses.SoftmaxMSE(weights=weights)
    elif crit_name == 'bce':
    	criterion = torch.nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Criterion {crit_name} is not supported.")

    return criterion


def get_scheduler(cfg, optimizer):
    sched = cfg['train']['scheduler']['name']
    t = cfg['train']['start_epoch']
    T = cfg['train']['epochs']
    rampup_rates = cfg['train']['scheduler']['rampup_rates']
    minlr = cfg['train']['scheduler']['min_lr']
    
    if sched == 'uniform':
        scheduler = schedulers.Uniform(
            optimizer,
            rampup_rates=rampup_rates
        )
    elif 'exponential' in sched:
        factor = cfg['train']['scheduler'][sched]['exp_factor']
        scheduler = schedulers.ExponentialDecay(
            optimizer,
            t=t,
            exp_factor=factor,
            minlr=minlr,
            rampup_rates=rampup_rates
        )
    elif 'linear' in sched:
        end_factor = cfg['train']['scheduler'][sched]['end_factor']
        scheduler = schedulers.LinearDecay(
            optimizer,
            T=T,
            end_factor=end_factor,
            minlr=minlr,
            rampup_rates=rampup_rates
        )
    elif 'consistencycosine' in sched:
        scheduler = schedulers.ConsistencyCosineDecay(
            optimizer,
            T, 
            t=t,
            minlr=minlr,
            rampup_rates=rampup_rates
        )
    elif 'plateau' in sched:
        factor = cfg['train']['scheduler'][sched]['factor']
        scheduler = schedulers.ReduceOnPlateau(
            optimizer,
            factor=factor,
            patience=cfg['train']['scheduler']['plateau']['patience'],
            lowerbetter=True,
            rampup_rates=rampup_rates
        )
    elif 'step' in sched:
        factor = cfg['train']['scheduler'][sched]['factor']
        scheduler = schedulers.StepDecay(
            optimizer,
            factor=factor,
            T=T,
            steps=cfg['train']['scheduler']['step']['steps'],
            rampup_rates=rampup_rates
        )
    
    return scheduler


def get_optimizer(cfg, params):
    opt = cfg['train']['optimizer']['name']
    lr = cfg['train']['optimizer']['lr']
    mom = cfg['train']['optimizer']['momentum']
    wdecay = cfg['train']['optimizer']['wt_decay']

    if 'adam' in opt:
        optimizer = torch.optim.Adam(
            params, 
            lr=lr, 
            weight_decay=wdecay,
            betas=cfg['train']['optimizer']['adam']['betas']
        )
    elif 'nesterov' in opt:
        optimizer = torch.optim.SGD(params, 
            lr=lr, 
            momentum=mom, 
            weight_decay=wdecay,
            nesterov=True
        )
    else:  # sgd
        optimizer = torch.optim.SGD(
            params, 
            lr=lr, 
            momentum=mom, 
            weight_decay=wdecay
        )

    return optimizer


