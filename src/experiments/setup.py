""" src/experiments/setup.py (By: Charley Zhang)
Boilerplate for setting up common experiment components:
  - Gathers the necessary resources, modules, and utilities.
  - Configures all essential training components 
    (model architecture + params, criterion, optimizer, schedulers)
  - Initializes stat trackers and experiment trackers
"""

import torch

from lib.utils.train import schedulers



# ------------------ ##  Training Components  ## ------------------ #


def get_criterion(cfg):
    crit_name = cfg.train.criterion.name
    if crit_name in cfg.train.criterion:
        crit_cfg = cfg.train.criterion[crit_name]

    # ------ #  3D Losses  # ------ #
    if crit_name == 'byol':
        from lib.assess.losses3d import BYOL3d
        criterion = BYOL3d()
    elif crit_name == 'dice-ce':
        from lib.assess.losses3d import DiceCrossEntropyLoss3d
        criterion = DiceCrossEntropyLoss3d(
            alpha=crit_cfg.alpha
        )
    
    # ------ #  2D Segmentation Losses  # ------ #
    
    # ------ #  2D Classification Losses  # ------ #
    else:
        raise ValueError(f"Criterion {crit_name} is not supported.")

    return criterion


def get_scheduler(cfg, optimizer):
    sched = cfg.train.scheduler.name
    t = cfg.train.start_epoch
    T = cfg.train.epochs
    rampup_rates = cfg.train.scheduler.rampup_rates
    min_lr = cfg.train.scheduler.min_lr
    
    if sched == 'uniform':
        scheduler = schedulers.Uniform(
            optimizer,
            rampup_rates=rampup_rates
        )
    elif 'exponential' in sched:
        sched_cfg = cfg.train.scheduler.exponential
        scheduler = schedulers.ExponentialDecay(
            optimizer,
            t=t,
            exp_factor=sched_cfg.exp_factor,
            minlr=min_lr,
            rampup_rates=rampup_rates
        )
    elif 'linear' in sched:
        sched_cfg = cfg.train.scheduler.linear
        scheduler = schedulers.LinearDecay(
            optimizer,
            T=T,
            end_factor=sched_cfg.end_factor,
            minlr=min_lr,
            rampup_rates=rampup_rates
        )
    elif 'consistencycosine' in sched:
        scheduler = schedulers.ConsistencyCosineDecay(
            optimizer,
            T, 
            t=t,
            minlr=min_lr,
            rampup_rates=rampup_rates
        )
    elif 'plateau' in sched:
        sched_cfg = cfg.train.scheduler.plateau
        scheduler = schedulers.ReduceOnPlateau(
            optimizer,
            factor=sched_cfg.factor,
            patience=sched_cfg.patience,
            lowerbetter=True,
            rampup_rates=rampup_rates
        )
    elif 'step' in sched:
        sched_cfg = cfg.train.scheduler.step
        scheduler = schedulers.StepDecay(
            optimizer,
            factor=sched_cfg.factor,
            T=T,
            steps=sched_cfg.steps,
            rampup_rates=rampup_rates
        )
    
    return scheduler


def get_optimizer(cfg, params):
    opt = cfg.train.optimizer.name
    lr = cfg.train.optimizer.lr
    wdecay = cfg.train.optimizer.wt_decay

    if 'adam' in opt:
        opt_cfg = cfg.train.optimizer.adam
        optimizer = torch.optim.Adam(
            params, 
            lr=lr, 
            weight_decay=wdecay,
            betas=opt_cfg.betas
        )
        print(f'💠 Adam optimizer initiated with lr={lr}, wd={wdecay}, \n'
              f'   betas={opt_cfg.betas}.')
    elif 'nesterov' in opt:
        opt_cfg = cfg.train.optimizer.nesterov
        optimizer = torch.optim.SGD(
            params, 
            lr=lr, 
            momentum=opt_cfg.momentum, 
            weight_decay=wdecay,
            nesterov=True
        )
        print(f'💠 Nesterov optimizer initiated with lr={lr}, wd={wdecay}, \n'
              f'   momentum={opt_cfg.momentum}.')
    elif 'sgd' in opt:  # sgd
        opt_cfg = cfg.train.optimizer.sgd
        optimizer = torch.optim.SGD(
            params, 
            lr=lr, 
            momentum=opt_cfg.momentum, 
            weight_decay=wdecay
        )
        print(f'💠 SGD optimizer initiated with lr={lr}, wd={wdecay}, \n'
              f'   momentum={opt_cfg.momentum}.')
    else:
        raise ValueError(f'Optimizer "{opt}" is not supported')

    return optimizer


