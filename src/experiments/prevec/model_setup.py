""" Module experiments/prevec/model_setup.py (By: Charley Zhang)

Top level model-grabbing API as well as task-specific module components
for spatial pretraining tasks and potentially others. 
"""

import os
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.nets import init as init_net

from experiments.prevec import data_setup


def get_model(config, class_bal_bias=False):
    num_classes = config.data[config.data.name].num_classes
    in_channels, deep_sup = 1, config.train.deep_sup
    img_size = config.train.patch_size

    norm = config.model.norm
    act = config.model.act
    
    print(f'[NET] Model={config.model.name}, '
          f'Class-Balanced Biases={class_bal_bias}')
    
    if config.model.name == 'res2unet3d':
        from lib.nets.volumetric.res2unet3d import res2net50_v1b, res2net101_v1b
        if config.model.res2unet3d.layers == 50:
            model = res2net50_v1b(pretrained=False,  # no pretrained 3d net
                                  base_width=config.model.res2unet3d.base_width,
                                  act=act, norm=norm,
                                  in_channels=in_channels, 
                                  num_classes=1,
                                  deep_sup=False)
        else:
            model = res2net101_v1b(pretrained=False,  # no pretrained 3d net
                                   base_width=config.model.res2unet3d.base_width,
                                   act=act, norm=norm,
                                   in_channels=in_channels, 
                                   num_classes=1,
                                   deep_sup=False)
        
        from experiments.prevec.models.prevec_model import PreVecV1
        n_classes = 3 * len(config.tasks.prevec.pred_indices) 
        if config.tasks.prevec.pred_mag:
            n_classes += len(config.tasks.prevec.pred_indices)
        print(f' (get_model) n_cls={n_classes}, {config.tasks.prevec}')
        model = PreVecV1(config, model, n_classes)
    elif config.model.name == 'denseunet3d':
        from lib.nets.volumetric.denseunet3d import get_model as get_dunet
        model = get_dunet(config.model.denseunet3d.layers, 
                          num_classes=num_classes, 
                          deconv=True, 
                          deep_sup=deep_sup, 
                          norm=norm, act=act)
    elif config.model.name == 'hrnet3d':
        import yaml
        from lib.nets.volumetric.hrnet.seg_hrnet3d import get_model as get_hrnet
        hr_config_file = config.model.hrnet3d.config
        if not os.path.isfile(hr_config_file):
            src_path = pathlib.Path(__file__).parent.parent.parent
            hr_path = src_path / 'lib' / 'nets' / 'volumetric' / 'hrnet'
            hr_config_file = str(hr_path / hr_config_file)
        with open(hr_config_file, 'r') as f:
            hr_config = yaml.safe_load(f)
        model = get_hrnet(hr_config, in_channels, num_classes, pretrained=False)
    else:
        raise ValueError(f'Model {config.model.name} is not supported.')

    # Initialize parameters
    if config.experiment.checkpoint.file:  # Checkpoint handling
        filename = config.experiment.checkpoint.file
        curr_path = pathlib.Path(__file__).parent.absolute()
        if pathlib.Path(filename).exists():
            filepath = str(pathlib.Path(filename).absolute())
        elif (curr_path / filename).exists():
            filepath = str(curr_path / filename)
        elif (curr_path / 'artifacts' / filename).exists():
            filepath = str(curr_path / 'artifacts' / filename)
        else:
            filepath = None
            print(f'Give filename {filename} could not be found.')

        if filepath:
            checkpoint_d = torch.load(filepath, map_location='cpu')
            state_dict = checkpoint_d['state_dict']
            print(model.load_state_dict(state_dict))
    else:  # Initialize weights
        init_type = config.model.init
        if init_type:
            init_net.init_weights(model, init_type=init_type)
        print(f' (Model) Successfully initialized weights via {init_type}.')

    param_counts = sum([p.numel() for p in model.parameters()])
    print(f" (Model) {config.model.name} model "
          f" loaded w/{param_counts:,} params.")
    
    return {
        'model': model
    }