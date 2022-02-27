""" Module experiments/prevec/model_setup.py (By: Charley Zhang)

Top level model-grabbing API as well as task-specific module components
for spatial pretraining tasks and potentially others. 
"""

import os
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from lib.nets import init as init_net

from experiments.prevec import data_setup


def get_model_components(config, class_bal_bias=False):
    
    model = get_model(config)
    task_config = config.tasks[config.tasks.name]
    
    # Pretraining weight loading before pretraining
    cp_exp_name = config.experiment.checkpoint.file
    if cp_exp_name:
        print(f'⭐  Loading checkpoint! {cp_exp_name}')
        curr_path = pathlib.Path(__file__).parent
        pre_artifact_path = curr_path.parent / 'prevec' / 'artifacts'
        if not (pre_artifact_path / cp_exp_name).exists():
            pre_artifact_path = curr_path.parent / 'rubik' / 'artifacts'
        assert (pre_artifact_path / cp_exp_name).exists()

        checkpoint_path = None
        for f in [pre_artifact_path / cp_exp_name / f for f in 
                os.listdir(pre_artifact_path / cp_exp_name)]:
            if str(f)[-3:] == 'pth':
                checkpoint_path = str(pre_artifact_path / f)
                print(f'\t Loaded Checkpoint: {checkpoint_path}')
        assert checkpoint_path, f'No .pth files in {pre_artifact_path / cp_exp_name}'

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        cp_config = checkpoint['config']

        if 'name' not in cp_config.tasks:
            print('⭐  Loading state dict for an old ass task.')
            compat_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                if 'backbone'in k:
                    new_k = k.replace('backbone.','')
                    if new_k.split('.')[0] == 'final_conv':
                        continue
                    compat_state_dict[new_k] = v 

            print(model.load_state_dict(compat_state_dict, strict=False))
        elif cp_config.tasks.name in('mg'):
            print('⭐  Loading state dict for {cp_config.tasks.name}.')
            compat_state_dict = OrderedDict()  # remove final conv weights and biases
            for k, v in checkpoint['state_dict'].items():
                if 'final_conv' == k.split('.')[0]:
                    # print(k)
                    continue
                compat_state_dict[k] = v 
            print(model.load_state_dict(compat_state_dict, strict=False))
        elif cp_config.tasks.name in('sar'):
            print('⭐  Loading state dict for {cp_config.tasks.name}.')

            compat_state_dict = OrderedDict()  # remove final conv weights and biases
            for k, v in checkpoint['state_dict'].items():
                if 'backbone'in k:
                    k = k.replace('backbone.','')
                if 'final_conv' == k.split('.')[0] or 'scale_' in k:
                    # print(k)
                    continue
                compat_state_dict[k] = v 

            print(model.load_state_dict(compat_state_dict, strict=False))
        elif 'rubik' in cp_config.tasks.name:
            print('⭐  Loading state dict for {cp_config.tasks.name}.')

            compat_state_dict = OrderedDict()  # remove final conv weights and biases
            for k, v in checkpoint['state_dict'].items():
                if 'backbone'in k:
                    k = k.replace('backbone.','')
                if 'final_conv' == k.split('.')[0] or 'scale_' in k:
                    # print(k)
                    continue
                compat_state_dict[k] = v 

            print(model.load_state_dict(compat_state_dict, strict=False))
        elif cp_config.tasks.name in('vec'):
            print('⭐  Loading state dict for {cp_config.tasks.name}.')
            compat_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                if 'backbone'in k:
                    k = k.replace('backbone.','')
                if k.split('.')[0] in ('final_conv', 'prediction'):
                    # print(k)
                    continue
                compat_state_dict[k] = v 
            print(model.load_state_dict(compat_state_dict, strict=False))
        else:
            raise NotImplementedError()
    
    # Task-specific modifications
    if config.tasks.name == 'vec':
        print(f'🪧  Model prevec modifications, {task_config}')
        
        # MG backbone modifications
        num_classes = task_config.num_classes
        if num_classes > 0:
            assert config.model.name == 'res2unet3d'
            print(f'🪧  VecPred - Modifying backbone to have {num_classes} out-dims.')
            model.final_conv = nn.Conv3d(32, num_classes, 1, bias=True)
        
        # Vec modifications & wrapper
        if task_config.model_version == 'v1':
            print(f'🪧  VecPred - Using model V1')
            from experiments.prevec.model_vec import PreVecV1 as VecModel
        else:
            print(f'🪧  VecPred - Using model V2')
            from experiments.prevec.model_vec import PreVecV2 as VecModel
        n_classes = 3 * len(task_config.pred_indices) 
        if task_config.pred_mag:
            raise NotImplementedError()
            n_classes += len(config.tasks.vec.pred_indices)
        model = VecModel(config, model, n_classes)
                
        print(f'🪧  VecPred - n_cls={n_classes}')
        
    elif config.tasks.name == 'mg':
        print(f'🪧  Model MG modifications, {task_config}')
        
        num_classes = task_config.num_classes
        assert config.model.name == 'res2unet3d'
        if task_config.prediction_head:
            pred_dims = task_config.prediction_head_dims
            model.final_conv = nn.Sequential([
                nn.Conv3d(32, pred_dims, 1, bias=False),
                nn.BatchNorm3d(pred_dims),
                nn.ReLU(inplace=True),
                nn.Conv3d(pred_dims, num_classes, 1, bias=True)
            ])
        else:
            model.final_conv = nn.Conv3d(32, num_classes, 1, bias=True)
    
    elif config.tasks.name == 'sar':
        from experiments.prevec.model_sar import SARModel
        print(f'🪧  Model SAR modifications, {task_config}')
        
        num_recon_classes = task_config.num_recon_classes
        model.final_conv = nn.Conv3d(32, num_recon_classes, 1, bias=True)  # recon pred
        
        num_scale_classes = len(task_config.t_scales)
        model = SARModel(config, model, num_scale_classes)

    else:
        assert False, f'Task name {config.tasks.name} not supported.'
    
    # Initialize parameters
    # if config.experiment.checkpoint.file:  # Checkpoint handling
    #     filename = config.experiment.checkpoint.file
    #     curr_path = pathlib.Path(__file__).parent.absolute()
    #     if pathlib.Path(filename).exists():
    #         filepath = str(pathlib.Path(filename).absolute())
    #     elif (curr_path / filename).exists():
    #         filepath = str(curr_path / filename)
    #     elif (curr_path / 'artifacts' / filename).exists():
    #         filepath = str(curr_path / 'artifacts' / filename)
    #     else:
    #         filepath = None
    #         print(f'Give filename {filename} could not be found.')

    #     if filepath:
    #         checkpoint_d = torch.load(filepath, map_location='cpu')
    #         state_dict = checkpoint_d['state_dict']
    #         print(model.load_state_dict(state_dict))
    # else:  # Initialize weights
    #     init_type = config.model.init
    #     if init_type:
    #         init_net.init_weights(model, init_type=init_type)
    #     print(f' (Model) Successfully initialized weights via {init_type}.')

    param_counts = sum([p.numel() for p in model.parameters()])
    print(f"🪧  Model {config.model.name} loaded w/{param_counts:,} params.")
    
    return {
        'model': model
    }


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
    
    return model
    