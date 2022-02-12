# ------------------------------------------------------------------------------
# Modified by Charley Zhang Dec 2021 from:
# https://github.com/HRNet/HRNet-Semantic-Segmentation/blob/pytorch-v1.1/lib/models/seg_hrnet.py
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

"""
List of Modifications
<> Changed init_weights to ignore classification layers & input channel compat.
<> Changed output to no longer be 4x downsampled. 


Sample Code for loading:
------------------------
model_cfg_file = './lib/nets/planar/hrnet/seg_hrnetv2_w32.yaml'
with open(model_cfg_file, 'r') as f:
    model_cfg = yaml.safe_load(f)
pretrained = model_cfg['PRETRAINED'] if pretrained else False
model = get_model(model_cfg, in_planes, num_classes, pretrained=pretrained)

Models
------
Seg3d_HRNetv2_W16 (modified ~22.6M params)
Seg3d_HRNetv2_W24 (modified ~49.7M params)
Seg3d_HRNetv2_W32 (modified ~87.5M params)
Seg3d_HRNetv2_W48 (modified ~195.5M params)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import functools

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F

from lib.nets.basemodel import BaseModel

__all__ = ['get_model', 'HighResolutionNet']

BatchNorm3d = nn.BatchNorm3d
BN_MOMENTUM = 0.01
logger = logging.getLogger(__name__)

model_params_path = '/afs/crc.nd.edu/user/y/yzhang46/_DLResources/Models/'
model_params = {
    'hrnetv2_32': os.path.join(model_params_path, 'hrnetv2_w32.pth'),
    'hrnetv2_48': os.path.join(model_params_path, 'hrnetv2_w48.pth'),
}


def get_model(model_cfg, in_planes, num_classes, pretrained='', **kwargs):
    """
    Args:
        model_cfg (dict-like): only cfg.MODEL.EXTRA in original config file.
        in_planes (int): number of input channels.
        num_classes (int): number of class channels in final output.
        **kwargs (opt): additional arguments.
    """
    msg = f'model_cfg does not have "MODEL" or "MODEL.EXTRA" keys.'
    assert 'MODEL' in model_cfg and 'EXTRA' in model_cfg['MODEL'], msg

    model = HighResolutionNet(model_cfg, in_planes, num_classes, **kwargs)
    model.init_weights(pretrained)
    return model


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm3d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm3d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm3d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm3d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm3d(planes * self.expansion,
                               momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm3d(num_channels[branch_index] * block.expansion,
                            momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv3d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        BatchNorm3d(num_inchannels[i], momentum=BN_MOMENTUM)))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv3d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm3d(num_outchannels_conv3x3, 
                                            momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv3d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm3d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    depth_output = x[i].shape[-3]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[depth_output, height_output, width_output],
                        mode='trilinear',
                        align_corners=True)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionNet(BaseModel):

    def __init__(self, config, in_channels, num_classes, **kwargs):
        extra = config['MODEL']['EXTRA']
        self.in_channels = in_channels
        self.num_classes = num_classes
        super(HighResolutionNet, self).__init__()

        # stem net
        self.conv1 = nn.Conv3d(self.in_channels, 64, kernel_size=3, stride=2, 
                               padding=1, bias=False)
        self.bn1 = BatchNorm3d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv3d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = BatchNorm3d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        
        self.stage1_cfg = extra['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion*num_channels

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        self.W = num_channels[0]
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)
        
        last_inp_channels = np.int(np.sum(pre_stage_channels))
        
        # Old last layer code
        # self.last_layer = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=last_inp_channels,
        #         out_channels=last_inp_channels,
        #         kernel_size=1,
        #         stride=1,
        #         padding=0),
        #     BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(
        #         in_channels=last_inp_channels,
        #         out_channels=self.num_classes,
        #         kernel_size=extra['FINAL_CONV_KERNEL'],
        #         stride=1,
        #         padding=1 if extra['FINAL_CONV_KERNEL'] == 3 else 0)
        # )

        # My modified code
        self.last_layer = nn.Sequential(
            nn.Conv3d(
                in_channels=last_inp_channels,
                out_channels=self.W,
                kernel_size=3,
                padding=1,
                bias=False),
            BatchNorm3d(self.W, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(self.W, self.W, 4, stride=2, padding=1, 
                               bias=False),
            BatchNorm3d(self.W, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(self.W, self.W, 4, stride=2, padding=1, 
                               bias=False),
            BatchNorm3d(self.W, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                in_channels=self.W,
                out_channels=self.num_classes,
                kernel_size=extra['FINAL_CONV_KERNEL'],
                stride=1,
                padding=1 if extra['FINAL_CONV_KERNEL'] == 3 else 0)
        )

        tot_params, tot_tparams = self.param_counts
        print(f'💠 HRNetSeg_3d model initiated with n_classes={num_classes}, '
                f'in_chans={in_channels}, W={self.W}, \n'
              f'   params={tot_params:,}, trainable_params={tot_tparams:,}.')

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv3d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        BatchNorm3d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv3d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        BatchNorm3d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm3d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches,
                                      block,
                                      num_blocks,
                                      num_inchannels,
                                      num_channels,
                                      fuse_method,
                                      reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)

        # Upsampling
        x0_d, x0_h, x0_w = x[0].shape[-3], x[0].shape[-2], x[0].shape[-1]
        x1 = F.interpolate(x[1], size=(x0_d, x0_h, x0_w), mode='trilinear',
                        align_corners=True)
        x2 = F.interpolate(x[2], size=(x0_d, x0_h, x0_w), mode='trilinear',
                        align_corners=True)
        x3 = F.interpolate(x[3], size=(x0_d, x0_h, x0_w), mode='trilinear',
                        align_corners=True)

        agg = torch.cat([x[0], x1, x2, x3], 1)
        out = self.last_layer(agg)
        return {'out': out}

    def init_weights(self, pretrained=''):
        logger.info('=> init weghts from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            print(f'  [HRNet-Seg3d] Loading pretrained model {pretrained}..')
            pretrained_dict = torch.load(pretrained)
            model_dict = self.state_dict()
            final_pretrained_dict = {}
            for k, v in pretrained_dict.items():
                if k == 'conv1.weight' and v.shape != model_dict[k].shape:
                    print(f'  [!] conv1 pretrained weights mismatched {v.shape}'
                          f' vs orig {model_dict[k].shape}. Skipping layer.')
                    continue 
                elif 'incre_modules' in k or 'downsamp_modules' in k:
                    continue  # these modules only needed for cls_hrnet
                else:
                    final_pretrained_dict[k] = v
            model_dict.update(final_pretrained_dict)
            print(self.load_state_dict(model_dict, strict=False))
        elif pretrained:
            print(f'[HRNet-Seg3d] Given pretrain file "{pretrained}" is bad.')