""" denseunet3d.py (modified version of 2D DenseUNet by: Charley Zhang)
Modified for readability + added modular functionality for densenet
backbones other than 161.

Original DenseUNet 161 implementation:
https://github.com/xmengli999/TCSM/blob/master/models/network.py

DenseUNet-121: ~M params
DenseUNet-169: ~M params
DenseUNet-201: 59.5M params
DenseUNet-161: ~M params
"""

from torch import nn
import torch.nn.functional as F
import torch

from . import densenet3d as densenet
from ..basemodel import BaseModel


densenet_outdims = {
    # [7x7+MP, denseblock1, denseblock2, denseblock3, denseblock4 output dims]
    'densenet121': [64, 256, 512, 1024, 1024],
    'densenet169': [64, 256, 512, 1280, 1664],
    'densenet201': [64, 256, 512, 1792, 1920],
    'densenet161': [96, 384, 768, 2112, 2208]
}


class UpBlock(nn.Module):
    def __init__(self, side_in_dim, bot_in_dim, out_dim, deconv=False):
        super(UpBlock, self).__init__()
        
        self.dim_mismatched = side_in_dim != bot_in_dim
        if self.dim_mismatched:
            self.side_conv = nn.Conv3d(side_in_dim, bot_in_dim, 1, padding=0)
            nn.init.xavier_normal_(self.side_conv.weight)
        
        self.aggregated_conv = nn.Conv3d(bot_in_dim, out_dim, 3, padding=1)
        nn.init.xavier_normal_(self.aggregated_conv.weight)

        self.bn = nn.BatchNorm3d(out_dim)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

        self.use_deconv = deconv
        if self.use_deconv:
            self.deconv = nn.ConvTranspose3d(bot_in_dim, bot_in_dim, 3, 
                stride=2, padding=1, output_padding=1)
            nn.init.xavier_normal_(self.deconv.weight)
        
    def forward(self, side_in, up_in):
        if self.use_deconv:
            up_in = self.deconv(up_in)
        else:
            up_in = F.interpolate(up_in, scale_factor=2, mode='trilinear', 
                align_corners=True)
        
        if self.dim_mismatched:
            side_in = self.side_conv(side_in)
        
        agg = torch.add(up_in, side_in)   # no cat like U-Net
        out = F.relu(self.bn(self.aggregated_conv(agg)))

        return out


class ForwardHook():
    features = None
    
    def __init__(self, module): 
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output): 
        self.features = output

    def remove(self): 
        self.hook.remove()


class DenseUNet(BaseModel):

    def __init__(self, name='densenet161', out_channels=1, deconv=False):
        super(DenseUNet, self).__init__()
        
        self.features, self.encoder_hooks, ldims = self._setup_encoder(name)
        
        self.up1 = UpBlock(ldims[3], ldims[4], ldims[2], deconv=deconv)
        self.up2 = UpBlock(ldims[2], ldims[2], ldims[1], deconv=deconv)  
        self.up3 = UpBlock(ldims[1], ldims[1], ldims[0], deconv=deconv)
        self.up4 = UpBlock(ldims[0], ldims[0], ldims[0], deconv=deconv)

        self.conv1 = nn.Conv3d(ldims[0], 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.conv2 = nn.Conv3d(64, out_channels, kernel_size=1, padding=0)
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)

        self.use_deconv = deconv
        if self.use_deconv:
            self.final_deconv = nn.ConvTranspose2d(ldims[0], ldims[0], 3, 
                stride=2, padding=1, output_padding=1)
            nn.init.xavier_normal_(self.final_deconv.weight)

        print(f"💠 DenseUNet3d-{name} initialized.")

    def forward(self, x, dropout=False):
        f = F.relu(self.features(x))  # 32x downsampled -> ReLU (is BN'd)
        x = self.up1(self.encoder_hooks[3].features, f)  # cat block3 out -> 16x down
        x = self.up2(self.encoder_hooks[2].features, x)  # cat block2 out -> 8x down
        x = self.up3(self.encoder_hooks[1].features, x)  # cat block1 out -> 4x down
        x = self.up4(self.encoder_hooks[0].features, x)  # cat activated 7x7 out -> 2x

        if self.use_deconv:
            x_fea = self.final_deconv(x)
        else:
            x_fea = F.interpolate(x, scale_factor=2, mode='trilinear', 
                align_corners=True)
        x_fea = self.conv1(x_fea)
        
        if dropout:
            x_fea = F.dropout3d(x_fea, p=0.3)
        
        x_fea = F.relu(self.bn1(x_fea))
        x_out = self.conv2(x_fea)
        return {
            'out': x_out
        }

    def close(self):
        for hook in self.encoder_hooks: 
            hook.remove()

    def _setup_encoder(self, name):
        if '121' in name:
            base_model = densenet.get_model(
                121, n_input_channels=1, num_classes=1,
                conv1_t_size=7, conv1_t_stride=2)
            layer_dims = densenet_outdims['densenet121']
        elif '169' in name:
            base_model = densenet.get_model(
                169, n_input_channels=1, num_classes=1,
                conv1_t_size=7, conv1_t_stride=2)
            layer_dims = densenet_outdims['densenet169']
        elif '201' in name:
            base_model = densenet.get_model(
                201, n_input_channels=1, num_classes=1,
                conv1_t_size=7, conv1_t_stride=2)
            layer_dims = densenet_outdims['densenet201']
        elif '161' in name:
            base_model = densenet.get_model(
                161, n_input_channels=1, num_classes=1,
                conv1_t_size=7, conv1_t_stride=2)
            layer_dims = densenet_outdims['densenet161']
        else:
            raise Exception(f"Invalid DenseNet name: {name}.")

        # layers is a list of 2 modules: dnet.features, dnet.classifier
        comps = list(base_model.children())
        layers = nn.Sequential(*comps)  # 0: feat extractor, 1: lin classifier
        
        encoder_hooks = [
            ForwardHook(layers[0][2]),  # ReLU after 7x7
            ForwardHook(layers[0][4]),  # DenseBlock 1 out
            ForwardHook(layers[0][6]),  # DenseBlock 2 out
            ForwardHook(layers[0][8]),  # DenseBlock 3 out
        ]
        return layers[0], encoder_hooks, layer_dims


def get_model(name, num_classes=1, deconv=False):
    model = DenseUNet(str(name), out_channels=num_classes, deconv=deconv)
    return model
