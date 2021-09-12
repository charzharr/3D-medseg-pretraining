"""
Implementing some features from nnUNet
    - Instance Norm
    - Leakly Relu
    - Initial FM of 32
    - Transposed convolutions
    - Lowest resolution # FMs capped at 320

"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.nets.basemodel import BaseModel


activation = nn.LeakyReLU(negative_slope=0.01, inplace=True)
norm = nn.InstanceNorm3d


# ------------------ ##  3D UNet (from Model Genesis)  ## ------------------ #


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, depth, act):
        super(UpTransition, self).__init__()
        self.depth = depth
        self.up_conv = nn.ConvTranspose3d(inChans, outChans, kernel_size=2, 
                                          stride=2)
        # self.up_conv = nn.Upsample(scale_factor=2, mode='trilinear',
        #                             align_corners=False)
        self.ops = _make_nConv(inChans+ outChans//2,depth, act, 
                               double_chnnel=True)

    def forward(self, x, skip_x):
        out_up_conv = self.up_conv(x)
        concat = torch.cat((out_up_conv,skip_x),1)
        out = self.ops(concat)
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, n_labels):

        super(OutputTransition, self).__init__()
        self.final_conv = nn.Conv3d(inChans, n_labels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # out = self.sigmoid(self.final_conv(x))
        out = self.final_conv(x)
        return out


class EncoderModule(nn.Module):
    def __init__(self, in_channels, out_channels, first_conv_stride=(2, 2, 2), 
                 kernel_size=(3, 3, 3)):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, 
            stride=first_conv_stride, kernel_size=kernel_size,
            padding=[k // 2 for k in kernel_size])
        self.norm1 = norm(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 
            stride=(1, 1, 1), kernel_size=kernel_size,
            padding=[k // 2 for k in kernel_size])
        self.norm2 = norm(out_channels)

    def forward(self, x):
        x = activation(self.norm1(self.conv1(x)))
        x = activation(self.norm2(self.conv2(x)))
        return x


class DecoderModule(nn.Module):
    def __init__(self, in_channels_side, in_channels_bot, out_channels, 
                 upscale_stride=(2, 2, 2), kernel_size=(3, 3, 3)):
        super().__init__()

        # dout = (din - 1) * stride - 2 * pad + dilation * (kernel_size - 1) + out_padding + 1
        # self.up = nn.ConvTranspose3d(in_channels_bot, in_channels_bot,
        #             kernel_size=kernel_size, upscale_stride=upscale_stride,
        #             padding=[])
        self.up_op = nn.Upsample(scale_factor=upscale_stride, mode='trilinear',
                              align_corners=False)
        self.up_norm = norm(in_channels_bot)

        self.conv1 = nn.Conv3d(in_channels_side + in_channels_bot, out_channels, 
            stride=(1, 1, 1), kernel_size=kernel_size,
            padding=[k // 2 for k in kernel_size])
        self.norm1 = norm(out_channels)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, 
            stride=(1, 1, 1), kernel_size=kernel_size,
            padding=[k // 2 for k in kernel_size])
        self.norm2 = norm(out_channels)
        

    def forward(self, side_in, below_in):
        below_x = activation(self.up_norm(self.up_op(below_in)))

        x = torch.cat([side_in, below_x], 1)
        x = activation(self.norm1(self.conv1(x)))
        x = activation(self.norm2(self.conv2(x)))
        return x



class UNet3D(BaseModel):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, n_input=1, n_class=1, deep_sup=False):
        super(UNet3D, self).__init__()

        self.down_0 = EncoderModule(n_input, 32, first_conv_stride=(1, 1, 1),
                                    kernel_size=(1, 3, 3)) # (1 1 1)x down
        self.down_1 = EncoderModule(32, 64, first_conv_stride=(1, 2, 2),
                                    kernel_size=(3, 3, 3)) # (1 2 2)x down
        self.down_2 = EncoderModule(64, 128, first_conv_stride=(2, 2, 2),
                                    kernel_size=(3, 3, 3)) # (2 4 4)x down
        self.down_3 = EncoderModule(128, 256, first_conv_stride=(2, 2, 2),
                                    kernel_size=(3, 3, 3)) # (4 8 8)x down
        self.bottom = EncoderModule(256, 320, first_conv_stride=(2, 2, 2),
                                    kernel_size=(3, 3, 3))# (8 16 16)x down

        self.up_3 = DecoderModule(256, 320, 256, upscale_stride=(2, 2, 2),
                                 kernel_size=(3, 3, 3))
        self.up_2 = DecoderModule(128, 256, 128, upscale_stride=(2, 2, 2),
                                 kernel_size=(3, 3, 3))
        self.up_1 = DecoderModule(64, 128, 64, upscale_stride=(2, 2, 2),
                                 kernel_size=(3, 3, 3))
        self.up_0 = DecoderModule(32, 64, 32, upscale_stride=(1, 2, 2),
                                 kernel_size=(1, 3, 3))

        self.deep_sup = deep_sup
        if deep_sup:
            self.final_8x = nn.Conv3d(256, n_class, kernel_size=1)
            self.final_4x = nn.Conv3d(128, n_class, kernel_size=1)
            self.final_2x = nn.Conv3d(64, n_class, kernel_size=1)
        self.final_1x = nn.Conv3d(32, n_class, kernel_size=1)
        
        tot_params, tot_tparams = self.param_counts
        print(f'💠 nnUNet3D model initiated with n_classes={n_class}, \n'
              f'   n_input={n_input}, \n'
              f'   params={tot_params:,}, trainable_params={tot_tparams:,}.')

    
    def forward(self, x, enc_only=False):
        out_0 = self.down_0(x)
        out_1 = self.down_1(out_0)
        out_2 = self.down_2(out_1)
        out_3 = self.down_3(out_2)
        feats = self.bottom(out_3)

        if enc_only:
            return {
                'out': feats
            }

        up_8x = self.up_3(out_3, feats)
        up_4x = self.up_2(out_2, up_8x)
        up_2x = self.up_1(out_1, up_4x)
        up_1x = self.up_0(out_0, up_2x)

        out_1x = self.final_1x(up_1x)
        if self.deep_sup:
            up_8x = self.final_8x(up_8x)
            up_4x = self.final_4x(up_4x)
            up_2x = self.final_2x(up_2x)

            return {
                'out': out_1x,  
                '2x': up_2x,  
                '4x': up_4x, 
                '8x': up_8x
            }
        return {
            'out': out_1x
        }
                