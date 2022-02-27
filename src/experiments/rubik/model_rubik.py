

import torch
import torch.nn as nn
import torch.nn.functional as F


# class Rubikpp(nn.Module):
#     """
#     Model container main module components and losses for Rubik++ pretraining.
#     Assumes that we are pretraining on 3D grayscale images.
#     """
#     def __init__(self, config, generator, discriminator):
#         self.config = config
#         self.generator = generator 
#         self.discriminator = discriminator 
    
#     def forward(self, x, y):
#         g_out = self.generator(x)
        
#         d_out_gen = self.discriminator(g_out['out'].sigmoid(), x)
#         d_out_orig = self.discriminator(y, x)
        
#         return {
#             'out': g_out['out'],
#             'g_out': g_out,          
#             'd_out_generated': d_out_gen,
#             'd_out_original': d_out_orig,
#             'y': y
#         }
    
#     def loss(self, pred_logits, targ):
#         return {
#             'loss': None
#         }


class Discriminator(nn.Module):
    
    def __init__(self, config, in_channels=2, out_channels=1, base_channels=32):
        """
        Args:
            in_channels: number of channels after real images is concatenated
                with the predicted one.
            out_channels: number of classes to output (1 for real/fake)
        """
        assert out_channels == 1
        
        self.config = config
        self.in_channels = in_channels

        super().__init__()

        # Modules
        self.model = nn.Sequential(
            self._create_layer(in_channels, base_channels),
            self._create_layer(base_channels, 2 * base_channels),
            self._create_layer(2 * base_channels, 4 * base_channels),
            self._create_layer(4 * base_channels, 8 * base_channels),
            nn.Conv3d(8 * base_channels, 1, 1)
        )
        
        num_params = sum([p.numel() for p in self.parameters()])
        print(f'💠 Rubik++ Discriminator initialized ({num_params:,} params)!\n'
            f'   in_channels={in_channels}. ')
        print(self)
        
    def forward(self, x1, x2):
        """
        Args:
            x1 (tensor): predicted or original volume
            x2 (tensor): the disarranged volume
        """
        x = torch.cat([x1, x2], 1)
        assert x.shape[1] == self.in_channels
        
        out = self.model(x)
        return out
        
    def _create_layer(self, in_channels, out_channels):
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        return layer