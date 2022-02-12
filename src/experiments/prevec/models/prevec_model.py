

import torch
import torch.nn as nn
import torch.nn.functional as F



# ============================================================================ #
# * ### * ### * ### *  Res2UNet101 Pretext Model Wrappers  * ### * ### * ### * #
# ============================================================================ #
    

class PreVecV1(nn.Module):
    """
    Changes & Comments
      - Use 8x downsampled for resolution of choice
      - Hao's method has 64, 128, 256, 512 channels in ResNet-34 encoder.
         - He also
    """
    
    def __init__(self, config, unet_like, num_classes):
        super().__init__()
        
        self.config = config 
        self.backbone = unet_like
        
        self.relu = nn.ReLU(inplace=True)
        
        # Pre-cat Operations
        self.cat_dims = [32, 32, 32, 64, 64, 64]
        self.precat_downsamp_1x = nn.AvgPool3d(8, stride=8)
        self.precat_bn_1x = nn.BatchNorm3d(32)
        self.precat_conv_1x = nn.Conv3d(32, self.cat_dims[0], 3, 1, 1, 
                                        bias=False)
        
        self.precat_downsamp_2x = nn.AvgPool3d(4, stride=4)
        self.precat_bn_2x = nn.BatchNorm3d(64)
        self.precat_conv_2x = nn.Conv3d(64, self.cat_dims[1], 3, 1, 1, 
                                        bias=False)
        
        self.precat_downsamp_4x = nn.AvgPool3d(2, stride=2)
        self.precat_bn_4x = nn.BatchNorm3d(64)
        self.precat_conv_4x = nn.Conv3d(64, self.cat_dims[2], 3, 1, 1, 
                                        bias=False)
        
        self.precat_bn_8x = nn.BatchNorm3d(128)
        self.precat_conv_8x = nn.Conv3d(128, self.cat_dims[3], 3, 1, 1, 
                                        bias=False)
        
        self.precat_bn_16x = nn.BatchNorm3d(256)
        self.precat_conv_16x = nn.Conv3d(256, self.cat_dims[4], 3, 1, 1, 
                                        bias=False)
        
        self.precat_bn_32x = nn.BatchNorm3d(2048)
        self.precat_conv_32x = nn.Conv3d(2048, self.cat_dims[5], 3, 1, 1, 
                                        bias=False)
        
        # Post-cat Prediction
        postcat_dims = sum(self.cat_dims)
        self.postcat_bn1 = nn.BatchNorm3d(postcat_dims)
        self.postcat_conv1 = nn.Conv3d(postcat_dims, 32, 3, 2, 1, bias=False)
        
        self.final_bn = nn.BatchNorm3d(32)
        self.projection_1 = nn.Linear(32 * 6 ** 3, 1024)
        self.projection_bn = nn.BatchNorm1d(1024)
        self.projection_2 = nn.Linear(1024, num_classes)
        
        
    def forward(self, x):
        out_d = self.backbone(x)
        
        # Downsample or Reduce dimensions before MS-concat
        precat_1x = self.precat_conv_1x(self.relu(self.precat_bn_1x(
            self.precat_downsamp_1x(out_d['1x'])
        )))
        precat_2x = self.precat_conv_2x(self.relu(self.precat_bn_2x(
            self.precat_downsamp_2x(out_d['2x'])
        )))
        precat_4x = self.precat_conv_4x(self.relu(self.precat_bn_4x(
            self.precat_downsamp_4x(out_d['4x'])
        )))
        precat_8x = self.precat_conv_8x(self.relu(self.precat_bn_8x(
            out_d['8x']
        )))
        precat_16x = F.interpolate(
            self.precat_conv_16x(self.relu(self.precat_bn_16x(out_d['16x']))),
            scale_factor=2, mode='trilinear', align_corners=False)
        precat_32x = F.interpolate(
            self.precat_conv_32x(self.relu(self.precat_bn_32x(out_d['32x']))),
            scale_factor=4, mode='trilinear', align_corners=False)
        
        # Concat & process
        cat = torch.cat([precat_1x, precat_2x, precat_4x, precat_8x,
                         precat_16x, precat_32x], dim=1)
        postcat = self.relu(self.postcat_bn1(cat))
        
        postcat = self.relu(self.final_bn(self.postcat_conv1(postcat)))
        postcat_flat = torch.flatten(postcat, 1)
        out = self.projection_1(postcat_flat)
        out = self.projection_2(self.relu(self.projection_bn(out)))
        
        return {
            'out': out
        }
    