

import torch
import torch.nn as nn
import torch.nn.functional as F



# ============================================================================ #
# * ### * ### * ### *  Res2UNet101 Pretext Model Wrappers  * ### * ### * ### * #
# ============================================================================ #
    

class SARModel(nn.Module):
    """
    Wraps a UNet-like model with a 2 layer classifier on the lowest 
    resolution features. Hard-coded to work with res2unet3d.
    """
    
    def __init__(self, config, unet_like, num_classes,
                 recon_weight=10, scale_weight=1):
        super().__init__()
        
        self.config = config 
        self.backbone = unet_like
        
        self.relu = nn.ReLU(inplace=True)
        
        # Scale prediction
        self.scale_pool = nn.AdaptiveAvgPool3d(1)
        self.scale_fc1 = nn.Linear(2048, 512, bias=False)
        self.scale_bn1 = nn.BatchNorm1d(512)
        self.scale_fc2 = nn.Linear(512, num_classes)
        
        # Losses
        self.recon_loss = nn.MSELoss()
        self.recon_weight = recon_weight
        self.scale_loss = nn.CrossEntropyLoss()
        self.scale_weight = scale_weight
        
        
    def forward(self, x):
        out_d = self.backbone(x)
        recon_out = out_d['out']
        
        # Scale prediction from encoder 32x features
        feats = self.scale_pool(out_d['32x']).flatten(1)  # activated
        scale_out = self.relu(self.scale_bn1(self.scale_fc1(feats)))
        scale_out = self.scale_fc2(scale_out)
        
        return {
            'out': recon_out,
            'unet_out': out_d,
            'recon': recon_out,
            'scale': scale_out
        }
    
    def loss(self, out_d, recon_targs, scale_targs):
        recon_preds = out_d['recon'].sigmoid()
        recon_loss = self.recon_weight * self.recon_loss(recon_preds, recon_targs)
        
        scale_logits = out_d['scale']
        scale_loss = self.scale_weight * self.scale_loss(scale_logits, scale_targs)
        
        loss = recon_loss + scale_loss
        
        return {
            'loss': loss,
            'scale': scale_loss,
            'recon': recon_loss
        }
        
    