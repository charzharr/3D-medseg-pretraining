""" Module basemodel.py (By: Charley Zhang, July 2020)
Implements basic functionality for all models subclasses.
"""

import sys, os
import torch, torch.nn as nn
import torchsummary


class BaseModel(nn.Module):
    r""" Pytorch basemodel with useful customized functionalities."""
    def __init__(self, *args, **kwargs):
        super(BaseModel, self).__init__(*args, **kwargs)

    def forward(self, *args):
        raise NotImplementedError(f"forward() func requires definition.")

    @property
    def device(self):
        return next(self.parameters()).device if self.parameters() else None

    @property
    def param_counts(self):
        tot_params = sum(p.numel() for p in self.parameters())
        tot_train_params = sum(p.numel() for p in self.parameters()
            if p.requires_grad
        )
        return tot_params, tot_train_params

    @property
    def size(self):
        r""" Gets total parameter and buffer memory usage in bytes. """
        params_mem = sum(
            [p.nelement() * p.element_size() for p in self.parameters()]
        )
        bufs_mem = sum(
            [buf.nelement() * buf.element_size() for buf in self.buffers()]
        )
        return params_mem + bufs_mem 

    def summary(self, input_size=(3,256,256), batch_size=-1, device='cpu'):
        if 'cuda' in device:
            device = 'cuda'  # summary does not support targ device assignment
        torchsummary.summary(
            self, 
            input_size=input_size, 
            batch_size=batch_size,
            device=device
        )

        
    