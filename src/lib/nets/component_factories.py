""" 
Objects that create neural network components or modules so that architecture
definitions can be cleaner & more generalizable. 

e.g. BN -> Norm('batchnorm', **kwargs) in network definition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NormFactory3d:
    def __init__(self, identifier):
        """
        Args:
            Size (int): either #channels or tensor shape for layer norm.
        """
        if isinstance(identifier, str):
            name = identifier.lower()
            if 'batch' in name:
                self.norm = nn.BatchNorm3d
            elif 'instance' in name:
                self.norm = nn.InstanceNorm3d
            # elif 'layer' in name:
            #     self.norm = nn.LayerNorm
            # elif 'group' in name:
            #     self.norm = nn.GroupNorm
            else:
                raise ValueError(f'Given norm name "{name}" is not supported.')
        elif isinstance(identifier, NormFactory3d):
            self.norm = identifier.norm
        elif isinstance(identifier, nn.Module):
            self.norm = identifier
        else:
            raise ValueError(f'Constructor must be given str, module, or fact.')
    
    def create(self, *args, **kwargs):
        return self.norm(*args, **kwargs)


class ActFactory:
    def __init__(self, identifier):
        if isinstance(identifier, str):
            name = identifier.lower()
            if name == 'relu':
                self.act = nn.ReLU
            elif name == 'leakyrelu':
                self.act = nn.LeakyReLU
            elif name == 'prelu':
                self.act = nn.PReLU
            elif name == 'sigmoid':
                self.act = nn.Sigmoid
            elif name == 'elu':
                self.act = nn.ELU
            elif name == 'gelu':
                self.act = nn.GELU
            else:
                raise ValueError(f'Given act name "{name}" is not supported.')
        elif isinstance(identifier, ActFactory):
            self.act = identifier.act
        elif isinstance(identifier, nn.Module):
            self.act = identifier
        else:
            raise ValueError(f'Constructor must be given str, module, or fact.')

    def create(self, *args, **kwargs):
        return self.act(*args, **kwargs)

