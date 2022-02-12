"""
Losses for unsupervised spatial pretraining. 
""" 

import torch
import torch.nn as nn



# ============================================================================ #
# * ### * ### * ### *        Vector Prediction Losses      * ### * ### * ### * #
# ============================================================================ #


class SphericalCriterion(torch.nn.Module):
    """
    Criterion to predict a position vector expressed in spherical coordinates.
        r = sqrt(x^2 + y^2 + z^2)
        θ ∈ [0, π], φ ∈ [0, 2π]
        ... https://www.wikiwand.com/en/Spherical_coordinate_system
        
    Experiment Loop Sample Code:
        targs = torch.tensor([
            r / max_dist, theta / math.pi, phi / (2 * math.pi)
        ])
        preds = model(x)
        loss = spherical_crit(preds, targs, logits=True)['loss']
    """
    def __init__(self, r_loss='l1', angle_loss='l1',
                 r_weight=1, theta_weight=1, phi_weight=1):
        super().__init__()
        self.r_w, self.t_w, self.p_w = r_weight, theta_weight, phi_weight

        if r_loss == 'l1':
            print(f'(SphericalCriterion) Using L1 for r loss.')
            self.r_criterion = torch.nn.L1Loss()
        else:
            print(f'(SphericalCriterion) Using MSE for r loss.')
            self.r_criterion = torch.nn.MSELoss()
        
        if angle_loss == 'l1':
            print(f'(SphericalCriterion) Using L1 for angle losses.')
            self.theta_criterion = torch.nn.L1Loss()
            self.phi_criterion = torch.nn.L1Loss()
        else:
            print(f'(SphericalCriterion) Using MSE for angle losses.')
            self.theta_criterion = torch.nn.MSELoss()
            self.phi_criterion = torch.nn.MSELoss()
            
        print(f'💠 SphericalCriterion initiated with r_loss={r_loss}, \n'
              f'   ang_loss={angle_loss}, \n'
              f'   r_w={r_weight}, t_w={theta_weight}, p_w={phi_weight}.')

    def forward(self, preds, targs, logits=True):
        """
        Note:
            - 3 edge cases for phi. Assume we have radians rescaled to [0,1]
                1. pred=0.01, targ=0.99 -> take dist(0.01, 0.99 - 1)
                    1 here is equivalent to 2π, AKA same angle
                2. pred=0.99, targ=0.01 -> take dist(0.99, 0.01 + 1)
        Args:
            preds (tensor): B x (n_vecs * 3)
            targs (tensor): B x (n_vecs * 3)
            logits (bool): if preds are logits, then we apply a sigmoid to them,
                else we assume they are between 0 & 1.
        """
        assert preds.shape == targs.shape
        assert preds.ndim == 2
        assert preds.shape[-1] % 3 == 0
        
        if logits:
            preds = preds.sigmoid()
        else:
            assert 0. <= preds.min() <= preds.max() <= 1.
            
        B, N_vecs = preds.shape[0], preds.shape[-1] // 3
        
        r_loss = self.r_w * self.r_criterion(preds[0::3], targs[0::3])
        theta_loss = self.t_w * self.theta_criterion(preds[1::3], targs[1::3])
        
        phi_loss = 0
        for b in range(B):
            for v in range(2, B, 3):
                phi_loss += min(
                    self.phi_criterion(preds[b, v], targs[b, v]),
                    self.phi_criterion(preds[b, v], targs[b, v] + 1),
                    self.phi_criterion(preds[b, v], targs[b, v] - 1),
                )
        phi_loss = self.p_w * phi_loss / (B * N_vecs)
        
        # print(r_loss, theta_loss, phi_loss)
        
        loss =  r_loss + theta_loss + phi_loss
        return {
            'loss': loss,
            'r': r_loss,
            'theta': theta_loss,
            'phi': phi_loss
        }
        
