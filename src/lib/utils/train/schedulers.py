""" schedulers.py (By: Charley Zhang, June 2020)
Various effective learning rate schedulers taken from literature.
"""

import sys
import math


class BaseScheduler:
    def step(self, epoch=-1, val=-1):
        raise NotImplementedError()
    
    def set_lr(self, lr):
        """(1) set all optim param_groups to lr; (2) update self.lr"""
        for g in self.optimizer.param_groups: # list of dicts with 'lr', 'params'
            g['lr'] = lr
        self.lr = self.get_lr()

    def get_lr(self):
        return next(iter(self.optimizer.param_groups))['lr']


class Uniform(BaseScheduler):
    """ Keeps uniform learning rate throughout. """
    def __init__(self, optimizer, rampup_rates=[]):
        self.optimizer = optimizer
        self.lr = self.get_lr()
        self.rampup_rates = rampup_rates
        if self.rampup_rates:
            if self.rampup_rates[-1] != self.lr:
                self.rampup_rates.append(self.lr)
            self.set_lr(self.rampup_rates.pop(0))

    def step(self, **kws):
        if self.rampup_rates:
            self.set_lr(self.rampup_rates.pop(0))


class ExponentialDecay(BaseScheduler):
    """ LR_t = orig_lr * (exp_fac) ^ t """
    def __init__(self, optimizer, exp_factor=0.9, t=0, minlr=1e-8, rampup_rates=[]):
        self.exp_factor = exp_factor
        self.minlr = minlr
        self.optimizer = optimizer
        self.lr = self.orig_lr = self.get_lr()

        self.rampup_rates = rampup_rates
        self.step(epoch=t)   # in-case we start from epoch > 1

    def step(self, epoch=None, **kws):
        assert epoch is not None, "Need to give epoch."
        
        if self.rampup_rates:
            self.set_lr(self.rampup_rates.pop(0))
        else:
            new_lr = max(self.minlr, self.orig_lr * (self.exp_factor) ** epoch)
            if new_lr != self.lr:
                self.set_lr(new_lr)


class LinearDecay(BaseScheduler):
    """ LR_t = LR_t+1 - (end_factor / T)  """
    def __init__(self, optimizer, end_factor=0.01, T=60, minlr=1e-8, 
                 rampup_rates=[]):
        self.end_factor = end_factor
        self.minlr = minlr
        self.optimizer = optimizer
        self.lr = self.orig_lr = self.get_lr()
        self.rampup_rates = rampup_rates

        self.step_reduction = (self.lr - self.lr * self.end_factor) / T
        print(f"\t (LinearDecay) Scheduler initialized with final lr={self.lr * self.end_factor:.8f}.")

    def step(self, **kws):
        if self.rampup_rates:
            self.set_lr(self.rampup_rates.pop(0))
        else:
            new_lr = max(self.minlr, self.lr - self.step_reduction)
            if new_lr != self.lr:
                self.set_lr(new_lr)


class ConsistencyCosineDecay(BaseScheduler):
    """ Scheduler used commonly in consistency regularization methods.
    LR_t = min_lr + orig_lr * cos(7*pi*t/(16*T)) 
    """
    def __init__(self, optimizer, T, t=0, minlr=1e-8, rampup_rates=[]):
        self.T = T
        self.minlr = minlr
        self.optimizer = optimizer
        self.lr = self.orig_lr = self.get_lr()

        self.rampup_rates = rampup_rates
        self.step(epoch=t)   # in-case we start from epoch > 1

    def step(self, epoch=None, **kws):
        assert epoch is not None, "Need to give epoch."
        
        if self.rampup_rates:
            self.set_lr(self.rampup_rates.pop(0))
        else:
            new_lr = max(self.minlr, self.orig_lr * math.cos(
                            7 * math.pi * epoch / (16 * self.T)))
            if new_lr != self.lr:
                self.set_lr(new_lr)


class CosineDecay(BaseScheduler):
    """ LR_t = min_lr + 0.5(1 + cos(pi*t/T)) * LR_0 """
    def __init__(self, optimizer, T, t=0, minlr=1e-8, rampup_rates=[]):
        self.T = T
        self.minlr = minlr
        self.optimizer = optimizer
        self.lr = self.orig_lr = self.get_lr()

        self.rampup_rates = rampup_rates
        self.step(epoch=t)   # in-case we start from epoch > 1

    def step(self, epoch=None, **kws):
        assert epoch is not None, "Need to give epoch."
        
        if self.rampup_rates:
            self.set_lr(self.rampup_rates.pop(0))
        else:
            new_lr = self.minlr + 0.5 * (1 + math.cos(
                math.pi * epoch / self.T)) * self.orig_lr
            if new_lr != self.lr:
                self.set_lr(new_lr)

    
class StepDecay(BaseScheduler):
    
    def __init__(self, optimizer, T, factor=0.2, steps=[0.33, 0.66], 
            rampup_rates=[]):
        self.T = T
        self.optimizer = optimizer
        self.factor = factor
        self.lr = self.start_lr = self.get_lr()

        if not steps or not sum(steps):
            self.step_epochs = []
        else:
            mult_fac = T if steps[0] < 1 else 1
            self.step_epochs = [int(s * mult_fac) for s in steps]

        self.rampup_rates = rampup_rates
        if self.rampup_rates:
            if self.rampup_rates[-1] != self.lr:
                self.rampup_rates.append(self.lr)
            self.set_lr(self.rampup_rates.pop(0))

    def step(self, epoch=None, **kws):
        assert epoch is not None
        if self.rampup_rates:
            self.set_lr(self.rampup_rates.pop(0))
            return

        num_lrsteps = sum([epoch >= s for s in self.step_epochs])
        correct_lr = self.start_lr * self.factor ** num_lrsteps
        if self.lr != correct_lr:
            self.set_lr(correct_lr)


class ReduceOnPlateau(BaseScheduler):
    """ Reduce LR on Plateue scheduler based on loss."""
    
    def __init__(self, optimizer, factor=0.5, patience=3, lowerbetter=True,
            rampup_rates=[]):
        self.factor = 0.5
        self.patience = patience
        self.lowerbetter = lowerbetter
        
        self.best_val = sys.float_info.max if lowerbetter else -sys.float_info.max
        self.bad_iter_count = 0

        self.optimizer = optimizer
        self.lr = self.get_lr()

        self.rampup_rates = rampup_rates
        if self.rampup_rates:
            if self.rampup_rates[-1] != self.lr:
                self.rampup_rates.append(self.lr)
            self.set_lr(self.rampup_rates.pop(0))
    
    def step(self, val=None, **kws):
        assert val is not None, "Need to give step a val."

        if self.rampup_rates:
            self.set_lr(self.rampup_rates.pop(0))
            return

        if self.lowerbetter:
            improved = True if val <= self.best_val else False
        else: 
            improved = True if val >= self.best_val else False
        
        if improved:
            self.best_val = val
            self.bad_iter_count = 0
        else:
            self.bad_iter_count += 1
            if self.bad_iter_count >= self.patience:
                self.set_lr(self.lr * self.factor)
                self.bad_iter_count = 0
                self.best_val = val
        

# Rudimentary Tests
if __name__ == '__main__':

    import torch

    T=20
    LR=1.
    FACTOR=0.5

    def get_optim():
        net = torch.nn.Sequential(
            torch.nn.Conv2d(12, 24, kernel_size=(3,3), stride=1, padding=1)
        )
        return torch.optim.SGD(net.parameters(), lr=LR)
    
    # stand-alone tests
    optimizer = get_optim()
    step = StepDecay(optimizer, T, factor=FACTOR, steps=[5,10,15,25])
    assert step.step_epochs == [5,10,15,25], f'{step.step_epochs}'
    step = StepDecay(optimizer, T, factor=FACTOR, steps=[.5,.8,2.])
    assert step.step_epochs == [10,16,40], f'{step.step_epochs}'
    cos = CosineDecay(optimizer, T, t=0)
    assert cos.lr == 1., f'{cos.lr}'
    cos = CosineDecay(optimizer, T, t=5)
    assert cos.lr == 0.5*(1+math.cos(math.pi*5/T))*LR, f'{cos.lr}'

    # start epoch = 0
    optimizer = get_optim()
    cos = CosineDecay(optimizer, T, t=0)
    step = StepDecay(optimizer, T, factor=FACTOR, steps=[5,10,15,25])
    plat_l = ReduceOnPlateau(optimizer, factor=FACTOR, patience=5, lowerbetter=True)
    plat_h = ReduceOnPlateau(optimizer, factor=FACTOR, patience=5, lowerbetter=False)
    
    schedulers = [cos, step, plat_l, plat_h]
    lrs = []
    for epoch in range(1, T+1):
        [s.step(epoch=epoch, val=epoch) for s in schedulers]
        lrs.append([s.lr for s in schedulers])

    cos_ans = [0.5 * (1 + math.cos(math.pi*e/T)) * LR for e in range(1, T+1)]
    step_ans = [LR]*4 + [LR*FACTOR]*5 + [LR*FACTOR**2]*5 + [LR*FACTOR**3]*6
    plat_lower_ans = [LR]*5 + [LR*FACTOR]*5 + [LR*FACTOR**2]*5 + [LR*FACTOR**3]*5
    plat_higher_ans = [LR]*20
    ans = [cos_ans, step_ans, plat_lower_ans, plat_higher_ans]

    for i, l in enumerate(zip(*lrs)):
        print(f"Testing schedule #{i}")
        assert ans[i] == list(l), f"\nAns{ans[i]}\nOut{l}"

    # start epoc_h = 5
    optimizer = get_optim()
    cos = CosineDecay(optimizer, T, t=5)
    optimizer = get_optim()
    step = StepDecay(optimizer, T, factor=FACTOR, steps=[5,10,15,25])
    plat_l = ReduceOnPlateau(optimizer, factor=FACTOR, patience=5, lowerbetter=True)
    plat_h = ReduceOnPlateau(optimizer, factor=FACTOR, patience=5, lowerbetter=False)
    
    schedulers = [cos, step, plat_l, plat_h]
    lrs = []
    for epoch in range(5, T+1):
        [s.step(epoch=epoch, val=epoch) for s in schedulers]
        lrs.append([s.lr for s in schedulers])

    cos_ans = [0.5 * (1 + math.cos(math.pi*e/T)) * LR for e in range(5, T+1)]
    step_ans = [LR*FACTOR]*5 + [LR*FACTOR**2]*5 + [LR*FACTOR**3]*6
    plat_lower_ans = [LR]*5 + [LR*FACTOR**1]*5 + [LR*FACTOR**2]*5 + [LR*FACTOR**3]
    plat_higher_ans = [LR]*16
    ans = [cos_ans, step_ans, plat_lower_ans, plat_higher_ans]

    for i, l in enumerate(zip(*lrs)):
        print(f"Testing schedule #{i}")
        assert ans[i] == list(l), f"\nAns{ans[i]}\nOut{l}"
    
    
    print("✔ All passed ✔")

    