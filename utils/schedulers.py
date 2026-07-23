"""
schedulers.py
-------------
Advanced learning rate schedulers for training stability.


Includes:
- Warmup + Cosine Annealing (standard for transformers)
- Warmup + Linear Decay
- One-Cycle with warmup
"""


import math
from torch.optim.lr_scheduler import _LRScheduler



class WarmupCosineScheduler(_LRScheduler):
    """
    Linear warmup followed by cosine annealing.
    
    This is the standard scheduler for training transformers and is critical
    for SwinUNet — without warmup, the initial large gradients from random
    attention patterns can destabilize training.


    Parameters
    ----------
    optimizer     : torch optimizer
    warmup_epochs : int — number of warmup epochs (linear ramp from 0 to base_lr)
    total_epochs  : int — total training epochs
    eta_min       : float — minimum LR at end of cosine annealing
    warmup_start_lr : float — LR at step 0 (default: base_lr / 100)
    """


    def __init__(
        self,
        optimizer,
        warmup_epochs: int = 5,
        total_epochs: int = 50,
        eta_min: float = 1e-7,
        warmup_start_lr: float = None,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        self.warmup_start_lr = warmup_start_lr
        super().__init__(optimizer, last_epoch)


    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            if self.warmup_start_lr is not None:
                start_lrs = [self.warmup_start_lr] * len(self.base_lrs)
            else:
                start_lrs = [lr / 100.0 for lr in self.base_lrs]


            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [start + (base - start) * alpha
                    for start, base in zip(start_lrs, self.base_lrs)]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            return [
                self.eta_min + (base - self.eta_min) * 0.5 * (1 + math.cos(math.pi * progress))
                for base in self.base_lrs
            ]



class WarmupLinearScheduler(_LRScheduler):
    """Linear warmup then linear decay to eta_min."""


    def __init__(
        self,
        optimizer,
        warmup_epochs: int = 5,
        total_epochs: int = 50,
        eta_min: float = 1e-7,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)


    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [lr * alpha for lr in self.base_lrs]
        else:
            progress = (self.last_epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            return [
                self.eta_min + (base - self.eta_min) * (1 - progress)
                for base in self.base_lrs
            ]