"""
ema.py
------
Exponential Moving Average (EMA) for model weights.


EMA maintains a shadow copy of model parameters that is a running
exponential average of the training weights. At inference time, the
EMA weights typically generalize better (smoother loss landscape).


This technique is used in nearly all modern SOTA models (diffusion models,
image restoration, etc.) and typically provides 0.1-0.5 dB PSNR improvement
"for free" with no extra training cost.


Usage
-----
    ema = EMAModel(model, decay=0.999)
    
    # During training:
    optimizer.step()
    ema.update()
    
    # For evaluation:
    with ema.average_parameters():
        val_loss = evaluate(model, val_loader)
"""


import copy
from contextlib import contextmanager
from typing import Iterable


import torch
import torch.nn as nn



class EMAModel:
    """
    Exponential Moving Average of model parameters.


    Parameters
    ----------
    model       : nn.Module — the model being trained
    decay       : float — EMA decay rate (0.999 = slow update, 0.99 = fast)
    warmup      : int — number of steps before EMA starts (use training weights until then)
    """


    def __init__(self, model: nn.Module, decay: float = 0.999, warmup: int = 1000):
        self.model = model
        self.decay = decay
        self.warmup = warmup
        self.step_count = 0


        # Shadow parameters (deep copy of model state)
        self.shadow = {}
        self.backup = {}
        self._init_shadow()


    def _init_shadow(self):
        """Initialize shadow parameters as a copy of model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()


    u/torch.no_grad()
    def update(self):
        """Update shadow parameters with current model parameters."""
        self.step_count += 1


        # Dynamic decay with warmup (ramps up from 0 to target decay)
        decay = min(self.decay, (1 + self.step_count) / (10 + self.step_count))
        if self.step_count < self.warmup:
            decay = 0.0  # Just copy during warmup


        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].lerp_(param.data, 1.0 - decay)


    
    def average_parameters(self):
        """
        Context manager to temporarily swap model params with EMA params.


        Usage:
            with ema.average_parameters():
                evaluate(model)
        """
        # Backup current model params
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])


        try:
            yield
        finally:
            # Restore original params
            for name, param in self.model.named_parameters():
                if name in self.backup:
                    param.data.copy_(self.backup[name])
            self.backup = {}


    def state_dict(self) -> dict:
        """Get EMA state for checkpointing."""
        return {
            "shadow": self.shadow,
            "step_count": self.step_count,
            "decay": self.decay,
        }


    def load_state_dict(self, state: dict):
        """Load EMA state from checkpoint."""
        self.shadow = state["shadow"]
        self.step_count = state["step_count"]
        self.decay = state.get("decay", self.decay)


    def apply_shadow(self):
        """Permanently replace model weights with EMA weights (for export/inference)."""
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])