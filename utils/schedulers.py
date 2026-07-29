"""
schedulers.py
-------------
Learning-rate schedules for stable transformer training.

Why warmup matters here
-----------------------
At initialisation a Swin block's attention logits are near-uniform, so the
softmax gradient is large and poorly conditioned; combined with Adam's small
initial second-moment estimate this produces an enormous effective step in the
first few hundred iterations. Linear warmup keeps that transient bounded. This
is not a cosmetic detail — it is the difference between the SwinUNet converging
and diverging within the first epoch.

Granularity
-----------
All schedulers here are unit-agnostic: construct them with ``warmup=<n>`` and
``total=<n>`` in whatever unit you intend to call ``.step()`` with. The trainer
drives them **per optimiser step** rather than per epoch, which gives the warmup
ramp hundreds of points of resolution instead of five.
"""

from __future__ import annotations

import math
import warnings

try:  # torch >= 2.0
    from torch.optim.lr_scheduler import LRScheduler as _BaseLRScheduler
except ImportError:  # pragma: no cover - older torch
    from torch.optim.lr_scheduler import _LRScheduler as _BaseLRScheduler


class _WarmupScheduler(_BaseLRScheduler):
    """Shared warmup plumbing for the concrete schedules below."""

    def __init__(
        self,
        optimizer,
        warmup: int = 500,
        total: int = 10_000,
        eta_min: float = 1e-7,
        warmup_start_lr: float | None = None,
        last_epoch: int = -1,
    ):
        if total <= 0:
            raise ValueError(f"total must be positive, got {total}")
        if warmup < 0:
            raise ValueError(f"warmup must be non-negative, got {warmup}")
        if warmup >= total:
            warnings.warn(
                f"warmup ({warmup}) >= total ({total}); the schedule will never "
                f"reach its decay phase. Clamping warmup to total // 10.",
                stacklevel=2,
            )
            warmup = max(1, total // 10)

        self.warmup = warmup
        self.total = total
        self.eta_min = eta_min
        self.warmup_start_lr = warmup_start_lr
        super().__init__(optimizer, last_epoch)

    def _warmup_lrs(self) -> list[float]:
        starts = (
            [self.warmup_start_lr] * len(self.base_lrs)
            if self.warmup_start_lr is not None
            else [lr / 100.0 for lr in self.base_lrs]
        )
        alpha = self.last_epoch / max(1, self.warmup)
        return [start + (base - start) * alpha for start, base in zip(starts, self.base_lrs)]

    def _progress(self) -> float:
        """Fraction of the post-warmup schedule completed, clamped to [0, 1]."""
        span = max(1, self.total - self.warmup)
        return min(1.0, max(0.0, (self.last_epoch - self.warmup) / span))


class WarmupCosineScheduler(_WarmupScheduler):
    """
    Linear warmup followed by cosine annealing to ``eta_min``.

    Parameters
    ----------
    optimizer
        Optimiser whose ``param_groups`` will be updated.
    warmup
        Steps (or epochs) of linear ramp from ``warmup_start_lr`` to the base LR.
    total
        Total steps (or epochs) in the schedule.
    eta_min
        Floor learning rate at the end of annealing.
    warmup_start_lr
        LR at step 0. Defaults to ``base_lr / 100``.
    """

    def get_lr(self) -> list[float]:
        if self.last_epoch < self.warmup:
            return self._warmup_lrs()
        progress = self._progress()
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [self.eta_min + (base - self.eta_min) * cosine for base in self.base_lrs]


class WarmupLinearScheduler(_WarmupScheduler):
    """Linear warmup then linear decay to ``eta_min``."""

    def get_lr(self) -> list[float]:
        if self.last_epoch < self.warmup:
            return self._warmup_lrs()
        progress = self._progress()
        return [self.eta_min + (base - self.eta_min) * (1.0 - progress) for base in self.base_lrs]


class WarmupCosineRestartScheduler(_WarmupScheduler):
    """
    Warmup followed by cosine annealing with warm restarts (SGDR).

    Each cycle is ``cycle_mult`` times longer than the last. Restarts help escape
    the sharp minima that long single-cycle schedules can settle into, and give
    a free ensemble: the weights at the end of each cycle are diverse enough to
    average usefully.

    Parameters
    ----------
    first_cycle
        Length of the first cosine cycle, in the same unit as ``total``.
    cycle_mult
        Multiplier applied to the cycle length after each restart.
    gamma
        Multiplier applied to the peak LR after each restart (``1.0`` = no decay).
    """

    def __init__(
        self,
        optimizer,
        warmup: int = 500,
        total: int = 10_000,
        first_cycle: int = 2_000,
        cycle_mult: float = 2.0,
        gamma: float = 0.8,
        eta_min: float = 1e-7,
        warmup_start_lr: float | None = None,
        last_epoch: int = -1,
    ):
        self.first_cycle = max(1, first_cycle)
        self.cycle_mult = cycle_mult
        self.gamma = gamma
        super().__init__(optimizer, warmup, total, eta_min, warmup_start_lr, last_epoch)

    def _cycle_position(self) -> tuple[float, int]:
        """Return (progress within current cycle, number of completed restarts)."""
        t = self.last_epoch - self.warmup
        cycle_len = float(self.first_cycle)
        restarts = 0
        while t >= cycle_len:
            t -= cycle_len
            cycle_len *= self.cycle_mult
            restarts += 1
        return t / cycle_len, restarts

    def get_lr(self) -> list[float]:
        if self.last_epoch < self.warmup:
            return self._warmup_lrs()
        progress, restarts = self._cycle_position()
        peak_scale = self.gamma**restarts
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [
            self.eta_min + (base * peak_scale - self.eta_min) * cosine for base in self.base_lrs
        ]


def build_scheduler(optimizer, name: str, total_steps: int, **kwargs):
    """
    Factory used by the trainer.

    Recognised names: ``warmup_cosine``, ``warmup_linear``, ``warmup_restart``,
    ``cosine``, ``step``, ``plateau``, ``none``.

    ``plateau`` is returned as-is and must be stepped with a metric; every other
    schedule is stepped per optimiser step. The trainer distinguishes them by
    isinstance check rather than by name.
    """
    from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR

    name = (name or "warmup_cosine").lower()
    warmup = kwargs.pop("warmup", max(1, total_steps // 20))
    eta_min = kwargs.pop("eta_min", 1e-7)

    if name == "warmup_cosine":
        return WarmupCosineScheduler(optimizer, warmup, total_steps, eta_min)
    if name == "warmup_linear":
        return WarmupLinearScheduler(optimizer, warmup, total_steps, eta_min)
    if name == "warmup_restart":
        return WarmupCosineRestartScheduler(
            optimizer,
            warmup,
            total_steps,
            first_cycle=kwargs.pop("first_cycle", max(1, total_steps // 4)),
            cycle_mult=kwargs.pop("cycle_mult", 2.0),
            gamma=kwargs.pop("gamma", 0.8),
            eta_min=eta_min,
        )
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=eta_min)
    if name == "step":
        return StepLR(optimizer, step_size=max(1, total_steps // 3), gamma=0.1)
    if name == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
    if name == "none":
        return None
    raise ValueError(
        f"Unknown scheduler '{name}'. Choose from: warmup_cosine, warmup_linear, "
        f"warmup_restart, cosine, step, plateau, none."
    )
