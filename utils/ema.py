"""
ema.py
------
Exponential Moving Average (EMA) of model weights.

EMA maintains a shadow copy of the model parameters that tracks a running
exponential average of the training trajectory. Averaged weights sit closer to
the centre of a flat minimum than any single SGD iterate, which is why EMA
reliably buys a few tenths of a dB on reconstruction tasks at zero training cost.

Usage
-----
    ema = EMAModel(model, decay=0.999)

    # During training, immediately after each optimiser step:
    optimizer.step()
    ema.update()

    # For evaluation / checkpointing:
    with ema.average_parameters():
        metrics = evaluate(model, val_loader)

Notes on correctness
--------------------
* ``average_parameters`` is a **context manager**; it is decorated with
  ``@contextlib.contextmanager`` so that ``with ema.average_parameters():``
  actually swaps weights. Without the decorator the method returns a bare
  generator, the body never executes, and validation silently runs on raw
  training weights — a failure mode that produces no error and no warning.
* The shadow state also tracks non-parameter **buffers** (e.g. running statistics)
  when ``include_buffers=True``, so that swapping in EMA weights yields a fully
  self-consistent model rather than a mix of averaged parameters and live buffers.
* ``state_dict`` emits both ``shadow`` (name -> tensor) and ``shadow_params``
  (ordered list) so checkpoints stay loadable by name, which is robust to
  parameter reordering in a way that positional zipping is not.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class EMAModel:
    """
    Exponential Moving Average of model parameters.

    Parameters
    ----------
    model
        The model being trained. The EMA holds a reference, not a copy of the
        module, and allocates shadow tensors on the same device as each source.
    decay
        Target EMA decay. Higher means slower adaptation (0.999 is a good default
        for runs of tens of thousands of steps; use 0.99 for short runs).
    warmup
        Number of optimiser steps during which the shadow simply tracks the live
        weights. Averaging from step 0 would drag the shadow toward the random
        initialisation for a long time.
    include_buffers
        Also shadow non-parameter buffers so the swapped-in model is coherent.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        warmup: int = 1000,
        include_buffers: bool = True,
    ):
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}")

        self.model = model
        self.decay = decay
        self.warmup = max(0, int(warmup))
        self.include_buffers = include_buffers
        self.step_count = 0

        self.shadow: dict[str, torch.Tensor] = {}
        self._backup: dict[str, torch.Tensor] = {}
        self._init_shadow()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tracked(self) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield (name, tensor) for every tensor under EMA control."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                yield name, param.data
        if self.include_buffers:
            for name, buf in self.model.named_buffers():
                # Integer buffers (counters, precomputed indices, attention masks)
                # cannot be meaningfully averaged, so they are left alone.
                if buf is not None and buf.is_floating_point():
                    yield f"__buffer__{name}", buf

    def _init_shadow(self) -> None:
        self.shadow = {name: tensor.detach().clone() for name, tensor in self._tracked()}

    def _current_decay(self) -> float:
        """
        Decay schedule: 0 during warmup (pure copy), then ramped toward the target.

        The ``(1 + t) / (10 + t)`` ramp is the standard warm-up used by timm and
        the diffusion literature: it prevents the average from being dominated by
        the first few (near-random) iterates.
        """
        if self.step_count <= self.warmup:
            return 0.0
        t = self.step_count - self.warmup
        return min(self.decay, (1.0 + t) / (10.0 + t))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update(self) -> None:
        """Fold the current model weights into the running average."""
        self.step_count += 1
        decay = self._current_decay()

        for name, tensor in self._tracked():
            shadow = self.shadow.get(name)
            if shadow is None:
                # A parameter appeared after construction (e.g. lazy module).
                self.shadow[name] = tensor.detach().clone()
                continue
            if shadow.device != tensor.device:
                shadow = shadow.to(tensor.device)
                self.shadow[name] = shadow
            # shadow <- decay * shadow + (1 - decay) * tensor
            shadow.lerp_(tensor, 1.0 - decay)

    @contextmanager
    def average_parameters(self):
        """
        Temporarily swap the live model weights for the EMA weights.

        Restores the original weights on exit, including when the body raises.
        """
        self.store()
        try:
            self.copy_to()
            yield self.model
        finally:
            self.restore()

    @torch.no_grad()
    def store(self) -> None:
        """Back up the live weights so they can be restored later."""
        self._backup = {name: tensor.detach().clone() for name, tensor in self._tracked()}

    @torch.no_grad()
    def copy_to(self) -> None:
        """Write the EMA weights into the live model."""
        for name, tensor in self._tracked():
            shadow = self.shadow.get(name)
            if shadow is not None:
                tensor.copy_(shadow.to(tensor.device))

    @torch.no_grad()
    def restore(self) -> None:
        """Undo :meth:`copy_to`, restoring the weights saved by :meth:`store`."""
        if not self._backup:
            return
        for name, tensor in self._tracked():
            saved = self._backup.get(name)
            if saved is not None:
                tensor.copy_(saved.to(tensor.device))
        self._backup = {}

    # Kept as an explicit alias: "apply permanently" reads very differently from
    # "apply temporarily", and conflating them has burned people before.
    apply_shadow = copy_to

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """
        Serialise EMA state.

        Emits the shadow weights twice on purpose:

        * ``shadow``       — name -> tensor, the authoritative form. Loading by
          name is immune to parameter reordering.
        * ``shadow_params``— ordered list matching ``model.parameters()``, kept
          for compatibility with loaders that zip positionally.
        """
        param_names = [n for n, p in self.model.named_parameters() if p.requires_grad]
        return {
            "shadow": {k: v.detach().cpu() for k, v in self.shadow.items()},
            "shadow_params": [
                self.shadow[n].detach().cpu() for n in param_names if n in self.shadow
            ],
            "param_names": param_names,
            "step_count": self.step_count,
            "decay": self.decay,
            "warmup": self.warmup,
            "include_buffers": self.include_buffers,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore EMA state saved by :meth:`state_dict`."""
        shadow = state.get("shadow")
        if shadow is None:
            # Tolerate checkpoints that only carried the positional form.
            names = state.get("param_names") or [
                n for n, p in self.model.named_parameters() if p.requires_grad
            ]
            params = state.get("shadow_params") or []
            shadow = dict(zip(names, params))

        self.shadow = {k: v.clone() for k, v in shadow.items()}
        self.step_count = int(state.get("step_count", 0))
        self.decay = float(state.get("decay", self.decay))
        self.warmup = int(state.get("warmup", self.warmup))

        missing = [name for name, _ in self._tracked() if name not in self.shadow]
        if missing:
            log.warning(
                "EMA checkpoint is missing %d tracked tensor(s) (e.g. %s); "
                "they will keep their current values.",
                len(missing),
                missing[0],
            )


def load_ema_weights_into(model: nn.Module, ema_state: dict | None) -> bool:
    """
    Apply EMA weights from a checkpoint onto ``model``, in place.

    This is the one function evaluation and inference should call: it understands
    every EMA payload format this project has emitted, prefers name-based
    matching, and reports honestly whether it did anything.

    Returns
    -------
    True if EMA weights were applied, False if the checkpoint carried none (in
    which case the caller should fall back to ``model_state``).
    """
    if not ema_state:
        return False

    shadow = ema_state.get("shadow")
    if isinstance(shadow, dict) and shadow:
        own = dict(model.named_parameters())
        own.update({f"__buffer__{n}": b for n, b in model.named_buffers()})
        applied = 0
        with torch.no_grad():
            for name, value in shadow.items():
                target = own.get(name)
                if target is not None and target.shape == value.shape:
                    target.copy_(value.to(target.device))
                    applied += 1
        if applied:
            log.info("Applied %d EMA tensors by name", applied)
            return True

    params = ema_state.get("shadow_params")
    if params:
        trainable = [p for p in model.parameters() if p.requires_grad]
        if len(trainable) != len(params):
            log.warning(
                "EMA shadow_params length %d != model parameter count %d; "
                "refusing to zip positionally.",
                len(params),
                len(trainable),
            )
            return False
        with torch.no_grad():
            for param, value in zip(trainable, params):
                param.copy_(value.to(param.device))
        log.info("Applied %d EMA tensors positionally", len(params))
        return True

    return False
