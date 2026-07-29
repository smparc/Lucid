"""
registry.py
-----------
Model construction from configuration.

One place decides how a config block becomes a module, so the trainer, the
evaluator, the inference pipeline and the tests can never disagree about what
``model.name = "swinunet"`` means. Previously each of those four call sites
built models independently with its own copy of the default arguments, and they
had already drifted apart (``dropout`` was forced to 0 in one, ``n_levels``
defaulted differently in another).

Interface contract
------------------
Every model returned by :func:`build_model` satisfies:

* ``model.expects_kspace : bool`` — whether ``forward`` takes ``(x, k, mask)``
  or just ``(x)``. Callers branch on this attribute rather than guessing.
* ``model.in_channels : int`` — 1 for magnitude input, 2 for complex.
* Output is ``(B, 1, H, W)`` real magnitude, ready for the loss and metrics,
  unless ``output="complex"`` is requested explicitly.
"""

from __future__ import annotations

import logging
from typing import Callable

import torch
import torch.nn as nn

from models.bt_unet import BTUNet
from models.data_consistency import CascadedNet, ResidualDCWrapper
from models.fourier import complex_abs_chan
from models.swinunet import SwinUNet
from models.unet import NormUNet, UNet

log = logging.getLogger(__name__)

# name -> (class, default kwargs). Defaults live here and nowhere else.
_BACKBONES: dict[str, tuple[type[nn.Module], dict]] = {
    "unet": (UNet, dict(base_ch=32, n_levels=4, norm="instance")),
    "norm_unet": (NormUNet, dict(base_ch=32, n_levels=4, norm="instance")),
    "bt_unet": (
        BTUNet,
        dict(base_ch=32, n_levels=4, tf_heads=8, tf_layers=4, tf_dropout=0.1),
    ),
    "swinunet": (
        SwinUNet,
        dict(
            img_size=320,
            patch_size=4,
            embed_dim=64,
            ws=8,
            head_dim=8,
            n_levels=3,
            mlp_ratio=4.0,
            dropout=0.0,
            attn_dropout=0.0,
            drop_path_rate=0.1,
        ),
    ),
}

# Channel arguments differ by backbone; normalise them here.
_CHANNEL_KEYS: dict[str, tuple[str, str]] = {
    "unet": ("in_channels", "out_channels"),
    "norm_unet": ("in_channels", "out_channels"),
    "bt_unet": ("in_channels", "out_channels"),
    "swinunet": ("in_ch", "out_ch"),
}


def available_models() -> list[str]:
    """Names accepted by :func:`build_model`."""
    return sorted(_BACKBONES)


class MagnitudeAdapter(nn.Module):
    """
    Take the magnitude of a complex-valued backbone's output.

    Used when a model runs on complex data but no data-consistency step is
    configured, so that its output still matches the ``(B, 1, H, W)`` real
    target the losses and metrics expect.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return complex_abs_chan(self.model(x))


def build_backbone(
    name: str,
    params: dict | None = None,
    in_channels: int = 1,
    out_channels: int = 1,
) -> nn.Module:
    """
    Instantiate a bare backbone, without any physics wrapper.

    Unknown keys in ``params`` are rejected rather than ignored: a silently
    dropped ``embed_dim`` produces a model that trains fine and is simply not
    the model that was asked for, which is the worst kind of configuration bug.
    """
    key = (name or "").lower()
    if key not in _BACKBONES:
        raise ValueError(
            f"Unknown model {name!r}. Choose from: {', '.join(available_models())}"
        )

    cls, defaults = _BACKBONES[key]
    in_key, out_key = _CHANNEL_KEYS[key]

    kwargs = dict(defaults)
    kwargs.update({k: v for k, v in (params or {}).items() if k not in _CHANNEL_KEYS[key]})
    # Channel counts are controlled by the caller (they depend on complex mode),
    # so drop any copies that came in through params.
    for alias in ("in_ch", "out_ch", "in_channels", "out_channels"):
        kwargs.pop(alias, None)
    kwargs[in_key] = in_channels
    kwargs[out_key] = out_channels

    import inspect

    signature = inspect.signature(cls.__init__ if cls is not NormUNet else UNet.__init__)
    valid = set(signature.parameters) - {"self"}
    unknown = set(kwargs) - valid
    if unknown:
        raise ValueError(
            f"Unknown parameter(s) for model {key!r}: {sorted(unknown)}. "
            f"Accepted: {sorted(valid)}"
        )

    return cls(**kwargs)


def build_model(cfg) -> nn.Module:
    """
    Build the full model described by a config object.

    Expects ``cfg.model`` with keys::

        name              : backbone name
        complex           : bool, run on 2-channel complex data
        params            : backbone keyword arguments
        data_consistency  : {enabled, mode, n_cascades, share_weights, hard}

    ``mode`` is ``"cascade"`` for an unrolled network (denoiser -> DC, repeated)
    or ``"single"`` for one denoiser followed by one DC step.
    """
    model_cfg = cfg.model
    name = str(model_cfg.get("name", "swinunet")).lower()
    params = dict(model_cfg.get("params", {}) or {})
    dc_cfg = dict(model_cfg.get("data_consistency", {}) or {})
    dc_enabled = bool(dc_cfg.get("enabled", False))

    # Data consistency writes measured Fourier coefficients back into the
    # estimate, which is only meaningful if the estimate carries phase.
    use_complex = bool(model_cfg.get("complex", dc_enabled))
    if dc_enabled and not use_complex:
        log.warning(
            "data_consistency.enabled=true forces complex mode: writing measured "
            "k-space back into a magnitude-only image is not well defined."
        )
        use_complex = True

    channels = 2 if use_complex else 1

    def make_backbone() -> nn.Module:
        return build_backbone(name, params, in_channels=channels, out_channels=channels)

    if not dc_enabled:
        model = make_backbone()
        model = MagnitudeAdapter(model) if use_complex else model
        model.expects_kspace = False
        model.in_channels = channels
        log.info("Built %s (complex=%s, data consistency: off)", name, use_complex)
        return model

    mode = str(dc_cfg.get("mode", "cascade")).lower()
    if mode == "cascade":
        n_cascades = int(dc_cfg.get("n_cascades", 4))
        model = CascadedNet(
            denoiser_fn=make_backbone,
            n_cascades=n_cascades,
            share_weights=bool(dc_cfg.get("share_weights", False)),
            hard_dc=bool(dc_cfg.get("hard", False)),
            output="magnitude",
        )
        log.info(
            "Built %s as a %d-stage unrolled cascade (share_weights=%s, hard_dc=%s)",
            name,
            n_cascades,
            dc_cfg.get("share_weights", False),
            dc_cfg.get("hard", False),
        )
    elif mode == "single":
        model = ResidualDCWrapper(make_backbone(), use_dc=True, output="magnitude")
        log.info("Built %s with a single data-consistency step", name)
    else:
        raise ValueError(f"Unknown data_consistency.mode {mode!r}; use 'cascade' or 'single'")

    model.expects_kspace = True
    model.in_channels = channels
    return model


def forward_model(
    model: nn.Module,
    x: torch.Tensor,
    kspace: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Call ``model`` with whatever inputs it actually needs.

    Central helper so training, evaluation and inference dispatch identically.
    """
    if getattr(model, "expects_kspace", False):
        return model(x, kspace, mask)
    return model(x)


def make_denoiser_factory(name: str, params: dict, channels: int = 2) -> Callable[[], nn.Module]:
    """Return a zero-argument factory, for building cascades manually."""

    def factory() -> nn.Module:
        return build_backbone(name, params, in_channels=channels, out_channels=channels)

    return factory
