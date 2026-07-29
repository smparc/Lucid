"""
layers.py
---------
Shared building blocks used across the reconstruction architectures.

Everything here is deliberately small and dependency-free so the models stay
readable, and so each component can be unit-tested in isolation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Stochastic depth
# ---------------------------------------------------------------------------


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Per-sample stochastic depth on residual branches.

    Drops the *whole residual branch* for a random subset of the batch, so the
    network is trained as an implicit ensemble of varying depths. This is the
    regularizer that makes deep Swin stacks trainable; plain dropout inside the
    MLP is a much weaker substitute and is what the original code relied on.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    # Shape broadcasts over every non-batch dimension.
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        mask.div_(keep_prob)  # keep the expectation unchanged
    return x * mask


class DropPath(nn.Module):
    """Module wrapper around :func:`drop_path`."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.3f}"


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def build_norm(kind: str, channels: int, groups: int = 8) -> nn.Module:
    """
    Build a 2D normalisation layer by name.

    ``instance`` matches the original U-Net baseline. ``group`` is generally the
    better choice for reconstruction: it is batch-size independent (important at
    the batch sizes a 320x320 transformer permits) while still sharing statistics
    across channels, which InstanceNorm does not.
    """
    kind = (kind or "instance").lower()
    if kind == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    if kind == "group":
        return nn.GroupNorm(num_groups=min(groups, channels), num_channels=channels)
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"Unknown norm '{kind}'. Choose from: instance, group, batch, none.")


# ---------------------------------------------------------------------------
# Convolutional blocks
# ---------------------------------------------------------------------------


class ConvBlock(nn.Module):
    """
    Two 3x3 convolutions with normalisation and LeakyReLU.

    Parameters
    ----------
    in_ch, out_ch
        Channel counts.
    norm
        Normalisation kind, see :func:`build_norm`.
    dropout
        2D (channel-wise) dropout applied between the two convolutions.
    residual
        Add a projected identity path. Residual conv blocks train noticeably
        more stably at depth and cost one 1x1 convolution.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        norm: str = "instance",
        dropout: float = 0.0,
        residual: bool = False,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            build_norm(norm, out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            build_norm(norm, out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.skip = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
            if residual and in_ch != out_ch
            else (nn.Identity() if residual else None)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.skip is not None:
            out = out + self.skip(x)
        return out


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def init_transformer_weights(module: nn.Module) -> None:
    """
    Truncated-normal initialisation for transformer submodules.

    Applied via ``model.apply(init_transformer_weights)``. PyTorch's default
    ``nn.Linear`` init is Kaiming-uniform tuned for ReLU MLPs; on a
    pre-LayerNorm transformer it produces activation variance large enough to
    saturate the attention softmax at initialisation. ``std=0.02`` is the value
    used by ViT, Swin and every derivative, and it matters more here than in
    classification because there is no pretraining to recover from a bad start.
    """
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="leaky_relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# ---------------------------------------------------------------------------
# Padding helpers
# ---------------------------------------------------------------------------


def pad_to_multiple(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """
    Reflection-pad a ``(B, C, H, W)`` tensor so H and W are multiples of ``multiple``.

    Returns the padded tensor and the ``(pad_h, pad_w)`` amounts so the caller
    can crop back. Reflection padding is used rather than zeros because a hard
    zero border injects a step edge, and edges are exactly what a
    high-frequency-sensitive reconstruction loss will chase.
    """
    h, w = x.shape[-2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h or pad_w:
        # Reflection padding requires the pad to be smaller than the dimension.
        mode = "reflect" if (pad_h < h and pad_w < w) else "constant"
        x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
    return x, (pad_h, pad_w)


def unpad(x: torch.Tensor, pads: tuple[int, int]) -> torch.Tensor:
    """Invert :func:`pad_to_multiple`."""
    pad_h, pad_w = pads
    if pad_h:
        x = x[..., : x.shape[-2] - pad_h, :]
    if pad_w:
        x = x[..., :, : x.shape[-1] - pad_w]
    return x


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
