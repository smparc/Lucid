"""
unet.py
-------
Baseline U-Net for accelerated MRI reconstruction.

Architecture
------------
Encoder   : ``n_levels`` downsampling stages, channels double each stage
            (32 -> 64 -> 128 -> 256 -> 512 at the default settings).
Bottleneck: a convolutional block at the deepest width.
Decoder   : ``n_levels`` upsampling stages, each concatenating the encoder skip.
Output    : 1x1 convolution to ``out_channels``.

Each convolutional block is ``Conv3x3 -> Norm -> LeakyReLU`` twice.

Notes
-----
* **Global residual.** The network predicts a correction to its input rather
  than the image itself. The zero-filled reconstruction is already close to the
  target, so predicting the (small, structured) aliasing artefact is a much
  better-conditioned problem than predicting the (large, high-dynamic-range)
  image. The output convolution is zero-initialised so the model starts as an
  exact identity.
* **Skip alignment.** Odd input sizes make the pooled resolution round down, so
  the decoder can come back one pixel short. Sizes are reconciled by
  interpolation against the skip rather than by one-sided padding, which would
  shift the image by half a pixel per level.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import ConvBlock, count_parameters


class UNet(nn.Module):
    """
    Standard U-Net for MRI reconstruction.

    Parameters
    ----------
    in_channels
        Input channels (1 for magnitude, 2 for complex real/imag).
    out_channels
        Output channels.
    base_ch
        Feature width of the first encoder stage; doubles each stage.
    n_levels
        Number of encoder/decoder stages.
    norm
        Normalisation kind: ``instance`` (default, matches the original
        baseline), ``group``, ``batch`` or ``none``.
    dropout
        Channel dropout inside each convolutional block.
    residual
        Predict a residual correction to the input. Requires
        ``in_channels == out_channels``.
    residual_blocks
        Use residual convolutional blocks internally.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_ch: int = 32,
        n_levels: int = 4,
        norm: str = "instance",
        dropout: float = 0.0,
        residual: bool = True,
        residual_blocks: bool = False,
    ):
        super().__init__()
        if n_levels < 1:
            raise ValueError(f"n_levels must be >= 1, got {n_levels}")
        if residual and in_channels != out_channels:
            raise ValueError(
                f"residual=True requires in_channels == out_channels, "
                f"got {in_channels} and {out_channels}"
            )

        self.n_levels = n_levels
        self.residual = residual

        block_kwargs = dict(norm=norm, dropout=dropout, residual=residual_blocks)

        # ── Encoder ────────────────────────────────────────────────────────
        self.enc_blocks = nn.ModuleList()
        self.pool = nn.ModuleList()
        ch, prev_ch = base_ch, in_channels
        for _ in range(n_levels):
            self.enc_blocks.append(ConvBlock(prev_ch, ch, **block_kwargs))
            self.pool.append(nn.MaxPool2d(2, ceil_mode=True))
            prev_ch, ch = ch, ch * 2

        # ── Bottleneck ─────────────────────────────────────────────────────
        self.bottleneck = ConvBlock(prev_ch, ch, **block_kwargs)
        prev_ch = ch

        # ── Decoder ────────────────────────────────────────────────────────
        self.up_convs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for _ in range(n_levels):
            out_ch = prev_ch // 2
            self.up_convs.append(nn.ConvTranspose2d(prev_ch, out_ch, kernel_size=2, stride=2))
            self.dec_blocks.append(ConvBlock(out_ch * 2, out_ch, **block_kwargs))
            prev_ch = out_ch

        # ── Output head ────────────────────────────────────────────────────
        self.out_conv = nn.Conv2d(prev_ch, out_channels, kernel_size=1)
        if residual:
            nn.init.zeros_(self.out_conv.weight)
            nn.init.zeros_(self.out_conv.bias)

    @staticmethod
    def _align(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Resize ``x`` to match ``ref``'s spatial size, if they differ."""
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        skips = []

        for i in range(self.n_levels):
            x = self.enc_blocks[i](x)
            skips.append(x)
            x = self.pool[i](x)

        x = self.bottleneck(x)

        for i in range(self.n_levels):
            x = self.up_convs[i](x)
            skip = skips[self.n_levels - 1 - i]
            x = self._align(x, skip)
            x = self.dec_blocks[i](torch.cat([x, skip], dim=1))

        out = self.out_conv(x)
        return identity + out if self.residual else out


class NormUNet(nn.Module):
    """
    U-Net wrapped in per-instance normalisation, as used by the fastMRI baselines.

    Each input is standardised by its own mean and standard deviation before the
    network sees it, and the statistics are re-applied to the output. MRI
    intensities have no absolute physical scale and vary by orders of magnitude
    between scans, so without this the network must waste capacity learning to
    be scale-equivariant. This wrapper makes that equivariance exact by
    construction.
    """

    def __init__(self, **unet_kwargs):
        super().__init__()
        self.unet = UNet(**unet_kwargs)

    @staticmethod
    def _norm(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c = x.shape[:2]
        flat = x.reshape(b, c, -1)
        mean = flat.mean(dim=-1).view(b, c, 1, 1)
        std = flat.std(dim=-1).view(b, c, 1, 1).clamp_min(1e-6)
        return (x - mean) / std, mean, std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_n, mean, std = self._norm(x)
        return self.unet(x_n) * std + mean


if __name__ == "__main__":  # pragma: no cover
    model = UNet()
    dummy = torch.randn(2, 1, 320, 320)
    out = model(dummy)
    print(f"U-Net  | params: {count_parameters(model) / 1e6:.1f}M | output: {out.shape}")
