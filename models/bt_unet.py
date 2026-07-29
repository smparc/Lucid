"""
bt_unet.py
----------
U-Net with a Transformer Encoder at the Bottleneck (BT-UNet).

Architecture
------------
A standard U-Net encoder produces a bottleneck feature map, which is flattened
into ``H' * W'`` tokens, given positional information, processed by ``L_T``
Transformer encoder layers, reshaped, and handed to the U-Net decoder. The CNN
handles local texture; the Transformer supplies global context at the most
abstract (and cheapest) scale.

This is the middle rung of the architectural ladder: it buys global receptive
field for a modest cost, but attention only ever operates at 1/16th resolution,
so it cannot use long-range structure to resolve fine detail. That limitation is
what motivates the fully hierarchical SwinUNet.

Positional encoding
-------------------
The original implementation allocated a learned table sized from a hard-coded
``320 // 2**n_levels``, so any other input size either silently truncated the
encoding or indexed out of bounds. Here the table is 2D and **bicubically
interpolated** to whatever bottleneck grid actually arrives — the standard
recipe for transferring ViT position embeddings across resolutions — so a model
trained at 320x320 evaluates correctly at any size.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import ConvBlock, count_parameters


class PositionalEncoding2D(nn.Module):
    """
    Learned 2D positional embedding, interpolated to the incoming grid size.

    Parameters
    ----------
    grid_size
        Side length of the reference grid the table is learned at.
    d_model
        Token dimension.
    """

    def __init__(self, grid_size: int, d_model: int):
        super().__init__()
        self.grid_size = grid_size
        self.pe = nn.Parameter(torch.zeros(1, d_model, grid_size, grid_size))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """``x`` : ``(B, H*W, D)`` -> same, with position added."""
        pe = self.pe
        if (self.grid_size, self.grid_size) != (H, W):
            pe = F.interpolate(pe, size=(H, W), mode="bicubic", align_corners=False)
        pe = pe.flatten(2).transpose(1, 2)  # (1, H*W, D)
        return x + pe


class TransformerBottleneck(nn.Module):
    """
    Pre-norm Transformer encoder applied to the bottleneck's spatial tokens.

    Parameters
    ----------
    d_model
        Token dimension (equals the bottleneck channel count).
    n_heads
        Attention heads. Must divide ``d_model``.
    n_layers
        Number of encoder layers.
    mlp_mult
        Feed-forward hidden size multiplier.
    dropout
        Dropout inside the encoder layers.
    grid_size
        Reference bottleneck grid the positional table is learned at.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        n_layers: int = 4,
        mlp_mult: int = 4,
        dropout: float = 0.1,
        grid_size: int = 20,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads}). "
                f"With base_ch=B and n_levels=L the bottleneck width is B * 2**L."
            )

        self.pos_enc = PositionalEncoding2D(grid_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * mlp_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # Nested tensors are a padding optimisation for variable-length
        # sequences and are incompatible with norm_first; our token sequences
        # are dense and fixed-length, so there is nothing to gain from them.
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )
        # A final norm is required with norm_first=True: otherwise the residual
        # stream leaves the stack unnormalised and the decoder sees drifting scale.
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, C, H, W)`` -> ``(B, C, H, W)``."""
        B, C, H, W = x.shape
        tokens = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        tokens = self.pos_enc(tokens, H, W)
        tokens = self.norm(self.transformer(tokens))
        return tokens.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()


class BTUNet(nn.Module):
    """
    U-Net with a Transformer at the bottleneck.

    Parameters
    ----------
    in_channels, out_channels
        Channel counts.
    base_ch
        Base feature width; doubles at each encoder stage.
    n_levels
        Encoder/decoder depth.
    tf_heads, tf_layers, tf_dropout
        Transformer configuration.
    img_size
        Nominal input size, used only to size the positional table. The forward
        pass accepts any resolution.
    norm
        Normalisation kind for the convolutional blocks.
    residual
        Predict a residual correction to the input.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_ch: int = 32,
        n_levels: int = 4,
        tf_heads: int = 8,
        tf_layers: int = 4,
        tf_dropout: float = 0.1,
        img_size: int = 320,
        norm: str = "instance",
        residual: bool = True,
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

        # ── Encoder ────────────────────────────────────────────────────────
        self.enc_blocks = nn.ModuleList()
        self.pool = nn.ModuleList()
        ch, prev_ch = base_ch, in_channels
        for _ in range(n_levels):
            self.enc_blocks.append(ConvBlock(prev_ch, ch, norm=norm))
            self.pool.append(nn.MaxPool2d(2, ceil_mode=True))
            prev_ch, ch = ch, ch * 2

        # ── Bottleneck: convolution then Transformer ───────────────────────
        self.bottleneck_conv = ConvBlock(prev_ch, ch, norm=norm)
        btl_ch = ch
        self.transformer = TransformerBottleneck(
            d_model=btl_ch,
            n_heads=tf_heads,
            n_layers=tf_layers,
            dropout=tf_dropout,
            grid_size=max(1, img_size // (2**n_levels)),
        )
        prev_ch = btl_ch

        # ── Decoder ────────────────────────────────────────────────────────
        self.up_convs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for _ in range(n_levels):
            out_ch = prev_ch // 2
            self.up_convs.append(nn.ConvTranspose2d(prev_ch, out_ch, kernel_size=2, stride=2))
            self.dec_blocks.append(ConvBlock(out_ch * 2, out_ch, norm=norm))
            prev_ch = out_ch

        self.out_conv = nn.Conv2d(prev_ch, out_channels, kernel_size=1)
        if residual:
            nn.init.zeros_(self.out_conv.weight)
            nn.init.zeros_(self.out_conv.bias)

    @staticmethod
    def _align(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
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

        x = self.transformer(self.bottleneck_conv(x))

        for i in range(self.n_levels):
            x = self.up_convs[i](x)
            skip = skips[self.n_levels - 1 - i]
            x = self._align(x, skip)
            x = self.dec_blocks[i](torch.cat([x, skip], dim=1))

        out = self.out_conv(x)
        return identity + out if self.residual else out


if __name__ == "__main__":  # pragma: no cover
    model = BTUNet()
    dummy = torch.randn(2, 1, 320, 320)
    out = model(dummy)
    print(f"BT-UNet | params: {count_parameters(model) / 1e6:.1f}M | output: {out.shape}")
