"""
swinunet.py
-----------
SwinUNet: Hierarchical Swin Transformer in a U-Net-like encoder-decoder structure
for accelerated MRI reconstruction.


Key ideas
---------
1. Patch Embedding     — divide image into non-overlapping patches, project to d_model
2. Swin Transformer    — shifted-window self-attention (linear complexity in image size)
3. Patch Merging       — halve spatial resolution, double channels (encoder downsampling)
4. Patch Expanding     — double spatial resolution, halve channels (decoder upsampling)
5. Skip Connections    — concatenate encoder outputs into decoder at each scale
6. Output Projection   — map final features back to image pixel space


Shifted Window Attention
------------------------
Instead of global self-attention (quadratic in tokens), each Swin block computes
attention within local windows of size (ws × ws). Consecutive blocks alternate between
regular and shifted window partitioning, enabling cross-window information flow
while keeping complexity linear in image size.


References
----------
Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows",
ICCV 2021.
Cao et al., "Swin-UNet: Unet-like Pure Transformer for Medical Image Segmentation",
arXiv 2021.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def window_partition(x: torch.Tensor, ws: int) -> torch.Tensor:
    """
    Partition feature map into non-overlapping windows.


    x  : (B, H, W, C)
    ws : window size


    Returns : (num_windows * B, ws, ws, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, ws, ws, C)



def window_reverse(windows: torch.Tensor, ws: int, H: int, W: int) -> torch.Tensor:
    """
    Reverse window partition back to feature map.


    windows : (num_windows * B, ws, ws, C)
    Returns : (B, H, W, C)
    """
    B_times_nW, _, _, C = windows.shape
    B = int(B_times_nW / (H * W / ws / ws))
    x = windows.view(B, H // ws, W // ws, ws, ws, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(B, H, W, C)



# ---------------------------------------------------------------------------
# Window Multi-Head Self-Attention
# ---------------------------------------------------------------------------


class WindowAttention(nn.Module):
    """
    Window-based Multi-Head Self-Attention with relative position bias.


    Parameters
    ----------
    dim      : int — input feature dimension
    ws       : int — window size
    n_heads  : int — number of attention heads
    head_dim : int — dimension per head (if None, dim // n_heads)
    dropout  : float
    """


    def __init__(
        self,
        dim: int,
        ws: int,
        n_heads: int,
        head_dim: int = None,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        super().__init__()
        self.ws = ws
        self.n_heads = n_heads
        self.head_dim = head_dim or (dim // n_heads)
        self.scale = self.head_dim ** -0.5


        inner_dim = self.n_heads * self.head_dim


        self.qkv  = nn.Linear(dim, inner_dim * 3, bias=True)
        self.proj = nn.Linear(inner_dim, dim)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)


        # Relative position bias table: (2ws-1) × (2ws-1) × n_heads
        self.rel_pos_bias_table = nn.Parameter(
            torch.zeros((2 * ws - 1) ** 2, n_heads)
        )
        nn.init.trunc_normal_(self.rel_pos_bias_table, std=0.02)


        # Precompute relative position indices
        coords = torch.arange(ws)
        grid   = torch.stack(torch.meshgrid(coords, coords, indexing="ij"))  # (2, ws, ws)
        flat   = grid.flatten(1)                                             # (2, ws²)
        rel    = flat[:, :, None] - flat[:, None, :]                        # (2, ws², ws²)
        rel    = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += ws - 1
        rel[:, :, 1] += ws - 1
        rel[:, :, 0] *= 2 * ws - 1
        self.register_buffer("rel_pos_idx", rel.sum(-1))                    # (ws², ws²)


    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        x    : (B_nW, ws², dim)
        mask : (nW, ws², ws²) or None   — attention mask for shifted windows
        """
        Bnw, N, _ = x.shape
        qkv = self.qkv(x).reshape(Bnw, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                # each (Bnw, heads, N, head_dim)


        attn = (q @ k.transpose(-2, -1)) * self.scale


        # Relative position bias
        bias = self.rel_pos_bias_table[self.rel_pos_idx.view(-1)]
        bias = bias.view(self.ws ** 2, self.ws ** 2, self.n_heads)
        bias = bias.permute(2, 0, 1).unsqueeze(0)                           # (1, heads, N, N)
        attn = attn + bias


        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(Bnw // nW, nW, self.n_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.n_heads, N, N)


        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)


        x = (attn @ v).transpose(1, 2).reshape(Bnw, N, self.n_heads * self.head_dim)
        return self.proj_drop(self.proj(x))



# ---------------------------------------------------------------------------
# Swin Transformer Block
# ---------------------------------------------------------------------------


class SwinBlock(nn.Module):
    """
    One Swin Transformer block: LayerNorm → W-MSA → residual → LayerNorm → MLP → residual.


    shift=False → regular window attention
    shift=True  → shifted window attention (offset by ws//2)
    """


    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        n_heads: int,
        ws: int = 7,
        head_dim: int = None,
        mlp_ratio: float = 4.0,
        shift: bool = False,
        attn_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.H, self.W = input_resolution
        self.ws = ws
        self.shift = shift
        self.shift_size = ws // 2 if shift else 0


        self.norm1 = nn.LayerNorm(dim)
        self.attn  = WindowAttention(
            dim, ws=ws, n_heads=n_heads, head_dim=head_dim,
            attn_dropout=attn_dropout, proj_dropout=mlp_dropout,
        )
        self.norm2 = nn.LayerNorm(dim)


        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(mlp_dropout),
        )


        # Precompute attention mask for shifted windows
        if self.shift_size > 0:
            img_mask = torch.zeros(1, self.H, self.W, 1)
            h_slices = (
                slice(0, -ws),
                slice(-ws, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -ws),
                slice(-ws, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for hs in h_slices:
                for ws_ in w_slices:
                    img_mask[:, hs, ws_, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.ws)  # (nW, ws, ws, 1)
            mask_windows = mask_windows.view(-1, self.ws * self.ws)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
        else:
            attn_mask = None


        self.register_buffer("attn_mask", attn_mask)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, H*W, C)"""
        B, L, C = x.shape
        H, W = self.H, self.W


        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)


        # Cyclic shift for shifted-window attention
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))


        # Partition into windows and apply attention
        x_windows = window_partition(x, self.ws)            # (nW*B, ws, ws, C)
        x_windows = x_windows.view(-1, self.ws * self.ws, C)
        attn_out  = self.attn(x_windows, mask=self.attn_mask)
        attn_out  = attn_out.view(-1, self.ws, self.ws, C)


        # Reverse windows
        x = window_reverse(attn_out, self.ws, H, W)         # (B, H, W, C)


        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))


        x = x.view(B, H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x



# ---------------------------------------------------------------------------
# Patch operations
# ---------------------------------------------------------------------------


class PatchEmbed(nn.Module):
    """Split image into patches and embed."""


    def __init__(self, img_size: int = 320, patch_size: int = 4,
                 in_ch: int = 1, embed_dim: int = 64):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches  = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.proj(x)                        # (B, embed_dim, H/P, W/P)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)        # (B, H*W, C)
        x = self.norm(x)
        return x, H, W



class PatchMerging(nn.Module):
    """Merge 2×2 patches to halve spatial resolution and double channels."""


    def __init__(self, dim: int, resolution: Tuple[int, int]):
        super().__init__()
        self.H, self.W = resolution
        self.norm  = nn.LayerNorm(4 * dim)
        self.linear = nn.Linear(4 * dim, 2 * dim, bias=False)


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """x : (B, H*W, C)"""
        B, L, C = x.shape
        H, W = self.H, self.W
        x = x.view(B, H, W, C)


        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x  = torch.cat([x0, x1, x2, x3], dim=-1)   # (B, H/2, W/2, 4C)
        x  = x.view(B, -1, 4 * C)
        x  = self.norm(x)
        x  = self.linear(x)
        return x, H // 2, W // 2



class PatchExpanding(nn.Module):
    """Expand patches to double spatial resolution and halve channels."""


    def __init__(self, dim: int):
        super().__init__()
        self.norm   = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, 4 * dim, bias=False)


    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        """x : (B, H*W, C)"""
        B, L, C = x.shape
        x = self.norm(x)
        x = self.linear(x)                     # (B, H*W, 4C)
        x = x.view(B, H, W, 4 * C)
        # Pixel shuffle: rearrange channels into spatial dims
        x = x.view(B, H, W, 2, 2, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H * 2, W * 2, C)
        x = x.view(B, -1, C)
        return x, H * 2, W * 2



# ---------------------------------------------------------------------------
# Swin Encoder / Decoder Layers
# ---------------------------------------------------------------------------


class SwinEncoderLayer(nn.Module):
    """One encoder stage: 2 Swin blocks (regular + shifted) then patch merging."""


    def __init__(self, dim, resolution, n_heads, ws=8, head_dim=8, dropout=0.0):
        super().__init__()
        H, W = resolution
        self.blocks = nn.ModuleList([
            SwinBlock(dim, (H, W), n_heads, ws=ws, head_dim=head_dim, shift=False, mlp_dropout=dropout),
            SwinBlock(dim, (H, W), n_heads, ws=ws, head_dim=head_dim, shift=True,  mlp_dropout=dropout),
        ])
        self.merge = PatchMerging(dim, (H, W))


    def forward(self, x, H, W):
        for blk in self.blocks:
            x = blk(x)
        skip = x
        x, H_new, W_new = self.merge(x)
        return x, H_new, W_new, skip, H, W



class SwinDecoderLayer(nn.Module):
    """One decoder stage: patch expanding then 2 Swin blocks, with skip connection."""


    def __init__(self, dim, resolution, n_heads, ws=8, head_dim=8, dropout=0.0):
        super().__init__()
        H, W = resolution
        self.expand = PatchExpanding(dim)
        # After concatenation with skip, channels = dim + dim/2 (we project back)
        self.concat_proj = nn.Linear(dim + dim // 2, dim // 2, bias=False)
        out_dim = dim // 2
        self.blocks = nn.ModuleList([
            SwinBlock(out_dim, (H * 2, W * 2), n_heads, ws=ws, head_dim=head_dim, shift=False, mlp_dropout=dropout),
            SwinBlock(out_dim, (H * 2, W * 2), n_heads, ws=ws, head_dim=head_dim, shift=True,  mlp_dropout=dropout),
        ])


    def forward(self, x, H, W, skip):
        x, H_new, W_new = self.expand(x, H, W)
        x = torch.cat([x, skip], dim=-1)
        x = self.concat_proj(x)
        for blk in self.blocks:
            x = blk(x)
        return x, H_new, W_new



# ---------------------------------------------------------------------------
# SwinUNet
# ---------------------------------------------------------------------------


class SwinUNet(nn.Module):
    """
    SwinUNet for accelerated MRI reconstruction.


    Parameters
    ----------
    img_size    : int — input image size (assumed square)
    patch_size  : int — patch size for initial embedding
    in_ch       : int — input channels
    out_ch      : int — output channels
    embed_dim   : int — base embedding dimension (doubles with each encoder stage)
    depths      : list[int] — not used here (2 blocks per stage by default)
    n_heads     : list[int] — attention heads per stage
    ws          : int — window size
    head_dim    : int — dimension per attention head
    mlp_ratio   : float — MLP expansion ratio
    dropout     : float — dropout rate
    n_levels    : int — number of encoder/decoder stages
    """


    def __init__(
        self,
        img_size: int = 320,
        patch_size: int = 4,
        in_ch: int = 1,
        out_ch: int = 1,
        embed_dim: int = 64,
        n_heads: list = None,
        ws: int = 8,
        head_dim: int = 8,
        dropout: float = 0.1,
        n_levels: int = 3,
    ):
        super().__init__()
        self.n_levels  = n_levels
        self.embed_dim = embed_dim


        if n_heads is None:
            n_heads = [max(1, embed_dim * (2 ** i) // head_dim) for i in range(n_levels + 1)]


        # ── Patch Embedding ──────────────────────────────────────────────────
        self.patch_embed = PatchEmbed(img_size, patch_size, in_ch, embed_dim)
        init_H = img_size // patch_size


        # ── Encoder ──────────────────────────────────────────────────────────
        self.encoder_layers = nn.ModuleList()
        dim = embed_dim
        res = init_H
        for i in range(n_levels):
            self.encoder_layers.append(
                SwinEncoderLayer(dim, (res, res), n_heads[i], ws=ws, head_dim=head_dim, dropout=dropout)
            )
            dim *= 2
            res //= 2


        # ── Bottleneck ────────────────────────────────────────────────────────
        self.bottleneck = nn.ModuleList([
            SwinBlock(dim, (res, res), n_heads[n_levels], ws=ws, head_dim=head_dim, shift=False, mlp_dropout=dropout),
            SwinBlock(dim, (res, res), n_heads[n_levels], ws=ws, head_dim=head_dim, shift=True,  mlp_dropout=dropout),
        ])
        self.btl_H = res
        self.btl_W = res


        # ── Decoder ──────────────────────────────────────────────────────────
        self.decoder_layers = nn.ModuleList()
        for i in range(n_levels):
            self.decoder_layers.append(
                SwinDecoderLayer(dim, (res, res), n_heads[n_levels - 1 - i],
                                 ws=ws, head_dim=head_dim, dropout=dropout)
            )
            dim //= 2
            res *= 2


        # ── Output ────────────────────────────────────────────────────────────
        self.norm = nn.LayerNorm(dim)
        # Expand patches back to pixel space
        self.output_expand = nn.Sequential(
            nn.Linear(dim, patch_size * patch_size * out_ch),
        )
        self.patch_size = patch_size
        self.out_ch     = out_ch
        self.init_H     = init_H


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H_in, W_in = x.shape


        # Patch embedding
        x, H, W = self.patch_embed(x)


        # Encoder
        skips     = []
        skip_HWs  = []
        for layer in self.encoder_layers:
            x, H, W, skip, skip_H, skip_W = layer(x, H, W)
            skips.append(skip)
            skip_HWs.append((skip_H, skip_W))


        # Bottleneck
        for blk in self.bottleneck:
            x = blk(x)


        # Decoder
        for i, layer in enumerate(self.decoder_layers):
            skip      = skips[self.n_levels - 1 - i]
            x, H, W = layer(x, H, W, skip)


        # Output projection
        x = self.norm(x)
        x = self.output_expand(x)                         # (B, H*W, P*P*out_ch)
        B_, L, _ = x.shape
        x = x.view(B_, H, W, self.patch_size, self.patch_size, self.out_ch)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B_, self.out_ch, H * self.patch_size, W * self.patch_size)


        # Ensure output matches input spatial size
        if x.shape[-2:] != (H_in, W_in):
            x = F.interpolate(x, size=(H_in, W_in), mode="bilinear", align_corners=False)


        return x



if __name__ == "__main__":
    from models.unet import count_parameters
    model = SwinUNet(img_size=320, embed_dim=64, n_levels=3, ws=8, head_dim=8)
    dummy = torch.randn(2, 1, 320, 320)
    out = model(dummy)
    print(f"SwinUNet | params: {count_parameters(model)/1e6:.1f}M | output: {out.shape}")