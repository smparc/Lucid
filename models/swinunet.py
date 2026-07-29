"""
swinunet.py
-----------
SwinUNet: a hierarchical Swin Transformer arranged as a U-Net encoder-decoder,
for accelerated MRI reconstruction.

Pipeline
--------
1. **Patch Embedding**   — non-overlapping patches projected to ``embed_dim``.
2. **Swin blocks**       — shifted-window self-attention, linear in image area.
3. **Patch Merging**     — 2x downsample, 2x channels (encoder).
4. **Patch Expanding**   — 2x upsample, halve channels (decoder).
5. **Skip Connections**  — encoder features concatenated into the decoder.
6. **Output Projection**  — features mapped back to pixel space.
7. **Global Residual**   — the network predicts a *correction* to the zero-filled
   input rather than the image itself.

What changed relative to the original implementation
----------------------------------------------------
The previous version could not be constructed at its own documented
configuration. With ``img_size=320, patch_size=4, n_levels=3, ws=8`` the stage
resolutions are 80 -> 40 -> 20, and ``window_partition`` reshapes with
``x.view(B, H // ws, ws, ...)``, which requires ``H`` to be an exact multiple of
``ws``. At the third stage 20 % 8 != 0 and construction raised
``RuntimeError: shape '[1, 2, 8, 2, 8, 1]' is invalid for input of size 400``.
Because the failure occurred in ``__init__``, no forward pass was ever possible.

The fixes, and the upgrades they enabled:

* **Windows are padded, not assumed.** ``window_partition`` pads to a multiple of
  the window size and the attention mask marks padded tokens invalid, so *any*
  resolution works and the relative-position-bias table keeps a single fixed
  shape.
* **Resolution is a forward-time argument, not a constructor constant.** Nothing
  bakes 320 into a buffer, so one trained model runs at any input size.
* **Attention masks are computed lazily and cached per resolution**, instead of
  being precomputed for a resolution that may never be seen.
* **Stochastic depth (DropPath)** on residual branches, with the standard
  linearly increasing schedule across depth.
* **Truncated-normal initialisation** — a pre-LN transformer trained from
  scratch is genuinely sensitive to this.
* **Scaled dot-product attention** for fused/memory-efficient kernels.
* **Optional gradient checkpointing** to trade compute for activation memory.

References
----------
Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted
Windows", ICCV 2021.
Cao et al., "Swin-Unet: Unet-like Pure Transformer for Medical Image
Segmentation", ECCVW 2022.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from models.layers import DropPath, count_parameters, init_transformer_weights

# ---------------------------------------------------------------------------
# Window helpers
# ---------------------------------------------------------------------------


def window_partition(x: torch.Tensor, ws: int) -> torch.Tensor:
    """
    Partition an ``(B, H, W, C)`` feature map into non-overlapping windows.

    ``H`` and ``W`` must already be multiples of ``ws`` — callers use
    :func:`pad_for_windows` first. Returns ``(B * nW, ws, ws, C)``.
    """
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C)


def window_reverse(windows: torch.Tensor, ws: int, H: int, W: int) -> torch.Tensor:
    """Inverse of :func:`window_partition`. Returns ``(B, H, W, C)``."""
    C = windows.shape[-1]
    n_windows = (H // ws) * (W // ws)
    B = windows.shape[0] // n_windows  # integer division; no float round-trip
    x = windows.view(B, H // ws, W // ws, ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)


def pad_for_windows(x: torch.Tensor, ws: int) -> tuple[torch.Tensor, int, int]:
    """
    Zero-pad ``(B, H, W, C)`` on the bottom/right so both dims are multiples of ``ws``.

    Returns ``(padded, pad_h, pad_w)``. The padded tokens are excluded from
    attention by the mask built in :meth:`SwinBlock._build_attn_mask`, so their
    value is irrelevant and zeros are the cheapest choice.
    """
    _, H, W, _ = x.shape
    pad_h = (ws - H % ws) % ws
    pad_w = (ws - W % ws) % ws
    if pad_h or pad_w:
        # F.pad on a NHWC tensor pads from the last dim backwards:
        # (C_left, C_right, W_left, W_right, H_left, H_right)
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    return x, pad_h, pad_w


# ---------------------------------------------------------------------------
# Window Multi-Head Self-Attention
# ---------------------------------------------------------------------------


class WindowAttention(nn.Module):
    """
    Window-based multi-head self-attention with relative position bias.

    Parameters
    ----------
    dim
        Input/output feature dimension.
    ws
        Window side length. Attention is computed over ``ws * ws`` tokens.
    n_heads
        Number of attention heads.
    head_dim
        Dimension per head. Defaults to ``dim // n_heads``. Decoupling it from
        ``dim`` lets width and head count be tuned independently.
    attn_dropout, proj_dropout
        Dropout on attention weights and on the output projection.
    use_sdpa
        Use ``F.scaled_dot_product_attention`` (fused/flash kernels where
        available) instead of an explicit softmax. Numerically equivalent.
    """

    def __init__(
        self,
        dim: int,
        ws: int,
        n_heads: int,
        head_dim: int | None = None,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        use_sdpa: bool = True,
    ):
        super().__init__()
        self.ws = ws
        self.n_heads = n_heads
        self.head_dim = head_dim or max(1, dim // n_heads)
        self.scale = self.head_dim**-0.5
        self.attn_dropout = attn_dropout
        self.use_sdpa = use_sdpa

        inner_dim = self.n_heads * self.head_dim
        self.qkv = nn.Linear(dim, inner_dim * 3, bias=True)
        self.proj = nn.Linear(inner_dim, dim)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)

        # Relative position bias: one learned scalar per (head, relative offset).
        # A window has (2*ws - 1)^2 distinct offsets in 2D.
        self.rel_pos_bias_table = nn.Parameter(torch.zeros((2 * ws - 1) ** 2, n_heads))
        nn.init.trunc_normal_(self.rel_pos_bias_table, std=0.02)
        self.register_buffer("rel_pos_idx", self._build_rel_pos_index(ws), persistent=False)

    @staticmethod
    def _build_rel_pos_index(ws: int) -> torch.Tensor:
        """Map each (query, key) token pair to an index into the bias table."""
        coords = torch.arange(ws)
        grid = torch.stack(torch.meshgrid(coords, coords, indexing="ij"))  # (2, ws, ws)
        flat = grid.flatten(1)  # (2, ws^2)
        rel = flat[:, :, None] - flat[:, None, :]  # (2, ws^2, ws^2)
        rel = rel.permute(1, 2, 0).contiguous()  # (ws^2, ws^2, 2)
        rel[:, :, 0] += ws - 1  # shift to non-negative
        rel[:, :, 1] += ws - 1
        rel[:, :, 0] *= 2 * ws - 1  # row-major flatten
        return rel.sum(-1)  # (ws^2, ws^2)

    def _bias(self) -> torch.Tensor:
        """Relative position bias as ``(1, heads, N, N)``."""
        bias = self.rel_pos_bias_table[self.rel_pos_idx.view(-1)]
        bias = bias.view(self.ws**2, self.ws**2, self.n_heads)
        return bias.permute(2, 0, 1).unsqueeze(0).contiguous()

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            ``(B * nW, ws^2, dim)`` window tokens.
        mask
            ``(nW, ws^2, ws^2)`` additive mask (0 to attend, -inf-ish to block),
            or None for unmasked regular windows.
        """
        Bnw, N, _ = x.shape
        qkv = self.qkv(x).reshape(Bnw, N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each (Bnw, heads, N, hd)

        attn_bias = self._bias()
        if mask is not None:
            nW = mask.shape[0]
            # Broadcast the per-window mask across the batch and heads, then add
            # it to the shared positional bias so a single tensor carries both.
            m = mask.unsqueeze(1)  # (nW, 1, N, N)
            m = m.repeat(Bnw // nW, 1, 1, 1)  # (Bnw, 1, N, N)
            attn_bias = attn_bias + m

        if self.use_sdpa:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_bias,
                dropout_p=self.attn_dropout if self.training else 0.0,
                scale=self.scale,
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale + attn_bias
            attn = self.attn_drop(attn.softmax(dim=-1))
            out = attn @ v

        out = out.transpose(1, 2).reshape(Bnw, N, self.n_heads * self.head_dim)
        return self.proj_drop(self.proj(out))


# ---------------------------------------------------------------------------
# Swin Transformer block
# ---------------------------------------------------------------------------


class SwinBlock(nn.Module):
    """
    One Swin block: ``x + DropPath(WMSA(LN(x)))`` then ``x + DropPath(MLP(LN(x)))``.

    ``shift=True`` offsets the window grid by ``ws // 2`` so that information
    crosses window boundaries; consecutive blocks alternate.

    The block is **resolution-agnostic**: ``(H, W)`` arrive with the input and the
    shift mask is built on demand and cached. This is what allows one trained
    model to run on any input size, and it is why nothing here can be
    mis-specified at construction time.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        ws: int = 8,
        head_dim: int | None = None,
        mlp_ratio: float = 4.0,
        shift: bool = False,
        attn_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        use_sdpa: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.ws = ws
        self.shift_size = ws // 2 if shift else 0

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim,
            ws=ws,
            n_heads=n_heads,
            head_dim=head_dim,
            attn_dropout=attn_dropout,
            proj_dropout=mlp_dropout,
            use_sdpa=use_sdpa,
        )
        self.norm2 = nn.LayerNorm(dim)

        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(mlp_dropout),
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

        # Attention masks depend only on the padded resolution, so they are
        # cached rather than recomputed every forward pass.
        self._mask_cache: dict[tuple[int, int, int], torch.Tensor] = {}

    def _build_attn_mask(
        self, Hp: int, Wp: int, pad_h: int, pad_w: int, device: torch.device
    ) -> torch.Tensor | None:
        """
        Build the additive attention mask for a padded ``(Hp, Wp)`` feature map.

        Two effects are folded into one mask:

        1. **Cyclic-shift regions.** After ``torch.roll`` a window can contain
           tokens that are not spatially adjacent (they wrapped around the
           image). Those pairs must not attend to each other.
        2. **Padding.** Tokens added by :func:`pad_for_windows` carry no signal
           and must be invisible to real tokens.

        Both are expressed by painting a region id per pixel and blocking any
        pair whose ids differ.
        """
        if self.shift_size == 0 and pad_h == 0 and pad_w == 0:
            return None

        key = (Hp, Wp, self.shift_size)
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached.to(device)

        img_mask = torch.zeros(1, Hp, Wp, 1)
        if self.shift_size > 0:
            slices = (
                slice(0, -self.ws),
                slice(-self.ws, -self.shift_size),
                slice(-self.shift_size, None),
            )
            region = 0
            for hs in slices:
                for wsl in slices:
                    img_mask[:, hs, wsl, :] = region
                    region += 1
        else:
            region = 1

        # Give every padded pixel its own region so real tokens never see them.
        if pad_h:
            img_mask[:, Hp - pad_h :, :, :] = region
            region += 1
        if pad_w:
            img_mask[:, :, Wp - pad_w :, :] = region

        mask_windows = window_partition(img_mask, self.ws).view(-1, self.ws * self.ws)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        # -100 rather than -inf: softmax(-inf) over an all-masked row yields NaN,
        # which can happen for a fully padded window. -100 makes those rows
        # uniform-but-harmless instead of poisoning the graph with NaNs.
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(
            attn_mask == 0, 0.0
        )

        self._mask_cache[key] = attn_mask
        return attn_mask.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` : ``(B, H, W, C)`` -> ``(B, H, W, C)``."""
        B, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)

        x, pad_h, pad_w = pad_for_windows(x, self.ws)
        Hp, Wp = x.shape[1], x.shape[2]

        # A single window spans the whole map, so shifting cannot expose any new
        # neighbours; skipping it avoids pointless masked-out attention.
        shift = self.shift_size if (Hp > self.ws and Wp > self.ws) else 0
        if shift > 0:
            x = torch.roll(x, shifts=(-shift, -shift), dims=(1, 2))

        saved_shift, self.shift_size = self.shift_size, shift
        attn_mask = self._build_attn_mask(Hp, Wp, pad_h, pad_w, x.device)
        self.shift_size = saved_shift

        x_windows = window_partition(x, self.ws).view(-1, self.ws * self.ws, C)
        attn_out = self.attn(x_windows, mask=attn_mask).view(-1, self.ws, self.ws, C)
        x = window_reverse(attn_out, self.ws, Hp, Wp)

        if shift > 0:
            x = torch.roll(x, shifts=(shift, shift), dims=(1, 2))

        if pad_h or pad_w:
            x = x[:, :H, :W, :].contiguous()

        x = shortcut + self.drop_path(x)
        return x + self.drop_path(self.mlp(self.norm2(x)))


# ---------------------------------------------------------------------------
# Patch operations
# ---------------------------------------------------------------------------


class PatchEmbed(nn.Module):
    """Split the image into ``patch_size`` patches and project to ``embed_dim``."""

    def __init__(self, patch_size: int = 4, in_ch: int = 1, embed_dim: int = 64):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, C, H, W)`` -> ``(B, H/P, W/P, embed_dim)``."""
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
        return self.norm(x)


class PatchMerging(nn.Module):
    """Merge each 2x2 neighbourhood: halve resolution, double channels."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.linear = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, H, W, C)`` -> ``(B, ceil(H/2), ceil(W/2), 2C)``."""
        B, H, W, C = x.shape
        # Pad odd dimensions so the 2x2 stride-2 gather is well defined.
        if H % 2 or W % 2:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x = torch.cat(
            [
                x[:, 0::2, 0::2, :],
                x[:, 1::2, 0::2, :],
                x[:, 0::2, 1::2, :],
                x[:, 1::2, 1::2, :],
            ],
            dim=-1,
        )
        return self.linear(self.norm(x))


class PatchExpanding(nn.Module):
    """
    Double the resolution via pixel shuffle, keeping the channel count.

    ``Linear(C -> 4C)`` then a ``(2, 2)`` spatial unfold. Channel count is
    unchanged; the decoder's concat-projection is what halves it, after the skip
    connection has been fused in.
    """

    def __init__(self, dim: int, scale: int = 2):
        super().__init__()
        self.scale = scale
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, scale * scale * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, H, W, C)`` -> ``(B, H*scale, W*scale, C)``."""
        B, H, W, C = x.shape
        s = self.scale
        x = self.linear(self.norm(x))  # (B, H, W, s*s*C)
        x = x.view(B, H, W, s, s, C).permute(0, 1, 3, 2, 4, 5).contiguous()
        return x.view(B, H * s, W * s, C)


# ---------------------------------------------------------------------------
# Encoder / decoder stages
# ---------------------------------------------------------------------------


class SwinStage(nn.Module):
    """A run of Swin blocks alternating regular and shifted windows."""

    def __init__(
        self,
        dim: int,
        depth: int,
        n_heads: int,
        ws: int = 8,
        head_dim: int | None = None,
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        drop_path_rates: list[float] | None = None,
        use_sdpa: bool = True,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        rates = drop_path_rates or [0.0] * depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            [
                SwinBlock(
                    dim,
                    n_heads=n_heads,
                    ws=ws,
                    head_dim=head_dim,
                    mlp_ratio=mlp_ratio,
                    shift=(i % 2 == 1),
                    attn_dropout=attn_dropout,
                    mlp_dropout=mlp_dropout,
                    drop_path_rate=rates[i],
                    use_sdpa=use_sdpa,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return x


class SwinDecoderStage(nn.Module):
    """Patch-expand, fuse the encoder skip, then run Swin blocks at ``dim // 2``."""

    def __init__(
        self,
        dim: int,
        depth: int,
        n_heads: int,
        ws: int = 8,
        head_dim: int | None = None,
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        drop_path_rates: list[float] | None = None,
        use_sdpa: bool = True,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        out_dim = dim // 2
        self.expand = PatchExpanding(dim)
        # After expanding we hold `dim` channels; the skip contributes `dim // 2`.
        self.concat_proj = nn.Linear(dim + out_dim, out_dim, bias=False)
        self.blocks = SwinStage(
            out_dim,
            depth=depth,
            n_heads=n_heads,
            ws=ws,
            head_dim=head_dim,
            mlp_ratio=mlp_ratio,
            attn_dropout=attn_dropout,
            mlp_dropout=mlp_dropout,
            drop_path_rates=drop_path_rates,
            use_sdpa=use_sdpa,
            use_checkpoint=use_checkpoint,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)
        # Odd input sizes make PatchMerging round up, so the expanded map can be
        # one pixel larger than its skip. Crop to the skip, which carries the
        # authoritative resolution for this scale.
        x = x[:, : skip.shape[1], : skip.shape[2], :]
        x = self.concat_proj(torch.cat([x, skip], dim=-1))
        return self.blocks(x)


# ---------------------------------------------------------------------------
# SwinUNet
# ---------------------------------------------------------------------------


class SwinUNet(nn.Module):
    """
    SwinUNet for accelerated MRI reconstruction.

    Parameters
    ----------
    img_size
        Nominal input size. Retained for configuration bookkeeping and ONNX
        tracing only — the forward pass accepts any size.
    patch_size
        Patch embedding stride.
    in_ch, out_ch
        Input/output channels. Use ``in_ch=2, out_ch=2`` for complex
        (real, imaginary) reconstruction, ``1`` for magnitude-only.
    embed_dim
        Channel width after patch embedding; doubles at each encoder stage.
    depths
        Blocks per encoder stage. Defaults to 2 per stage.
    n_heads
        Heads per stage. Defaults to ``dim // head_dim`` at each scale.
    ws
        Window side length.
    head_dim
        Channels per attention head.
    mlp_ratio
        MLP expansion inside each block.
    dropout, attn_dropout
        Dropout on MLP/projection and on attention weights.
    drop_path_rate
        Maximum stochastic-depth rate; ramped linearly across depth.
    n_levels
        Number of encoder (and decoder) stages.
    residual
        Predict a residual correction to the input rather than the image. For
        MRI reconstruction the zero-filled input is already a decent estimate,
        so learning the artefact pattern is a far easier target than learning
        the image. Requires ``in_ch == out_ch``.
    use_checkpoint
        Gradient checkpointing on the Swin stages.
    use_sdpa
        Use fused scaled-dot-product attention.
    """

    def __init__(
        self,
        img_size: int = 320,
        patch_size: int = 4,
        in_ch: int = 1,
        out_ch: int = 1,
        embed_dim: int = 64,
        depths: list[int] | None = None,
        n_heads: list[int] | None = None,
        ws: int = 8,
        head_dim: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path_rate: float = 0.1,
        n_levels: int = 3,
        residual: bool = True,
        use_checkpoint: bool = False,
        use_sdpa: bool = True,
    ):
        super().__init__()
        if n_levels < 1:
            raise ValueError(f"n_levels must be >= 1, got {n_levels}")
        if residual and in_ch != out_ch:
            raise ValueError(
                f"residual=True requires in_ch == out_ch, got {in_ch} and {out_ch}"
            )

        self.img_size = img_size
        self.patch_size = patch_size
        self.n_levels = n_levels
        self.embed_dim = embed_dim
        self.out_ch = out_ch
        self.residual = residual

        depths = list(depths) if depths else [2] * (n_levels + 1)
        if len(depths) < n_levels + 1:
            depths = depths + [depths[-1]] * (n_levels + 1 - len(depths))

        dims = [embed_dim * (2**i) for i in range(n_levels + 1)]
        if n_heads is None:
            n_heads = [max(1, d // head_dim) for d in dims]
        elif len(n_heads) < n_levels + 1:
            n_heads = list(n_heads) + [n_heads[-1]] * (n_levels + 1 - len(n_heads))

        # Linearly increasing stochastic depth: shallow blocks are kept almost
        # always, deep blocks are dropped most often.
        total_blocks = sum(depths[: n_levels + 1])
        dpr = torch.linspace(0, drop_path_rate, max(1, total_blocks)).tolist()
        cursor = 0

        self.patch_embed = PatchEmbed(patch_size, in_ch, embed_dim)

        common = dict(
            ws=ws,
            head_dim=head_dim,
            mlp_ratio=mlp_ratio,
            attn_dropout=attn_dropout,
            mlp_dropout=dropout,
            use_sdpa=use_sdpa,
            use_checkpoint=use_checkpoint,
        )

        # ── Encoder ────────────────────────────────────────────────────────
        self.encoder_stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(n_levels):
            self.encoder_stages.append(
                SwinStage(
                    dims[i],
                    depth=depths[i],
                    n_heads=n_heads[i],
                    drop_path_rates=dpr[cursor : cursor + depths[i]],
                    **common,
                )
            )
            cursor += depths[i]
            self.downsamples.append(PatchMerging(dims[i]))

        # ── Bottleneck ─────────────────────────────────────────────────────
        self.bottleneck = SwinStage(
            dims[n_levels],
            depth=depths[n_levels],
            n_heads=n_heads[n_levels],
            drop_path_rates=dpr[cursor : cursor + depths[n_levels]],
            **common,
        )

        # ── Decoder ────────────────────────────────────────────────────────
        self.decoder_stages = nn.ModuleList()
        for i in range(n_levels):
            level = n_levels - i
            self.decoder_stages.append(
                SwinDecoderStage(
                    dims[level],
                    depth=depths[level - 1],
                    n_heads=n_heads[level - 1],
                    drop_path_rates=dpr[: depths[level - 1]],
                    **common,
                )
            )

        # ── Output head ────────────────────────────────────────────────────
        self.norm = nn.LayerNorm(embed_dim)
        self.up_to_pixels = PatchExpanding(embed_dim, scale=patch_size)
        # A 3x3 convolution after the transformer suppresses the blocking
        # artefacts a purely patch-wise projection tends to leave behind.
        self.head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(embed_dim // 2, out_ch, kernel_size=3, padding=1),
        )

        self.apply(init_transformer_weights)
        # Zero-init the final projection so a residual model starts as the exact
        # identity: at step 0 it reproduces the zero-filled input rather than
        # noise, which removes the initial loss spike entirely.
        if residual:
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, in_ch, H, W)`` -> ``(B, out_ch, H, W)``."""
        identity = x
        _, _, H_in, W_in = x.shape

        # Patch embedding needs the image to divide evenly into patches; each of
        # the n_levels merges then halves the resolution.
        multiple = self.patch_size * (2**self.n_levels)
        pad_h = (multiple - H_in % multiple) % multiple
        pad_w = (multiple - W_in % multiple) % multiple
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        x = self.patch_embed(x)  # (B, H', W', C)

        skips = []
        for stage, down in zip(self.encoder_stages, self.downsamples, strict=True):
            x = stage(x)
            skips.append(x)
            x = down(x)

        x = self.bottleneck(x)

        for i, stage in enumerate(self.decoder_stages):
            x = stage(x, skips[self.n_levels - 1 - i])

        x = self.norm(x)
        x = self.up_to_pixels(x)  # (B, H, W, embed_dim) at full resolution
        x = x.permute(0, 3, 1, 2).contiguous()  # NHWC -> NCHW
        x = self.head(x)

        if pad_h or pad_w:
            x = x[..., :H_in, :W_in]

        return identity + x if self.residual else x

    @torch.no_grad()
    def flops_estimate(self, H: int = 320, W: int = 320) -> float:
        """
        Rough forward-pass FLOP estimate (multiply-accumulates counted as 2).

        Useful for the efficiency table in the paper; approximate by design —
        it counts attention and MLP matmuls and ignores norms and activations.
        """
        total = 0.0
        h, w = H // self.patch_size, W // self.patch_size
        for i, stage in enumerate(self.encoder_stages):
            dim = self.embed_dim * (2**i)
            n_tok = h * w
            for blk in stage.blocks:
                ws2 = blk.ws**2
                total += 2 * n_tok * dim * (3 * dim)  # qkv
                total += 2 * n_tok * ws2 * dim * 2  # attention matmuls
                total += 2 * n_tok * dim * dim  # output projection
                total += 2 * n_tok * dim * int(dim * 4) * 2  # MLP
            h, w = (h + 1) // 2, (w + 1) // 2
        return total * 2  # encoder + decoder are near-symmetric


if __name__ == "__main__":  # pragma: no cover
    model = SwinUNet(img_size=320, embed_dim=64, n_levels=3, ws=8, head_dim=8)
    dummy = torch.randn(2, 1, 320, 320)
    out = model(dummy)
    print(f"SwinUNet | params: {count_parameters(model) / 1e6:.1f}M | output: {out.shape}")
