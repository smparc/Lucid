"""
bt_unet.py
----------
U-Net with a Transformer Encoder at the Bottleneck (BT-UNet).


Architecture
------------
Standard U-Net encoder → flatten bottleneck features into tokens →
add learnable positional embeddings → Multi-Head Self-Attention + FFN
(repeated L_T times) → reshape back → U-Net decoder.


The transformer at the bottleneck provides global context modeling while
the CNN handles local feature extraction.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import math


from models.unet import ConvBlock



class PositionalEncoding1D(nn.Module):
    """Learnable 1D positional embedding added to token sequences."""


    def __init__(self, n_tokens: int, d_model: int):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, n_tokens, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        return x + self.pe[:, :x.shape[1], :]



class TransformerBottleneck(nn.Module):
    """
    Standard Transformer Encoder applied to spatial tokens from the bottleneck.


    Parameters
    ----------
    d_model  : int — token feature dimension (= bottleneck channels)
    n_heads  : int — number of attention heads
    n_layers : int — number of transformer encoder layers
    mlp_mult : int — MLP hidden dim = d_model * mlp_mult
    dropout  : float
    max_tokens : int — max spatial tokens (H' * W')
    """


    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        n_layers: int = 4,
        mlp_mult: int = 4,
        dropout: float = 0.1,
        max_tokens: int = 400,
    ):
        super().__init__()
        self.pos_enc = PositionalEncoding1D(max_tokens, d_model)


        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * mlp_mult,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H', W')
        B, C, H, W = x.shape
        # Tokenize: flatten spatial dims → (B, H*W, C)
        tokens = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        tokens = self.pos_enc(tokens)
        tokens = self.transformer(tokens)
        # Reshape back: (B, C, H', W')
        out = tokens.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return out



class BTUNet(nn.Module):
    """
    U-Net with Transformer at Bottleneck.


    Parameters
    ----------
    in_channels  : int
    out_channels : int
    base_ch      : int — base feature channels
    n_levels     : int — encoder/decoder depth
    tf_heads     : int — transformer attention heads
    tf_layers    : int — transformer encoder layers
    tf_dropout   : float
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
    ):
        super().__init__()
        self.n_levels = n_levels


        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc_blocks = nn.ModuleList()
        self.pool       = nn.ModuleList()
        ch = base_ch
        prev_ch = in_channels
        for _ in range(n_levels):
            self.enc_blocks.append(ConvBlock(prev_ch, ch))
            self.pool.append(nn.MaxPool2d(2))
            prev_ch = ch
            ch *= 2


        # ── Bottleneck CNN ────────────────────────────────────────────────────
        self.bottleneck_conv = ConvBlock(prev_ch, ch)
        btl_ch = ch


        # ── Transformer ───────────────────────────────────────────────────────
        # Bottleneck spatial size after n_levels poolings: 320 / 2^4 = 20
        bottleneck_h = 320 // (2 ** n_levels)
        max_tokens = bottleneck_h * bottleneck_h
        self.transformer = TransformerBottleneck(
            d_model=btl_ch,
            n_heads=tf_heads,
            n_layers=tf_layers,
            dropout=tf_dropout,
            max_tokens=max_tokens,
        )
        prev_ch = btl_ch


        # ── Decoder ──────────────────────────────────────────────────────────
        self.up_convs   = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for _ in range(n_levels):
            out_ch = prev_ch // 2
            self.up_convs.append(
                nn.ConvTranspose2d(prev_ch, out_ch, kernel_size=2, stride=2)
            )
            self.dec_blocks.append(ConvBlock(out_ch * 2, out_ch))
            prev_ch = out_ch


        self.out_conv = nn.Conv2d(prev_ch, out_channels, kernel_size=1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []


        # Encoder
        for i in range(self.n_levels):
            x = self.enc_blocks[i](x)
            skips.append(x)
            x = self.pool[i](x)


        # Bottleneck CNN → Transformer
        x = self.bottleneck_conv(x)
        x = self.transformer(x)


        # Decoder
        for i in range(self.n_levels):
            x = self.up_convs[i](x)
            skip = skips[self.n_levels - 1 - i]
            if x.shape != skip.shape:
                x = F.pad(x, [0, skip.shape[-1] - x.shape[-1],
                               0, skip.shape[-2] - x.shape[-2]])
            x = torch.cat([x, skip], dim=1)
            x = self.dec_blocks[i](x)


        return self.out_conv(x)



if __name__ == "__main__":
    from models.unet import count_parameters
    model = BTUNet()
    dummy = torch.randn(2, 1, 320, 320)
    out = model(dummy)
    print(f"BT-UNet | params: {count_parameters(model)/1e6:.1f}M | output: {out.shape}")