"""
unet.py
-------
Baseline U-Net for accelerated MRI reconstruction.


Architecture
------------
Encoder:  4 downsampling stages, feature channels double each stage (32→64→128→256→512)
Bottleneck: convolutional block at 512 channels
Decoder:  4 upsampling stages with skip connections
Output:   1×1 conv → single-channel grayscale image


Each convolutional block:
    Conv2d(3x3) → InstanceNorm2d → LeakyReLU
    Conv2d(3x3) → InstanceNorm2d → LeakyReLU
"""


import torch
import torch.nn as nn
import torch.nn.functional as F



class ConvBlock(nn.Module):
    """Two 3x3 convolutions with InstanceNorm and LeakyReLU."""


    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)



class UNet(nn.Module):
    """
    Standard U-Net for single-channel MRI reconstruction.


    Parameters
    ----------
    in_channels  : int — input channels (1 for single-coil MRI)
    out_channels : int — output channels (1 for grayscale)
    base_ch      : int — base feature channels (doubled at each encoder stage)
    n_levels     : int — number of encoder/decoder stages
    """


    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_ch: int = 32,
        n_levels: int = 4,
    ):
        super().__init__()
        self.n_levels = n_levels


        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc_blocks = nn.ModuleList()
        self.pool = nn.ModuleList()
        ch = base_ch
        prev_ch = in_channels
        for _ in range(n_levels):
            self.enc_blocks.append(ConvBlock(prev_ch, ch))
            self.pool.append(nn.MaxPool2d(2))
            prev_ch = ch
            ch *= 2


        # ── Bottleneck ────────────────────────────────────────────────────────
        self.bottleneck = ConvBlock(prev_ch, ch)
        prev_ch = ch


        # ── Decoder ──────────────────────────────────────────────────────────
        self.up_convs   = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in range(n_levels):
            out_ch = prev_ch // 2
            self.up_convs.append(
                nn.ConvTranspose2d(prev_ch, out_ch, kernel_size=2, stride=2)
            )
            # skip connection doubles the channels
            self.dec_blocks.append(ConvBlock(out_ch * 2, out_ch))
            prev_ch = out_ch


        # ── Output head ───────────────────────────────────────────────────────
        self.out_conv = nn.Conv2d(prev_ch, out_channels, kernel_size=1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []


        # Encoder
        for i in range(self.n_levels):
            x = self.enc_blocks[i](x)
            skips.append(x)
            x = self.pool[i](x)


        # Bottleneck
        x = self.bottleneck(x)


        # Decoder
        for i in range(self.n_levels):
            x = self.up_convs[i](x)
            skip = skips[self.n_levels - 1 - i]


            # Pad if spatial sizes differ (edge case for odd input sizes)
            if x.shape != skip.shape:
                x = F.pad(x, [0, skip.shape[-1] - x.shape[-1],
                               0, skip.shape[-2] - x.shape[-2]])
            x = torch.cat([x, skip], dim=1)
            x = self.dec_blocks[i](x)


        return self.out_conv(x)



def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



if __name__ == "__main__":
    model = UNet()
    dummy = torch.randn(2, 1, 320, 320)
    out = model(dummy)
    print(f"U-Net  | params: {count_parameters(model)/1e6:.1f}M | output: {out.shape}")