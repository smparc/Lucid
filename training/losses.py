"""
losses.py
---------
Advanced loss functions for MRI reconstruction.


Includes:
- L1 Loss
- SSIM Loss (differentiable)
- Perceptual Loss (VGG feature matching)
- Frequency Domain Loss (k-space MSE)
- Combined loss with configurable weights
- Charbonnier Loss (smooth L1 alternative)
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional



# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------


def gaussian_kernel(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = g.outer(g)
    return kernel.unsqueeze(0).unsqueeze(0)



class SSIMLoss(nn.Module):
    """Structural Similarity Index loss (1 - SSIM)."""


    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        kernel = gaussian_kernel(window_size, sigma)
        self.register_buffer("kernel", kernel)
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2


    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        k = self.kernel.expand(pred.shape[1], 1, -1, -1)
        pad = self.window_size // 2


        mu1 = F.conv2d(pred, k, padding=pad, groups=pred.shape[1])
        mu2 = F.conv2d(target, k, padding=pad, groups=pred.shape[1])


        mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2


        sigma1_sq = F.conv2d(pred * pred, k, padding=pad, groups=pred.shape[1]) - mu1_sq
        sigma2_sq = F.conv2d(target * target, k, padding=pad, groups=pred.shape[1]) - mu2_sq
        sigma12 = F.conv2d(pred * target, k, padding=pad, groups=pred.shape[1]) - mu1_mu2


        ssim_map = (
            (2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)
        ) / (
            (mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2)
        )
        return 1 - ssim_map.mean()



# ---------------------------------------------------------------------------
# Charbonnier Loss (smooth L1)
# ---------------------------------------------------------------------------


class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss: L(x) = sqrt(x^2 + eps^2)
    
    Smoother than L1 at zero, more robust than L2 to outliers.
    Used in many SOTA image restoration methods (SwinIR, Restormer).
    """


    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps_sq = eps ** 2


    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.sqrt(diff ** 2 + self.eps_sq).mean()



# ---------------------------------------------------------------------------
# Frequency Domain Loss
# ---------------------------------------------------------------------------


class FrequencyLoss(nn.Module):
    """
    Loss computed in the Fourier domain (k-space).
    
    Penalizes reconstruction errors in frequency space, which helps
    recover high-frequency details (edges, textures) that pixel-space
    losses often underweight.
    
    L_freq = || F(pred) - F(target) ||_1
    """


    def __init__(self, loss_type: str = "l1", focus_high_freq: bool = False):
        super().__init__()
        self.loss_type = loss_type
        self.focus_high_freq = focus_high_freq


    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Compute 2D FFT
        pred_fft = torch.fft.fft2(pred, norm="ortho")
        target_fft = torch.fft.fft2(target, norm="ortho")


        # Compute magnitude difference
        if self.loss_type == "l1":
            diff = torch.abs(pred_fft - target_fft)
        else:
            diff = (pred_fft - target_fft).abs() ** 2


        # Optionally weight high frequencies more
        if self.focus_high_freq:
            B, C, H, W = pred.shape
            # Create frequency weight map (higher weight for high frequencies)
            freq_weight = self._get_freq_weight(H, W, pred.device)
            diff = diff * freq_weight


        return diff.mean()


    @staticmethod
    def _get_freq_weight(H: int, W: int, device: torch.device) -> torch.Tensor:
        """Create a weight map that emphasizes high frequencies."""
        cy, cx = H // 2, W // 2
        y = torch.arange(H, device=device).float() - cy
        x = torch.arange(W, device=device).float() - cx
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        # Normalized distance from center (0=DC, 1=Nyquist)
        dist = torch.sqrt(yy ** 2 + xx ** 2) / max(cy, cx)
        # Weight: 1 + dist (low freq=1x, high freq=2x)
        weight = 1.0 + dist.clamp(0, 1)
        return weight.unsqueeze(0).unsqueeze(0)



# ---------------------------------------------------------------------------
# Perceptual Loss (VGG-based)
# ---------------------------------------------------------------------------


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 features.
    
    Compares feature representations at multiple layers rather than
    raw pixels. Helps produce perceptually sharper reconstructions.
    
    Note: VGG expects 3-channel input, so grayscale is repeated 3x.
    """


    def __init__(self, layers: list = None, weights: list = None):
        super().__init__()
        self.layers = layers or [3, 8, 15, 22]  # relu1_2, relu2_2, relu3_3, relu4_3
        self.weights = weights or [1.0, 1.0, 1.0, 1.0]


        # Import here to avoid slow import at module level
        from torchvision.models import vgg16, VGG16_Weights
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features


        # Freeze VGG
        for param in vgg.parameters():
            param.requires_grad = False


        # Split into sub-networks at each desired layer
        self.blocks = nn.ModuleList()
        prev_layer = 0
        for layer_idx in self.layers:
            self.blocks.append(nn.Sequential(*list(vgg.children())[prev_layer:layer_idx + 1]))
            prev_layer = layer_idx + 1


        # ImageNet normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))


    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Convert grayscale to 3-channel and normalize for VGG."""
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return (x - self.mean) / self.std


    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = self._preprocess(pred)
        target = self._preprocess(target)


        loss = 0.0
        x, y = pred, target
        for block, weight in zip(self.blocks, self.weights):
            x = block(x)
            y = block(y)
            loss += weight * F.l1_loss(x, y)


        return loss



# ---------------------------------------------------------------------------
# Edge Loss (Sobel-based)
# ---------------------------------------------------------------------------


class EdgeLoss(nn.Module):
    """
    Sobel edge loss for preserving structural boundaries.
    
    Critical for medical imaging where edge definition determines
    diagnostic quality (cartilage boundaries, fractures, etc.).
    """


    def __init__(self):
        super().__init__()
        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))


    def _edges(self, x: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)


    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self._edges(pred), self._edges(target))



# ---------------------------------------------------------------------------
# Combined Loss
# ---------------------------------------------------------------------------


class CombinedLoss(nn.Module):
    """
    Flexible combined loss with configurable components and weights.


    Default: L1 + SSIM (as in paper)
    Advanced: L1 + SSIM + Frequency + Edge + Perceptual
    """


    def __init__(
        self,
        l1_weight: float = 0.7,
        ssim_weight: float = 0.3,
        freq_weight: float = 0.0,
        edge_weight: float = 0.0,
        perceptual_weight: float = 0.0,
        charbonnier: bool = False,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.freq_weight = freq_weight
        self.edge_weight = edge_weight
        self.perceptual_weight = perceptual_weight


        self.l1 = CharbonnierLoss() if charbonnier else nn.L1Loss()
        self.ssim = SSIMLoss()


        if freq_weight > 0:
            self.freq = FrequencyLoss(focus_high_freq=True)
        if edge_weight > 0:
            self.edge = EdgeLoss()
        if perceptual_weight > 0:
            self.perceptual = PerceptualLoss()


    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0


        if self.l1_weight > 0:
            loss += self.l1_weight * self.l1(pred, target)
        if self.ssim_weight > 0:
            loss += self.ssim_weight * self.ssim(pred, target)
        if self.freq_weight > 0:
            loss += self.freq_weight * self.freq(pred, target)
        if self.edge_weight > 0:
            loss += self.edge_weight * self.edge(pred, target)
        if self.perceptual_weight > 0:
            loss += self.perceptual_weight * self.perceptual(pred, target)


        return loss