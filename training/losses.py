"""
losses.py
---------
Loss functions for MRI reconstruction.

Components
----------
* **L1 / Charbonnier** — pixel fidelity. Charbonnier is a smooth L1 used by most
  modern restoration models (SwinIR, Restormer).
* **SSIM** — structural fidelity. This is the metric fastMRI ranks on, so
  optimising it directly is well motivated.
* **Frequency** — error in k-space, which weights high spatial frequencies far
  more heavily than a pixel loss does.
* **Edge (Sobel)** — boundary sharpness, the property radiologists actually read.
* **Perceptual (VGG)** — deep feature matching, for perceptual sharpness.

Weighting
---------
:class:`CombinedLoss` reports every active component separately through
:attr:`CombinedLoss.last_components`, so the trainer can log them. Watching a
combined scalar fall while one component silently diverges is a common and
easily avoided way to waste a training run.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.metrics import gaussian_kernel

__all__ = [
    "CharbonnierLoss",
    "CombinedLoss",
    "EdgeLoss",
    "FrequencyLoss",
    "PerceptualLoss",
    "SSIMLoss",
    "gaussian_kernel",
]


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------


class SSIMLoss(nn.Module):
    """
    Structural similarity loss, ``1 - SSIM``.

    Parameters
    ----------
    window_size, sigma
        Gaussian window geometry.
    data_range
        Dynamic range of the reference. ``None`` (the default) derives it per
        image from the target's maximum, which is correct under this project's
        per-slice scaling; passing a fixed value reproduces the classic
        constant-range formulation.
    """

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        data_range: float | None = None,
    ):
        super().__init__()
        self.window_size = window_size
        self.data_range = data_range
        self.register_buffer("kernel", gaussian_kernel(window_size, sigma), persistent=False)
        self.C1 = 0.01**2
        self.C2 = 0.03**2

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        data_range: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        C = pred.shape[1]
        k = self.kernel.to(dtype=pred.dtype, device=pred.device).expand(C, 1, -1, -1)
        pad = self.window_size // 2

        rng = data_range if data_range is not None else self.data_range
        if rng is None:
            rng = target.reshape(target.shape[0], -1).amax(dim=1).view(-1, 1, 1, 1)
        elif isinstance(rng, torch.Tensor):
            rng = rng.reshape(-1, 1, 1, 1).to(pred.device, pred.dtype)
        else:
            rng = torch.as_tensor(float(rng), device=pred.device, dtype=pred.dtype)
        rng = torch.clamp(rng, min=1e-8)

        c1 = (0.01 * rng) ** 2
        c2 = (0.03 * rng) ** 2

        def blur(x: torch.Tensor) -> torch.Tensor:
            return F.conv2d(x, k, padding=pad, groups=C)

        mu1, mu2 = blur(pred), blur(target)
        mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2

        sigma1_sq = (blur(pred * pred) - mu1_sq).clamp_min(0)
        sigma2_sq = (blur(target * target) - mu2_sq).clamp_min(0)
        sigma12 = blur(pred * target) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        )
        return 1 - ssim_map.mean()


# ---------------------------------------------------------------------------
# Charbonnier
# ---------------------------------------------------------------------------


class CharbonnierLoss(nn.Module):
    """
    Charbonnier loss, ``sqrt((pred - target)^2 + eps^2)``.

    Smoother than L1 at the origin (so gradients do not chatter near
    convergence) and far more robust than L2 to the bright outliers that fat and
    fluid produce in knee MRI.
    """

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps_sq = eps**2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((pred - target) ** 2 + self.eps_sq).mean()


# ---------------------------------------------------------------------------
# Frequency domain
# ---------------------------------------------------------------------------


class FrequencyLoss(nn.Module):
    """
    Error measured in the Fourier domain.

    Pixel losses are dominated by the low frequencies that carry most of an MR
    image's energy, so a network can minimise them while leaving fine texture
    unresolved. Penalising the k-space residual directly re-weights the problem
    toward the high frequencies that undersampling actually destroyed.

    Parameters
    ----------
    loss_type
        ``l1`` or ``l2`` on the complex residual.
    focus_high_freq
        Multiply the residual by a radial ramp so Nyquist-adjacent errors count
        roughly twice as much as DC.
    """

    def __init__(self, loss_type: str = "l1", focus_high_freq: bool = True):
        super().__init__()
        if loss_type not in ("l1", "l2"):
            raise ValueError(f"loss_type must be 'l1' or 'l2', got {loss_type!r}")
        self.loss_type = loss_type
        self.focus_high_freq = focus_high_freq
        self._weight_cache: dict[tuple[int, int], torch.Tensor] = {}

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # fft has no half-precision kernel and k-space has enormous dynamic
        # range, so this must run outside autocast.
        with torch.autocast(device_type=pred.device.type, enabled=False):
            pred_fft = torch.fft.fft2(pred.float(), norm="ortho")
            target_fft = torch.fft.fft2(target.float(), norm="ortho")

            residual = pred_fft - target_fft
            diff = residual.abs() if self.loss_type == "l1" else residual.abs() ** 2

            if self.focus_high_freq:
                H, W = pred.shape[-2:]
                diff = diff * self._freq_weight(H, W, pred.device)

            return diff.mean()

    def _freq_weight(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Radial ramp from 1 at DC to 2 at the corners; cached per shape."""
        key = (H, W)
        cached = self._weight_cache.get(key)
        if cached is not None and cached.device == device:
            return cached

        # torch.fft.fft2 leaves DC at index 0, so build the weight in unshifted
        # coordinates rather than centring it — a mismatch here would invert the
        # intended emphasis and down-weight exactly the frequencies we care about.
        fy = torch.fft.fftfreq(H, device=device).abs() * 2  # -> [0, 1]
        fx = torch.fft.fftfreq(W, device=device).abs() * 2
        dist = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2) / (2**0.5)
        weight = (1.0 + dist.clamp(0, 1)).unsqueeze(0).unsqueeze(0)

        self._weight_cache[key] = weight
        return weight


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------


class EdgeLoss(nn.Module):
    """
    Sobel gradient-magnitude loss.

    Edge definition is what determines whether a reconstruction is diagnostically
    useful — cartilage boundaries, meniscal tears, cortical margins — and it is
    exactly what an L1 loss is happy to blur away.
    """

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1.0, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = torch.tensor([[-1.0, -2, -1], [0, 0, 0], [1, 2, 1]])
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3), persistent=False)

    def _edges(self, x: torch.Tensor) -> torch.Tensor:
        C = x.shape[1]
        kx = self.sobel_x.to(x.dtype).expand(C, 1, -1, -1)
        ky = self.sobel_y.to(x.dtype).expand(C, 1, -1, -1)
        gx = F.conv2d(x, kx, padding=1, groups=C)
        gy = F.conv2d(x, ky, padding=1, groups=C)
        return torch.sqrt(gx**2 + gy**2 + 1e-8)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self._edges(pred), self._edges(target))


# ---------------------------------------------------------------------------
# Perceptual
# ---------------------------------------------------------------------------


class PerceptualLoss(nn.Module):
    """
    VGG16 feature-matching loss.

    Grayscale input is replicated to three channels and ImageNet-normalised.
    Note the domain gap: VGG features are trained on natural images, so this is
    a heuristic prior for MRI, not a principled one. It sharpens texture but can
    hallucinate it, which is why the default weight is zero.

    Parameters
    ----------
    layers
        Indices into ``vgg16.features`` at which to compare.
    weights
        Per-layer weights.
    resize
        Downsample inputs larger than 320 px before the VGG pass, to bound cost.
    """

    def __init__(
        self,
        layers: list[int] | None = None,
        weights: list[float] | None = None,
        resize: bool = True,
    ):
        super().__init__()
        self.layers = layers or [3, 8, 15, 22]  # relu1_2, relu2_2, relu3_3, relu4_3
        self.weights = weights or [1.0] * len(self.layers)
        self.resize = resize

        from torchvision.models import VGG16_Weights, vgg16

        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.eval()
        for param in vgg.parameters():
            param.requires_grad = False

        children = list(vgg.children())
        self.blocks = nn.ModuleList()
        prev = 0
        for idx in self.layers:
            self.blocks.append(nn.Sequential(*children[prev : idx + 1]))
            prev = idx + 1

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        """Keep VGG frozen in eval mode regardless of the parent's state."""
        super().train(mode)
        for block in self.blocks:
            block.eval()
        return self

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if self.resize and max(x.shape[-2:]) > 320:
            x = F.interpolate(x, size=(320, 320), mode="bilinear", align_corners=False)
        # VGG expects roughly [0, 1] inputs; reconstructions can exceed that.
        return (x.clamp(0, 1) - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x, y = self._preprocess(pred), self._preprocess(target)
        loss = pred.new_zeros(())
        for block, weight in zip(self.blocks, self.weights):
            x = block(x)
            with torch.no_grad():
                y = block(y)
            loss = loss + weight * F.l1_loss(x, y)
        return loss


# ---------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------


class CombinedLoss(nn.Module):
    """
    Weighted sum of the components above.

    Default is ``0.7 * L1 + 0.3 * SSIM``, matching the original study. Setting
    any other weight above zero activates that component; components with zero
    weight are never constructed, so the VGG weights are not downloaded unless
    a perceptual term is actually requested.

    Parameters
    ----------
    l1_weight, ssim_weight, freq_weight, edge_weight, perceptual_weight
        Component weights.
    charbonnier
        Use Charbonnier in place of L1.
    lambda1, lambda2
        Deprecated aliases for ``l1_weight`` and ``ssim_weight``.
    """

    def __init__(
        self,
        l1_weight: float = 0.7,
        ssim_weight: float = 0.3,
        freq_weight: float = 0.0,
        edge_weight: float = 0.0,
        perceptual_weight: float = 0.0,
        charbonnier: bool = False,
        lambda1: float | None = None,
        lambda2: float | None = None,
    ):
        super().__init__()
        if lambda1 is not None:
            l1_weight = lambda1
        if lambda2 is not None:
            ssim_weight = lambda2

        weights = {
            "l1": float(l1_weight),
            "ssim": float(ssim_weight),
            "freq": float(freq_weight),
            "edge": float(edge_weight),
            "perceptual": float(perceptual_weight),
        }
        if all(w <= 0 for w in weights.values()):
            raise ValueError("CombinedLoss needs at least one positive component weight")
        if any(w < 0 for w in weights.values()):
            raise ValueError(f"Component weights must be non-negative, got {weights}")

        self.weights = weights
        self.last_components: dict[str, float] = {}

        self.l1 = (CharbonnierLoss() if charbonnier else nn.L1Loss()) if weights["l1"] > 0 else None
        self.ssim = SSIMLoss() if weights["ssim"] > 0 else None
        self.freq = FrequencyLoss(focus_high_freq=True) if weights["freq"] > 0 else None
        self.edge = EdgeLoss() if weights["edge"] > 0 else None
        self.perceptual = PerceptualLoss() if weights["perceptual"] > 0 else None

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        data_range: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        total = pred.new_zeros(())
        components: dict[str, float] = {}

        if self.l1 is not None:
            value = self.l1(pred, target)
            components["l1"] = float(value.detach())
            total = total + self.weights["l1"] * value

        if self.ssim is not None:
            value = self.ssim(pred, target, data_range=data_range)
            components["ssim"] = float(value.detach())
            total = total + self.weights["ssim"] * value

        if self.freq is not None:
            value = self.freq(pred, target)
            components["freq"] = float(value.detach())
            total = total + self.weights["freq"] * value

        if self.edge is not None:
            value = self.edge(pred, target)
            components["edge"] = float(value.detach())
            total = total + self.weights["edge"] * value

        if self.perceptual is not None:
            value = self.perceptual(pred, target)
            components["perceptual"] = float(value.detach())
            total = total + self.weights["perceptual"] * value

        self.last_components = components
        return total

    def extra_repr(self) -> str:
        active = {k: v for k, v in self.weights.items() if v > 0}
        return ", ".join(f"{k}={v:g}" for k, v in active.items())
