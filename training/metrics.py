"""
metrics.py
----------
Reconstruction quality metrics, computed the way the fastMRI benchmark defines
them.

Three defects in the original implementation made the reported numbers
incomparable to published results, and in one case incomparable to *themselves*
across configurations:

1. **PSNR was computed over the whole batch at once.** ``F.mse_loss(pred, target)``
   averages the squared error across every image in the batch before taking the
   log, which yields ``10 log10(1 / mean_b MSE_b)`` rather than the correct
   ``mean_b 10 log10(1 / MSE_b)``. Because the logarithm is concave, the batch
   form is systematically *lower* than the true mean PSNR, and — worse — the gap
   depends on batch size. Two models trained at different batch sizes could not
   be compared, which matters here because the U-Net baseline used batch 8 and
   the SwinUNet used batch 6.

2. **The data range was hard-coded to 1.0.** PSNR is defined against the dynamic
   range of the reference. After per-slice scaling the target maximum is not
   1.0, so a fixed denominator silently changes the metric per sample. The
   dataset now ships ``max_value`` for exactly this purpose.

3. **SSIM allocated a fresh module on every call** — including re-creating the
   Gaussian kernel and moving it to the device — inside the validation loop.
   Correct, but it made validation several times slower than it needed to be.

Additionally, metrics here always run in float32 even under autocast. A PSNR
above ~40 dB cannot be represented meaningfully by an fp16 MSE, so evaluating
metrics inside an autocast region quietly caps the number you can report.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


def gaussian_kernel(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Normalised 2D Gaussian window of shape ``(1, 1, size, size)``."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return g.outer(g).unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Per-image metrics
# ---------------------------------------------------------------------------


def _data_range(target: torch.Tensor, data_range: torch.Tensor | float | None) -> torch.Tensor:
    """Resolve the per-image dynamic range as a ``(B, 1, 1, 1)`` tensor."""
    B = target.shape[0]
    if data_range is None:
        flat = target.reshape(B, -1)
        rng = flat.amax(dim=1)
    elif isinstance(data_range, torch.Tensor):
        rng = data_range.reshape(-1).to(target.device, torch.float32)
        if rng.numel() == 1:
            rng = rng.expand(B)
    else:
        rng = torch.full((B,), float(data_range), device=target.device)
    return rng.clamp_min(1e-8).view(B, 1, 1, 1)


def psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: torch.Tensor | float | None = None,
    reduce: bool = True,
) -> torch.Tensor:
    """
    Peak signal-to-noise ratio, computed **per image** and then averaged.

    Parameters
    ----------
    pred, target
        ``(B, C, H, W)`` tensors.
    data_range
        Peak signal value. Scalar, per-image tensor, or None to use each
        target's own maximum.
    reduce
        Return the batch mean rather than the per-image vector.

    Returns
    -------
    Scalar mean PSNR in dB, or a ``(B,)`` vector when ``reduce=False``.
    """
    pred = pred.float()
    target = target.float()
    B = pred.shape[0]

    mse = ((pred - target) ** 2).reshape(B, -1).mean(dim=1)
    rng = _data_range(target, data_range).reshape(B)

    # Clamp instead of branching on mse == 0: identical images would give +inf,
    # which poisons any subsequent mean. 1e-12 corresponds to ~120 dB, far above
    # anything achievable, so it never affects a real measurement.
    values = 10.0 * torch.log10(rng**2 / mse.clamp_min(1e-12))
    return values.mean() if reduce else values


def nmse(pred: torch.Tensor, target: torch.Tensor, reduce: bool = True) -> torch.Tensor:
    """
    Normalised mean squared error, ``||pred - target||^2 / ||target||^2``.

    Reported by the fastMRI leaderboard and scale-invariant, so it is the metric
    least sensitive to the normalisation choices this pipeline makes.
    """
    pred = pred.float()
    target = target.float()
    B = pred.shape[0]
    num = ((pred - target) ** 2).reshape(B, -1).sum(dim=1)
    den = (target**2).reshape(B, -1).sum(dim=1).clamp_min(1e-12)
    values = num / den
    return values.mean() if reduce else values


class SSIM(torch.nn.Module):
    """
    Structural similarity, computed per image with a Gaussian window.

    Instantiate once and reuse: the kernel is a registered buffer, so it moves
    with ``.to(device)`` and is never rebuilt.

    Parameters
    ----------
    window_size, sigma
        Gaussian window geometry. ``11`` / ``1.5`` are the values from Wang et
        al. and the ones fastMRI uses.
    k1, k2
        Stabilising constants.
    """

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
    ):
        super().__init__()
        self.window_size = window_size
        self.k1 = k1
        self.k2 = k2
        self.register_buffer("kernel", gaussian_kernel(window_size, sigma), persistent=False)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        data_range: torch.Tensor | float | None = None,
        reduce: bool = True,
    ) -> torch.Tensor:
        pred = pred.float()
        target = target.float()
        B, C = pred.shape[:2]

        rng = _data_range(target, data_range)
        c1 = (self.k1 * rng) ** 2
        c2 = (self.k2 * rng) ** 2

        k = self.kernel.to(pred.dtype).expand(C, 1, -1, -1)
        pad = self.window_size // 2

        def blur(x: torch.Tensor) -> torch.Tensor:
            return F.conv2d(x, k, padding=pad, groups=C)

        mu1, mu2 = blur(pred), blur(target)
        mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2

        # Clamp the variances: the "E[x^2] - E[x]^2" identity is numerically
        # unstable and can go slightly negative on near-constant regions such as
        # image background, which would produce NaNs downstream.
        sigma1_sq = (blur(pred * pred) - mu1_sq).clamp_min(0)
        sigma2_sq = (blur(target * target) - mu2_sq).clamp_min(0)
        sigma12 = blur(pred * target) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        )
        values = ssim_map.reshape(B, -1).mean(dim=1)
        return values.mean() if reduce else values


# A shared instance for callers that just want a number. Metric modules are
# stateless apart from the kernel buffer, so this is safe to share.
_DEFAULT_SSIM: dict[torch.device, SSIM] = {}


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: torch.Tensor | float | None = None,
    reduce: bool = True,
) -> torch.Tensor:
    """Functional SSIM, backed by a per-device cached module."""
    device = pred.device
    module = _DEFAULT_SSIM.get(device)
    if module is None:
        module = SSIM().to(device)
        _DEFAULT_SSIM[device] = module
    return module(pred, target, data_range=data_range, reduce=reduce)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass
class MetricAccumulator:
    """
    Collect per-image metrics across a whole evaluation pass.

    Retaining every per-image value, rather than a running mean, is what makes
    confidence intervals and paired significance tests possible later — see
    :mod:`training.stats`. It costs a few kilobytes.
    """

    psnr: list[float] = field(default_factory=list)
    ssim: list[float] = field(default_factory=list)
    nmse: list[float] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    fnames: list[str] = field(default_factory=list)
    accelerations: list[int] = field(default_factory=list)

    @torch.no_grad()
    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        data_range: torch.Tensor | float | None = None,
        loss: float | None = None,
        fnames: list[str] | None = None,
        accelerations: torch.Tensor | None = None,
    ) -> None:
        """Add one batch. ``pred`` and ``target`` are ``(B, 1, H, W)``."""
        pred = pred.float()
        target = target.float()

        self.psnr.extend(psnr(pred, target, data_range, reduce=False).tolist())
        self.ssim.extend(ssim(pred, target, data_range, reduce=False).tolist())
        self.nmse.extend(nmse(pred, target, reduce=False).tolist())

        if loss is not None:
            self.loss.append(float(loss))
        if fnames:
            self.fnames.extend(fnames)
        if accelerations is not None:
            self.accelerations.extend(accelerations.reshape(-1).tolist())

    def compute(self) -> dict[str, float]:
        """Mean of each metric, plus the sample count."""
        import statistics

        def mean(xs: list[float]) -> float:
            return float(statistics.fmean(xs)) if xs else float("nan")

        out = {
            "psnr": mean(self.psnr),
            "ssim": mean(self.ssim),
            "nmse": mean(self.nmse),
            "n": len(self.psnr),
        }
        if self.loss:
            out["loss"] = mean(self.loss)
        if len(self.psnr) > 1:
            out["psnr_std"] = float(statistics.stdev(self.psnr))
            out["ssim_std"] = float(statistics.stdev(self.ssim))
        return out

    def by_acceleration(self) -> dict[int, dict[str, float]]:
        """
        Break the metrics down per acceleration factor.

        Essential when training across multiple accelerations: a single averaged
        number hides whether a model is uniformly good or excellent at R=4 and
        useless at R=8.
        """
        import statistics
        from collections import defaultdict

        groups: dict[int, dict[str, list[float]]] = defaultdict(
            lambda: {"psnr": [], "ssim": [], "nmse": []}
        )
        for i, acc in enumerate(self.accelerations):
            groups[int(acc)]["psnr"].append(self.psnr[i])
            groups[int(acc)]["ssim"].append(self.ssim[i])
            groups[int(acc)]["nmse"].append(self.nmse[i])

        return {
            acc: {
                "psnr": float(statistics.fmean(v["psnr"])),
                "ssim": float(statistics.fmean(v["ssim"])),
                "nmse": float(statistics.fmean(v["nmse"])),
                "n": len(v["psnr"]),
            }
            for acc, v in sorted(groups.items())
        }

    def worst(self, k: int = 5, metric: str = "psnr") -> list[tuple[str, float]]:
        """
        The ``k`` worst-scoring samples, with filenames.

        Failure analysis is where reconstruction models actually get improved;
        an average hides the slices a clinician would object to.
        """
        values = getattr(self, metric)
        if not self.fnames or len(self.fnames) != len(values):
            return []
        order = sorted(range(len(values)), key=lambda i: values[i])
        return [(self.fnames[i], values[i]) for i in order[:k]]


def evaluate_batch(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: torch.Tensor | float | None = None,
) -> dict[str, float]:
    """Convenience one-shot evaluation of a single batch."""
    return {
        "psnr": float(psnr(pred, target, data_range)),
        "ssim": float(ssim(pred, target, data_range)),
        "nmse": float(nmse(pred, target)),
    }
