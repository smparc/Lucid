"""
visualizations.py
-----------------
Publication-quality figures for reconstruction analysis.

Contents
--------
* Reconstruction comparison panels (input / prediction / target / error / k-space).
* Training curves.
* k-space sampling diagrams.
* Architecture comparison charts.
* Swin attention maps and attention rollout.

Attention capture
-----------------
The previous ``AttentionExtractor`` registered forward hooks and then looked for
attention weights in the module's *output*::

    if isinstance(output, tuple) and len(output) >= 2:   # never true
        ...
    elif hasattr(module, "_attention_weights"):          # never set

``WindowAttention.forward`` returns a single tensor and never stored its weights,
so the hook matched nothing, ``get_attention_maps()`` always returned an empty
list, and every plotting function short-circuited on "No attention maps
captured". The extractor here instead switches the attention modules out of
their fused path and recomputes the softmax explicitly, so the maps are real —
and it restores the fast path afterwards.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)

__all__ = [
    "AttentionExtractor",
    "plot_architecture_comparison",
    "plot_attention_maps",
    "plot_attention_rollout",
    "plot_kspace_analysis",
    "plot_reconstruction_comparison",
    "plot_training_curves",
]


def _save(fig, save_path: str | None, dpi: int = 200) -> str | None:
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return save_path


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu().numpy()
    return np.asarray(x).squeeze()


# ---------------------------------------------------------------------------
# Reconstruction comparison
# ---------------------------------------------------------------------------


def plot_reconstruction_comparison(
    input_img,
    prediction,
    ground_truth,
    psnr_val: float | None = None,
    ssim_val: float | None = None,
    save_path: str | None = None,
    show_error: bool = True,
    show_frequency: bool = True,
    error_scale: float = 0.25,
) -> str | None:
    """
    A publication panel: input, prediction, ground truth, error, spectral error.

    Parameters
    ----------
    error_scale
        Upper limit of the error colour map as a fraction of the target's
        dynamic range. Autoscaling each error map to its own maximum makes
        panels from different models incomparable — a good reconstruction and a
        bad one both render as a full-range heatmap. A fixed fraction keeps the
        colour scale meaningful across figures.
    """
    input_img = _to_numpy(input_img)
    prediction = _to_numpy(prediction)
    ground_truth = _to_numpy(ground_truth)

    n_cols = 3 + int(show_error) + int(show_frequency)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4.2))

    vmax = float(ground_truth.max()) or 1.0

    axes[0].imshow(input_img, cmap="gray", vmin=0, vmax=vmax)
    axes[0].set_title("Input\n(zero-filled)", fontsize=10)

    title = "Prediction"
    if psnr_val is not None and ssim_val is not None:
        title += f"\nPSNR {psnr_val:.2f} dB | SSIM {ssim_val:.4f}"
    axes[1].imshow(prediction, cmap="gray", vmin=0, vmax=vmax)
    axes[1].set_title(title, fontsize=10)

    axes[2].imshow(ground_truth, cmap="gray", vmin=0, vmax=vmax)
    axes[2].set_title("Ground truth", fontsize=10)

    col = 3
    if show_error:
        error = np.abs(prediction - ground_truth)
        im = axes[col].imshow(error, cmap="inferno", vmin=0, vmax=vmax * error_scale)
        axes[col].set_title(f"|Error|\n(MAE {error.mean():.4f})", fontsize=10)
        plt.colorbar(im, ax=axes[col], fraction=0.046, pad=0.04)
        col += 1

    if show_frequency:
        pred_fft = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(prediction))))
        gt_fft = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(ground_truth))))
        im = axes[col].imshow(np.abs(pred_fft - gt_fft), cmap="viridis")
        axes[col].set_title("Spectral error\n(log magnitude)", fontsize=10)
        plt.colorbar(im, ax=axes[col], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    return _save(fig, save_path)


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------


def plot_training_curves(
    history: dict,
    save_path: str | None = None,
    title: str = "Training progress",
) -> str | None:
    """Loss, PSNR, SSIM and LR panels from a run's history dict."""

    def series(*names: str):
        for name in names:
            if history.get(name):
                return history[name]
        return None

    train_loss = series("epoch/train_loss", "train_loss")
    val_loss = series("epoch/val_loss", "val_loss")
    val_psnr = series("epoch/val_psnr", "val_psnr")
    val_ssim = series("epoch/val_ssim", "val_ssim")
    lr = series("epoch/lr", "lr")
    epochs = history.get("epoch") or list(range(1, len(train_loss or []) + 1))

    panels = []
    if train_loss or val_loss:
        panels.append(
            ("Loss", [(train_loss, "Train", "C0"), (val_loss, "Validation", "C3")], None)
        )
    if val_psnr:
        panels.append(("PSNR (dB)", [(val_psnr, "Validation", "C2")], "max"))
    if val_ssim:
        panels.append(("SSIM", [(val_ssim, "Validation", "C4")], "max"))
    if lr:
        panels.append(("Learning rate", [(lr, "LR", "k")], "log"))

    if not panels:
        raise ValueError("History contains none of the expected metric series")

    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 4), squeeze=False)
    for ax, (name, curves, style) in zip(axes[0], panels, strict=True):
        for values, label, colour in curves:
            if values:
                ax.plot(epochs[: len(values)], values, color=colour, label=label, linewidth=1.6)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
        if style == "log":
            ax.set_yscale("log")
        elif style == "max" and curves[0][0]:
            best = max(curves[0][0])
            ax.axhline(best, color="grey", linestyle="--", alpha=0.6)
            name = f"{name} (best {best:.4g})"
        ax.set_title(name)
        if len(curves) > 1:
            ax.legend(frameon=False)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, save_path)


# ---------------------------------------------------------------------------
# k-space
# ---------------------------------------------------------------------------


def plot_kspace_analysis(
    kspace_full,
    mask,
    reconstruction,
    save_path: str | None = None,
) -> str | None:
    """Show the sampling pattern and its consequence for the reconstruction."""
    kspace_full = np.asarray(kspace_full).squeeze()
    mask = _to_numpy(mask)
    reconstruction = _to_numpy(reconstruction)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))

    kspace_mag = np.log1p(np.abs(np.fft.fftshift(kspace_full)))
    axes[0].imshow(kspace_mag, cmap="gray")
    axes[0].set_title("Full k-space\n(log magnitude)")

    mask_2d = np.tile(mask.reshape(1, -1), (kspace_full.shape[0], 1)) if mask.ndim == 1 else mask
    sampled = mask_2d.sum() / mask_2d.size
    axes[1].imshow(mask_2d, cmap="gray", aspect="auto")
    axes[1].set_title(f"Sampling mask\n{sampled * 100:.1f}% acquired (R={1 / sampled:.1f})")

    axes[2].imshow(kspace_mag * mask_2d, cmap="gray")
    axes[2].set_title("Acquired k-space")

    axes[3].imshow(reconstruction, cmap="gray", vmin=0, vmax=float(reconstruction.max()) or 1.0)
    axes[3].set_title("Reconstruction")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    return _save(fig, save_path)


# ---------------------------------------------------------------------------
# Architecture comparison
# ---------------------------------------------------------------------------


def plot_architecture_comparison(
    results: dict,
    save_path: str | None = None,
    metric_errors: dict | None = None,
) -> str | None:
    """
    Bar and scatter comparison across architectures.

    Parameters
    ----------
    results
        ``{name: {"psnr": float, "ssim": float, "params_m": float, ...}}``
    metric_errors
        Optional ``{name: {"psnr": (low, high), "ssim": (low, high)}}`` giving
        confidence intervals, drawn as error bars. A bar chart of point
        estimates invites the reader to believe differences the data may not
        support; show the interval whenever you have it.
    """
    models = list(results)
    if not models:
        raise ValueError("No results to plot")

    palette = plt.cm.tab10(np.linspace(0, 1, 10))[: len(models)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    for ax, key, label, fmt in (
        (axes[0], "psnr", "PSNR (dB)", "{:.2f}"),
        (axes[1], "ssim", "SSIM", "{:.4f}"),
    ):
        values = [results[m][key] for m in models]
        errors = None
        if metric_errors:
            errors = np.array(
                [
                    [
                        values[i] - metric_errors[m][key][0],
                        metric_errors[m][key][1] - values[i],
                    ]
                    for i, m in enumerate(models)
                ]
            ).T

        bars = ax.bar(
            models,
            values,
            color=palette,
            edgecolor="black",
            linewidth=0.6,
            yerr=errors,
            capsize=4,
        )
        ax.set_ylabel(label)
        ax.set_title(f"{label} comparison")
        ax.tick_params(axis="x", rotation=20)
        span = max(values) - min(values) or 1.0
        ax.set_ylim(min(values) - 0.35 * span, max(values) + 0.35 * span)
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.04 * span,
                fmt.format(value),
                ha="center",
                fontsize=9,
            )

    if all("params_m" in results[m] for m in models):
        params = [results[m]["params_m"] for m in models]
        psnrs = [results[m]["psnr"] for m in models]
        axes[2].scatter(params, psnrs, c=palette, s=180, edgecolors="black", zorder=5)
        for i, name in enumerate(models):
            axes[2].annotate(
                name,
                (params[i], psnrs[i]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=9,
            )
        axes[2].set_xlabel("Parameters (M)")
        axes[2].set_ylabel("PSNR (dB)")
        axes[2].set_title("Quality vs capacity")
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].axis("off")

    fig.tight_layout()
    return _save(fig, save_path)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class AttentionExtractor:
    """
    Capture attention maps from Swin ``WindowAttention`` modules.

    Attention weights are not part of the forward output, so a plain forward
    hook cannot see them. This switches each module out of its fused
    scaled-dot-product path (which never materialises the attention matrix at
    all) and recomputes the softmax explicitly during the hook.

    Usage
    -----
        with AttentionExtractor(model) as extractor:
            model(input_tensor)
            maps = extractor.get_attention_maps()
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.attention_maps: list[torch.Tensor] = []
        self._patched: list[tuple[torch.nn.Module, bool]] = []
        self._hooks: list = []
        self._install()

    def _install(self) -> None:
        from models.swinunet import WindowAttention

        for module in self.model.modules():
            if not isinstance(module, WindowAttention):
                continue
            # Remember the fast-path setting so it can be restored on exit.
            self._patched.append((module, module.use_sdpa))
            module.use_sdpa = False
            self._hooks.append(module.register_forward_hook(self._capture))

        if not self._patched:
            log.warning(
                "No WindowAttention modules found; attention extraction only "
                "supports Swin-based models."
            )

    def _capture(self, module, inputs, output) -> None:
        """
        Recompute the attention matrix for the captured window tokens.

        Storing it inside ``forward`` instead would cost memory on every
        training step; this pays the cost only while extracting.
        """
        x = inputs[0]
        mask = inputs[1] if len(inputs) > 1 else None
        with torch.no_grad():
            Bnw, N, _ = x.shape
            qkv = module.qkv(x).reshape(Bnw, N, 3, module.n_heads, module.head_dim)
            q, k, _ = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            attn = (q @ k.transpose(-2, -1)) * module.scale + module._bias()
            if mask is not None:
                n_windows = mask.shape[0]
                attn = attn + mask.unsqueeze(1).repeat(Bnw // n_windows, 1, 1, 1)
            self.attention_maps.append(attn.softmax(dim=-1).detach().cpu())

    def get_attention_maps(self) -> list[torch.Tensor]:
        """Return the captured maps and clear the buffer."""
        maps = list(self.attention_maps)
        self.attention_maps.clear()
        return maps

    def remove_hooks(self) -> None:
        """Remove hooks and restore the fused attention path."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        for module, original in self._patched:
            module.use_sdpa = original
        self._patched.clear()

    def __enter__(self) -> AttentionExtractor:
        return self

    def __exit__(self, *exc_info) -> None:
        self.remove_hooks()


def _attention_to_image(attn: torch.Tensor, height: int, width: int) -> np.ndarray:
    """Reduce an attention tensor to a normalised ``(height, width)`` heatmap."""
    if attn.dim() == 4:  # (windows, heads, N, N)
        attn = attn.mean(dim=(0, 1))
    elif attn.dim() == 3:
        attn = attn.mean(dim=0)

    received = attn.mean(dim=0)  # mean attention each token receives
    side = int(np.sqrt(received.shape[0]))
    grid = received[: side * side].reshape(side, side).float()

    resized = (
        F.interpolate(
            grid.unsqueeze(0).unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        .squeeze()
        .numpy()
    )

    span = resized.max() - resized.min()
    return (resized - resized.min()) / (span + 1e-8)


def plot_attention_maps(
    input_img,
    attention_maps: list[torch.Tensor],
    layer_indices: list[int] | None = None,
    save_path: str | None = None,
    max_display: int = 6,
) -> str | None:
    """Overlay attention heatmaps from selected layers on the input image."""
    if not attention_maps:
        raise ValueError(
            "No attention maps supplied. Capture them with AttentionExtractor "
            "while running a forward pass."
        )

    input_img = _to_numpy(input_img)
    height, width = input_img.shape[:2]

    if layer_indices is None:
        step = max(1, len(attention_maps) // max_display)
        layer_indices = list(range(0, len(attention_maps), step))[:max_display]

    n_show = len(layer_indices)
    fig, axes = plt.subplots(2, n_show, figsize=(3.2 * n_show, 6.4), squeeze=False)
    vmax = float(input_img.max()) or 1.0

    for col, layer_idx in enumerate(layer_indices):
        heatmap = _attention_to_image(attention_maps[layer_idx], height, width)

        axes[0, col].imshow(input_img, cmap="gray", vmin=0, vmax=vmax)
        axes[0, col].imshow(heatmap, cmap="jet", alpha=0.45)
        axes[0, col].set_title(f"Layer {layer_idx}", fontsize=9)

        axes[1, col].imshow(heatmap, cmap="inferno")
        axes[1, col].set_title(f"Attention L{layer_idx}", fontsize=9)

        for row in (0, 1):
            axes[row, col].axis("off")

    fig.suptitle("Swin attention", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, save_path)


def plot_attention_rollout(
    input_img,
    attention_maps: list[torch.Tensor],
    save_path: str | None = None,
    discard_ratio: float = 0.9,
) -> str | None:
    """
    Attention rollout: propagate attention multiplicatively through the layers.

    Following Abnar & Zuidema (2020), an identity term is added at each layer to
    account for the residual connection, and the lowest-weight connections are
    discarded before renormalisation to suppress the diffuse background that
    otherwise dominates after several matrix products.
    """
    if not attention_maps:
        raise ValueError("No attention maps supplied")

    input_img = _to_numpy(input_img)
    height, width = input_img.shape[:2]

    result = None
    for attn in attention_maps:
        if attn.dim() == 4:
            averaged = attn.mean(dim=(0, 1))
        elif attn.dim() == 3:
            averaged = attn.mean(dim=0)
        else:
            continue

        seq_len = averaged.shape[0]
        averaged = 0.5 * averaged + 0.5 * torch.eye(seq_len)

        if discard_ratio > 0:
            threshold = averaged.flatten().quantile(discard_ratio)
            averaged = averaged * (averaged > threshold).float()

        averaged = averaged / (averaged.sum(dim=-1, keepdim=True) + 1e-8)
        if result is None:
            result = averaged
        elif result.shape == averaged.shape:
            result = averaged @ result

    if result is None:
        raise ValueError("Attention maps had unexpected dimensionality")

    heatmap = _attention_to_image(result.unsqueeze(0).unsqueeze(0), height, width)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    vmax = float(input_img.max()) or 1.0

    axes[0].imshow(input_img, cmap="gray", vmin=0, vmax=vmax)
    axes[0].set_title("Input")

    axes[1].imshow(input_img, cmap="gray", vmin=0, vmax=vmax)
    axes[1].imshow(heatmap, cmap="jet", alpha=0.45)
    axes[1].set_title("Rollout overlay")

    im = axes[2].imshow(heatmap, cmap="inferno")
    axes[2].set_title("Rollout map")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.axis("off")

    fig.suptitle("Attention rollout (all layers)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, save_path)
