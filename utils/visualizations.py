"""
visualization.py
----------------
Advanced visualization tools for MRI reconstruction analysis.


Features:
- Attention map visualization (what the SwinUNet is "looking at")
- Error heatmaps (where the model fails)
- Frequency-domain error analysis
- Multi-scale feature visualization
- Interactive comparison grids
"""


import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional, List
from pathlib import Path



def plot_reconstruction_comparison(
    input_img: np.ndarray,
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    psnr_val: float = None,
    ssim_val: float = None,
    save_path: str = None,
    show_error: bool = True,
    show_frequency: bool = True,
):
    """
    Publication-quality reconstruction comparison figure.


    Shows: Input | Prediction | Ground Truth | Error Map | Freq Error
    """
    n_cols = 3 + int(show_error) + int(show_frequency)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))


    # Input
    axes[0].imshow(input_img, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Input\n(Undersampled)", fontsize=10)
    axes[0].axis("off")


    # Prediction
    axes[1].imshow(prediction, cmap="gray", vmin=0, vmax=1)
    title = "Prediction"
    if psnr_val and ssim_val:
        title += f"\nPSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}"
    axes[1].set_title(title, fontsize=10)
    axes[1].axis("off")


    # Ground truth
    axes[2].imshow(ground_truth, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Ground Truth", fontsize=10)
    axes[2].axis("off")


    col = 3


    # Error heatmap
    if show_error:
        error = np.abs(prediction - ground_truth)
        im = axes[col].imshow(error, cmap="hot", vmin=0, vmax=error.max() * 0.8)
        axes[col].set_title(f"Error Map\n(MAE: {error.mean():.4f})", fontsize=10)
        axes[col].axis("off")
        plt.colorbar(im, ax=axes[col], fraction=0.046, pad=0.04)
        col += 1


    # Frequency domain error
    if show_frequency:
        pred_fft = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(prediction))))
        gt_fft = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(ground_truth))))
        freq_error = np.abs(pred_fft - gt_fft)
        im = axes[col].imshow(freq_error, cmap="inferno")
        axes[col].set_title("Frequency Error\n(log |ΔF|)", fontsize=10)
        axes[col].axis("off")
        plt.colorbar(im, ax=axes[col], fraction=0.046, pad=0.04)


    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()



def plot_training_curves(
    history: dict,
    save_path: str = None,
    title: str = "Training Progress",
):
    """
    Publication-quality training curves with dual y-axis for loss and metrics.
    """
    fig = plt.figure(figsize=(16, 4))
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.35)


    epochs = range(1, len(history["train_loss"]) + 1)


    # Loss curves
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(epochs, history["train_loss"], "b-", alpha=0.8, label="Train", linewidth=1.5)
    ax1.plot(epochs, history["val_loss"], "r-", alpha=0.8, label="Val", linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend(frameon=False)
    ax1.grid(True, alpha=0.3)


    # PSNR
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(epochs, history["val_psnr"], "g-", linewidth=1.5)
    ax2.axhline(y=max(history["val_psnr"]), color="g", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("PSNR (dB)")
    ax2.set_title(f"PSNR (best: {max(history['val_psnr']):.2f} dB)")
    ax2.grid(True, alpha=0.3)


    # SSIM
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(epochs, history["val_ssim"], "m-", linewidth=1.5)
    ax3.axhline(y=max(history["val_ssim"]), color="m", linestyle="--", alpha=0.5)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("SSIM")
    ax3.set_title(f"SSIM (best: {max(history['val_ssim']):.4f})")
    ax3.grid(True, alpha=0.3)


    # LR schedule
    if "lr" in history:
        ax4 = fig.add_subplot(gs[3])
        ax4.plot(epochs, history["lr"], "k-", linewidth=1.5)
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Learning Rate")
        ax4.set_title("LR Schedule")
        ax4.set_yscale("log")
        ax4.grid(True, alpha=0.3)


    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)


    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()



def plot_kspace_analysis(
    kspace_full: np.ndarray,
    mask: np.ndarray,
    reconstruction: np.ndarray,
    save_path: str = None,
):
    """
    Visualize the k-space sampling pattern and its effect on reconstruction.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))


    # Full k-space (log magnitude)
    kspace_mag = np.log1p(np.abs(np.fft.fftshift(kspace_full)))
    axes[0].imshow(kspace_mag, cmap="gray")
    axes[0].set_title("Full k-space\n(log magnitude)")
    axes[0].axis("off")


    # Sampling mask
    if mask.ndim == 1:
        mask_2d = np.tile(mask, (kspace_full.shape[0], 1))
    else:
        mask_2d = mask
    axes[1].imshow(mask_2d, cmap="gray")
    axes[1].set_title(f"Sampling Mask\n({mask.sum()/mask.size*100:.1f}% sampled)")
    axes[1].axis("off")


    # Undersampled k-space
    axes[2].imshow(kspace_mag * mask_2d, cmap="gray")
    axes[2].set_title("Undersampled k-space")
    axes[2].axis("off")


    # Reconstruction
    axes[3].imshow(reconstruction, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("Reconstruction")
    axes[3].axis("off")


    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()



def plot_architecture_comparison(
    results: dict,
    save_path: str = None,
):
    """
    Bar chart comparing architectures across metrics.
    
    results: {model_name: {"psnr": float, "ssim": float, "val_loss": float, "params_m": float}}
    """
    models = list(results.keys())
    n = len(models)


    fig, axes = plt.subplots(1, 3, figsize=(12, 4))


    # PSNR
    psnrs = [results[m]["psnr"] for m in models]
    colors = ["#2196F3", "#FF9800", "#4CAF50"][:n]
    bars = axes[0].bar(models, psnrs, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_title("PSNR Comparison")
    for bar, val in zip(bars, psnrs):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     f"{val:.2f}", ha="center", fontsize=9)


    # SSIM
    ssims = [results[m]["ssim"] for m in models]
    bars = axes[1].bar(models, ssims, color=colors, edgecolor="black", linewidth=0.5)
    axes[1].set_ylabel("SSIM")
    axes[1].set_title("SSIM Comparison")
    for bar, val in zip(bars, ssims):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                     f"{val:.4f}", ha="center", fontsize=9)


    # Parameters vs PSNR (efficiency scatter)
    if all("params_m" in results[m] for m in models):
        params = [results[m]["params_m"] for m in models]
        axes[2].scatter(params, psnrs, c=colors, s=200, edgecolors="black", linewidth=0.5, zorder=5)
        for i, m in enumerate(models):
            axes[2].annotate(m, (params[i], psnrs[i]), textcoords="offset points",
                           xytext=(5, 5), fontsize=9)
        axes[2].set_xlabel("Parameters (M)")
        axes[2].set_ylabel("PSNR (dB)")
        axes[2].set_title("Efficiency: Params vs Quality")
        axes[2].grid(True, alpha=0.3)


    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()



# ---------------------------------------------------------------------------
# Attention Visualization
# ---------------------------------------------------------------------------


class AttentionExtractor:
    """
    Hook-based attention map extractor for Swin Transformer models.


    Usage:
        extractor = AttentionExtractor(model)
        output = model(input_tensor)
        attn_maps = extractor.get_attention_maps()
        plot_attention_maps(input_img, attn_maps)
        extractor.remove_hooks()
    """


    def __init__(self, model: torch.nn.Module):
        self.attention_maps: List[torch.Tensor] = []
        self.hooks = []
        self._register_hooks(model)


    def _register_hooks(self, model: torch.nn.Module):
        """Register forward hooks on all attention softmax outputs."""
        for name, module in model.named_modules():
            # Match common attention patterns in Swin/ViT
            if "attn" in name.lower() and hasattr(module, "softmax"):
                hook = module.register_forward_hook(self._hook_fn)
                self.hooks.append(hook)
            # Also match nn.MultiheadAttention or custom WindowAttention
            elif module.__class__.__name__ in ("WindowAttention", "MultiheadAttention"):
                hook = module.register_forward_hook(self._hook_fn)
                self.hooks.append(hook)


    def _hook_fn(self, module, input, output):
        """Capture attention weights from the module output."""
        if isinstance(output, tuple) and len(output) >= 2:
            # MultiheadAttention returns (attn_output, attn_weights)
            attn_weights = output[1]
            if attn_weights is not None:
                self.attention_maps.append(attn_weights.detach().cpu())
        elif hasattr(module, "_attention_weights"):
            # Some implementations store weights as attribute
            self.attention_maps.append(module._attention_weights.detach().cpu())


    def get_attention_maps(self) -> List[torch.Tensor]:
        """Return collected attention maps and clear buffer."""
        maps = self.attention_maps.copy()
        self.attention_maps.clear()
        return maps


    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()



def plot_attention_maps(
    input_img: np.ndarray,
    attention_maps: List[torch.Tensor],
    layer_indices: Optional[List[int]] = None,
    head_indices: Optional[List[int]] = None,
    save_path: str = None,
    max_display: int = 8,
):
    """
    Visualize attention maps from transformer layers overlaid on the input.


    Parameters
    ----------
    input_img : (H, W) array, input MR image
    attention_maps : list of attention tensors from AttentionExtractor
    layer_indices : which layers to display (None = evenly spaced selection)
    head_indices : which attention heads to display (None = average across heads)
    save_path : path to save figure
    max_display : maximum number of attention maps to show
    """
    if not attention_maps:
        print("No attention maps captured. Ensure hooks are registered correctly.")
        return


    # Select layers to display
    n_layers = len(attention_maps)
    if layer_indices is None:
        step = max(1, n_layers // max_display)
        layer_indices = list(range(0, n_layers, step))[:max_display]


    n_show = len(layer_indices)
    fig, axes = plt.subplots(2, n_show, figsize=(3 * n_show, 6))
    if n_show == 1:
        axes = axes.reshape(2, 1)


    img_h, img_w = input_img.shape[:2]


    for col, layer_idx in enumerate(layer_indices):
        if layer_idx >= len(attention_maps):
            continue


        attn = attention_maps[layer_idx]  # (B, nH, seq_len, seq_len) or (B, seq, seq)


        # Average over batch
        if attn.dim() == 4:
            # (B, nH, seq, seq)
            if head_indices is not None:
                attn = attn[:, head_indices].mean(dim=(0, 1))
            else:
                attn = attn.mean(dim=(0, 1))  # (seq, seq)
        elif attn.dim() == 3:
            attn = attn.mean(dim=0)  # (seq, seq)


        # Compute attention rollout: mean attention received per token
        attn_map = attn.mean(dim=0)  # (seq_len,) — avg attention each token receives
        seq_len = attn_map.shape[0]
        side = int(np.sqrt(seq_len))


        if side * side == seq_len:
            attn_2d = attn_map.reshape(side, side).numpy()
        else:
            # Non-square: best effort reshape
            attn_2d = attn_map[:side * side].reshape(side, side).numpy()


        # Resize to image dimensions
        attn_resized = np.array(
            F.interpolate(
                torch.from_numpy(attn_2d).unsqueeze(0).unsqueeze(0).float(),
                size=(img_h, img_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze()
        )


        # Normalize to [0, 1]
        attn_resized = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)


        # Row 1: Input with attention overlay
        axes[0, col].imshow(input_img, cmap="gray", vmin=0, vmax=1)
        axes[0, col].imshow(attn_resized, cmap="jet", alpha=0.5)
        axes[0, col].set_title(f"Layer {layer_idx}", fontsize=9)
        axes[0, col].axis("off")


        # Row 2: Raw attention heatmap
        im = axes[1, col].imshow(attn_resized, cmap="inferno")
        axes[1, col].set_title(f"Attn Map L{layer_idx}", fontsize=9)
        axes[1, col].axis("off")


    axes[0, 0].set_ylabel("Overlay", fontsize=10)
    axes[1, 0].set_ylabel("Raw Attention", fontsize=10)


    fig.suptitle("Swin Transformer Attention Visualization", fontsize=12, fontweight="bold")
    plt.tight_layout()


    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()



def plot_attention_rollout(
    input_img: np.ndarray,
    attention_maps: List[torch.Tensor],
    save_path: str = None,
    discard_ratio: float = 0.9,
):
    """
    Attention rollout: multiplicative propagation of attention through layers.


    This gives a more accurate picture of what the final layer 'sees' by
    combining attention from all layers multiplicatively.


    Parameters
    ----------
    input_img : (H, W) input image
    attention_maps : list of (B, nH, seq, seq) attention tensors
    save_path : output path
    discard_ratio : fraction of lowest-attention connections to discard (sparsify)
    """
    if not attention_maps:
        return


    result = None
    for attn in attention_maps:
        # Average over batch and heads
        if attn.dim() == 4:
            attn_avg = attn.mean(dim=(0, 1))  # (seq, seq)
        elif attn.dim() == 3:
            attn_avg = attn.mean(dim=0)
        else:
            continue


        seq_len = attn_avg.shape[0]


        # Add identity (residual connection contribution)
        attn_avg = 0.5 * attn_avg + 0.5 * torch.eye(seq_len)


        # Discard low-attention connections
        if discard_ratio > 0:
            flat = attn_avg.flatten()
            threshold = flat.quantile(discard_ratio)
            attn_avg = attn_avg * (attn_avg > threshold).float()


        # Re-normalize rows
        attn_avg = attn_avg / (attn_avg.sum(dim=-1, keepdim=True) + 1e-8)


        # Multiply through layers
        if result is None:
            result = attn_avg
        else:
            # Only multiply if dimensions match
            if result.shape == attn_avg.shape:
                result = torch.matmul(attn_avg, result)


    if result is None:
        return


    # Take attention from CLS token (first row) or average
    rollout_map = result.mean(dim=0)  # Average attention received
    seq_len = rollout_map.shape[0]
    side = int(np.sqrt(seq_len))


    if side * side != seq_len:
        side = int(np.ceil(np.sqrt(seq_len)))
        rollout_map = torch.cat([rollout_map, torch.zeros(side * side - seq_len)])


    rollout_2d = rollout_map[:side * side].reshape(side, side).numpy()


    img_h, img_w = input_img.shape[:2]
    rollout_resized = np.array(
        F.interpolate(
            torch.from_numpy(rollout_2d).unsqueeze(0).unsqueeze(0).float(),
            size=(img_h, img_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze()
    )
    rollout_resized = (rollout_resized - rollout_resized.min()) / (rollout_resized.max() - rollout_resized.min() + 1e-8)


    fig, axes = plt.subplots(1, 3, figsize=(12, 4))


    axes[0].imshow(input_img, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Input")
    axes[0].axis("off")


    axes[1].imshow(input_img, cmap="gray", vmin=0, vmax=1)
    axes[1].imshow(rollout_resized, cmap="jet", alpha=0.5)
    axes[1].set_title("Attention Rollout Overlay")
    axes[1].axis("off")


    im = axes[2].imshow(rollout_resized, cmap="inferno")
    axes[2].set_title("Rollout Map")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)


    fig.suptitle("Attention Rollout (All Layers Combined)", fontsize=12, fontweight="bold")
    plt.tight_layout()


    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()