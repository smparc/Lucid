"""
evaluate.py
-----------
Evaluation and visualization for trained MRI reconstruction models.


Metrics
-------
- PSNR  (Peak Signal-to-Noise Ratio)     — pixel-wise accuracy
- SSIM  (Structural Similarity Index)    — structural fidelity
- Val Loss                               — combined L1 + SSIM loss


Outputs
-------
- Per-sample PSNR / SSIM printed to stdout
- Visual grid: [Input | Prediction | Ground Truth] saved to file
- Architecture comparison summary table
"""


import os
import sys
import json
import argparse


import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.unet import UNet
from models.bt_unet import BTUNet
from models.swinunet import SwinUNet
from training.train import build_model, psnr as compute_psnr, ssim_metric
from data.preprocessing import FastMRIKneeDataset



# ---------------------------------------------------------------------------
# Load model from checkpoint
# ---------------------------------------------------------------------------


def load_model(model_name: str, ckpt_path: str, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)


    # Use config from checkpoint if available (avoids param mismatch)
    config = ckpt.get("config", {})
    if config and "model" in config:
        from config import Config
        cfg = Config(config)
        model = build_model(cfg)
    else:
        # Fallback: build from model_name with defaults
        name = model_name.lower()
        if name == "unet":
            model = UNet(in_channels=1, out_channels=1, base_ch=32, n_levels=4)
        elif name == "bt_unet":
            model = BTUNet(in_channels=1, out_channels=1, base_ch=32, n_levels=4)
        elif name == "swinunet":
            model = SwinUNet(img_size=320, patch_size=4, in_ch=1, out_ch=1,
                             embed_dim=64, ws=8, head_dim=8, dropout=0.0, n_levels=3)
        else:
            raise ValueError(f"Unknown model: {model_name}")


    # Prefer EMA weights for evaluation
    if ckpt.get("ema_state") is not None:
        ema_state = ckpt["ema_state"]
        shadow_params = ema_state.get("shadow_params")
        if shadow_params is not None:
            for param, shadow in zip(model.parameters(), shadow_params):
                param.data.copy_(shadow)
            print(f"  Loaded EMA weights from {ckpt_path}")
        else:
            model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt["model_state"])


    model.to(device).eval()
    print(f"  Loaded {model_name} from {ckpt_path}  (epoch {ckpt.get('epoch', '?')}, "
          f"val_loss={ckpt.get('val_loss', 0):.4f}, PSNR={ckpt.get('val_psnr', 0):.2f}dB)")
    return model
    return model



# ---------------------------------------------------------------------------
# Quantitative evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_model(model, loader, device, n_samples: int = None):
    try:
        from training.losses import CombinedLoss
        criterion = CombinedLoss().to(device)
    except ImportError:
        from training.train import CombinedLoss
        criterion = CombinedLoss().to(device)
    total_loss = total_psnr = total_ssim = 0.0
    count = 0


    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        total_loss += criterion(pred, y).item()
        total_psnr += compute_psnr(pred, y)
        total_ssim += ssim_metric(pred, y)
        count += 1
        if n_samples and count >= n_samples:
            break


    return {
        "val_loss": total_loss / count,
        "psnr_db":  total_psnr / count,
        "ssim":     total_ssim / count,
    }



# ---------------------------------------------------------------------------
# Qualitative visualization
# ---------------------------------------------------------------------------


@torch.no_grad()
def visualize_reconstructions(
    model,
    dataset,
    device,
    n_examples: int = 4,
    save_path: str = "results/reconstructions.png",
):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    indices = np.random.choice(len(dataset), min(n_examples, len(dataset)), replace=False)


    fig, axes = plt.subplots(n_examples, 3, figsize=(12, 4 * n_examples))
    fig.suptitle("FastMRI Reconstruction", fontsize=16, fontweight="bold")


    col_titles = ["Input (Undersampled)", "SwinUNet Reconstruction", "Ground Truth"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12)


    model.eval()
    for row, idx in enumerate(indices):
        x, y = dataset[idx]
        x_in   = x.unsqueeze(0).to(device)
        y_true = y.numpy().squeeze()
        y_pred = model(x_in).cpu().numpy().squeeze()
        x_np   = x.numpy().squeeze()


        p = compute_psnr(
            torch.tensor(y_pred).unsqueeze(0).unsqueeze(0),
            torch.tensor(y_true).unsqueeze(0).unsqueeze(0),
        )
        s = ssim_metric(
            torch.tensor(y_pred).unsqueeze(0).unsqueeze(0),
            torch.tensor(y_true).unsqueeze(0).unsqueeze(0),
        )


        for col, img in enumerate([x_np, y_pred, y_true]):
            axes[row, col].imshow(img, cmap="gray", vmin=0, vmax=1)
            axes[row, col].axis("off")


        axes[row, 1].set_title(
            f"PSNR: {p:.2f} dB  |  SSIM: {s:.4f}",
            fontsize=9, color="white",
            bbox=dict(facecolor="black", alpha=0.6, pad=2),
        )


    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved reconstruction grid → {save_path}")
    plt.close()



# ---------------------------------------------------------------------------
# Training curve plot
# ---------------------------------------------------------------------------


def plot_training_history(history_path: str, save_path: str = None):
    with open(history_path) as f:
        hist = json.load(f)


    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = range(1, len(hist["train_loss"]) + 1)


    axes[0].plot(epochs, hist["train_loss"], label="Train Loss")
    axes[0].plot(epochs, hist["val_loss"],   label="Val Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss"); axes[0].legend()


    axes[1].plot(epochs, hist["val_psnr"], color="C2")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("PSNR (dB)")
    axes[1].set_title("Validation PSNR")


    axes[2].plot(epochs, hist["val_ssim"], color="C3")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("SSIM")
    axes[2].set_title("Validation SSIM")


    plt.suptitle(os.path.basename(history_path).replace("_history.json", ""), fontsize=13)
    plt.tight_layout()


    out = save_path or history_path.replace("_history.json", "_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved training curves → {out}")
    plt.close()



# ---------------------------------------------------------------------------
# Architecture comparison table
# ---------------------------------------------------------------------------


def compare_architectures(results: dict):
    """results = {model_name: {val_loss, psnr_db, ssim}}"""
    print(f"\n{'Architecture':<20} {'Val Loss':>10} {'PSNR (dB)':>10} {'SSIM':>8}")
    print("─" * 52)
    for name, m in results.items():
        print(f"{name:<20} {m['val_loss']:>10.4f} {m['psnr_db']:>10.2f} {m['ssim']:>8.4f}")



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MRI Reconstruction Models")
    parser.add_argument("--model",      default="swinunet", choices=["unet", "bt_unet", "swinunet"])
    parser.add_argument("--ckpt",       required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--data_dir",   default="data/knee_singlecoil_val")
    parser.add_argument("--n_vis",      type=int, default=4, help="Number of visual examples")
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()


    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model   = load_model(args.model, args.ckpt, device)
    dataset = FastMRIKneeDataset(args.data_dir)
    loader  = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)


    print(f"\n── Quantitative Evaluation ({len(dataset)} volumes) ────────")
    metrics = evaluate_model(model, loader, device)
    print(f"  Val Loss : {metrics['val_loss']:.4f}")
    print(f"  PSNR     : {metrics['psnr_db']:.2f} dB")
    print(f"  SSIM     : {metrics['ssim']:.4f}")


    print(f"\n── Qualitative Visualization ──────────────────────────")
    vis_path = os.path.join(args.output_dir, f"{args.model}_reconstructions.png")
    visualize_reconstructions(model, dataset, device, n_examples=args.n_vis, save_path=vis_path)