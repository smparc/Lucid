"""
train.py
--------
Training loop for all three MRI reconstruction architectures.

Loss
----
Combined L1 + (1 - SSIM):
    L = lambda1 * L1 + lambda2 * (1 - SSIM)

Training strategy
-----------------
- Adam optimizer
- Cosine annealing LR scheduler
- Weight decay regularization
- Early stopping on validation loss
- Random flip/rotation augmentation
- Gradient clipping for stability
"""

import os
import sys
import time
import argparse
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.unet     import UNet
from models.bt_unet  import BTUNet
from models.swinunet import SwinUNet

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
from preprocessing import FastMRIKneeDataset


# ---------------------------------------------------------------------------
# SSIM Loss
# ---------------------------------------------------------------------------

def gaussian_kernel(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Create a 2D Gaussian kernel."""
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

        mu1 = F.conv2d(pred,   k, padding=pad, groups=pred.shape[1])
        mu2 = F.conv2d(target, k, padding=pad, groups=pred.shape[1])

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred   * pred,   k, padding=pad, groups=pred.shape[1]) - mu1_sq
        sigma2_sq = F.conv2d(target * target, k, padding=pad, groups=pred.shape[1]) - mu2_sq
        sigma12   = F.conv2d(pred   * target, k, padding=pad, groups=pred.shape[1]) - mu1_mu2

        ssim_map = (
            (2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)
        ) / (
            (mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2)
        )
        return 1 - ssim_map.mean()


class CombinedLoss(nn.Module):
    def __init__(self, lambda1: float = 0.7, lambda2: float = 0.3):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.l1   = nn.L1Loss()
        self.ssim = SSIMLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.lambda1 * self.l1(pred, target) + self.lambda2 * self.ssim(pred, target)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def augment(x: torch.Tensor, y: torch.Tensor):
    """Random horizontal flip and 90-degree rotation."""
    if torch.rand(1) > 0.5:
        x = torch.flip(x, dims=[-1])
        y = torch.flip(y, dims=[-1])
    if torch.rand(1) > 0.5:
        x = torch.flip(x, dims=[-2])
        y = torch.flip(y, dims=[-2])
    k = torch.randint(0, 4, (1,)).item()
    if k > 0:
        x = torch.rot90(x, k, dims=[-2, -1])
        y = torch.rot90(y, k, dims=[-2, -1])
    return x, y


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return float("inf")
    return 10 * torch.log10(torch.tensor(max_val ** 2 / mse)).item()


def ssim_metric(pred: torch.Tensor, target: torch.Tensor) -> float:
    loss_fn = SSIMLoss().to(pred.device)
    with torch.no_grad():
        return (1 - loss_fn(pred, target)).item()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model, loader, optimizer, criterion, device, augment_data=True
):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if augment_data:
            x, y = augment(x, y)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        total_loss += criterion(pred, y).item()
        total_psnr += psnr(pred, y)
        total_ssim += ssim_metric(pred, y)
    n = len(loader)
    return total_loss / n, total_psnr / n, total_ssim / n


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def build_model(name: str, device: torch.device) -> nn.Module:
    name = name.lower()
    if name == "unet":
        model = UNet(in_channels=1, out_channels=1, base_ch=32, n_levels=4)
    elif name == "bt_unet":
        model = BTUNet(in_channels=1, out_channels=1, base_ch=32, n_levels=4,
                       tf_heads=8, tf_layers=4)
    elif name == "swinunet":
        model = SwinUNet(img_size=320, patch_size=4, in_ch=1, out_ch=1,
                         embed_dim=64, ws=8, head_dim=8, dropout=0.1, n_levels=3)
    else:
        raise ValueError(f"Unknown model: {name}")
    return model.to(device)


def train(
    model_name: str = "swinunet",
    data_dir: str = "data/knee_singlecoil_train",
    output_dir: str = "checkpoints",
    epochs: int = 50,
    batch_size: int = 4,
    lr: float = 8e-5,
    weight_decay: float = 1e-4,
    val_split: float = 0.1,
    seed: int = 42,
    augment_data: bool = True,
):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n── Device: {device} ─────────────────────────────────────")

    # Dataset
    dataset = FastMRIKneeDataset(data_dir, seed=seed)
    n_val   = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    print(f"  Train: {n_train} | Val: {n_val}")

    # Model
    model = build_model(model_name, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {model_name} | Params: {n_params/1e6:.1f}M")

    # Optimizer & scheduler
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    criterion = CombinedLoss(lambda1=0.7, lambda2=0.3)

    os.makedirs(output_dir, exist_ok=True)
    history = {"train_loss": [], "val_loss": [], "val_psnr": [], "val_ssim": []}

    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0

    print(f"\n{'Epoch':>6} {'Train Loss':>11} {'Val Loss':>9} {'PSNR':>7} {'SSIM':>7}  {'Time':>6}")
    print("─" * 55)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, augment_data)
        val_loss, val_psnr, val_ssim = validate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_psnr"].append(val_psnr)
        history["val_ssim"].append(val_ssim)

        print(f"{epoch:>6} {train_loss:>11.4f} {val_loss:>9.4f} {val_psnr:>7.2f} {val_ssim:>7.4f}  {elapsed:>5.1f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            ckpt_path = os.path.join(output_dir, f"{model_name}_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_psnr": val_psnr,
                "val_ssim": val_ssim,
            }, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    # Save training history
    hist_path = os.path.join(output_dir, f"{model_name}_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n  Best val loss: {best_val_loss:.4f}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  History:    {hist_path}")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MRI Reconstruction Model")
    parser.add_argument("--model",      default="swinunet", choices=["unet", "bt_unet", "swinunet"])
    parser.add_argument("--data_dir",   default="data/knee_singlecoil_train")
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch_size", type=int,   default=4)
    parser.add_argument("--lr",         type=float, default=8e-5)
    parser.add_argument("--no_augment", action="store_true")
    args = parser.parse_args()

    train(
        model_name=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        augment_data=not args.no_augment,
    )
