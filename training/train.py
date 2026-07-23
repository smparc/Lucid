"""
train.py
--------
Production-grade training loop for accelerated MRI reconstruction.


Features
--------
- YAML config-driven training
- Automatic Mixed Precision (AMP) for faster training
- TensorBoard + W&B experiment tracking
- Cosine / Step / Plateau LR scheduling
- Gradient clipping, early stopping
- Top-K checkpoint management
- Resume from checkpoint
"""


import os
import sys
import time
import logging
from pathlib import Path


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from torch.amp import GradScaler, autocast
from tqdm import tqdm


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


from models.unet import UNet
from models.bt_unet import BTUNet
from models.swinunet import SwinUNet
from data.preprocessing import FastMRIKneeDataset
from config import load_config
from utils.logger import ExperimentLogger
from utils.reproducibility import seed_everything
from utils.ema import EMAModel
from utils.schedulers import WarmupCosineScheduler


from contextlib import nullcontext


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# Loss (import from training.losses; inline fallback for standalone usage)
# ---------------------------------------------------------------------------


try:
    from training.losses import SSIMLoss, CombinedLoss
except ImportError:
    # Minimal fallback for when training.losses is not available
    def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g.outer(g).unsqueeze(0).unsqueeze(0)


    class SSIMLoss(nn.Module):
        def __init__(self, window_size: int = 11, sigma: float = 1.5):
            super().__init__()
            self.window_size = window_size
            self.register_buffer("kernel", _gaussian_kernel(window_size, sigma))
            self.C1 = 0.01 ** 2
            self.C2 = 0.03 ** 2


        def forward(self, pred, target):
            k = self.kernel.expand(pred.shape[1], 1, -1, -1)
            pad = self.window_size // 2
            mu1 = F.conv2d(pred, k, padding=pad, groups=pred.shape[1])
            mu2 = F.conv2d(target, k, padding=pad, groups=pred.shape[1])
            sigma1_sq = F.conv2d(pred * pred, k, padding=pad, groups=pred.shape[1]) - mu1 ** 2
            sigma2_sq = F.conv2d(target * target, k, padding=pad, groups=pred.shape[1]) - mu2 ** 2
            sigma12 = F.conv2d(pred * target, k, padding=pad, groups=pred.shape[1]) - mu1 * mu2
            ssim_map = ((2 * mu1 * mu2 + self.C1) * (2 * sigma12 + self.C2)) / \
                       ((mu1 ** 2 + mu2 ** 2 + self.C1) * (sigma1_sq + sigma2_sq + self.C2))
            return 1 - ssim_map.mean()


    class CombinedLoss(nn.Module):
        def __init__(self, lambda1=0.7, lambda2=0.3, **kwargs):
            super().__init__()
            self.l1 = nn.L1Loss()
            self.ssim = SSIMLoss()
            self.lambda1 = lambda1
            self.lambda2 = lambda2


        def forward(self, pred, target):
            return self.lambda1 * self.l1(pred, target) + self.lambda2 * self.ssim(pred, target)



# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------


def augment(x: torch.Tensor, y: torch.Tensor):
    """Random horizontal flip and 90-degree rotation (applied consistently to input/target)."""
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
    """Peak Signal-to-Noise Ratio."""
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return float("inf")
    return 10 * torch.log10(torch.tensor(max_val ** 2 / mse)).item()



def ssim_metric(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute SSIM (higher is better)."""
    loss_fn = SSIMLoss().to(pred.device)
    with torch.no_grad():
        return (1 - loss_fn(pred, target)).item()



# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def build_model(cfg) -> nn.Module:
    """Build model from config, optionally wrapped with data consistency."""
    name = cfg.model.name.lower()
    params = dict(cfg.model.get("params", {}))


    if name == "unet":
        model = UNet(
            in_channels=params.get("in_channels", 1),
            out_channels=params.get("out_channels", 1),
            base_ch=params.get("base_ch", 32),
            n_levels=params.get("n_levels", 4),
        )
    elif name == "bt_unet":
        model = BTUNet(
            in_channels=params.get("in_channels", 1),
            out_channels=params.get("out_channels", 1),
            base_ch=params.get("base_ch", 32),
            n_levels=params.get("n_levels", 4),
            tf_heads=params.get("tf_heads", 8),
            tf_layers=params.get("tf_layers", 4),
            tf_dropout=params.get("tf_dropout", 0.1),
        )
    elif name == "swinunet":
        model = SwinUNet(
            img_size=params.get("img_size", 320),
            patch_size=params.get("patch_size", 4),
            in_ch=params.get("in_ch", params.get("in_channels", 1)),
            out_ch=params.get("out_ch", params.get("out_channels", 1)),
            embed_dim=params.get("embed_dim", 64),
            ws=params.get("ws", 8),
            head_dim=params.get("head_dim", 8),
            dropout=params.get("dropout", 0.1),
            n_levels=params.get("n_levels", 3),
        )
    else:
        raise ValueError(f"Unknown model: {name}. Choose from: unet, bt_unet, swinunet")


    # Optionally wrap with residual learning + data consistency
    dc_cfg = cfg.model.get("data_consistency", {})
    if dc_cfg.get("enabled", False):
        from models.data_consistency import ResidualDCWrapper
        model = ResidualDCWrapper(model, use_dc=dc_cfg.get("use_dc", True))
        log.info("Wrapped model with ResidualDCWrapper (residual learning + data consistency)")


    return model



# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """
    Configurable trainer with AMP, logging, and checkpoint management.
    """


    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        # Reproducibility
        seed_everything(cfg.training.get("seed", 42))


        # Logger
        self.logger = ExperimentLogger(cfg.logging, full_config=dict(cfg))


        # Model
        self.model = build_model(cfg).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        log.info(f"Model: {cfg.model.name} | Parameters: {n_params/1e6:.2f}M | Device: {self.device}")


        # Loss
        loss_cfg = cfg.training.get("loss", {})
        self.criterion = CombinedLoss(
            l1_weight=loss_cfg.get("l1_weight", 0.7),
            ssim_weight=loss_cfg.get("ssim_weight", 0.3),
            freq_weight=loss_cfg.get("freq_weight", 0.0),
            edge_weight=loss_cfg.get("edge_weight", 0.0),
            perceptual_weight=loss_cfg.get("perceptual_weight", 0.0),
            charbonnier=loss_cfg.get("charbonnier", False),
        ).to(self.device)


        # Optimizer
        opt_name = cfg.training.get("optimizer", "adam").lower()
        opt_cls = AdamW if opt_name == "adamw" else Adam
        betas = tuple(cfg.training.get("betas", [0.9, 0.999]))
        self.optimizer = opt_cls(
            self.model.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.get("weight_decay", 1e-4),
            betas=betas,
        )


        # Scheduler
        self._build_scheduler(cfg)


        # AMP
        self.use_amp = cfg.training.get("amp", False) and self.device.type == "cuda"
        self.scaler = GradScaler("cuda", enabled=self.use_amp)


        # EMA
        ema_cfg = cfg.training.get("ema", {})
        self.use_ema = ema_cfg.get("enabled", True)
        if self.use_ema:
            self.ema = EMAModel(
                self.model,
                decay=ema_cfg.get("decay", 0.999),
                warmup=ema_cfg.get("warmup_steps", 1000),
            )
            log.info(f"EMA enabled (decay={ema_cfg.get('decay', 0.999)})")


        # Gradient accumulation
        self.grad_accum_steps = cfg.training.get("gradient_accumulation", 1)
        if self.grad_accum_steps > 1:
            log.info(f"Gradient accumulation: {self.grad_accum_steps} steps "
                     f"(effective batch = {cfg.training.batch_size * self.grad_accum_steps})")


        # Training state
        self.start_epoch = 1
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.global_step = 0


    def _build_scheduler(self, cfg):
        sched_name = cfg.training.get("scheduler", "warmup_cosine")
        epochs = cfg.training.epochs
        lr = cfg.training.lr
        eta_min = lr * cfg.training.get("eta_min_factor", 0.01)
        warmup = cfg.training.get("warmup_epochs", 5)


        if sched_name == "warmup_cosine":
            self.scheduler = WarmupCosineScheduler(
                self.optimizer, warmup_epochs=warmup, total_epochs=epochs, eta_min=eta_min
            )
        elif sched_name == "cosine":
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=eta_min)
        elif sched_name == "step":
            self.scheduler = StepLR(self.optimizer, step_size=max(1, epochs // 3), gamma=0.1)
        elif sched_name == "plateau":
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=5, factor=0.5)
        else:
            self.scheduler = WarmupCosineScheduler(
                self.optimizer, warmup_epochs=warmup, total_epochs=epochs, eta_min=eta_min
            )


    def _get_dataloaders(self):
        """Build train/val dataloaders from config."""
        cfg = self.cfg


        train_dir = cfg.data.get("train_dir", "data/knee_singlecoil_train")
        val_dir = cfg.data.get("val_dir", None)


        data_kwargs = dict(
            center_fraction=cfg.data.get("center_fraction", 0.08),
            acceleration=cfg.data.get("acceleration", 4),
            crop_size=tuple(cfg.data.get("crop_size", [320, 320])),
            seed=cfg.training.get("seed", 42),
        )


        loader_kwargs = dict(
            batch_size=cfg.training.batch_size,
            num_workers=cfg.data.get("num_workers", 4),
            pin_memory=cfg.data.get("pin_memory", True),
        )


        # If separate val dir exists, use it; otherwise split train
        if val_dir and os.path.isdir(val_dir):
            train_ds = FastMRIKneeDataset(train_dir, **data_kwargs)
            val_ds = FastMRIKneeDataset(val_dir, **data_kwargs)
        else:
            full_ds = FastMRIKneeDataset(train_dir, **data_kwargs)
            val_frac = cfg.training.get("val_split", 0.1)
            n_val = max(1, int(len(full_ds) * val_frac))
            n_train = len(full_ds) - n_val
            train_ds, val_ds = random_split(
                full_ds, [n_train, n_val],
                generator=torch.Generator().manual_seed(cfg.training.get("seed", 42))
            )


        train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)


        log.info(f"Data: train={len(train_ds)}, val={len(val_ds)}, "
                 f"batch_size={cfg.training.batch_size}")
        return train_loader, val_loader


    def train_one_epoch(self, loader, epoch: int):
        """Train for one epoch with AMP + gradient accumulation + EMA."""
        self.model.train()
        total_loss = 0.0
        do_augment = self.cfg.training.get("augmentation", True)
        grad_clip = self.cfg.training.get("gradient_clip", 1.0)
        log_interval = self.cfg.logging.get("log_interval", 10)
        accum_steps = self.grad_accum_steps


        pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
        self.optimizer.zero_grad(set_to_none=True)


        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(self.device), y.to(self.device)
            if do_augment:
                x, y = augment(x, y)


            with autocast("cuda", enabled=self.use_amp):
                pred = self.model(x)
                loss = self.criterion(pred, y) / accum_steps


            self.scaler.scale(loss).backward()


            # Step optimizer every accum_steps batches
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(loader):
                if grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)


                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)


                # Update EMA after optimizer step
                if self.use_ema:
                    self.ema.update()


            total_loss += loss.item() * accum_steps
            self.global_step += 1


            # Batch-level logging
            if batch_idx % log_interval == 0:
                self.logger.log_scalars(
                    {"train/batch_loss": loss.item() * accum_steps,
                     "train/lr": self.optimizer.param_groups[0]["lr"]},
                    step=self.global_step,
                )


            pbar.set_postfix(loss=f"{loss.item() * accum_steps:.4f}")


        return total_loss / len(loader)


    u/torch.no_grad()
    def validate(self, loader):
        """Validate using EMA weights if available."""
        self.model.eval()
        total_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0


        # Use EMA weights for validation (better generalization)
        ctx = self.ema.average_parameters() if self.use_ema else nullcontext()
        with ctx:
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)


                with autocast("cuda", enabled=self.use_amp):
                    pred = self.model(x)
                    loss = self.criterion(pred, y)


                total_loss += loss.item()
                total_psnr += psnr(pred, y)
                total_ssim += ssim_metric(pred, y)


        n = len(loader)
        return {
            "val_loss": total_loss / n,
            "val_psnr": total_psnr / n,
            "val_ssim": total_ssim / n,
        }


    def fit(self):
        """Full training loop."""
        cfg = self.cfg
        epochs = cfg.training.epochs
        patience = cfg.training.get("patience", 10)


        train_loader, val_loader = self._get_dataloaders()


        log.info(f"Training for {epochs} epochs | AMP: {self.use_amp} | Patience: {patience}")
        log.info(f"Output: {self.logger.run_dir}")


        header = f"{'Epoch':>6} {'Train Loss':>11} {'Val Loss':>9} {'PSNR':>7} {'SSIM':>7} {'LR':>10} {'Time':>6}"
        print(header)
        print("─" * len(header))


        for epoch in range(self.start_epoch, epochs + 1):
            t0 = time.time()


            train_loss = self.train_one_epoch(train_loader, epoch)
            metrics = self.validate(val_loader)
            elapsed = time.time() - t0


            val_loss = metrics["val_loss"]
            val_psnr = metrics["val_psnr"]
            val_ssim = metrics["val_ssim"]
            current_lr = self.optimizer.param_groups[0]["lr"]


            # Step scheduler
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()


            # Epoch-level logging
            self.logger.log_scalars({
                "epoch/train_loss": train_loss,
                "epoch/val_loss": val_loss,
                "epoch/val_psnr": val_psnr,
                "epoch/val_ssim": val_ssim,
                "epoch/lr": current_lr,
            }, step=epoch)


            print(f"{epoch:>6} {train_loss:>11.4f} {val_loss:>9.4f} "
                  f"{val_psnr:>7.2f} {val_ssim:>7.4f} {current_lr:>10.2e} {elapsed:>5.1f}s")


            # Checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1


            self.logger.save_checkpoint(
                state={
                    "epoch": epoch,
                    "global_step": self.global_step,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "scheduler_state": self.scheduler.state_dict(),
                    "scaler_state": self.scaler.state_dict(),
                    "ema_state": self.ema.state_dict() if self.use_ema else None,
                    "val_loss": val_loss,
                    "val_psnr": val_psnr,
                    "val_ssim": val_ssim,
                    "config": dict(cfg),
                },
                filename=f"epoch_{epoch:03d}.pt",
                is_best=is_best,
            )


            # Early stopping
            if self.patience_counter >= patience:
                log.info(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break


        log.info(f"Training complete. Best val_loss: {self.best_val_loss:.4f}")
        log.info(f"Best checkpoint: {self.logger.checkpoint_dir / 'best.pt'}")
        self.logger.finish()


        return {
            "best_val_loss": self.best_val_loss,
            "final_epoch": epoch,
            "run_dir": str(self.logger.run_dir),
        }


    def resume(self, checkpoint_path: str):
        """Resume training from a checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
        if "scaler_state" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state"])
        if self.use_ema and ckpt.get("ema_state"):
            self.ema.load_state_dict(ckpt["ema_state"])
            log.info("Restored EMA state from checkpoint")
        self.start_epoch = ckpt["epoch"] + 1
        self.global_step = ckpt.get("global_step", 0)
        self.best_val_loss = ckpt.get("val_loss", float("inf"))
        log.info(f"Resumed from {checkpoint_path} at epoch {self.start_epoch} (global_step={self.global_step})")



# ---------------------------------------------------------------------------
# Legacy functional API (backward compatibility with main.py)
# ---------------------------------------------------------------------------


def train(
    model_name: str = "swinunet",
    data_dir: str = "data/knee_singlecoil_train",
    output_dir: str = "checkpoints",
    epochs: int = 50,
    batch_size: int = 4,
    lr: float = 8e-5,
    augment_data: bool = True,
):
    """Legacy training entry point for backward compatibility."""
    cfg = load_config(overrides={
        "model": {"name": model_name},
        "data": {"train_dir": data_dir},
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "augmentation": augment_data,
        },
        "logging": {
            "output_dir": output_dir,
            "experiment_name": model_name,
        },
    })
    trainer = Trainer(cfg)
    return trainer.fit()



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse


    parser = argparse.ArgumentParser(description="Train MRI Reconstruction Model")
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("overrides", nargs="*", help="Config overrides: key=value (dot notation)")
    args = parser.parse_args()


    cfg = load_config(args.config, cli_overrides=args.overrides)
    trainer = Trainer(cfg)


    if args.resume:
        trainer.resume(args.resume)


    trainer.fit()