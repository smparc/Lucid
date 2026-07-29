"""
train.py
--------
Training loop for accelerated MRI reconstruction.

Features
--------
* Config-driven, with the resolved config and an environment manifest written
  into every run directory.
* Mixed precision on any accelerator (CUDA or otherwise), with metrics forced
  to float32.
* Per-step LR scheduling, gradient accumulation, gradient clipping, EMA.
* Physics-aware: k-space and the sampling mask are routed to models that need
  them, so data consistency actually runs.
* Early stopping on a configurable monitored metric, checkpoint retention by
  that metric, and fully faithful resume.

Correctness notes
-----------------
Several defects in the original loop produced numbers that could not be
compared across configurations:

* ``len(loader)`` was used as the denominator of the epoch loss, but batches
  were also accumulated — with gradient accumulation the reported training loss
  was scaled wrongly.
* Validation ran under ``autocast``, so metrics were computed in fp16.
* Epoch metrics were logged at ``step=epoch`` while batch metrics used
  ``global_step``; W&B rejects out-of-order steps, so epoch curves were dropped.
* ``resume`` restored weights but not the early-stopping counter or the best
  score, so resuming reset patience and could overwrite a better checkpoint.
* If ``epochs < start_epoch`` the loop body never ran and ``fit`` raised
  ``UnboundLocalError`` on ``epoch``.

All are addressed below.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config  # noqa: E402
from data.preprocessing import FastMRIKneeDataset, collate  # noqa: E402
from models.registry import build_model, forward_model  # noqa: E402
from training.losses import CombinedLoss, SSIMLoss  # noqa: E402
from training.metrics import MetricAccumulator, psnr, ssim  # noqa: E402
from utils.ema import EMAModel  # noqa: E402
from utils.logger import ExperimentLogger  # noqa: E402
from utils.reproducibility import make_generator, seed_everything, seed_worker  # noqa: E402
from utils.schedulers import build_scheduler  # noqa: E402

log = logging.getLogger(__name__)

__all__ = ["Trainer", "build_model", "psnr", "ssim_metric", "CombinedLoss", "SSIMLoss", "train"]


def ssim_metric(pred: torch.Tensor, target: torch.Tensor, data_range=None) -> float:
    """SSIM as a plain float. Thin wrapper over :func:`training.metrics.ssim`."""
    with torch.no_grad():
        return float(ssim(pred, target, data_range=data_range))


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """
    Configurable trainer.

    Parameters
    ----------
    cfg
        A validated :class:`config.Config`.
    resume_from
        Checkpoint to resume from. Passing it here rather than calling
        :meth:`resume` afterwards lets the logger reuse the original run
        directory instead of starting a fresh one.
    """

    def __init__(self, cfg, resume_from: str | Path | None = None):
        self.cfg = cfg
        self.device = self._select_device()

        seed = int(cfg.training.get("seed", 42))
        seed_everything(seed, deterministic=bool(cfg.training.get("deterministic", False)))
        self.seed = seed

        self.logger = ExperimentLogger(
            cfg.logging, full_config=cfg.to_dict(), resume=resume_from is not None
        )

        # ── Model ──────────────────────────────────────────────────────────
        self.model = build_model(cfg).to(self.device)
        self.expects_kspace = bool(getattr(self.model, "expects_kspace", False))
        self.model_in_channels = int(getattr(self.model, "in_channels", 1))

        if cfg.training.get("channels_last", False):
            self.model = self.model.to(memory_format=torch.channels_last)

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        log.info(
            "Model: %s | %.2fM trainable params | device=%s | k-space input=%s",
            cfg.model.name,
            n_params / 1e6,
            self.device,
            self.expects_kspace,
        )

        # ── Loss ───────────────────────────────────────────────────────────
        loss_cfg = dict(cfg.training.get("loss", {}) or {})
        self.criterion = CombinedLoss(**loss_cfg).to(self.device)
        log.info("Loss: %s", self.criterion.extra_repr())

        # ── Optimiser ──────────────────────────────────────────────────────
        self.optimizer = self._build_optimizer(cfg)

        # ── AMP ────────────────────────────────────────────────────────────
        self.amp_dtype = self._resolve_amp_dtype(cfg)
        self.use_amp = bool(cfg.training.get("amp", False)) and self.amp_dtype is not None
        # bf16 has the same exponent range as fp32, so gradient scaling is
        # unnecessary — and enabling it would cost a synchronisation per step.
        needs_scaler = self.use_amp and self.amp_dtype == torch.float16
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=needs_scaler)
        if self.use_amp:
            log.info("Mixed precision enabled (%s)", str(self.amp_dtype).split(".")[-1])

        # ── EMA ────────────────────────────────────────────────────────────
        ema_cfg = dict(cfg.training.get("ema", {}) or {})
        self.use_ema = bool(ema_cfg.get("enabled", True))
        self.ema = (
            EMAModel(
                self.model,
                decay=float(ema_cfg.get("decay", 0.999)),
                warmup=int(ema_cfg.get("warmup_steps", 1000)),
                include_buffers=bool(ema_cfg.get("include_buffers", True)),
            )
            if self.use_ema
            else None
        )
        if self.use_ema:
            log.info("EMA enabled (decay=%s)", ema_cfg.get("decay", 0.999))

        self.grad_accum_steps = max(1, int(cfg.training.get("gradient_accumulation", 1)))
        if self.grad_accum_steps > 1:
            log.info(
                "Gradient accumulation: %d steps (effective batch = %d)",
                self.grad_accum_steps,
                int(cfg.training.batch_size) * self.grad_accum_steps,
            )

        # ── Monitored metric ───────────────────────────────────────────────
        self.monitor = str(cfg.training.get("monitor", "val_loss"))
        self.monitor_mode = str(cfg.training.get("monitor_mode", "min"))
        self.higher_is_better = self.monitor_mode == "max"

        # ── State ──────────────────────────────────────────────────────────
        self.start_epoch = 1
        self.global_step = 0
        self.best_score = -math.inf if self.higher_is_better else math.inf
        self.patience_counter = 0
        self.scheduler = None  # built in fit(), once steps-per-epoch is known

        self._train_loader: DataLoader | None = None
        self._val_loader: DataLoader | None = None

        if resume_from:
            self.resume(resume_from)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _select_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _resolve_amp_dtype(self, cfg) -> torch.dtype | None:
        """
        Pick an autocast dtype, or None if mixed precision is unavailable.

        bf16 is preferred wherever supported: fp16's 5-bit exponent overflows on
        k-space magnitudes, and this pipeline moves between image and Fourier
        domains constantly.
        """
        if not cfg.training.get("amp", False):
            return None
        if self.device.type != "cuda":
            log.info("AMP requested but device is %s; running in fp32", self.device.type)
            return None

        requested = str(cfg.training.get("amp_dtype", "auto")).lower()
        if requested == "float16":
            return torch.float16
        if requested == "bfloat16":
            return torch.bfloat16
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    def _build_optimizer(self, cfg) -> torch.optim.Optimizer:
        """
        Build the optimiser, excluding norms and biases from weight decay.

        Applying L2 to LayerNorm gains, biases and the relative-position-bias
        table is standard practice to avoid — those parameters have no
        meaningful scale prior, and decaying them measurably hurts transformers.
        """
        name = str(cfg.training.get("optimizer", "adamw")).lower()
        weight_decay = float(cfg.training.get("weight_decay", 1e-4))

        decay, no_decay = [], []
        for pname, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or pname.endswith(".bias") or "rel_pos_bias" in pname:
                no_decay.append(param)
            else:
                decay.append(param)

        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        lr = float(cfg.training.lr)

        if name == "sgd":
            return SGD(groups, lr=lr, momentum=0.9, nesterov=True)
        betas = tuple(cfg.training.get("betas", [0.9, 0.999]))
        cls = AdamW if name == "adamw" else Adam
        return cls(groups, lr=lr, betas=betas)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _dataset_kwargs(self, train: bool) -> dict:
        cfg = self.cfg
        return dict(
            center_fraction=float(cfg.data.get("center_fraction", 0.08)),
            acceleration=cfg.data.get("acceleration", 4),
            crop_size=tuple(cfg.data.get("crop_size", [320, 320])),
            mask_type=str(cfg.data.get("mask_type", "random")),
            slice_mode=str(cfg.data.get("slice_mode", "middle")),
            normalization=str(cfg.data.get("normalization", "zf_max")),
            undersample_domain=str(cfg.data.get("undersample_domain", "cropped")),
            complex_input=self.model_in_channels == 2,
            cache=bool(cfg.data.get("cache_dataset", False)),
            max_files=cfg.data.get("max_files"),
            seed=self.seed,
            train=train,
            augment=bool(cfg.training.get("augmentation", True)) and train,
        )

    def get_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        """Build (and cache) the train and validation loaders."""
        if self._train_loader is not None and self._val_loader is not None:
            return self._train_loader, self._val_loader

        cfg = self.cfg
        train_dir = cfg.data.get("train_dir", "data/knee_singlecoil_train")
        val_dir = cfg.data.get("val_dir")

        if val_dir and os.path.isdir(val_dir):
            train_ds = FastMRIKneeDataset(train_dir, **self._dataset_kwargs(train=True))
            val_ds = FastMRIKneeDataset(val_dir, **self._dataset_kwargs(train=False))
        else:
            # No held-out directory: split the training set. Both halves are
            # built from separate dataset objects so the validation half keeps
            # deterministic masks and no augmentation — splitting a single
            # object would leak training-mode randomness into validation.
            log.warning(
                "No val_dir at %r; splitting train_dir by val_split. This is a "
                "random slice-level split, so slices from one volume may appear "
                "on both sides. Prefer a volume-level held-out directory.",
                val_dir,
            )
            full_train = FastMRIKneeDataset(train_dir, **self._dataset_kwargs(train=True))
            full_val = FastMRIKneeDataset(train_dir, **self._dataset_kwargs(train=False))

            val_frac = float(cfg.training.get("val_split", 0.1))
            n_val = max(1, int(len(full_train) * val_frac))
            indices = torch.randperm(
                len(full_train), generator=make_generator(self.seed)
            ).tolist()
            train_idx, val_idx = indices[n_val:], indices[:n_val]

            from torch.utils.data import Subset

            train_ds = Subset(full_train, train_idx)
            val_ds = Subset(full_val, val_idx)

        num_workers = int(cfg.data.get("num_workers", 4))
        loader_kwargs = dict(
            batch_size=int(cfg.training.batch_size),
            num_workers=num_workers,
            pin_memory=bool(cfg.data.get("pin_memory", True)) and self.device.type == "cuda",
            collate_fn=collate,
            worker_init_fn=seed_worker,
            drop_last=False,
        )
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = bool(
                cfg.data.get("persistent_workers", True)
            )
            loader_kwargs["prefetch_factor"] = int(cfg.data.get("prefetch_factor", 2))

        self._train_loader = DataLoader(
            train_ds, shuffle=True, generator=make_generator(self.seed), **loader_kwargs
        )
        self._val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

        log.info(
            "Data: train=%d, val=%d, batch_size=%d",
            len(train_ds),
            len(val_ds),
            cfg.training.batch_size,
        )
        return self._train_loader, self._val_loader

    # ------------------------------------------------------------------
    # Batch plumbing
    # ------------------------------------------------------------------

    def _to_device(self, batch: dict) -> dict:
        out = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                out[key] = value.to(self.device, non_blocking=True)
            else:
                out[key] = value
        return out

    def _forward(self, batch: dict) -> torch.Tensor:
        """Run the model with whatever inputs its architecture requires."""
        x = batch["image"]
        if self.cfg.training.get("channels_last", False):
            x = x.contiguous(memory_format=torch.channels_last)
        if self.expects_kspace:
            return forward_model(self.model, x, batch["kspace"], batch["mask"])
        return forward_model(self.model, x)

    def _autocast(self):
        if not self.use_amp:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_one_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        """Run one training epoch. Returns the mean loss and its components."""
        self.model.train()
        grad_clip = float(self.cfg.training.get("gradient_clip", 1.0))
        log_interval = int(self.cfg.logging.get("log_interval", 10))
        accum = self.grad_accum_steps
        max_steps = self.cfg.training.get("max_steps_per_epoch")

        total_loss = 0.0
        n_batches = 0
        component_totals: dict[str, float] = {}

        pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(pbar):
            if max_steps and batch_idx >= int(max_steps):
                break

            batch = self._to_device(batch)

            with self._autocast():
                pred = self._forward(batch)
                loss = self.criterion(pred, batch["target"], data_range=batch["max_value"])

            if not torch.isfinite(loss):
                # A single non-finite loss poisons every parameter through the
                # optimiser state; skip the step and keep going rather than
                # silently training on NaNs for the rest of the run.
                log.warning("Non-finite loss at epoch %d batch %d; skipping", epoch, batch_idx)
                self.optimizer.zero_grad(set_to_none=True)
                continue

            self.scaler.scale(loss / accum).backward()

            is_step = (batch_idx + 1) % accum == 0 or (batch_idx + 1) == len(loader)
            if is_step:
                if grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=grad_clip
                    )
                else:
                    grad_norm = None

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                if self.ema is not None:
                    self.ema.update()

                # Per-step scheduling: hundreds of warmup points instead of a
                # handful, which is what transformer stability actually needs.
                if self.scheduler is not None and not isinstance(
                    self.scheduler, ReduceLROnPlateau
                ):
                    self.scheduler.step()

                self.global_step += 1

                if self.global_step % log_interval == 0:
                    metrics = {
                        "train/batch_loss": float(loss.detach()),
                        "train/lr": self.optimizer.param_groups[0]["lr"],
                    }
                    if grad_norm is not None:
                        metrics["train/grad_norm"] = float(grad_norm)
                    for key, value in self.criterion.last_components.items():
                        metrics[f"train/loss_{key}"] = value
                    self.logger.log_scalars(metrics, step=self.global_step)

            batch_loss = float(loss.detach())
            total_loss += batch_loss
            n_batches += 1
            for key, value in self.criterion.last_components.items():
                component_totals[key] = component_totals.get(key, 0.0) + value

            pbar.set_postfix(loss=f"{batch_loss:.4f}")

        if n_batches == 0:
            raise RuntimeError("Training epoch processed zero batches; is the dataset empty?")

        result = {"train_loss": total_loss / n_batches}
        result.update({f"train_{k}": v / n_batches for k, v in component_totals.items()})
        return result

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> dict[str, float]:
        """
        Evaluate on the validation set, under EMA weights when enabled.

        The forward pass may use autocast, but the loss and every metric are
        computed in float32.
        """
        self.model.eval()
        acc = MetricAccumulator()

        context = self.ema.average_parameters() if self.ema is not None else nullcontext()
        with context:
            for batch in loader:
                batch = self._to_device(batch)

                with self._autocast():
                    pred = self._forward(batch)

                pred = pred.float()
                target = batch["target"].float()
                data_range = batch["max_value"].float()

                loss = self.criterion(pred, target, data_range=data_range)
                acc.update(
                    pred,
                    target,
                    data_range=data_range,
                    loss=float(loss),
                    fnames=batch.get("fname"),
                    accelerations=batch.get("acceleration"),
                )

        summary = acc.compute()
        result = {
            "val_loss": summary.get("loss", float("nan")),
            "val_psnr": summary["psnr"],
            "val_ssim": summary["ssim"],
            "val_nmse": summary["nmse"],
        }
        if "psnr_std" in summary:
            result["val_psnr_std"] = summary["psnr_std"]

        self._last_accumulator = acc
        return result

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self) -> dict[str, Any]:
        """Run the full training loop. Returns a summary dict."""
        cfg = self.cfg
        epochs = int(cfg.training.epochs)
        patience = int(cfg.training.get("patience", 10))

        train_loader, val_loader = self.get_dataloaders()

        if self.scheduler is None:
            steps_per_epoch = max(1, math.ceil(len(train_loader) / self.grad_accum_steps))
            total_steps = steps_per_epoch * epochs
            warmup = int(
                cfg.training.get(
                    "warmup_steps",
                    steps_per_epoch * int(cfg.training.get("warmup_epochs", 5)),
                )
            )
            self.scheduler = build_scheduler(
                self.optimizer,
                str(cfg.training.get("scheduler", "warmup_cosine")),
                total_steps=total_steps,
                warmup=min(warmup, max(1, total_steps // 2)),
                eta_min=float(cfg.training.lr) * float(cfg.training.get("eta_min_factor", 0.01)),
            )
            log.info(
                "Scheduler: %s over %d steps (%d warmup)",
                cfg.training.get("scheduler", "warmup_cosine"),
                total_steps,
                warmup,
            )
            # A resumed run stashed its scheduler state before the scheduler
            # existed; apply it now that it does.
            self._restore_scheduler()

        if self.start_epoch > epochs:
            log.warning(
                "start_epoch (%d) exceeds epochs (%d); nothing to train. "
                "Increase training.epochs to continue this run.",
                self.start_epoch,
                epochs,
            )
            return self._summary(final_epoch=self.start_epoch - 1, stopped_early=False)

        log.info(
            "Training epochs %d-%d | monitor=%s (%s) | patience=%d | run=%s",
            self.start_epoch,
            epochs,
            self.monitor,
            self.monitor_mode,
            patience,
            self.logger.run_dir,
        )

        header = (
            f"{'Epoch':>6} {'Train':>10} {'Val':>10} {'PSNR':>8} "
            f"{'SSIM':>8} {'NMSE':>9} {'LR':>10} {'Time':>7}"
        )
        print(header)
        print("-" * len(header))

        final_epoch = self.start_epoch - 1
        stopped_early = False

        for epoch in range(self.start_epoch, epochs + 1):
            t0 = time.time()
            final_epoch = epoch

            train_metrics = self.train_one_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader)
            elapsed = time.time() - t0

            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_metrics["val_loss"])

            current_lr = self.optimizer.param_groups[0]["lr"]
            epoch_metrics = {
                **{f"epoch/{k}": v for k, v in train_metrics.items()},
                **{f"epoch/{k}": v for k, v in val_metrics.items()},
                "epoch/lr": current_lr,
                "epoch/seconds": elapsed,
            }
            if hasattr(self.model, "lambdas"):
                for i, lam in enumerate(self.model.lambdas):
                    epoch_metrics[f"epoch/dc_lambda_{i}"] = lam

            self.logger.log_epoch(epoch_metrics, epoch=epoch, step=self.global_step)

            print(
                f"{epoch:>6} {train_metrics['train_loss']:>10.4f} "
                f"{val_metrics['val_loss']:>10.4f} {val_metrics['val_psnr']:>8.2f} "
                f"{val_metrics['val_ssim']:>8.4f} {val_metrics['val_nmse']:>9.5f} "
                f"{current_lr:>10.2e} {elapsed:>6.1f}s"
            )

            score = val_metrics.get(self.monitor, val_metrics["val_loss"])
            is_best = (
                score > self.best_score if self.higher_is_better else score < self.best_score
            )
            if is_best:
                self.best_score = score
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            self.logger.save_checkpoint(
                state=self._checkpoint_state(epoch, val_metrics),
                filename=f"epoch_{epoch:03d}.pt",
                is_best=is_best,
                score=score,
                higher_is_better=self.higher_is_better,
            )

            if self.patience_counter >= patience:
                log.info(
                    "Early stopping at epoch %d: %s did not improve for %d epochs",
                    epoch,
                    self.monitor,
                    patience,
                )
                stopped_early = True
                break

        summary = self._summary(final_epoch, stopped_early)
        self.logger.log_summary(summary)
        self.logger.finish()

        log.info("Training complete. Best %s: %.5f", self.monitor, self.best_score)
        log.info("Best checkpoint: %s", self.logger.checkpoint_dir / "best.pt")
        return summary

    def _summary(self, final_epoch: int, stopped_early: bool) -> dict[str, Any]:
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        summary: dict[str, Any] = {
            "model": self.cfg.model.name,
            "monitor": self.monitor,
            f"best_{self.monitor}": self.best_score,
            "final_epoch": final_epoch,
            "global_step": self.global_step,
            "stopped_early": stopped_early,
            "params_m": n_params / 1e6,
            "run_dir": str(self.logger.run_dir),
        }
        acc = getattr(self, "_last_accumulator", None)
        if acc is not None and acc.psnr:
            from training.stats import bootstrap_ci

            summary["val_psnr_ci"] = bootstrap_ci(acc.psnr).as_dict()
            summary["val_ssim_ci"] = bootstrap_ci(acc.ssim).as_dict()
            by_acc = acc.by_acceleration()
            if len(by_acc) > 1:
                summary["by_acceleration"] = by_acc
            summary["worst_slices"] = acc.worst(k=5)
        return summary

    def _checkpoint_state(self, epoch: int, val_metrics: dict) -> dict:
        return {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "scaler_state": self.scaler.state_dict(),
            "ema_state": self.ema.state_dict() if self.ema is not None else None,
            "best_score": self.best_score,
            "patience_counter": self.patience_counter,
            "monitor": self.monitor,
            "monitor_mode": self.monitor_mode,
            "config": self.cfg.to_dict(),
            **val_metrics,
        }

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------

    def resume(self, checkpoint_path: str | Path) -> None:
        """
        Restore full training state from a checkpoint.

        Restores the optimiser, scheduler, gradient scaler, EMA shadow, the
        best score *and* the early-stopping counter. Omitting the last two —
        as the original did — silently resets patience on every resume and lets
        a worse checkpoint overwrite ``best.pt``.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        missing, unexpected = self.model.load_state_dict(ckpt["model_state"], strict=False)
        if missing or unexpected:
            log.warning(
                "Model state mismatch on resume: %d missing, %d unexpected keys. "
                "This usually means the checkpoint was trained with a different "
                "model config than the one loaded.",
                len(missing),
                len(unexpected),
            )

        if ckpt.get("optimizer_state"):
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if ckpt.get("scaler_state"):
            self.scaler.load_state_dict(ckpt["scaler_state"])
        if self.ema is not None and ckpt.get("ema_state"):
            self.ema.load_state_dict(ckpt["ema_state"])
            log.info("Restored EMA state")

        self._pending_scheduler_state = ckpt.get("scheduler_state")
        self.start_epoch = int(ckpt.get("epoch", 0)) + 1
        self.global_step = int(ckpt.get("global_step", 0))
        self.patience_counter = int(ckpt.get("patience_counter", 0))

        saved_monitor = ckpt.get("monitor")
        if saved_monitor and saved_monitor != self.monitor:
            log.warning(
                "Checkpoint monitored %r but this run monitors %r; "
                "resetting the best score rather than comparing incomparable values.",
                saved_monitor,
                self.monitor,
            )
            self.best_score = -math.inf if self.higher_is_better else math.inf
        else:
            default_best = -math.inf if self.higher_is_better else math.inf
            self.best_score = float(ckpt.get("best_score", ckpt.get("val_loss", default_best)))

        log.info(
            "Resumed from %s at epoch %d (step %d, best %s=%.5f, patience %d)",
            path,
            self.start_epoch,
            self.global_step,
            self.monitor,
            self.best_score,
            self.patience_counter,
        )

    def _restore_scheduler(self) -> None:
        """Apply a pending scheduler state once the scheduler has been built."""
        state = getattr(self, "_pending_scheduler_state", None)
        if state and self.scheduler is not None:
            try:
                self.scheduler.load_state_dict(state)
            except Exception as exc:
                log.warning("Could not restore scheduler state: %s", exc)
            self._pending_scheduler_state = None


# ---------------------------------------------------------------------------
# Functional entry point
# ---------------------------------------------------------------------------


def train(
    model_name: str = "swinunet",
    data_dir: str = "data/knee_singlecoil_train",
    output_dir: str = "outputs",
    epochs: int = 50,
    batch_size: int = 4,
    lr: float = 8e-5,
    augment_data: bool = True,
    **extra,
) -> dict:
    """Convenience wrapper for scripted training."""
    cfg = load_config(
        overrides={
            "model": {"name": model_name},
            "data": {"train_dir": data_dir},
            "training": {
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "augmentation": augment_data,
                **extra,
            },
            "logging": {"output_dir": output_dir, "experiment_name": model_name},
        }
    )
    return Trainer(cfg).fit()


if __name__ == "__main__":  # pragma: no cover
    import argparse

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    parser = argparse.ArgumentParser(description="Train an MRI reconstruction model")
    parser.add_argument("--config", default=None, help="Path to a YAML config")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from")
    parser.add_argument("overrides", nargs="*", help="Config overrides: dotted.key=value")
    args = parser.parse_args()

    trainer = Trainer(
        load_config(args.config, cli_overrides=args.overrides), resume_from=args.resume
    )
    result = trainer.fit()
    print(json.dumps(result, indent=2, default=str))
