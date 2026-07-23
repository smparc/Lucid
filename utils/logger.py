"""
logger.py
---------
Unified experiment logging with TensorBoard and Weights & Biases support.


Usage
-----
    from utils.logger import ExperimentLogger
    logger = ExperimentLogger(cfg.logging)
    logger.log_scalars({"train/loss": 0.05, "val/psnr": 30.5}, step=10)
    logger.log_image("val/reconstruction", img_tensor, step=10)
    logger.finish()
"""


import os
import json
import logging
from pathlib import Path
from typing import Optional


import torch
import numpy as np


log = logging.getLogger(__name__)



class ExperimentLogger:
    """
    Unified logger that dispatches to TensorBoard and/or W&B.


    Parameters
    ----------
    config : dict-like with keys:
        output_dir, experiment_name, tensorboard, wandb,
        wandb_project, wandb_entity, log_interval, save_top_k
    """


    def __init__(self, config, full_config: dict = None):
        self.cfg = config
        self.output_dir = Path(config.get("output_dir", "outputs"))
        self.experiment_name = config.get("experiment_name", "experiment")
        self.run_dir = self.output_dir / self.experiment_name
        self.run_dir.mkdir(parents=True, exist_ok=True)


        self.tb_writer = None
        self.wandb_run = None


        # Save full config
        if full_config:
            config_path = self.run_dir / "config.yaml"
            import yaml
            with open(config_path, "w") as f:
                yaml.dump(dict(full_config), f, default_flow_style=False)


        # TensorBoard
        if config.get("tensorboard", False):
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = self.run_dir / "tensorboard"
                self.tb_writer = SummaryWriter(log_dir=str(tb_dir))
                log.info(f"TensorBoard logging to {tb_dir}")
            except ImportError:
                log.warning("tensorboard not installed, skipping TB logging")


        # Weights & Biases
        if config.get("wandb", False):
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=config.get("wandb_project", "lucid-mri"),
                    entity=config.get("wandb_entity"),
                    name=self.experiment_name,
                    config=full_config or {},
                    dir=str(self.run_dir),
                    reinit=True,
                )
                log.info(f"W&B logging to project={config.get('wandb_project')}")
            except ImportError:
                log.warning("wandb not installed, skipping W&B logging")
            except Exception as e:
                log.warning(f"wandb init failed: {e}")


    u/property
    def checkpoint_dir(self) -> Path:
        d = self.run_dir / "checkpoints"
        d.mkdir(exist_ok=True)
        return d


    def log_scalars(self, metrics: dict, step: int):
        """Log scalar metrics to all active backends."""
        if self.tb_writer:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(key, value, global_step=step)


        if self.wandb_run:
            import wandb
            wandb.log(metrics, step=step)


    def log_image(self, tag: str, image: torch.Tensor, step: int):
        """
        Log a single image.


        image : (C, H, W) or (H, W) tensor in [0, 1]
        """
        if image.dim() == 2:
            image = image.unsqueeze(0)


        if self.tb_writer:
            self.tb_writer.add_image(tag, image, global_step=step)


        if self.wandb_run:
            import wandb
            img_np = image.permute(1, 2, 0).cpu().numpy()
            if img_np.shape[-1] == 1:
                img_np = img_np.squeeze(-1)
            wandb.log({tag: wandb.Image(img_np)}, step=step)


    def log_images_grid(self, tag: str, images: list, step: int, captions: list = None):
        """Log a grid of images."""
        if self.wandb_run:
            import wandb
            wandb_images = []
            for i, img in enumerate(images):
                if isinstance(img, torch.Tensor):
                    img = img.cpu().numpy()
                if img.ndim == 3 and img.shape[0] in (1, 3):
                    img = img.transpose(1, 2, 0)
                if img.ndim == 3 and img.shape[-1] == 1:
                    img = img.squeeze(-1)
                caption = captions[i] if captions else None
                wandb_images.append(wandb.Image(img, caption=caption))
            wandb.log({tag: wandb_images}, step=step)


    def log_model_graph(self, model: torch.nn.Module, input_tensor: torch.Tensor):
        """Log model computation graph (TensorBoard only)."""
        if self.tb_writer:
            try:
                self.tb_writer.add_graph(model, input_tensor)
            except Exception as e:
                log.warning(f"Failed to log model graph: {e}")


    def save_checkpoint(
        self,
        state: dict,
        filename: str,
        is_best: bool = False,
    ) -> Path:
        """Save checkpoint and manage top-K retention."""
        path = self.checkpoint_dir / filename
        torch.save(state, path)


        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(state, best_path)


        # Top-K retention: delete oldest non-best epoch checkpoints
        save_top_k = self.cfg.get("save_top_k", 0)
        if save_top_k > 0:
            epoch_ckpts = sorted(
                [f for f in self.checkpoint_dir.glob("epoch_*.pt")],
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )
            for old_ckpt in epoch_ckpts[save_top_k:]:
                old_ckpt.unlink()


        return path


    def log_hyperparams(self, hparams: dict, metrics: dict = None):
        """Log hyperparameters (for HP search dashboards)."""
        if self.tb_writer and metrics:
            self.tb_writer.add_hparams(hparams, metrics)


    def finish(self):
        """Close all logging backends."""
        if self.tb_writer:
            self.tb_writer.close()
        if self.wandb_run:
            import wandb
            wandb.finish()