"""
logger.py
---------
Unified experiment logging: TensorBoard, Weights & Biases, and a plain-JSON
history file that is always written regardless of which backends are enabled.

Design notes
------------
* **One monotonic step axis.** W&B rejects out-of-order steps: logging batch
  metrics at ``global_step`` (tens of thousands) and then epoch metrics at
  ``epoch`` (tens) causes every epoch-level point after the first to be silently
  dropped. Everything here is logged against ``global_step``, with ``epoch``
  recorded as an ordinary metric so it remains available as an x-axis.
* **History is a first-class artifact.** ``history.json`` is written after every
  epoch, so plotting and post-hoc analysis never depend on a live TensorBoard
  event file or a network service. It is also crash-resilient: a run killed at
  epoch 30 still leaves 30 usable epochs on disk.
* **Top-k checkpoints by metric, not by mtime.** Retaining the *most recent* k
  checkpoints is not the same as retaining the *best* k, and quietly deleting a
  better checkpoint because it happens to be older is exactly the kind of bug
  that surfaces only when you need the weights back.
* **Unique run directories.** Reusing a run directory silently interleaves two
  experiments' checkpoints and TensorBoard scalars. New runs get a timestamp
  suffix unless the caller explicitly opts into resuming.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

log = logging.getLogger(__name__)


class ExperimentLogger:
    """
    Dispatches metrics to TensorBoard and/or W&B, and owns the run directory.

    Parameters
    ----------
    config
        dict-like with keys: ``output_dir``, ``experiment_name``, ``tensorboard``,
        ``wandb``, ``wandb_project``, ``wandb_entity``, ``log_interval``,
        ``save_top_k``, ``save_every``, ``unique_run_dir``.
    full_config
        The complete resolved config, persisted next to the checkpoints.
    resume
        If True, reuse the existing run directory instead of creating a new one.
    """

    def __init__(self, config, full_config: dict | None = None, resume: bool = False):
        self.cfg = config
        self.output_dir = Path(config.get("output_dir", "outputs"))
        self.experiment_name = config.get("experiment_name", "experiment")

        base = self.output_dir / self.experiment_name
        if base.exists() and not resume and config.get("unique_run_dir", True):
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            base = self.output_dir / f"{self.experiment_name}_{stamp}"
        self.run_dir = base
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.tb_writer = None
        self.wandb_run = None
        self.history: dict[str, list] = {}
        self._ckpt_scores: dict[str, float] = {}
        self._history_path = self.run_dir / "history.json"

        if resume and self._history_path.exists():
            try:
                self.history = json.loads(self._history_path.read_text())
                log.info("Resumed history with %d epochs", len(self.history.get("epoch", [])))
            except Exception as exc:
                log.warning("Could not read existing history.json: %s", exc)

        if full_config:
            self._write_yaml(self.run_dir / "config.yaml", dict(full_config))

        self._write_manifest()
        self._init_tensorboard(config)
        self._init_wandb(config, full_config)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    @staticmethod
    def _write_yaml(path: Path, payload: dict) -> None:
        import yaml

        def _plain(obj):
            """Strip Config/Path/tensor wrappers so yaml emits clean scalars."""
            if isinstance(obj, dict):
                return {k: _plain(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_plain(v) for v in obj]
            if isinstance(obj, Path):
                return str(obj)
            return obj

        with open(path, "w") as f:
            yaml.safe_dump(_plain(payload), f, default_flow_style=False, sort_keys=False)

    def _write_manifest(self) -> None:
        """Record the environment fingerprint so results stay traceable."""
        try:
            from utils.reproducibility import collect_environment

            manifest = collect_environment()
            manifest["created"] = datetime.now().isoformat(timespec="seconds")
            manifest["run_dir"] = str(self.run_dir)
            (self.run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        except Exception as exc:  # pragma: no cover - never block a run on this
            log.debug("Could not write manifest: %s", exc)

    def _init_tensorboard(self, config) -> None:
        if not config.get("tensorboard", False):
            return
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_dir = self.run_dir / "tensorboard"
            self.tb_writer = SummaryWriter(log_dir=str(tb_dir))
            log.info("TensorBoard logging to %s", tb_dir)
        except ImportError:
            log.warning("tensorboard not installed; skipping TensorBoard logging")

    def _init_wandb(self, config, full_config) -> None:
        if not config.get("wandb", False):
            return
        try:
            import wandb

            self.wandb_run = wandb.init(
                project=config.get("wandb_project", "lucid-mri"),
                entity=config.get("wandb_entity"),
                name=self.run_dir.name,
                config=dict(full_config or {}),
                dir=str(self.run_dir),
                reinit=True,
            )
            log.info("W&B logging to project=%s", config.get("wandb_project"))
        except ImportError:
            log.warning("wandb not installed; skipping W&B logging")
        except Exception as exc:
            log.warning("wandb init failed (%s); continuing without it", exc)

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    @property
    def checkpoint_dir(self) -> Path:
        d = self.run_dir / "checkpoints"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ------------------------------------------------------------------
    # Scalar / image logging
    # ------------------------------------------------------------------

    def log_scalars(self, metrics: dict[str, float], step: int) -> None:
        """Log scalars to every active backend against a single monotonic step."""
        clean = {k: float(v) for k, v in metrics.items() if v is not None}
        if not clean:
            return

        if self.tb_writer:
            for key, value in clean.items():
                self.tb_writer.add_scalar(key, value, global_step=step)

        if self.wandb_run:
            import wandb

            wandb.log(clean, step=step)

    def log_epoch(self, metrics: dict[str, float], epoch: int, step: int) -> None:
        """
        Record one epoch of results: appended to ``history`` and flushed to disk.

        ``epoch`` is logged as a metric rather than used as the step axis, so
        that batch- and epoch-level series share one monotonic ``global_step``.
        """
        payload = {"epoch": epoch, **{k: float(v) for k, v in metrics.items() if v is not None}}
        for key, value in payload.items():
            self.history.setdefault(key, []).append(value)
        self.log_scalars(payload, step=step)
        self._flush_history()

    def _flush_history(self) -> None:
        tmp = self._history_path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(self.history, indent=2))
            tmp.replace(self._history_path)  # atomic: never leave a truncated file
        except Exception as exc:  # pragma: no cover
            log.warning("Could not write history.json: %s", exc)

    def log_image(self, tag: str, image: torch.Tensor, step: int) -> None:
        """Log a single image, given as ``(C, H, W)`` or ``(H, W)`` in [0, 1]."""
        image = image.detach().float().cpu()
        if image.dim() == 2:
            image = image.unsqueeze(0)
        image = image.clamp(0, 1)

        if self.tb_writer:
            self.tb_writer.add_image(tag, image, global_step=step)

        if self.wandb_run:
            import wandb

            arr = image.permute(1, 2, 0).numpy()
            if arr.shape[-1] == 1:
                arr = arr.squeeze(-1)
            wandb.log({tag: wandb.Image(arr)}, step=step)

    def log_images_grid(
        self, tag: str, images: list, step: int, captions: list[str] | None = None
    ) -> None:
        """Log a row of images side by side (e.g. input / prediction / target)."""
        prepared = []
        for img in images:
            if isinstance(img, torch.Tensor):
                img = img.detach().float().cpu().numpy()
            if img.ndim == 3 and img.shape[0] in (1, 3):
                img = img.transpose(1, 2, 0)
            if img.ndim == 3 and img.shape[-1] == 1:
                img = img.squeeze(-1)
            prepared.append(img)

        if self.tb_writer:
            import numpy as np

            stacked = np.concatenate(
                [(a - a.min()) / (a.ptp() + 1e-8) for a in prepared], axis=1
            )
            self.tb_writer.add_image(
                tag, torch.from_numpy(stacked).unsqueeze(0), global_step=step
            )

        if self.wandb_run:
            import wandb

            wandb.log(
                {
                    tag: [
                        wandb.Image(a, caption=captions[i] if captions else None)
                        for i, a in enumerate(prepared)
                    ]
                },
                step=step,
            )

    def log_model_graph(self, model: torch.nn.Module, input_tensor: torch.Tensor) -> None:
        """Log the computation graph (TensorBoard only); best-effort."""
        if self.tb_writer:
            try:
                self.tb_writer.add_graph(model, input_tensor)
            except Exception as exc:
                log.debug("Could not log model graph: %s", exc)

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        state: dict,
        filename: str,
        is_best: bool = False,
        score: float | None = None,
        higher_is_better: bool = False,
    ) -> Path:
        """
        Save a checkpoint and prune to the top-k by ``score``.

        Parameters
        ----------
        state
            Payload to serialise.
        filename
            Name within the checkpoint directory.
        is_best
            Also copy to ``best.pt``.
        score
            Metric used for top-k retention. If None, retention is skipped —
            better to keep everything than to delete by an unrelated criterion.
        higher_is_better
            Direction of ``score``.
        """
        path = self.checkpoint_dir / filename
        torch.save(state, path)

        if is_best:
            shutil.copyfile(path, self.checkpoint_dir / "best.pt")

        # `last.pt` always points at the newest checkpoint, so resuming never
        # requires knowing the epoch number.
        shutil.copyfile(path, self.checkpoint_dir / "last.pt")

        if score is not None:
            self._ckpt_scores[filename] = float(score)
            self._prune_checkpoints(higher_is_better)

        return path

    def _prune_checkpoints(self, higher_is_better: bool) -> None:
        """Delete all but the top-k scoring epoch checkpoints."""
        save_top_k = int(self.cfg.get("save_top_k", 0) or 0)
        if save_top_k <= 0:
            return

        existing = {
            name: score
            for name, score in self._ckpt_scores.items()
            if (self.checkpoint_dir / name).exists()
        }
        if len(existing) <= save_top_k:
            return

        ranked = sorted(existing.items(), key=lambda kv: kv[1], reverse=higher_is_better)
        for name, _ in ranked[save_top_k:]:
            try:
                (self.checkpoint_dir / name).unlink()
                self._ckpt_scores.pop(name, None)
            except OSError as exc:  # pragma: no cover
                log.debug("Could not prune %s: %s", name, exc)

    def log_hyperparams(self, hparams: dict, metrics: dict | None = None) -> None:
        """Log hyperparameters for HP-search dashboards."""
        if self.tb_writer and metrics:
            flat = {
                k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                for k, v in hparams.items()
            }
            try:
                self.tb_writer.add_hparams(flat, metrics)
            except Exception as exc:  # pragma: no cover
                log.debug("Could not log hparams: %s", exc)

    def log_summary(self, summary: dict[str, Any]) -> None:
        """Write the final result summary to ``summary.json`` and the backends."""
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        if self.wandb_run:
            import wandb

            for key, value in summary.items():
                if isinstance(value, (int, float)):
                    wandb.run.summary[key] = value

    def finish(self) -> None:
        """Flush and close every backend."""
        self._flush_history()
        if self.tb_writer:
            self.tb_writer.flush()
            self.tb_writer.close()
            self.tb_writer = None
        if self.wandb_run:
            import wandb

            wandb.finish()
            self.wandb_run = None
