"""
evaluate.py
-----------
Evaluation, visualisation and statistically grounded architecture comparison.

Metrics
-------
PSNR, SSIM and NMSE, all computed per image against each target's own dynamic
range (see :mod:`training.metrics` for why that matters), plus the combined
validation loss.

Beyond point estimates
----------------------
:func:`compare_architectures` reports bootstrap confidence intervals and paired
permutation tests rather than bare means. On ~199 validation slices the
per-slice PSNR spread is several dB, so a difference of a few tenths between two
architectures is not distinguishable from noise — and saying so is a stronger
result than quietly reporting the larger number.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config, load_config
from data.preprocessing import FastMRIKneeDataset, collate
from models.registry import build_model, forward_model
from training.losses import CombinedLoss
from training.metrics import MetricAccumulator
from training.stats import compare_models, format_comparison_table
from utils.ema import load_ema_weights_into

log = logging.getLogger(__name__)

__all__ = [
    "compare_architectures",
    "evaluate_model",
    "load_model",
    "plot_training_history",
    "visualize_reconstructions",
]


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def load_model(
    model_name: str | None,
    ckpt_path: str | Path,
    device: torch.device,
    use_ema: bool = True,
) -> torch.nn.Module:
    """
    Rebuild a model from a checkpoint.

    The architecture is reconstructed from the config stored inside the
    checkpoint whenever one is present, so the weights can never be loaded into
    a differently shaped model. ``model_name`` is only a fallback for legacy
    checkpoints that predate config embedding.

    Parameters
    ----------
    use_ema
        Prefer the EMA (averaged) weights. These are what the trainer validated
        against and what ``best.pt`` was selected on, so evaluating the raw
        weights instead would report a different — usually worse — model than
        the one the run claims to have produced.
    """
    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt.get("config") or {}

    if config and "model" in config:
        cfg = config if isinstance(config, Config) else Config(config)
    else:
        if not model_name:
            raise ValueError(
                f"{path} has no embedded config; pass --model to specify the architecture."
            )
        log.warning("Checkpoint has no embedded config; building %r from defaults", model_name)
        cfg = load_config(overrides={"model": {"name": model_name}}, validate=False)

    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state"])

    applied_ema = False
    if use_ema:
        applied_ema = load_ema_weights_into(model, ckpt.get("ema_state"))

    model.to(device).eval()

    log.info(
        "Loaded %s from %s (epoch %s, val_psnr=%s, weights=%s)",
        cfg.model.get("name", model_name),
        path,
        ckpt.get("epoch", "?"),
        f"{ckpt.get('val_psnr', float('nan')):.2f}" if ckpt.get("val_psnr") else "?",
        "EMA" if applied_ema else "raw",
    )
    return model


# ---------------------------------------------------------------------------
# Quantitative evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
    criterion: torch.nn.Module | None = None,
) -> dict[str, Any]:
    """
    Evaluate a model over a loader.

    Returns
    -------
    dict with mean ``psnr``/``ssim``/``nmse``/``val_loss``, the per-sample
    vectors under ``per_sample`` (needed for paired significance tests), and a
    per-acceleration breakdown when the loader mixes accelerations.
    """
    model.eval()
    criterion = criterion or CombinedLoss().to(device)
    acc = MetricAccumulator()

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        batch = {
            k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }

        if getattr(model, "expects_kspace", False):
            pred = forward_model(model, batch["image"], batch["kspace"], batch["mask"])
        else:
            pred = forward_model(model, batch["image"])

        pred = pred.float()
        target = batch["target"].float()
        data_range = batch["max_value"].float()

        acc.update(
            pred,
            target,
            data_range=data_range,
            loss=float(criterion(pred, target, data_range=data_range)),
            fnames=batch.get("fname"),
            accelerations=batch.get("acceleration"),
        )

    summary = acc.compute()
    result: dict[str, Any] = {
        "val_loss": summary.get("loss", float("nan")),
        "psnr_db": summary["psnr"],
        "ssim": summary["ssim"],
        "nmse": summary["nmse"],
        "n_samples": summary["n"],
        "per_sample": {"psnr": acc.psnr, "ssim": acc.ssim, "nmse": acc.nmse},
        "worst": acc.worst(k=5),
    }
    by_acc = acc.by_acceleration()
    if len(by_acc) > 1:
        result["by_acceleration"] = by_acc
    return result


# ---------------------------------------------------------------------------
# Qualitative visualisation
# ---------------------------------------------------------------------------


@torch.no_grad()
def visualize_reconstructions(
    model: torch.nn.Module,
    dataset: FastMRIKneeDataset,
    device: torch.device,
    n_examples: int = 4,
    save_path: str = "results/reconstructions.png",
    seed: int = 0,
    show_error: bool = True,
) -> str:
    """
    Save a grid of ``[input | prediction | target | error]`` rows.

    The error column is what makes the figure diagnostic rather than decorative:
    two reconstructions can look identical at display contrast while differing
    substantially at the tissue boundaries a radiologist reads.
    """
    from training.metrics import psnr as psnr_fn
    from training.metrics import ssim as ssim_fn

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    n_examples = min(n_examples, len(dataset))
    if n_examples == 0:
        raise ValueError("Dataset is empty; nothing to visualise")

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), n_examples, replace=False)

    n_cols = 4 if show_error else 3
    fig, axes = plt.subplots(
        n_examples, n_cols, figsize=(4 * n_cols, 4 * n_examples), squeeze=False
    )
    fig.suptitle("fastMRI reconstruction", fontsize=15, fontweight="bold")

    titles = ["Input (zero-filled)", "Reconstruction", "Ground truth", "|Error|"]
    for col in range(n_cols):
        axes[0, col].set_title(titles[col], fontsize=11)

    model.eval()
    for row, idx in enumerate(indices):
        sample = dataset[int(idx)]
        image = sample["image"].unsqueeze(0).to(device)
        target = sample["target"].unsqueeze(0).to(device)
        data_range = sample["max_value"].reshape(1).to(device)

        if getattr(model, "expects_kspace", False):
            pred = forward_model(
                model,
                image,
                sample["kspace"].unsqueeze(0).to(device),
                sample["mask"].unsqueeze(0).to(device),
            )
        else:
            pred = forward_model(model, image)
        pred = pred.float()

        p = float(psnr_fn(pred, target, data_range))
        s = float(ssim_fn(pred, target, data_range))

        # A complex input has no single displayable channel; show its magnitude.
        inp = sample["image"]
        inp_np = (
            torch.sqrt((inp**2).sum(0)).numpy() if inp.shape[0] == 2 else inp.squeeze(0).numpy()
        )
        pred_np = pred.squeeze().cpu().numpy()
        target_np = target.squeeze().cpu().numpy()

        vmax = float(target_np.max()) or 1.0
        for col, img in enumerate([inp_np, pred_np, target_np]):
            axes[row, col].imshow(img, cmap="gray", vmin=0, vmax=vmax)
            axes[row, col].axis("off")

        if show_error:
            error = np.abs(pred_np - target_np)
            im = axes[row, 3].imshow(error, cmap="inferno", vmin=0, vmax=vmax * 0.25)
            axes[row, 3].axis("off")
            plt.colorbar(im, ax=axes[row, 3], fraction=0.046, pad=0.04)

        axes[row, 1].set_title(
            f"PSNR {p:.2f} dB | SSIM {s:.4f}",
            fontsize=9,
            color="white",
            bbox=dict(facecolor="black", alpha=0.65, pad=2),
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved reconstruction grid to %s", save_path)
    return save_path


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------


def plot_training_history(history_path: str | Path, save_path: str | None = None) -> str:
    """
    Plot loss, PSNR, SSIM and LR from a run's ``history.json``.

    The trainer writes this file every epoch, so curves can be regenerated from
    any run without a live TensorBoard process.
    """
    path = Path(history_path)
    if path.is_dir():
        path = path / "history.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No history at {path}. The trainer writes history.json into the run directory."
        )

    hist = json.loads(path.read_text())

    def series(*names: str) -> list[float] | None:
        for name in names:
            if hist.get(name):
                return hist[name]
        return None

    train_loss = series("epoch/train_loss", "train_loss")
    val_loss = series("epoch/val_loss", "val_loss")
    val_psnr = series("epoch/val_psnr", "val_psnr")
    val_ssim = series("epoch/val_ssim", "val_ssim")
    lr = series("epoch/lr", "lr")
    epochs = hist.get("epoch") or list(range(1, len(train_loss or []) + 1))

    panels = [
        ("Loss", [(train_loss, "Train"), (val_loss, "Validation")], None),
        ("PSNR (dB)", [(val_psnr, "Validation")], "max"),
        ("SSIM", [(val_ssim, "Validation")], "max"),
        ("Learning rate", [(lr, "LR")], "log"),
    ]
    panels = [p for p in panels if any(s is not None for s, _ in p[1])]

    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 4), squeeze=False)
    for ax, (title, series_list, style) in zip(axes[0], panels):
        for values, label in series_list:
            if values:
                ax.plot(epochs[: len(values)], values, label=label, linewidth=1.6)
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if style == "log":
            ax.set_yscale("log")
        elif style == "max":
            best_values = series_list[0][0]
            if best_values:
                ax.axhline(max(best_values), color="grey", linestyle="--", alpha=0.6)
                ax.set_title(f"{title} (best {max(best_values):.4g})")
        if len(series_list) > 1:
            ax.legend(frameon=False)

    fig.suptitle(path.parent.name, fontsize=13, fontweight="bold")
    plt.tight_layout()

    out = str(save_path or path.with_name("training_curves.png"))
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved training curves to %s", out)
    return out


# ---------------------------------------------------------------------------
# Architecture comparison
# ---------------------------------------------------------------------------


def compare_architectures(
    results: dict[str, dict],
    metric: str = "psnr",
    reference: str | None = None,
    save_path: str | None = None,
) -> dict:
    """
    Print and return a statistically grounded comparison table.

    Parameters
    ----------
    results
        ``{model_name: evaluate_model(...) output}``. Per-sample vectors are
        used when present so the comparison is paired.
    metric
        ``psnr``, ``ssim`` or ``nmse``.
    reference
        Baseline model name. Defaults to the first entry.
    """
    print(f"\n{'Architecture':<22}{'Val loss':>11}{'PSNR (dB)':>12}{'SSIM':>10}{'NMSE':>11}")
    print("-" * 66)
    for name, m in results.items():
        print(
            f"{name:<22}{m['val_loss']:>11.4f}{m['psnr_db']:>12.2f}"
            f"{m['ssim']:>10.4f}{m['nmse']:>11.5f}"
        )

    per_sample = {
        name: m["per_sample"] for name, m in results.items() if m.get("per_sample")
    }

    comparison: dict = {}
    if len(per_sample) >= 2:
        lengths = {len(v[metric]) for v in per_sample.values()}
        if len(lengths) == 1:
            comparison = compare_models(per_sample, metric=metric, reference=reference)
            print()
            print(format_comparison_table(comparison))
        else:
            log.warning(
                "Models were evaluated on different numbers of samples %s; "
                "skipping the paired test, which requires identical sample sets.",
                sorted(lengths),
            )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": {
                name: {k: v for k, v in m.items() if k != "per_sample"}
                for name, m in results.items()
            },
            "statistics": comparison,
        }
        Path(save_path).write_text(json.dumps(payload, indent=2, default=str))
        log.info("Saved comparison to %s", save_path)

    return comparison


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Evaluate MRI reconstruction models")
    parser.add_argument("--model", default=None, help="Architecture (legacy checkpoints only)")
    parser.add_argument("--ckpt", required=True, help="Checkpoint .pt file")
    parser.add_argument("--data_dir", default="data/knee_singlecoil_val")
    parser.add_argument("--n_vis", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--no-ema", action="store_true", help="Evaluate raw, not EMA, weights")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, args.ckpt, device, use_ema=not args.no_ema)

    dataset = FastMRIKneeDataset(
        args.data_dir, complex_input=int(getattr(model, "in_channels", 1)) == 2
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate
    )

    print(f"\n-- Quantitative evaluation ({len(dataset)} slices) --")
    metrics = evaluate_model(model, loader, device)
    print(f"  Val loss : {metrics['val_loss']:.4f}")
    print(f"  PSNR     : {metrics['psnr_db']:.2f} dB")
    print(f"  SSIM     : {metrics['ssim']:.4f}")
    print(f"  NMSE     : {metrics['nmse']:.5f}")

    from training.stats import bootstrap_ci

    print(f"  PSNR 95% CI: {bootstrap_ci(metrics['per_sample']['psnr'])}")

    if metrics.get("worst"):
        print("\n  Worst slices by PSNR:")
        for fname, value in metrics["worst"]:
            print(f"    {fname:<32} {value:.2f} dB")

    name = args.model or "model"
    visualize_reconstructions(
        model,
        dataset,
        device,
        n_examples=args.n_vis,
        save_path=os.path.join(args.output_dir, f"{name}_reconstructions.png"),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
