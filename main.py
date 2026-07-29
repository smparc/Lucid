"""
main.py
-------
Unified entry point for Lucid, accelerated MRI reconstruction.

Commands
--------
    python main.py test_models                                   # sanity check, no data
    python main.py train --config configs/swinunet.yaml
    python main.py train --config configs/swinunet.yaml training.lr=5e-5
    python main.py train --config configs/swinunet.yaml --resume outputs/.../last.pt
    python main.py eval --ckpt outputs/.../best.pt
    python main.py compare --ckpt_dir outputs
    python main.py benchmark --ckpt outputs/.../best.pt
    python main.py export --ckpt outputs/.../best.pt --format onnx
    python main.py curves --run outputs/swinunet_dc

Exit codes
----------
Every command returns 0 on success and non-zero on failure. This matters: the
original ``test_models`` caught every exception, printed ``FAIL`` into a table,
and returned successfully — so the CI "model sanity" job passed green while the
flagship architecture could not even be constructed.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ConfigError, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("lucid")

BANNER = r"""
+----------------------------------------------------------+
|   Lucid : Accelerated MRI Reconstruction                 |
|   SwinUNet - BT-UNet - U-Net, with data consistency      |
+----------------------------------------------------------+
"""


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_test_models(args: argparse.Namespace) -> int:
    """Forward-pass every architecture and verify output shapes."""
    from models.data_consistency import CascadedNet
    from models.layers import count_parameters
    from models.registry import available_models, build_backbone

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = args.size
    dummy = torch.randn(2, 1, size, size, device=device)

    print(f"\n-- Architecture sanity check ({device}, {size}x{size}) --\n")
    print(f"  {'Architecture':<28}{'Params (M)':>12}{'Output':>20}  Status")
    print("  " + "-" * 74)

    cases: list[tuple[str, callable, torch.Tensor]] = []
    for name in available_models():
        cases.append(
            (name, lambda n=name: build_backbone(n, {}, 1, 1), dummy)
        )

    # Also exercise the physics path, which the plain backbones do not cover.
    complex_dummy = torch.randn(2, 2, size, size, device=device)
    cases.append(
        (
            "swinunet + DC cascade",
            lambda: CascadedNet(
                lambda: build_backbone("swinunet", {"embed_dim": 32, "n_levels": 2}, 2, 2),
                n_cascades=2,
            ),
            complex_dummy,
        )
    )

    failures = []
    for name, factory, sample in cases:
        try:
            model = factory().to(device).eval()
            with torch.no_grad():
                if isinstance(model, CascadedNet):
                    from models.fourier import fft2c_chan

                    mask = (torch.rand(2, 1, size, device=device) > 0.75).float()
                    kspace = fft2c_chan(sample) * mask.unsqueeze(1)
                    out = model(sample, kspace, mask)
                else:
                    out = model(sample)

            n_params = count_parameters(model)
            expected_spatial = sample.shape[-2:]
            if out.shape[-2:] != expected_spatial:
                raise AssertionError(
                    f"output spatial size {tuple(out.shape[-2:])} != input {tuple(expected_spatial)}"
                )
            if not torch.isfinite(out).all():
                raise AssertionError("output contains NaN or Inf")

            print(
                f"  {name:<28}{n_params / 1e6:>12.2f}{tuple(out.shape)!s:>20}  OK"
            )
        except Exception as exc:
            failures.append((name, exc))
            print(f"  {name:<28}{'-':>12}{'-':>20}  FAIL: {exc}")

    print()
    if failures:
        log.error("%d of %d architectures failed", len(failures), len(cases))
        return 1
    print(f"  All {len(cases)} architectures passed.\n")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Train a model."""
    from training.train import Trainer

    cfg = load_config(args.config, cli_overrides=args.overrides)
    trainer = Trainer(cfg, resume_from=args.resume)
    summary = trainer.fit()

    print("\n-- Training summary --")
    for key in ("model", "final_epoch", "params_m", "run_dir"):
        if key in summary:
            print(f"  {key:<14} {summary[key]}")
    monitor_key = f"best_{summary.get('monitor', 'val_loss')}"
    if monitor_key in summary:
        print(f"  {monitor_key:<14} {summary[monitor_key]:.5f}")
    if "val_psnr_ci" in summary:
        ci = summary["val_psnr_ci"]
        print(
            f"  {'val PSNR':<14} {ci['mean']:.2f} dB "
            f"[{ci['ci_low']:.2f}, {ci['ci_high']:.2f}] (95% CI)"
        )
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Evaluate a checkpoint on a validation directory."""
    from torch.utils.data import DataLoader

    from data.preprocessing import FastMRIKneeDataset, collate
    from training.evaluate import evaluate_model, load_model, visualize_reconstructions
    from training.stats import bootstrap_ci

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, args.ckpt, device, use_ema=not args.no_ema)

    dataset = FastMRIKneeDataset(
        args.data_dir,
        acceleration=args.acceleration,
        complex_input=int(getattr(model, "in_channels", 1)) == 2,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate
    )

    print(f"\n-- Evaluation: {args.model or 'checkpoint'} ({len(dataset)} slices) --")
    metrics = evaluate_model(model, loader, device)
    print(f"  Val loss : {metrics['val_loss']:.4f}")
    print(f"  PSNR     : {metrics['psnr_db']:.2f} dB   95% CI {bootstrap_ci(metrics['per_sample']['psnr'])}")
    print(f"  SSIM     : {metrics['ssim']:.4f}   95% CI {bootstrap_ci(metrics['per_sample']['ssim'])}")
    print(f"  NMSE     : {metrics['nmse']:.5f}")

    if "by_acceleration" in metrics:
        print("\n  By acceleration:")
        for acc, m in metrics["by_acceleration"].items():
            print(f"    R={acc}: PSNR {m['psnr']:.2f} dB, SSIM {m['ssim']:.4f}  (n={m['n']})")

    os.makedirs(args.output_dir, exist_ok=True)
    visualize_reconstructions(
        model,
        dataset,
        device,
        n_examples=args.n_vis,
        save_path=os.path.join(args.output_dir, f"{args.model or 'model'}_reconstructions.png"),
    )
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare every checkpoint found under a directory, with significance tests."""
    from torch.utils.data import DataLoader

    from data.preprocessing import FastMRIKneeDataset, collate
    from training.evaluate import compare_architectures, evaluate_model, load_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(args.ckpt_dir)

    candidates: dict[str, Path] = {}
    for path in sorted(ckpt_dir.glob("*/checkpoints/best.pt")):
        candidates[path.parent.parent.name] = path
    for path in sorted(ckpt_dir.glob("*_best.pt")):  # legacy layout
        candidates.setdefault(path.stem.replace("_best", ""), path)

    if not candidates:
        log.error("No checkpoints found under %s (looked for */checkpoints/best.pt)", ckpt_dir)
        return 1

    results = {}
    for name, path in candidates.items():
        try:
            model = load_model(None, path, device)
        except Exception as exc:
            log.warning("Skipping %s: %s", name, exc)
            continue

        dataset = FastMRIKneeDataset(
            args.data_dir,
            acceleration=args.acceleration,
            complex_input=int(getattr(model, "in_channels", 1)) == 2,
        )
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate
        )
        results[name] = evaluate_model(model, loader, device)

    if not results:
        log.error("No checkpoints could be loaded")
        return 1

    compare_architectures(
        results,
        metric=args.metric,
        reference=args.reference,
        save_path=os.path.join(args.output_dir, "comparison.json"),
    )
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Benchmark inference latency."""
    from inference import MRIReconstructionPipeline

    pipe = MRIReconstructionPipeline.from_checkpoint(args.ckpt, args.model)
    results = pipe.benchmark(
        n_runs=args.n_runs, input_size=(args.batch_size, 1, args.size, args.size)
    )

    print(f"\n-- Benchmark: {args.n_runs} runs on {results['device']} --")
    print(f"  Median      {results['median_ms']:.2f} ms")
    print(f"  Mean +/- SD {results['mean_ms']:.2f} +/- {results['std_ms']:.2f} ms")
    print(f"  Min / Max   {results['min_ms']:.2f} / {results['max_ms']:.2f} ms")
    print(f"  p95         {results['p95_ms']:.2f} ms")
    print(f"  Throughput  {results['throughput_fps']:.1f} images/s")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Export a checkpoint to ONNX or TorchScript."""
    from inference import MRIReconstructionPipeline

    pipe = MRIReconstructionPipeline.from_checkpoint(args.ckpt, args.model)
    if args.format == "onnx":
        pipe.export_onnx(args.output or "exports/model.onnx")
    else:
        pipe.export_torchscript(args.output or "exports/model_traced.pt")
    return 0


def cmd_curves(args: argparse.Namespace) -> int:
    """Plot training curves from a run's history.json."""
    from training.evaluate import plot_training_history

    print(f"  Wrote {plot_training_history(args.run, args.output)}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lucid",
        description="Lucid: accelerated MRI reconstruction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p_test = sub.add_parser("test_models", help="Sanity-check all architectures (no data)")
    p_test.add_argument("--size", type=int, default=320, help="Input size for the test pass")

    p_train = sub.add_parser("train", help="Train a model")
    p_train.add_argument("--config", default=None, help="YAML config file")
    p_train.add_argument("--resume", default=None, help="Checkpoint to resume from")
    p_train.add_argument("overrides", nargs="*", help="Config overrides: dotted.key=value")

    p_eval = sub.add_parser("eval", help="Evaluate a checkpoint")
    p_eval.add_argument("--ckpt", required=True)
    p_eval.add_argument("--model", default=None, help="Only needed for legacy checkpoints")
    p_eval.add_argument("--data_dir", default="data/knee_singlecoil_val")
    p_eval.add_argument("--acceleration", type=int, default=4)
    p_eval.add_argument("--batch_size", type=int, default=4)
    p_eval.add_argument("--n_vis", type=int, default=4)
    p_eval.add_argument("--output_dir", default="results")
    p_eval.add_argument("--no-ema", action="store_true", help="Use raw rather than EMA weights")

    p_cmp = sub.add_parser("compare", help="Compare architectures with significance tests")
    p_cmp.add_argument("--data_dir", default="data/knee_singlecoil_val")
    p_cmp.add_argument("--ckpt_dir", default="outputs")
    p_cmp.add_argument("--acceleration", type=int, default=4)
    p_cmp.add_argument("--batch_size", type=int, default=4)
    p_cmp.add_argument("--metric", default="psnr", choices=["psnr", "ssim", "nmse"])
    p_cmp.add_argument("--reference", default=None, help="Baseline model name")
    p_cmp.add_argument("--output_dir", default="results")

    p_bench = sub.add_parser("benchmark", help="Benchmark inference speed")
    p_bench.add_argument("--ckpt", required=True)
    p_bench.add_argument("--model", default=None)
    p_bench.add_argument("--n_runs", type=int, default=100)
    p_bench.add_argument("--batch_size", type=int, default=1)
    p_bench.add_argument("--size", type=int, default=320)

    p_export = sub.add_parser("export", help="Export to ONNX or TorchScript")
    p_export.add_argument("--ckpt", required=True)
    p_export.add_argument("--model", default=None)
    p_export.add_argument("--format", choices=["onnx", "torchscript"], default="onnx")
    p_export.add_argument("--output", default=None)

    p_curves = sub.add_parser("curves", help="Plot training curves from history.json")
    p_curves.add_argument("--run", required=True, help="Run directory or history.json path")
    p_curves.add_argument("--output", default=None)

    return parser


_COMMANDS = {
    "test_models": cmd_test_models,
    "train": cmd_train,
    "eval": cmd_eval,
    "compare": cmd_compare,
    "benchmark": cmd_benchmark,
    "export": cmd_export,
    "curves": cmd_curves,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        print(BANNER)
        parser.print_help()
        return 1

    print(BANNER)

    handler = _COMMANDS[args.command]
    try:
        return handler(args)
    except ConfigError as exc:
        log.error("%s", exc)
        return 2
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 3
    except KeyboardInterrupt:
        log.warning("Interrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
