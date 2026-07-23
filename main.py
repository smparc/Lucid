"""
main.py
-------
Unified entry point for the Accelerated MRI Reconstruction project (Lucid).


Commands
--------
    # Train with config file:
    python main.py train --config configs/swinunet.yaml


    # Train with CLI overrides:
    python main.py train --config configs/swinunet.yaml training.lr=5e-5 training.batch_size=6


    # Resume training from checkpoint:
    python main.py train --config configs/swinunet.yaml --resume outputs/swinunet/checkpoints/best.pt


    # Evaluate a checkpoint:
    python main.py eval --model swinunet --ckpt outputs/swinunet/checkpoints/best.pt


    # Benchmark inference speed:
    python main.py benchmark --ckpt outputs/swinunet/checkpoints/best.pt


    # Export to ONNX:
    python main.py export --ckpt outputs/swinunet/checkpoints/best.pt --format onnx


    # Compare all architectures:
    python main.py compare --data_dir data/knee_singlecoil_val


    # Sanity check (no data needed):
    python main.py test_models
"""


import os
import sys
import argparse
import logging


import torch


sys.path.insert(0, os.path.dirname(__file__))


from models.unet import UNet, count_parameters
from models.bt_unet import BTUNet
from models.swinunet import SwinUNet
from config import load_config


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_test_models(args):
    """Run a forward pass through all three architectures to verify shapes."""
    print("\n── Architecture Sanity Check ─────────────────────────────")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy = torch.randn(2, 1, 320, 320).to(device)


    configs = [
        ("U-Net Baseline", UNet(in_channels=1, out_channels=1, base_ch=32, n_levels=4)),
        ("BT-UNet", BTUNet(in_channels=1, out_channels=1, base_ch=32, n_levels=4, tf_heads=8, tf_layers=4)),
        ("SwinUNet", SwinUNet(img_size=320, patch_size=4, in_ch=1, out_ch=1, embed_dim=64, ws=8, head_dim=8, n_levels=3)),
    ]


    print(f"  {'Architecture':<20} {'Params (M)':>10} {'Output Shape':>16} {'Status':>8}")
    print("  " + "─" * 60)
    for name, model in configs:
        model = model.to(device)
        try:
            with torch.no_grad():
                out = model(dummy)
            status = "OK"
        except Exception as e:
            out = dummy
            status = f"FAIL: {e}"
        n = count_parameters(model)
        print(f"  {name:<20} {n/1e6:>10.1f} {str(tuple(out.shape)):>16} {status:>8}")
    print()



def cmd_train(args):
    """Train a model using the config-driven Trainer."""
    from training.train import Trainer


    cfg = load_config(args.config, cli_overrides=args.overrides)
    trainer = Trainer(cfg)


    if args.resume:
        trainer.resume(args.resume)


    trainer.fit()



def cmd_eval(args):
    """Evaluate a checkpoint on the validation set."""
    from training.evaluate import load_model, evaluate_model, visualize_reconstructions
    from data.preprocessing import FastMRIKneeDataset
    from torch.utils.data import DataLoader


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, args.ckpt, device)
    dataset = FastMRIKneeDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)


    print(f"\n── Evaluation: {args.model} ────────────────────────────────")
    metrics = evaluate_model(model, loader, device)
    print(f"  Val Loss : {metrics['val_loss']:.4f}")
    print(f"  PSNR     : {metrics['psnr_db']:.2f} dB")
    print(f"  SSIM     : {metrics['ssim']:.4f}")


    os.makedirs(args.output_dir, exist_ok=True)
    vis_path = os.path.join(args.output_dir, f"{args.model}_reconstructions.png")
    visualize_reconstructions(model, dataset, device, n_examples=4, save_path=vis_path)



def cmd_benchmark(args):
    """Benchmark inference speed."""
    from inference import MRIReconstructionPipeline


    pipe = MRIReconstructionPipeline.from_checkpoint(args.ckpt, args.model)
    results = pipe.benchmark(n_runs=args.n_runs)


    print(f"\n── Benchmark Results ({args.n_runs} runs) ──────────────────")
    print(f"  Mean:       {results['mean_ms']:.2f} ms")
    print(f"  Std:        {results['std_ms']:.2f} ms")
    print(f"  Min:        {results['min_ms']:.2f} ms")
    print(f"  Max:        {results['max_ms']:.2f} ms")
    print(f"  Throughput: {results['throughput_fps']:.1f} FPS")



def cmd_export(args):
    """Export model to ONNX or TorchScript."""
    from inference import MRIReconstructionPipeline


    pipe = MRIReconstructionPipeline.from_checkpoint(args.ckpt, args.model)


    if args.format == "onnx":
        output = args.output or "exports/model.onnx"
        pipe.export_onnx(output)
    elif args.format == "torchscript":
        output = args.output or "exports/model_traced.pt"
        pipe.export_torchscript(output)
    else:
        print(f"Unknown format: {args.format}")



def cmd_compare(args):
    """Compare all three architectures from their best checkpoints."""
    from training.evaluate import load_model, evaluate_model, compare_architectures
    from data.preprocessing import FastMRIKneeDataset
    from torch.utils.data import DataLoader


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = FastMRIKneeDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)


    model_names = ["unet", "bt_unet", "swinunet"]
    results = {}


    for name in model_names:
        ckpt = os.path.join(args.ckpt_dir, name, "checkpoints", "best.pt")
        if not os.path.exists(ckpt):
            # Fallback to old-style checkpoint path
            ckpt = os.path.join(args.ckpt_dir, f"{name}_best.pt")
        if not os.path.exists(ckpt):
            print(f"  [SKIP] Checkpoint not found for {name}")
            continue
        model = load_model(name, ckpt, device)
        metrics = evaluate_model(model, loader, device)
        results[name] = metrics


    if results:
        compare_architectures(results)



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Lucid: Accelerated MRI Reconstruction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")


    # test_models
    sub.add_parser("test_models", help="Sanity-check all architectures (no data needed)")


    # train
    p_train = sub.add_parser("train", help="Train a model")
    p_train.add_argument("--config", default=None, help="YAML config file")
    p_train.add_argument("--resume", default=None, help="Checkpoint to resume from")
    p_train.add_argument("overrides", nargs="*", help="Config overrides: key=value")


    # eval
    p_eval = sub.add_parser("eval", help="Evaluate a checkpoint")
    p_eval.add_argument("--model", required=True, choices=["unet", "bt_unet", "swinunet"])
    p_eval.add_argument("--ckpt", required=True, help="Checkpoint path")
    p_eval.add_argument("--data_dir", default="data/knee_singlecoil_val")
    p_eval.add_argument("--output_dir", default="results")


    # benchmark
    p_bench = sub.add_parser("benchmark", help="Benchmark inference speed")
    p_bench.add_argument("--ckpt", required=True)
    p_bench.add_argument("--model", default=None)
    p_bench.add_argument("--n_runs", type=int, default=100)


    # export
    p_export = sub.add_parser("export", help="Export model to ONNX/TorchScript")
    p_export.add_argument("--ckpt", required=True)
    p_export.add_argument("--model", default=None)
    p_export.add_argument("--format", choices=["onnx", "torchscript"], default="onnx")
    p_export.add_argument("--output", default=None)


    # compare
    p_cmp = sub.add_parser("compare", help="Compare all architectures")
    p_cmp.add_argument("--data_dir", default="data/knee_singlecoil_val")
    p_cmp.add_argument("--ckpt_dir", default="outputs")


    args = parser.parse_args()


    print("╔══════════════════════════════════════════════════════════╗")
    print("║       Lucid: Accelerated MRI Reconstruction             ║")
    print("║       SwinUNet • BT-UNet • U-Net Baseline               ║")
    print("╚══════════════════════════════════════════════════════════╝")


    if args.command == "test_models":
        cmd_test_models(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "export":
        cmd_export(args)
    elif args.command == "compare":
        cmd_compare(args)
    else:
        parser.print_help()



if __name__ == "__main__":
    main()
