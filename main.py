"""
main.py
-------
Unified entry point for the Accelerated MRI Reconstruction project.

Commands
--------
    # Train SwinUNet (best model):
    python main.py train --model swinunet --data_dir data/knee_singlecoil_train

    # Train baseline U-Net:
    python main.py train --model unet --data_dir data/knee_singlecoil_train

    # Evaluate a checkpoint:
    python main.py eval --model swinunet --ckpt checkpoints/swinunet_best.pt --data_dir data/knee_singlecoil_val

    # Compare all three architectures (requires all three checkpoints):
    python main.py compare --data_dir data/knee_singlecoil_val

    # Quick architecture sanity check (no data needed):
    python main.py test_models
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(__file__))

from models.unet     import UNet, count_parameters
from models.bt_unet  import BTUNet
from models.swinunet import SwinUNet


def cmd_test_models(args):
    """Run a forward pass through all three architectures to verify shapes."""
    print("\n── Architecture Sanity Check ─────────────────────────────")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy  = torch.randn(2, 1, 320, 320).to(device)

    configs = [
        ("U-Net Baseline", UNet(in_channels=1, out_channels=1, base_ch=32, n_levels=4)),
        ("BT-UNet",        BTUNet(in_channels=1, out_channels=1, base_ch=32, n_levels=4, tf_heads=8, tf_layers=4)),
        ("SwinUNet",       SwinUNet(img_size=320, patch_size=4, in_ch=1, out_ch=1, embed_dim=64, ws=8, head_dim=8, n_levels=3)),
    ]

    print(f"  {'Architecture':<20} {'Params (M)':>10} {'Output Shape':>16} {'Status':>8}")
    print("  " + "─" * 60)
    for name, model in configs:
        model = model.to(device)
        try:
            with torch.no_grad():
                out = model(dummy)
            status = "✓ OK"
        except Exception as e:
            out    = dummy  # placeholder
            status = f"✗ {e}"
        n = count_parameters(model)
        print(f"  {name:<20} {n/1e6:>10.1f} {str(tuple(out.shape)):>16} {status:>8}")


def cmd_train(args):
    from training.train import train
    train(
        model_name=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        augment_data=not args.no_augment,
    )


def cmd_eval(args):
    from training.evaluate import load_model, evaluate_model, visualize_reconstructions
    from data.preprocessing import FastMRIKneeDataset
    from torch.utils.data import DataLoader

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model   = load_model(args.model, args.ckpt, device)
    dataset = FastMRIKneeDataset(args.data_dir)
    loader  = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)

    print(f"\n── Evaluation: {args.model} ────────────────────────────────")
    from training.train import CombinedLoss, ssim_metric
    metrics = evaluate_model(model, loader, device)
    print(f"  Val Loss : {metrics['val_loss']:.4f}")
    print(f"  PSNR     : {metrics['psnr_db']:.2f} dB")
    print(f"  SSIM     : {metrics['ssim']:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    vis_path = os.path.join(args.output_dir, f"{args.model}_reconstructions.png")
    visualize_reconstructions(model, dataset, device, n_examples=4, save_path=vis_path)


def cmd_compare(args):
    """Compare all three architectures from their best checkpoints."""
    from training.evaluate import load_model, evaluate_model, compare_architectures
    from data.preprocessing import FastMRIKneeDataset
    from torch.utils.data import DataLoader

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = FastMRIKneeDataset(args.data_dir)
    loader  = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)

    model_names = ["unet", "bt_unet", "swinunet"]
    results = {}

    for name in model_names:
        ckpt = os.path.join(args.output_dir, f"{name}_best.pt")
        if not os.path.exists(ckpt):
            print(f"  ⚠ Checkpoint not found for {name}: {ckpt}")
            continue
        model   = load_model(name, ckpt, device)
        metrics = evaluate_model(model, loader, device)
        results[name] = metrics

    if results:
        compare_architectures(results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Accelerated MRI Reconstruction — SwinUNet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # test_models
    sub.add_parser("test_models", help="Sanity-check all architectures (no data needed)")

    # train
    p_train = sub.add_parser("train", help="Train a model")
    p_train.add_argument("--model",      default="swinunet", choices=["unet", "bt_unet", "swinunet"])
    p_train.add_argument("--data_dir",   default="data/knee_singlecoil_train")
    p_train.add_argument("--output_dir", default="checkpoints")
    p_train.add_argument("--epochs",     type=int,   default=50)
    p_train.add_argument("--batch_size", type=int,   default=4)
    p_train.add_argument("--lr",         type=float, default=8e-5)
    p_train.add_argument("--no_augment", action="store_true")

    # eval
    p_eval = sub.add_parser("eval", help="Evaluate a checkpoint")
    p_eval.add_argument("--model",      required=True, choices=["unet", "bt_unet", "swinunet"])
    p_eval.add_argument("--ckpt",       required=True)
    p_eval.add_argument("--data_dir",   default="data/knee_singlecoil_val")
    p_eval.add_argument("--output_dir", default="results")

    # compare
    p_cmp = sub.add_parser("compare", help="Compare all three model checkpoints")
    p_cmp.add_argument("--data_dir",   default="data/knee_singlecoil_val")
    p_cmp.add_argument("--output_dir", default="checkpoints")

    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════╗")
    print("║     Accelerated MRI Reconstruction — SwinUNet       ║")
    print("╚══════════════════════════════════════════════════════╝")

    if args.command == "test_models":
        cmd_test_models(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "compare":
        cmd_compare(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
