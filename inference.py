"""
inference.py
------------
Production inference pipeline with ONNX/TorchScript export support.


Features
--------
- Single-image and batch inference
- ONNX export for deployment
- TorchScript tracing for C++ deployment
- Timing/profiling utilities
- Input validation and preprocessing
"""


import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional, Union


import numpy as np
import torch
import torch.nn as nn


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


from models.unet import UNet
from models.bt_unet import BTUNet
from models.swinunet import SwinUNet
from data.preprocessing import (
    to_tensor, ifft2c, complex_abs, center_crop, normalize, random_mask
)


log = logging.getLogger(__name__)



class MRIReconstructionPipeline:
    """
    End-to-end inference pipeline for MRI reconstruction.


    Usage
    -----
        pipe = MRIReconstructionPipeline.from_checkpoint("checkpoints/best.pt")
        reconstruction = pipe.reconstruct(undersampled_kspace)
        pipe.export_onnx("exports/model.onnx")
    """


    def __init__(self, model: nn.Module, device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self._warmup_done = False


    
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        model_name: str = None,
        device: torch.device = None,
    ) -> "MRIReconstructionPipeline":
        """
        Load pipeline from a training checkpoint.


        Parameters
        ----------
        checkpoint_path : path to .pt checkpoint file
        model_name      : model architecture name (auto-detected from checkpoint if saved)
        device          : target device
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)


        # Auto-detect model name from config saved in checkpoint
        if model_name is None:
            config = ckpt.get("config", {})
            model_name = config.get("model", {}).get("name", "swinunet")


        model = cls._build_model(model_name, ckpt.get("config", {}))


        # Prefer EMA weights (better generalization) over raw training weights
        if ckpt.get("ema_state") is not None:
            ema_state = ckpt["ema_state"]
            shadow_params = ema_state.get("shadow_params")
            if shadow_params is not None:
                # Apply EMA shadow params to model
                for param, shadow in zip(model.parameters(), shadow_params):
                    param.data.copy_(shadow)
                log.info("Loaded EMA (shadow) weights for inference")
            else:
                model.load_state_dict(ckpt["model_state"])
        else:
            model.load_state_dict(ckpt["model_state"])


        log.info(f"Loaded {model_name} from {checkpoint_path} "
                 f"(epoch {ckpt.get('epoch', '?')}, PSNR={ckpt.get('val_psnr', 0):.2f}dB)")


        return cls(model, device)


    
    def _build_model(name: str, config: dict = None) -> nn.Module:
        """Build model from name and optional config."""
        params = config.get("model", {}).get("params", {}) if config else {}
        name = name.lower()


        if name == "unet":
            return UNet(
                in_channels=params.get("in_channels", 1),
                out_channels=params.get("out_channels", 1),
                base_ch=params.get("base_ch", 32),
                n_levels=params.get("n_levels", 4),
            )
        elif name == "bt_unet":
            return BTUNet(
                in_channels=params.get("in_channels", 1),
                out_channels=params.get("out_channels", 1),
                base_ch=params.get("base_ch", 32),
                n_levels=params.get("n_levels", 4),
                tf_heads=params.get("tf_heads", 8),
                tf_layers=params.get("tf_layers", 4),
            )
        elif name == "swinunet":
            return SwinUNet(
                img_size=params.get("img_size", 320),
                patch_size=params.get("patch_size", 4),
                in_ch=params.get("in_ch", 1),
                out_ch=params.get("out_ch", 1),
                embed_dim=params.get("embed_dim", 64),
                ws=params.get("ws", 8),
                head_dim=params.get("head_dim", 8),
                dropout=0.0,  # No dropout at inference
                n_levels=params.get("n_levels", 3),
            )
        else:
            raise ValueError(f"Unknown model: {name}")


    def warmup(self, input_size: tuple = (1, 1, 320, 320)):
        """Run a dummy forward pass to warm up CUDA kernels."""
        dummy = torch.randn(*input_size, device=self.device)
        with torch.no_grad():
            _ = self.model(dummy)
        self._warmup_done = True


    @torch.no_grad()
    def reconstruct(
        self,
        input_image: Union[torch.Tensor, np.ndarray],
        return_time: bool = False,
    ) -> Union[torch.Tensor, tuple]:
        """
        Reconstruct a full MRI image from an undersampled input.


        Parameters
        ----------
        input_image  : (H, W) or (1, H, W) or (B, 1, H, W) tensor/ndarray in [0, 1]
        return_time  : if True, also return inference time in ms


        Returns
        -------
        reconstruction : torch.Tensor (B, 1, H, W)
        time_ms        : float (only if return_time=True)
        """
        if not self._warmup_done:
            self.warmup()


        # Normalize input shape
        if isinstance(input_image, np.ndarray):
            input_image = torch.from_numpy(input_image).float()


        if input_image.dim() == 2:
            input_image = input_image.unsqueeze(0).unsqueeze(0)
        elif input_image.dim() == 3:
            input_image = input_image.unsqueeze(0)


        # Input validation
        if torch.isnan(input_image).any() or torch.isinf(input_image).any():
            raise ValueError("Input contains NaN or Inf values")
        input_image = input_image.clamp(0, 1)


        input_image = input_image.to(self.device)


        # Inference with timing
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()


        output = self.model(input_image)


        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000


        output = output.clamp(0, 1)


        if return_time:
            return output, elapsed_ms
        return output


    @torch.no_grad()
    def reconstruct_from_kspace(
        self,
        kspace: np.ndarray,
        center_fraction: float = 0.08,
        acceleration: int = 4,
        crop_size: tuple = (320, 320),
    ) -> torch.Tensor:
        """
        Full pipeline: raw k-space → undersampled → reconstruction.


        Parameters
        ----------
        kspace : complex numpy array (H, W)
        """
        kspace_t = to_tensor(kspace)
        mask = random_mask(kspace.shape, center_fraction, acceleration)
        kspace_us = kspace_t * mask.unsqueeze(-1)
        input_img = complex_abs(ifft2c(kspace_us))
        input_img, max_val = normalize(input_img)
        input_img = center_crop(input_img.unsqueeze(0), crop_size)


        output = self.reconstruct(input_img)
        return output * max_val


    def benchmark(self, n_runs: int = 100, input_size: tuple = (1, 1, 320, 320)) -> dict:
        """
        Benchmark inference speed.


        Returns
        -------
        dict with keys: mean_ms, std_ms, min_ms, max_ms, throughput_fps
        """
        self.warmup(input_size)
        dummy = torch.randn(*input_size, device=self.device)
        times = []


        for _ in range(n_runs):
            _, t = self.reconstruct(dummy, return_time=True)
            times.append(t)


        times = np.array(times)
        return {
            "mean_ms": float(times.mean()),
            "std_ms": float(times.std()),
            "min_ms": float(times.min()),
            "max_ms": float(times.max()),
            "throughput_fps": float(1000.0 / times.mean()),
        }


    # -----------------------------------------------------------------------
    # Export
    # -----------------------------------------------------------------------


    def export_onnx(
        self,
        output_path: str = "exports/model.onnx",
        input_size: tuple = (1, 1, 320, 320),
        opset_version: int = 17,
        dynamic_batch: bool = True,
    ) -> str:
        """
        Export model to ONNX format.


        Parameters
        ----------
        output_path   : where to save the .onnx file
        input_size    : dummy input shape for tracing
        opset_version : ONNX opset version
        dynamic_batch : allow variable batch size at runtime
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)


        dummy = torch.randn(*input_size, device=self.device)
        dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}} if dynamic_batch else None


        torch.onnx.export(
            self.model,
            dummy,
            output_path,
            opset_version=opset_version,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )


        # Validate
        import onnx
        model_onnx = onnx.load(output_path)
        onnx.checker.check_model(model_onnx)


        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        log.info(f"ONNX model exported to {output_path} ({file_size_mb:.1f} MB)")
        return output_path


    def export_torchscript(
        self,
        output_path: str = "exports/model_traced.pt",
        input_size: tuple = (1, 1, 320, 320),
    ) -> str:
        """Export model via TorchScript tracing."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)


        dummy = torch.randn(*input_size, device=self.device)
        traced = torch.jit.trace(self.model, dummy)
        traced.save(output_path)


        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        log.info(f"TorchScript model exported to {output_path} ({file_size_mb:.1f} MB)")
        return output_path



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse


    logging.basicConfig(level=logging.INFO)


    parser = argparse.ArgumentParser(description="MRI Reconstruction Inference & Export")
    sub = parser.add_subparsers(dest="command")


    # Benchmark
    p_bench = sub.add_parser("benchmark", help="Benchmark inference speed")
    p_bench.add_argument("--checkpoint", required=True)
    p_bench.add_argument("--model", default=None)
    p_bench.add_argument("--n_runs", type=int, default=100)


    # Export ONNX
    p_onnx = sub.add_parser("export-onnx", help="Export to ONNX")
    p_onnx.add_argument("--checkpoint", required=True)
    p_onnx.add_argument("--model", default=None)
    p_onnx.add_argument("--output", default="exports/model.onnx")


    # Export TorchScript
    p_ts = sub.add_parser("export-torchscript", help="Export to TorchScript")
    p_ts.add_argument("--checkpoint", required=True)
    p_ts.add_argument("--model", default=None)
    p_ts.add_argument("--output", default="exports/model_traced.pt")


    args = parser.parse_args()


    if args.command == "benchmark":
        pipe = MRIReconstructionPipeline.from_checkpoint(args.checkpoint, args.model)
        results = pipe.benchmark(n_runs=args.n_runs)
        print(f"\n── Benchmark Results ({args.n_runs} runs) ──")
        print(f"  Mean: {results['mean_ms']:.2f} ms")
        print(f"  Std:  {results['std_ms']:.2f} ms")
        print(f"  Min:  {results['min_ms']:.2f} ms")
        print(f"  Max:  {results['max_ms']:.2f} ms")
        print(f"  Throughput: {results['throughput_fps']:.1f} FPS")


    elif args.command == "export-onnx":
        pipe = MRIReconstructionPipeline.from_checkpoint(args.checkpoint, args.model)
        pipe.export_onnx(args.output)


    elif args.command == "export-torchscript":
        pipe = MRIReconstructionPipeline.from_checkpoint(args.checkpoint, args.model)
        pipe.export_torchscript(args.output)


    else:
        parser.print_help()