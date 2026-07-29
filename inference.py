"""
inference.py
------------
Production inference pipeline with ONNX / TorchScript export.

Features
--------
* Single-image and batch reconstruction, with or without k-space guidance.
* Test-time augmentation over the dihedral group.
* Monte-Carlo dropout uncertainty maps.
* Honest benchmarking (warmup, CUDA synchronisation, median and percentiles).
* ONNX and TorchScript export with output verification.

Fixed here
----------
``from_checkpoint`` and ``_build_model`` were missing their ``@classmethod`` and
``@staticmethod`` decorators. Both were therefore plain functions on the class,
so the documented call ``MRIReconstructionPipeline.from_checkpoint(path)`` bound
``path`` to the ``cls`` parameter and ``model_name`` to ``checkpoint_path``,
making every entry point in the README — ``benchmark``, ``export``, and the
Python API — raise immediately. The EMA loading path was also broken
independently: it looked for a ``shadow_params`` key that :class:`EMAModel` never
emitted (it wrote ``shadow``), so the ``else`` branch silently loaded raw
training weights while logging nothing.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from config import Config, load_config
from models.registry import build_model, forward_model
from utils.ema import load_ema_weights_into

log = logging.getLogger(__name__)

__all__ = ["MRIReconstructionPipeline"]


class MRIReconstructionPipeline:
    """
    End-to-end inference pipeline.

    Usage
    -----
        pipe = MRIReconstructionPipeline.from_checkpoint("checkpoints/best.pt")
        recon = pipe.reconstruct(zero_filled_image)
        pipe.export_onnx("exports/model.onnx")
    """

    def __init__(self, model: nn.Module, device: torch.device | None = None):
        self.device = device or self._default_device()
        self.model = model.to(self.device).eval()
        self.expects_kspace = bool(getattr(model, "expects_kspace", False))
        self.in_channels = int(getattr(model, "in_channels", 1))
        self._warmup_done = False

    @staticmethod
    def _default_device() -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        model_name: str | None = None,
        device: torch.device | None = None,
        use_ema: bool = True,
    ) -> MRIReconstructionPipeline:
        """
        Load a pipeline from a training checkpoint.

        Parameters
        ----------
        checkpoint_path
            Path to a ``.pt`` checkpoint.
        model_name
            Architecture override, needed only for legacy checkpoints that do
            not embed their config.
        device
            Target device; defaults to CUDA when available.
        use_ema
            Prefer the averaged weights the run was actually validated on.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        device = device or cls._default_device()
        ckpt = torch.load(path, map_location=device, weights_only=False)

        config = ckpt.get("config") or {}
        if config and "model" in config:
            cfg = config if isinstance(config, Config) else Config(config)
        else:
            name = model_name or "swinunet"
            log.warning("Checkpoint has no embedded config; building %r from defaults", name)
            cfg = load_config(overrides={"model": {"name": name}}, validate=False)

        model = build_model(cfg)
        model.load_state_dict(ckpt["model_state"])

        applied_ema = load_ema_weights_into(model, ckpt.get("ema_state")) if use_ema else False

        psnr_value = ckpt.get("val_psnr")
        log.info(
            "Loaded %s from %s (epoch %s, PSNR=%s, weights=%s)",
            cfg.model.get("name", model_name),
            path,
            ckpt.get("epoch", "?"),
            f"{psnr_value:.2f} dB" if isinstance(psnr_value, (int, float)) else "?",
            "EMA" if applied_ema else "raw",
        )
        return cls(model, device)

    @staticmethod
    def _build_model(name: str, config: dict | None = None) -> nn.Module:
        """Build a bare model by name. Retained for backward compatibility."""
        cfg = (
            config
            if isinstance(config, Config)
            else Config(config or {"model": {"name": name}})
        )
        cfg.setdefault("model", {})
        cfg["model"]["name"] = name
        return build_model(cfg)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def warmup(self, input_size: tuple[int, ...] = (1, 1, 320, 320)) -> None:
        """Run one dummy pass so lazy kernel selection does not pollute timings."""
        size = (input_size[0], self.in_channels, *input_size[2:])
        dummy = torch.randn(*size, device=self.device)
        kspace, mask = self._dummy_kspace(size) if self.expects_kspace else (None, None)
        with torch.no_grad():
            forward_model(self.model, dummy, kspace, mask)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self._warmup_done = True

    def _dummy_kspace(self, size: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        B, _, H, W = size
        kspace = torch.randn(B, 2, H, W, device=self.device)
        mask = (torch.rand(B, 1, W, device=self.device) > 0.75).float()
        return kspace, mask

    def _prepare(self, image: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Coerce an input to a batched ``(B, C, H, W)`` float tensor on device."""
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(np.ascontiguousarray(image))
        image = image.float()

        if image.dim() == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        elif image.dim() == 3:
            image = image.unsqueeze(0)
        elif image.dim() != 4:
            raise ValueError(f"Expected a 2D, 3D or 4D input, got shape {tuple(image.shape)}")

        if not torch.isfinite(image).all():
            raise ValueError("Input contains NaN or Inf values")

        if image.shape[1] != self.in_channels:
            raise ValueError(
                f"Model expects {self.in_channels} input channel(s) but got "
                f"{image.shape[1]}. Complex models need a (real, imaginary) pair."
            )

        return image.to(self.device)

    @torch.no_grad()
    def reconstruct(
        self,
        input_image: torch.Tensor | np.ndarray,
        kspace: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        return_time: bool = False,
        tta: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, float]:
        """
        Reconstruct from a zero-filled image.

        Parameters
        ----------
        input_image
            ``(H, W)``, ``(C, H, W)`` or ``(B, C, H, W)``, already scaled the way
            the model was trained.
        kspace, mask
            Measured k-space and its sampling mask. Required when the model
            includes data consistency.
        return_time
            Also return the wall-clock inference time in milliseconds.
        tta
            Average predictions over the eight flips and rotations of the
            dihedral group.

        Notes
        -----
        The input is **not** clamped to [0, 1]. The original implementation did,
        which corrupts any input whose per-slice normalisation places tissue
        above 1.0 — silently, and in a way that looks like a model quality
        problem rather than a preprocessing one.
        """
        if not self._warmup_done:
            self.warmup((1, self.in_channels, *tuple(np.shape(input_image))[-2:]))

        image = self._prepare(input_image)

        if self.expects_kspace and (kspace is None or mask is None):
            raise ValueError(
                "This model was trained with data consistency and needs `kspace` "
                "and `mask`. Use reconstruct_from_kspace(), or load a checkpoint "
                "whose config has data_consistency.enabled=false."
            )
        if kspace is not None:
            kspace = kspace.to(self.device).float()
        if mask is not None:
            mask = mask.to(self.device).float()

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        output = (
            self._forward_tta(image, kspace, mask)
            if tta
            else forward_model(self.model, image, kspace, mask)
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return (output, elapsed_ms) if return_time else output

    def _forward_tta(
        self, image: torch.Tensor, kspace: torch.Tensor | None, mask: torch.Tensor | None
    ) -> torch.Tensor:
        """
        Average predictions over the dihedral group of the square.

        Each transform is undone before averaging. Rotations are skipped when
        k-space guidance is active: rotating the image invalidates the
        correspondence with the measured phase-encode direction, so a rotated
        pass would apply data consistency to the wrong frequencies. Flips along
        the two axes remain valid because they map the Cartesian grid onto
        itself.
        """
        use_rotations = kspace is None
        outputs = []

        for flip_h in (False, True):
            for flip_v in (False, True):
                for k in ((0, 1, 2, 3) if use_rotations else (0,)):
                    x = image
                    if flip_h:
                        x = torch.flip(x, dims=[-1])
                    if flip_v:
                        x = torch.flip(x, dims=[-2])
                    if k:
                        x = torch.rot90(x, k, dims=[-2, -1])

                    y = forward_model(self.model, x, kspace, mask)

                    if k:
                        y = torch.rot90(y, -k, dims=[-2, -1])
                    if flip_v:
                        y = torch.flip(y, dims=[-2])
                    if flip_h:
                        y = torch.flip(y, dims=[-1])
                    outputs.append(y)

        return torch.stack(outputs).mean(dim=0)

    @torch.no_grad()
    def reconstruct_with_uncertainty(
        self,
        input_image: torch.Tensor | np.ndarray,
        kspace: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        n_samples: int = 16,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Monte-Carlo dropout reconstruction with a per-pixel uncertainty map.

        Dropout layers are re-enabled while everything else stays in eval mode,
        turning the network into an approximate posterior sample generator
        (Gal & Ghahramani, 2016). The returned standard-deviation map highlights
        exactly the regions where the model is interpolating rather than
        reconstructing — which is where a reader should be sceptical, and which
        the paper listed as future work.

        Returns
        -------
        ``(mean, std)``, both ``(B, 1, H, W)``.

        Raises
        ------
        RuntimeError
            If the model has no dropout, in which case every sample would be
            identical and the "uncertainty" map would be a misleading field of
            zeros.
        """
        image = self._prepare(input_image)

        dropouts = [m for m in self.model.modules() if isinstance(m, (nn.Dropout, nn.Dropout2d))]
        active = [m for m in dropouts if m.p > 0]
        if not active:
            raise RuntimeError(
                "Model contains no active dropout, so MC-dropout would return a "
                "zero uncertainty map. Train with model.params.dropout > 0, or "
                "use an ensemble instead."
            )

        for module in active:
            module.train()
        try:
            samples = torch.stack(
                [forward_model(self.model, image, kspace, mask) for _ in range(n_samples)]
            )
        finally:
            for module in active:
                module.eval()

        return samples.mean(dim=0), samples.std(dim=0)

    @torch.no_grad()
    def reconstruct_from_kspace(
        self,
        kspace: np.ndarray,
        center_fraction: float = 0.08,
        acceleration: int = 4,
        crop_size: tuple[int, int] = (320, 320),
        mask_type: str = "random",
        seed: int | None = 0,
    ) -> dict[str, Any]:
        """
        Full pipeline: raw complex k-space -> undersampled -> reconstruction.

        Mirrors the training-time simulation exactly (crop the complex image
        first, then undersample), so the model sees the distribution it was
        trained on.

        Returns
        -------
        dict with ``reconstruction`` (rescaled to the input's units),
        ``zero_filled``, ``mask`` and ``scale``.
        """
        from data.masks import build_mask
        from data.preprocessing import center_crop_complex, to_tensor
        from models.fourier import complex_abs, fft2c, ifft2c, last_to_chan

        kspace_t = to_tensor(np.asarray(kspace))
        image = center_crop_complex(ifft2c(kspace_t), crop_size)

        k_full = fft2c(image)
        mask = build_mask(
            mask_type, k_full.shape[:-1], center_fraction, acceleration, seed=seed
        )
        k_us = k_full * mask.unsqueeze(-1)
        zf = ifft2c(k_us)

        scale = float(complex_abs(zf).max()) or 1.0
        zf, k_us = zf / scale, k_us / scale

        zf_chan = last_to_chan(zf.unsqueeze(0))
        model_input = zf_chan if self.in_channels == 2 else complex_abs(zf).unsqueeze(0).unsqueeze(0)

        output = self.reconstruct(
            model_input,
            kspace=last_to_chan(k_us.unsqueeze(0)) if self.expects_kspace else None,
            mask=mask.unsqueeze(0) if self.expects_kspace else None,
        )

        return {
            "reconstruction": output * scale,
            "zero_filled": complex_abs(zf).unsqueeze(0).unsqueeze(0) * scale,
            "mask": mask,
            "scale": scale,
        }

    # ------------------------------------------------------------------
    # Benchmarking
    # ------------------------------------------------------------------

    def benchmark(
        self,
        n_runs: int = 100,
        input_size: tuple[int, ...] = (1, 1, 320, 320),
        warmup_runs: int = 10,
    ) -> dict[str, float]:
        """
        Measure inference latency.

        Reports the **median** alongside the mean: latency distributions are
        right-skewed (allocator activity, thermal throttling, other processes),
        so a mean over 100 runs can be dragged well above typical by a handful
        of outliers. p95 is included because that is what a deployment cares
        about.
        """
        size = (input_size[0], self.in_channels, *input_size[2:])
        self.warmup(size)

        dummy = torch.randn(*size, device=self.device)
        kspace, mask = self._dummy_kspace(size) if self.expects_kspace else (None, None)

        for _ in range(warmup_runs):
            self.reconstruct(dummy, kspace, mask)

        times = []
        for _ in range(n_runs):
            _, elapsed = self.reconstruct(dummy, kspace, mask, return_time=True)
            times.append(elapsed)

        arr = np.array(times)
        return {
            "mean_ms": float(arr.mean()),
            "median_ms": float(np.median(arr)),
            "std_ms": float(arr.std()),
            "min_ms": float(arr.min()),
            "max_ms": float(arr.max()),
            "p95_ms": float(np.percentile(arr, 95)),
            "throughput_fps": float(1000.0 * size[0] / np.median(arr)),
            "batch_size": size[0],
            "device": str(self.device),
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_onnx(
        self,
        output_path: str = "exports/model.onnx",
        input_size: tuple[int, ...] = (1, 1, 320, 320),
        opset_version: int = 17,
        dynamic_batch: bool = True,
        verify: bool = True,
    ) -> str:
        """
        Export to ONNX and verify the exported graph reproduces PyTorch output.

        Verification is not optional busywork: tracing silently bakes in any
        Python-level control flow, and an export that produces a *different*
        model is far more dangerous than one that fails outright.
        """
        if self.expects_kspace:
            raise NotImplementedError(
                "Models with data consistency take three inputs (image, k-space, "
                "mask) and are not exported by this single-input path. Export the "
                "backbone alone, or extend this method with the full signature."
            )

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        size = (input_size[0], self.in_channels, *input_size[2:])
        dummy = torch.randn(*size, device=self.device)

        dynamic_axes = (
            {"input": {0: "batch", 2: "height", 3: "width"},
             "output": {0: "batch", 2: "height", 3: "width"}}
            if dynamic_batch
            else None
        )

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

        try:
            import onnx

            onnx.checker.check_model(onnx.load(output_path))
        except ImportError:
            log.warning("onnx not installed; skipping graph validation")

        if verify:
            self._verify_onnx(output_path, dummy)

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        log.info("Exported ONNX model to %s (%.1f MB)", output_path, size_mb)
        return output_path

    def _verify_onnx(self, path: str, dummy: torch.Tensor, tol: float = 1e-3) -> None:
        """Compare ONNX Runtime output against PyTorch."""
        try:
            import onnxruntime as ort
        except ImportError:
            log.warning("onnxruntime not installed; skipping numerical verification")
            return

        with torch.no_grad():
            expected = self.model(dummy).cpu().numpy()

        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        actual = session.run(None, {"input": dummy.cpu().numpy()})[0]

        max_diff = float(np.abs(expected - actual).max())
        if max_diff > tol:
            raise RuntimeError(
                f"ONNX export verification failed: max difference {max_diff:.2e} "
                f"exceeds tolerance {tol:.0e}. The exported graph does not match "
                f"the PyTorch model."
            )
        log.info("ONNX output verified (max difference %.2e)", max_diff)

    def export_torchscript(
        self,
        output_path: str = "exports/model_traced.pt",
        input_size: tuple[int, ...] = (1, 1, 320, 320),
        verify: bool = True,
    ) -> str:
        """Export via TorchScript tracing, verifying the traced output."""
        if self.expects_kspace:
            raise NotImplementedError(
                "Models with data consistency take three inputs and are not "
                "supported by this single-input tracing path."
            )

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        size = (input_size[0], self.in_channels, *input_size[2:])
        dummy = torch.randn(*size, device=self.device)

        traced = torch.jit.trace(self.model, dummy)
        traced.save(output_path)

        if verify:
            with torch.no_grad():
                max_diff = float((self.model(dummy) - traced(dummy)).abs().max())
            if max_diff > 1e-4:
                raise RuntimeError(
                    f"TorchScript verification failed: max difference {max_diff:.2e}"
                )
            log.info("TorchScript output verified (max difference %.2e)", max_diff)

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        log.info("Exported TorchScript model to %s (%.1f MB)", output_path, size_mb)
        return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="MRI reconstruction inference and export")
    sub = parser.add_subparsers(dest="command", required=True)

    p_bench = sub.add_parser("benchmark", help="Benchmark inference speed")
    p_bench.add_argument("--checkpoint", required=True)
    p_bench.add_argument("--model", default=None)
    p_bench.add_argument("--n_runs", type=int, default=100)
    p_bench.add_argument("--batch_size", type=int, default=1)

    p_onnx = sub.add_parser("export-onnx", help="Export to ONNX")
    p_onnx.add_argument("--checkpoint", required=True)
    p_onnx.add_argument("--model", default=None)
    p_onnx.add_argument("--output", default="exports/model.onnx")

    p_ts = sub.add_parser("export-torchscript", help="Export to TorchScript")
    p_ts.add_argument("--checkpoint", required=True)
    p_ts.add_argument("--model", default=None)
    p_ts.add_argument("--output", default="exports/model_traced.pt")

    args = parser.parse_args()
    pipe = MRIReconstructionPipeline.from_checkpoint(args.checkpoint, args.model)

    if args.command == "benchmark":
        results = pipe.benchmark(n_runs=args.n_runs, input_size=(args.batch_size, 1, 320, 320))
        print(f"\n-- Benchmark ({args.n_runs} runs, {results['device']}) --")
        for key in ("median_ms", "mean_ms", "std_ms", "min_ms", "max_ms", "p95_ms"):
            print(f"  {key:<12} {results[key]:.2f}")
        print(f"  {'throughput':<12} {results['throughput_fps']:.1f} FPS")
    elif args.command == "export-onnx":
        pipe.export_onnx(args.output)
    elif args.command == "export-torchscript":
        pipe.export_torchscript(args.output)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
