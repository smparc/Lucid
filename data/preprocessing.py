"""
preprocessing.py
----------------
Data loading and retrospective undersampling for the fastMRI single-coil knee
dataset.

Pipeline
--------
1. Load complex k-space for one slice from an ``.h5`` volume.
2. Inverse FFT to a **complex** image and centre-crop the field of view.
3. (Training) apply a geometry augmentation to the complex image.
4. Forward FFT of the cropped image to obtain the simulation's k-space.
5. Apply the undersampling mask.
6. Inverse FFT to the zero-filled reconstruction, and scale everything by one
   scalar.

Sample contract
---------------
``__getitem__`` returns a dict, not a tuple::

    image      (C, H, W)  zero-filled reconstruction; C=2 complex, C=1 magnitude
    target     (1, H, W)  fully sampled magnitude ground truth
    kspace     (2, H, W)  the masked measurements the model must stay faithful to
    mask       (1, W)     binary sampling mask
    max_value  scalar     target maximum, the correct PSNR/SSIM data range
    scale      scalar     normalisation divisor, to undo scaling at inference
    fname, slice_idx      provenance

A dict is used deliberately: the tuple contract silently prevented data
consistency from ever being wired up, because there was nowhere to put k-space.
:func:`legacy_collate` restores the old ``(input, target)`` behaviour for code
that has not migrated.

Why undersample in the cropped domain
-------------------------------------
The original code masked full-FOV k-space and *then* cropped the image. That is
a faithful simulation, but it makes the measured coefficients unusable for data
consistency: after cropping, the measurements no longer correspond to the
Fourier transform of the image the network outputs, so writing them back is
mathematically wrong. Cropping first, then undersampling, means the model's
k-space and the measurements live in exactly the same space, so data
consistency is exact.

The cost is that cropping is a low-pass-plus-decimation of the readout axis, so
the simulated acquisition is of a slightly smaller FOV than the scanner's. Both
behaviours are available via ``undersample_domain``; ``"cropped"`` is the
default because exact physics is worth more than FOV fidelity here, and
``"full"`` reproduces the original numbers.

Normalisation
-------------
Scaling is always by a single positive scalar, never a mean shift. The Fourier
transform is linear, so ``x -> x / s`` implies ``k -> k / s`` and data
consistency survives untouched; subtracting a mean would break that
correspondence.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from data.masks import build_mask, effective_acceleration, equispaced_mask, random_mask
from models.fourier import complex_abs, fft2c, ifft2c, last_to_chan

log = logging.getLogger(__name__)

# Mask constructors and Fourier helpers are re-exported here because this module
# has always been their public entry point.
__all__ = [
    "FastMRIKneeDataset",
    "SampleSpec",
    "build_mask",
    "center_crop",
    "center_crop_complex",
    "collate",
    "complex_abs",
    "effective_acceleration",
    "equispaced_mask",
    "fft2c",
    "ifft2c",
    "legacy_collate",
    "normalize",
    "random_mask",
    "to_tensor",
]


# ---------------------------------------------------------------------------
# Tensor utilities
# ---------------------------------------------------------------------------


def to_tensor(kspace_np: np.ndarray) -> torch.Tensor:
    """Convert a complex numpy array to a real tensor of shape ``(..., 2)``."""
    if np.iscomplexobj(kspace_np):
        kspace_np = np.stack([kspace_np.real, kspace_np.imag], axis=-1)
    return torch.from_numpy(np.ascontiguousarray(kspace_np)).float()


def center_crop(img: torch.Tensor, crop_size: tuple[int, int] = (320, 320)) -> torch.Tensor:
    """
    Centre-crop ``(..., H, W)``, padding first if the image is smaller.

    Padding is symmetric so the anatomical centre stays at the image centre;
    an asymmetric pad would shift the k-space phase ramp and desynchronise the
    crop from the sampling mask.
    """
    h, w = img.shape[-2], img.shape[-1]
    ch, cw = crop_size

    if h < ch or w < cw:
        pad_h, pad_w = max(0, ch - h), max(0, cw - w)
        img = torch.nn.functional.pad(
            img, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
        )
        h, w = img.shape[-2], img.shape[-1]

    top, left = (h - ch) // 2, (w - cw) // 2
    return img[..., top : top + ch, left : left + cw]


def center_crop_complex(
    img: torch.Tensor, crop_size: tuple[int, int] = (320, 320)
) -> torch.Tensor:
    """Centre-crop a complex image stored as ``(H, W, 2)``."""
    return center_crop(img.permute(2, 0, 1), crop_size).permute(1, 2, 0).contiguous()


def normalize(
    img: torch.Tensor, max_val: float | None = None
) -> tuple[torch.Tensor, float]:
    """Scale by ``max_val`` (or the image maximum). Returns ``(scaled, divisor)``."""
    if max_val is None:
        max_val = float(img.abs().max())
    if max_val == 0 or not np.isfinite(max_val):
        return img, 1.0
    return img / max_val, float(max_val)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------


def augment_complex(img: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """
    Geometry augmentation applied to the **fully sampled complex image**, before
    undersampling.

    Order matters and the original code had it backwards. It augmented the
    already-undersampled input and target as a pair, which rotates the aliasing
    artefact away from the phase-encode direction that produced it. The network
    then sees artefact orientations that no Cartesian acquisition can generate,
    and — once data consistency is enabled — the measured k-space no longer
    corresponds to the augmented image at all.

    Augmenting first and undersampling afterwards keeps every sample a
    physically realisable acquisition.

    Parameters
    ----------
    img
        Complex image ``(H, W, 2)``.
    rng
        Generator supplying the augmentation decisions.
    """
    if rng.random() > 0.5:
        img = torch.flip(img, dims=[1])  # left-right
    if rng.random() > 0.5:
        img = torch.flip(img, dims=[0])  # up-down
    k = int(rng.integers(0, 4))
    if k:
        img = torch.rot90(img, k, dims=[0, 1])
    return img.contiguous()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleSpec:
    """One indexed slice: which file, which slice."""

    path: str
    slice_idx: int


class FastMRIKneeDataset(Dataset):
    """
    fastMRI single-coil knee reconstruction dataset.

    Parameters
    ----------
    root_dir
        Directory containing ``.h5`` volumes.
    center_fraction
        Fraction of low-frequency k-space lines always acquired.
    acceleration
        Acceleration factor. May be a list, in which case one value is drawn per
        sample — training across accelerations produces a single model that
        generalises across them instead of one model locked to a fixed R.
    crop_size
        Field of view after cropping.
    mask_type
        ``random``, ``equispaced`` or ``magic``.
    complex_input
        Return a 2-channel complex zero-filled image (required for data
        consistency) rather than 1-channel magnitude.
    train
        Training mode: masks are redrawn every access and augmentation is
        applied. In eval mode both are deterministic functions of the index, so
        validation numbers are exactly comparable across epochs and runs.
    augment
        Enable geometry augmentation (training mode only).
    seed
        Base seed for deterministic mask generation in eval mode.
    slice_mode
        ``middle`` (one slice per volume, as in the original study), ``all``, or
        ``range:start:end``.
    normalization
        ``zf_max`` scales by the zero-filled magnitude maximum (robust, always
        available); ``attr_max`` uses the volume's ``max`` HDF5 attribute, which
        is what the original code used; ``none`` disables scaling.
    undersample_domain
        ``cropped`` (default, exact data consistency) or ``full`` (mask the
        full-FOV k-space, then crop — reproduces the original simulation).
    cache
        Cache decoded complex images in memory. Caches the *pre-undersampling*
        image so masks can still vary per epoch.
    max_files
        Limit the number of volumes indexed. Useful for smoke tests.
    """

    def __init__(
        self,
        root_dir: str,
        center_fraction: float = 0.08,
        acceleration: int | Sequence[int] = 4,
        crop_size: tuple[int, int] = (320, 320),
        mask_type: str = "random",
        complex_input: bool = False,
        train: bool = False,
        augment: bool = False,
        seed: int = 42,
        slice_mode: str = "middle",
        normalization: str = "zf_max",
        undersample_domain: str = "cropped",
        cache: bool = False,
        max_files: int | None = None,
    ):
        self.root_dir = str(root_dir)
        self.center_fraction = float(center_fraction)
        self.accelerations = (
            [int(acceleration)]
            if isinstance(acceleration, (int, float))
            else [int(a) for a in acceleration]
        )
        self.crop_size = tuple(crop_size)
        self.mask_type = mask_type
        self.complex_input = bool(complex_input)
        self.train = bool(train)
        self.augment = bool(augment) and self.train
        self.seed = int(seed)
        self.slice_mode = slice_mode
        self.normalization = normalization
        self.cache_enabled = bool(cache)
        self._cache: dict[int, tuple[torch.Tensor, float]] = {}

        if undersample_domain not in ("cropped", "full"):
            raise ValueError(
                f"undersample_domain must be 'cropped' or 'full', got {undersample_domain!r}"
            )
        self.undersample_domain = undersample_domain

        if normalization not in ("zf_max", "attr_max", "none"):
            raise ValueError(
                f"normalization must be 'zf_max', 'attr_max' or 'none', got {normalization!r}"
            )

        if self.undersample_domain == "full" and self.complex_input:
            log.warning(
                "undersample_domain='full' with complex_input=True: the returned "
                "k-space is re-derived from the cropped image and is NOT the "
                "measured data, so data consistency will be approximate. Use "
                "undersample_domain='cropped' for exact data consistency."
            )

        self.samples = self._build_index(max_files)
        if not self.samples:
            raise FileNotFoundError(
                f"No readable .h5 files found in {root_dir!r}. "
                f"See data/README.md for download instructions."
            )

        log.info(
            "Dataset: %d samples from %s (slices=%s, R=%s, mask=%s, "
            "domain=%s, complex=%s, train=%s)",
            len(self.samples),
            root_dir,
            slice_mode,
            self.accelerations,
            mask_type,
            self.undersample_domain,
            self.complex_input,
            self.train,
        )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def _build_index(self, max_files: int | None) -> list[SampleSpec]:
        """Enumerate (file, slice) pairs without loading any k-space."""
        root = Path(self.root_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"Data directory does not exist: {root}")

        files = sorted(str(p) for p in root.glob("*.h5"))
        if max_files is not None:
            files = files[:max_files]

        samples: list[SampleSpec] = []
        skipped = 0
        for fpath in files:
            try:
                with h5py.File(fpath, "r") as hf:
                    if "kspace" not in hf:
                        raise KeyError("missing 'kspace' dataset")
                    n_slices = hf["kspace"].shape[0]
            except Exception as exc:
                log.warning("Skipping %s: %s", fpath, exc)
                skipped += 1
                continue
            samples.extend(SampleSpec(fpath, s) for s in self._slice_indices(n_slices))

        if skipped:
            log.warning("Skipped %d unreadable file(s)", skipped)
        return samples

    def _slice_indices(self, n_slices: int) -> list[int]:
        mode = self.slice_mode or "middle"
        if mode == "all":
            return list(range(n_slices))
        if mode.startswith("range:"):
            parts = mode.split(":")
            start = int(parts[1])
            end = int(parts[2]) if len(parts) > 2 and parts[2] else n_slices
            return list(range(max(0, start), min(end, n_slices)))
        if mode != "middle":
            log.warning("Unknown slice_mode %r; falling back to 'middle'", mode)
        return [n_slices // 2]

    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _rng(self, idx: int) -> np.random.Generator:
        """
        RNG for sample ``idx``.

        In eval mode the stream is a pure function of ``(seed, idx)``, so the
        validation set is byte-identical on every epoch and every run — without
        this, "val loss improved" can just mean "this epoch drew easier masks".

        In train mode entropy comes from the OS, so each epoch presents a fresh
        mask and a fresh augmentation. Per-worker seeding is handled by
        ``utils.reproducibility.seed_worker``.
        """
        if self.train:
            return np.random.default_rng()
        return np.random.default_rng(self.seed + idx)

    def _load_complex_image(self, idx: int) -> tuple[torch.Tensor, float]:
        """
        Load one slice as a cropped complex image ``(H, W, 2)`` plus its HDF5
        ``max`` attribute.
        """
        if self.cache_enabled and idx in self._cache:
            return self._cache[idx]

        spec = self.samples[idx]
        with h5py.File(spec.path, "r") as hf:
            kspace_slice = np.asarray(hf["kspace"][spec.slice_idx])
            attr_max = float(hf.attrs.get("max", 0.0))

        kspace = to_tensor(kspace_slice)  # (H, W, 2)
        image = ifft2c(kspace)  # complex image, full FOV

        if self.undersample_domain == "cropped":
            image = center_crop_complex(image, self.crop_size)

        result = (image, attr_max)
        if self.cache_enabled:
            self._cache[idx] = result
        return result

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rng = self._rng(idx)
        image, attr_max = self._load_complex_image(idx)

        if self.augment:
            image = augment_complex(image, rng)

        acceleration = int(rng.choice(self.accelerations))
        target_full = complex_abs(image)  # magnitude before cropping in 'full' mode

        # ── Undersample ────────────────────────────────────────────────────
        kspace_full = fft2c(image)
        mask = build_mask(
            self.mask_type,
            kspace_full.shape[:-1],
            center_fraction=self.center_fraction,
            acceleration=acceleration,
            rng=rng,
        )
        kspace_us = kspace_full * mask.unsqueeze(-1)
        zf_complex = ifft2c(kspace_us)

        if self.undersample_domain == "full":
            # Simulation happened at full FOV; crop everything afterwards. The
            # k-space returned is re-derived from the cropped image, so it is a
            # consistent pair with `image` but is not the literal measurement.
            zf_complex = center_crop_complex(zf_complex, self.crop_size)
            target_full = center_crop(target_full.unsqueeze(0), self.crop_size).squeeze(0)
            kspace_us = fft2c(zf_complex)
            mask = build_mask(
                self.mask_type,
                (self.crop_size[0], self.crop_size[1]),
                center_fraction=self.center_fraction,
                acceleration=acceleration,
                rng=np.random.default_rng(self.seed + idx),
            )

        # ── Scale ──────────────────────────────────────────────────────────
        scale = self._scale_for(zf_complex, attr_max)
        zf_complex = zf_complex / scale
        kspace_us = kspace_us / scale
        target = (target_full / scale).unsqueeze(0)  # (1, H, W)

        zf_mag = complex_abs(zf_complex).unsqueeze(0)  # (1, H, W)
        image_in = (
            last_to_chan(zf_complex.unsqueeze(0)).squeeze(0) if self.complex_input else zf_mag
        )

        return {
            "image": image_in.float().contiguous(),
            "target": target.float().contiguous(),
            "kspace": last_to_chan(kspace_us.unsqueeze(0)).squeeze(0).float().contiguous(),
            "mask": mask.float().contiguous(),
            "max_value": torch.tensor(float(target.max()), dtype=torch.float32),
            "scale": torch.tensor(float(scale), dtype=torch.float32),
            "acceleration": torch.tensor(acceleration, dtype=torch.int64),
            "fname": os.path.basename(self.samples[idx].path),
            "slice_idx": self.samples[idx].slice_idx,
        }

    def _scale_for(self, zf_complex: torch.Tensor, attr_max: float) -> float:
        """Choose the normalisation divisor; always positive and finite."""
        if self.normalization == "none":
            return 1.0
        if self.normalization == "attr_max" and attr_max > 0:
            return attr_max
        scale = float(complex_abs(zf_complex).max())
        return scale if scale > 0 and np.isfinite(scale) else 1.0

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_metadata(self, idx: int) -> dict[str, Any]:
        """Provenance and simulation settings for one sample."""
        spec = self.samples[idx]
        return {
            "file": os.path.basename(spec.path),
            "slice_idx": spec.slice_idx,
            "accelerations": self.accelerations,
            "center_fraction": self.center_fraction,
            "mask_type": self.mask_type,
            "undersample_domain": self.undersample_domain,
            "normalization": self.normalization,
        }

    def describe(self) -> dict[str, Any]:
        """Summary suitable for logging at the start of a run."""
        return {
            "n_samples": len(self),
            "n_volumes": len({s.path for s in self.samples}),
            "crop_size": list(self.crop_size),
            "accelerations": self.accelerations,
            "center_fraction": self.center_fraction,
            "mask_type": self.mask_type,
            "complex_input": self.complex_input,
            "undersample_domain": self.undersample_domain,
            "train_mode": self.train,
        }


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------


def legacy_collate(batch: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    """Collapse the sample dict to the old ``(input, target)`` tuple."""
    return (
        torch.stack([b["image"] for b in batch]),
        torch.stack([b["target"] for b in batch]),
    )


def collate(batch: list[dict]) -> dict[str, Any]:
    """
    Default collation: stack tensors, keep strings as lists.

    ``torch.utils.data.default_collate`` chokes on the string ``fname`` field in
    some versions; handling it explicitly keeps provenance attached to every
    batch, which is what makes per-file error analysis possible.
    """
    out: dict[str, Any] = {}
    for key in batch[0]:
        values = [b[key] for b in batch]
        if isinstance(values[0], torch.Tensor):
            out[key] = torch.stack(values)
        elif isinstance(values[0], (int, float)):
            out[key] = torch.tensor(values)
        else:
            out[key] = values
    return out
