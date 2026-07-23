"""
preprocessing.py
----------------
Data loading and preprocessing for the fastMRI single-coil knee dataset.


Features
--------
- Multi-slice support (configurable: middle, all, or range)
- In-memory caching for faster training after first epoch
- Configurable undersampling (center fraction, acceleration factor)
- Equispaced and random mask generation
- Proper normalization strategies


Pipeline
--------
1. Load complex k-space from .h5 files
2. Extract slices (middle by default)
3. Apply retrospective undersampling mask (center_fraction=0.08, acceleration=4)
4. Compute zero-filled IFFT reconstruction
5. Normalize by per-scan maximum
6. Center-crop to 320x320
"""


import os
import logging
from functools import lru_cache


import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, List


log = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# Mask generation
# ---------------------------------------------------------------------------


def random_mask(
    shape: Tuple[int, int],
    center_fraction: float = 0.08,
    acceleration: int = 4,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Generate a random undersampling mask for k-space.


    Parameters
    ----------
    shape            : (num_rows, num_cols) — k-space shape
    center_fraction  : float — fraction of low-frequency lines always sampled
    acceleration     : int — overall acceleration factor
    seed             : optional random seed for reproducibility


    Returns
    -------
    torch.Tensor (1, num_cols) — binary mask (1=sampled, 0=zero-filled)
    """
    num_cols = shape[-1]
    num_low_freqs = int(round(num_cols * center_fraction))


    # Compute probability for additional random lines
    prob = (num_cols / acceleration - num_low_freqs) / (num_cols - num_low_freqs)
    prob = max(0.0, min(1.0, prob))  # Clamp to valid range


    rng = np.random.default_rng(seed)
    mask = rng.uniform(size=num_cols) < prob


    # Always include centre frequencies
    pad = (num_cols - num_low_freqs + 1) // 2
    mask[pad: pad + num_low_freqs] = True


    return torch.from_numpy(mask.reshape(1, num_cols).astype(np.float32))



def equispaced_mask(
    shape: Tuple[int, int],
    center_fraction: float = 0.08,
    acceleration: int = 4,
) -> torch.Tensor:
    """
    Generate an equispaced undersampling mask (deterministic).


    Samples every `acceleration`-th line plus the center fraction.
    """
    num_cols = shape[-1]
    num_low_freqs = int(round(num_cols * center_fraction))


    mask = np.zeros(num_cols, dtype=np.float32)


    # Equispaced lines
    mask[::acceleration] = 1.0


    # Center frequencies
    pad = (num_cols - num_low_freqs + 1) // 2
    mask[pad: pad + num_low_freqs] = 1.0


    return torch.from_numpy(mask.reshape(1, num_cols))



# ---------------------------------------------------------------------------
# k-space ↔ image utilities
# ---------------------------------------------------------------------------


def ifft2c(kspace: torch.Tensor) -> torch.Tensor:
    """
    Centred 2D inverse FFT: k-space (complex) → image (complex).


    kspace : (..., H, W, 2)  where dim -1 = [real, imag]
    """
    x = torch.view_as_complex(kspace.float())
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    x = torch.fft.ifft2(x, norm="ortho")
    x = torch.fft.fftshift(x, dim=(-2, -1))
    return torch.view_as_real(x)



def complex_abs(x: torch.Tensor) -> torch.Tensor:
    """Magnitude of a complex tensor stored as (..., 2)."""
    return torch.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2)



def to_tensor(kspace_np: np.ndarray) -> torch.Tensor:
    """Convert complex numpy k-space array to a real tensor of shape (..., 2)."""
    if np.iscomplexobj(kspace_np):
        kspace_np = np.stack([kspace_np.real, kspace_np.imag], axis=-1)
    return torch.from_numpy(kspace_np.astype(np.float32))



def center_crop(img: torch.Tensor, crop_size: Tuple[int, int] = (320, 320)) -> torch.Tensor:
    """Center-crop a 2D image tensor (..., H, W)."""
    h, w = img.shape[-2], img.shape[-1]
    ch, cw = crop_size
    if h < ch or w < cw:
        # Pad if image is smaller than crop size
        pad_h = max(0, ch - h)
        pad_w = max(0, cw - w)
        img = torch.nn.functional.pad(img, [pad_w // 2, pad_w - pad_w // 2,
                                             pad_h // 2, pad_h - pad_h // 2])
        h, w = img.shape[-2], img.shape[-1]
    top = (h - ch) // 2
    left = (w - cw) // 2
    return img[..., top: top + ch, left: left + cw]



def normalize(img: torch.Tensor, max_val: Optional[float] = None) -> Tuple[torch.Tensor, float]:
    """Normalize image by its maximum value."""
    if max_val is None:
        max_val = float(img.max())
    if max_val == 0:
        return img, 1.0
    return img / max_val, max_val



# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class FastMRIKneeDataset(Dataset):
    """
    PyTorch Dataset for fastMRI single-coil knee reconstruction.


    Supports:
    - Middle slice only (default, as in paper)
    - All slices from each volume
    - Configurable slice range
    - In-memory caching for repeated access


    Returns
    -------
    (input_image, target_image)  — both torch.Tensor (1, 320, 320), float32
    """


    def __init__(
        self,
        root_dir: str,
        center_fraction: float = 0.08,
        acceleration: int = 4,
        crop_size: Tuple[int, int] = (320, 320),
        seed: int = 42,
        slice_mode: str = "middle",  # "middle", "all", or "range:start:end"
        mask_type: str = "random",   # "random" or "equispaced"
        cache: bool = False,
    ):
        """
        Parameters
        ----------
        root_dir         : directory containing .h5 files
        center_fraction  : fraction of k-space centre always sampled
        acceleration     : subsampling factor
        crop_size        : final image crop size
        seed             : random seed for mask generation
        slice_mode       : which slices to use per volume
        mask_type        : "random" or "equispaced" undersampling
        cache            : if True, cache processed samples in memory
        """
        self.root_dir = root_dir
        self.center_fraction = center_fraction
        self.acceleration = acceleration
        self.crop_size = crop_size
        self.seed = seed
        self.slice_mode = slice_mode
        self.mask_type = mask_type
        self.cache = cache
        self._cache_dict = {}


        # Build index: list of (file_path, slice_idx) pairs
        self.samples = self._build_index()
        if not self.samples:
            raise FileNotFoundError(f"No .h5 files found in {root_dir}")


        log.info(f"Dataset: {len(self.samples)} samples from {root_dir} "
                 f"(mode={slice_mode}, accel={acceleration}x)")


    def _build_index(self) -> List[Tuple[str, int]]:
        """Build a flat index of (filepath, slice_idx) pairs."""
        files = sorted([
            os.path.join(self.root_dir, f)
            for f in os.listdir(self.root_dir) if f.endswith(".h5")
        ])


        samples = []
        for fpath in files:
            try:
                with h5py.File(fpath, "r") as hf:
                    n_slices = hf["kspace"].shape[0]
            except Exception as e:
                log.warning(f"Skipping {fpath}: {e}")
                continue


            if self.slice_mode == "middle":
                samples.append((fpath, n_slices // 2))
            elif self.slice_mode == "all":
                for s in range(n_slices):
                    samples.append((fpath, s))
            elif self.slice_mode.startswith("range:"):
                parts = self.slice_mode.split(":")
                start = int(parts[1])
                end = int(parts[2]) if len(parts) > 2 else n_slices
                for s in range(start, min(end, n_slices)):
                    samples.append((fpath, s))
            else:
                samples.append((fpath, n_slices // 2))


        return samples


    def __len__(self) -> int:
        return len(self.samples)


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Check cache
        if self.cache and idx in self._cache_dict:
            return self._cache_dict[idx]


        path, slice_idx = self.samples[idx]


        with h5py.File(path, "r") as hf:
            kspace_slice = hf["kspace"][slice_idx]  # (H, W) complex
            max_val = float(hf.attrs.get("max", 1.0))


        kspace_t = to_tensor(kspace_slice)  # (H, W, 2)


        # Full reconstruction (target)
        full_image = complex_abs(ifft2c(kspace_t))  # (H, W)
        target, _ = normalize(full_image, max_val)
        target = center_crop(target.unsqueeze(0), self.crop_size)  # (1, 320, 320)


        # Undersampled reconstruction (input)
        if self.mask_type == "equispaced":
            mask = equispaced_mask(
                kspace_slice.shape,
                center_fraction=self.center_fraction,
                acceleration=self.acceleration,
            )
        else:
            mask = random_mask(
                kspace_slice.shape,
                center_fraction=self.center_fraction,
                acceleration=self.acceleration,
                seed=self.seed + idx,
            )


        kspace_us = kspace_t * mask.unsqueeze(-1)  # (H, W, 2)
        input_image = complex_abs(ifft2c(kspace_us))  # (H, W)
        input_image, _ = normalize(input_image, max_val)
        input_image = center_crop(input_image.unsqueeze(0), self.crop_size)  # (1, 320, 320)


        result = (input_image.float(), target.float())


        # Store in cache
        if self.cache:
            self._cache_dict[idx] = result


        return result


    def get_metadata(self, idx: int) -> dict:
        """Get metadata for a sample (useful for evaluation/reporting)."""
        path, slice_idx = self.samples[idx]
        return {
            "file": os.path.basename(path),
            "slice_idx": slice_idx,
            "acceleration": self.acceleration,
            "center_fraction": self.center_fraction,
        }