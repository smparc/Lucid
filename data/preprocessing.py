"""
preprocessing.py
----------------
Data loading and preprocessing for the fastMRI single-coil knee dataset.

Pipeline
--------
1. Load complex k-space from .h5 files
2. Extract middle slice from each volume
3. Convert to PyTorch tensor
4. Apply retrospective undersampling mask (center_fraction=0.08, acceleration=4)
5. Compute zero-filled IFFT reconstruction
6. Normalize by per-scan maximum
7. Center-crop to 320x320
"""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional


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
    seed             : optional random seed

    Returns
    -------
    torch.Tensor (1, num_cols) — binary mask (1=sampled, 0=zero-filled)
    """
    num_cols = shape[-1]
    num_low_freqs = int(round(num_cols * center_fraction))

    # Compute how many additional lines to sample
    prob = (num_cols / acceleration - num_low_freqs) / (num_cols - num_low_freqs)

    rng = np.random.default_rng(seed)
    mask = rng.uniform(size=num_cols) < prob

    # Always include centre frequencies
    pad = (num_cols - num_low_freqs + 1) // 2
    mask[pad: pad + num_low_freqs] = True

    return torch.from_numpy(mask.reshape(1, num_cols).astype(np.float32))


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
    """
    Convert complex numpy k-space array to a real tensor of shape (..., 2).
    """
    if np.iscomplexobj(kspace_np):
        kspace_np = np.stack([kspace_np.real, kspace_np.imag], axis=-1)
    return torch.from_numpy(kspace_np.astype(np.float32))


def center_crop(img: torch.Tensor, crop_size: Tuple[int, int] = (320, 320)) -> torch.Tensor:
    """
    Center-crop a 2D image tensor.

    img : (..., H, W)
    """
    h, w = img.shape[-2], img.shape[-1]
    ch, cw = crop_size
    top  = (h - ch) // 2
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
    ):
        """
        Parameters
        ----------
        root_dir         : directory containing .h5 files
        center_fraction  : fraction of k-space centre always sampled
        acceleration     : subsampling factor
        crop_size        : final image crop size
        seed             : random seed for mask generation
        """
        self.files = sorted([
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir) if f.endswith(".h5")
        ])
        if not self.files:
            raise FileNotFoundError(f"No .h5 files found in {root_dir}")

        self.center_fraction = center_fraction
        self.acceleration = acceleration
        self.crop_size = crop_size
        self.seed = seed

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.files[idx]

        with h5py.File(path, "r") as hf:
            kspace_np = hf["kspace"][()]          # (slices, H, W) complex
            max_val   = float(hf.attrs.get("max", 1.0))

        # Middle slice
        mid = kspace_np.shape[0] // 2
        kspace_slice = kspace_np[mid]              # (H, W) complex

        kspace_t = to_tensor(kspace_slice)         # (H, W, 2)

        # Full reconstruction (target)
        full_image = complex_abs(ifft2c(kspace_t)) # (H, W)
        target, _ = normalize(full_image, max_val)
        target = center_crop(target.unsqueeze(0), self.crop_size)  # (1, 320, 320)

        # Undersampled reconstruction (input)
        mask = random_mask(
            kspace_slice.shape,
            center_fraction=self.center_fraction,
            acceleration=self.acceleration,
            seed=self.seed + idx,
        )                                                  # (1, W)
        kspace_us = kspace_t * mask.unsqueeze(-1)          # (H, W, 2)
        input_image = complex_abs(ifft2c(kspace_us))       # (H, W)
        input_image, _ = normalize(input_image, max_val)
        input_image = center_crop(input_image.unsqueeze(0), self.crop_size)  # (1, 320, 320)

        return input_image.float(), target.float()
