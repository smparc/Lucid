"""
conftest.py
-----------
Shared pytest fixtures.

The most important thing here is :func:`synthetic_data_dir`, which writes real
HDF5 volumes in the fastMRI layout. Without it the dataset, the undersampling
simulation and the trainer can only be tested through mocks, which is exactly
how the original data-consistency path shipped completely non-functional: every
model test passed because none of them ever loaded a file.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

# Tests import top-level packages (models, data, training), so the repository
# root must be importable regardless of where pytest is invoked from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Tests run on CPU so results are hardware independent."""
    return torch.device("cpu")


@pytest.fixture
def small_image() -> torch.Tensor:
    """A small batch of single-channel images."""
    torch.manual_seed(0)
    return torch.rand(2, 1, 64, 64)


@pytest.fixture
def complex_image() -> torch.Tensor:
    """A small batch of 2-channel (real, imaginary) images."""
    torch.manual_seed(0)
    return torch.randn(2, 2, 64, 64)


def _make_phantom(h: int, w: int, seed: int) -> np.ndarray:
    """
    A smooth blob phantom.

    Smoothness matters: white noise has a flat k-space spectrum, so
    undersampling it produces no coherent aliasing and every reconstruction
    metric becomes meaningless. Gaussian blobs give the rapid spectral decay of
    real anatomy.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w), dtype=np.float64)
    for _ in range(6):
        cy = rng.integers(h // 5, h - h // 5)
        cx = rng.integers(w // 5, w - w // 5)
        radius = rng.integers(max(3, h // 16), max(5, h // 5))
        img += np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * radius**2))) * rng.uniform(
            0.3, 1.0
        )
    return img


@pytest.fixture(scope="session")
def synthetic_data_dir(tmp_path_factory) -> str:
    """
    A directory of synthetic ``.h5`` volumes in the fastMRI single-coil layout.

    Each file holds a ``kspace`` dataset of shape ``(slices, H, W)`` (complex64)
    and a ``max`` attribute, matching what ``FastMRIKneeDataset`` expects.
    """
    h5py = pytest.importorskip("h5py")

    root = tmp_path_factory.mktemp("fastmri")
    height, width, n_slices = 96, 80, 3

    for file_idx in range(3):
        volume = []
        for slice_idx in range(n_slices):
            img = _make_phantom(height, width, seed=file_idx * 10 + slice_idx)
            kspace = np.fft.fftshift(
                np.fft.fft2(np.fft.ifftshift(img), norm="ortho")
            ).astype(np.complex64)
            volume.append(kspace)

        with h5py.File(root / f"file_{file_idx}.h5", "w") as hf:
            hf.create_dataset("kspace", data=np.stack(volume))
            hf.attrs["max"] = float(
                np.abs(_make_phantom(height, width, seed=file_idx * 10)).max()
            )

    return str(root)


@pytest.fixture
def base_config():
    """A validated config with a tiny model, suitable for fast tests."""
    from config import load_config

    return load_config(
        overrides={
            "model": {
                "name": "swinunet",
                "complex": False,
                "params": {
                    "img_size": 64,
                    "patch_size": 2,
                    "embed_dim": 16,
                    "ws": 4,
                    "head_dim": 8,
                    "n_levels": 2,
                    "depths": [1, 1, 1],
                    "drop_path_rate": 0.0,
                },
            },
            "data": {"crop_size": [64, 64], "num_workers": 0, "persistent_workers": False},
            "training": {"epochs": 1, "batch_size": 2, "amp": False},
            "logging": {"tensorboard": False, "wandb": False},
        }
    )
