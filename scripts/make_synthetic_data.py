#!/usr/bin/env python
"""
make_synthetic_data.py
----------------------
Generate synthetic HDF5 volumes in the fastMRI single-coil layout.

Why this exists
---------------
fastMRI requires registration and a ~90 GB download, which means that without
it nothing in this repository can be exercised end to end — not the dataset, not
the undersampling simulation, not the trainer. That gap is precisely how a
data-consistency path that never received any data shipped unnoticed.

These phantoms are not anatomically meaningful, but they have the property that
matters for testing a reconstruction pipeline: a rapidly decaying k-space
spectrum. White noise has a flat spectrum, so undersampling it produces no
coherent aliasing and every reconstruction metric becomes uninformative. Smooth
elliptical structures reproduce the low-frequency dominance of real MR images,
so zero-filled reconstructions show genuine streaking artefacts and PSNR/SSIM
behave sensibly.

Usage
-----
    python scripts/make_synthetic_data.py --out data/synthetic --volumes 8
    python main.py train --config configs/smoke.yaml data.train_dir=data/synthetic
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def make_phantom(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    """
    A Shepp-Logan-flavoured phantom: overlapping ellipses plus a soft background.

    Ellipses (rather than circles) give anisotropic edges, so that direction
    dependent artefacts from Cartesian undersampling are actually visible.
    """
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    cy, cx = height / 2, width / 2

    # Soft elliptical "body" so the image is not surrounded by hard zeros.
    body = (((yy - cy) / (height * 0.42)) ** 2 + ((xx - cx) / (width * 0.38)) ** 2) < 1.0
    img = body.astype(np.float64) * 0.25

    for _ in range(rng.integers(5, 9)):
        ey = rng.uniform(height * 0.25, height * 0.75)
        ex = rng.uniform(width * 0.25, width * 0.75)
        ry = rng.uniform(height * 0.05, height * 0.20)
        rx = rng.uniform(width * 0.05, width * 0.20)
        theta = rng.uniform(0, np.pi)

        ct, st = np.cos(theta), np.sin(theta)
        y_rot = (yy - ey) * ct + (xx - ex) * st
        x_rot = -(yy - ey) * st + (xx - ex) * ct
        inside = ((y_rot / ry) ** 2 + (x_rot / rx) ** 2) < 1.0
        img[inside] += rng.uniform(0.2, 0.8)

    # A gentle blur removes the perfectly sharp ellipse boundaries, which would
    # otherwise put unrealistic energy at the very highest frequencies.
    kernel = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
    kernel /= kernel.sum()
    img = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), 0, img)
    img = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), 1, img)

    return img / (img.max() or 1.0)


def to_kspace(image: np.ndarray, rng: np.random.Generator, noise: float = 0.0) -> np.ndarray:
    """
    Centred FFT of an image with a smooth phase ramp.

    Real MRI is complex-valued: B0 inhomogeneity and receive-coil phase mean the
    image has non-trivial phase. Generating a purely real phantom would make the
    task artificially easy for a complex-valued model and would leave the phase
    handling in the data-consistency layer untested.
    """
    height, width = image.shape
    yy, xx = np.mgrid[0:height, 0:width]
    ramp = (
        rng.uniform(-1, 1) * yy / height
        + rng.uniform(-1, 1) * xx / width
        + rng.uniform(-0.3, 0.3) * (yy / height) * (xx / width)
    )
    complex_image = image * np.exp(1j * 2 * np.pi * ramp)

    kspace = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(complex_image), norm="ortho"))
    if noise > 0:
        scale = noise * np.abs(kspace).mean()
        kspace = kspace + scale * (
            rng.normal(size=kspace.shape) + 1j * rng.normal(size=kspace.shape)
        )
    return kspace.astype(np.complex64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--volumes", type=int, default=8, help="Number of .h5 files")
    parser.add_argument("--slices", type=int, default=4, help="Slices per volume")
    parser.add_argument("--height", type=int, default=128, help="Readout size")
    parser.add_argument("--width", type=int, default=112, help="Phase-encode size")
    parser.add_argument("--noise", type=float, default=0.01, help="Relative k-space noise")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    try:
        import h5py
    except ImportError:
        print("h5py is required: pip install h5py")
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    for volume_idx in range(args.volumes):
        base = make_phantom(args.height, args.width, rng)

        volume, magnitudes = [], []
        for slice_idx in range(args.slices):
            # Slices vary smoothly through the volume, as in a real acquisition.
            weight = 0.75 + 0.25 * np.cos(np.pi * slice_idx / max(1, args.slices - 1))
            image = base * weight
            volume.append(to_kspace(image, rng, args.noise))
            magnitudes.append(image.max())

        path = out_dir / f"synthetic_{volume_idx:03d}.h5"
        with h5py.File(path, "w") as hf:
            hf.create_dataset("kspace", data=np.stack(volume), compression="gzip")
            hf.attrs["max"] = float(np.max(magnitudes))
            hf.attrs["acquisition"] = "SYNTHETIC"
            hf.attrs["patient_id"] = f"synthetic_{volume_idx:03d}"

    total = args.volumes * args.slices
    print(
        f"Wrote {args.volumes} volumes ({total} slices, "
        f"{args.height}x{args.width}) to {out_dir}"
    )
    print("\nTrain on it with:")
    print(f"  python main.py train --config configs/smoke.yaml data.train_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
