"""
masks.py
--------
Cartesian undersampling masks for retrospective MRI acceleration simulation.

A Cartesian MRI acquisition samples one *phase-encode line* at a time, so
acceleration means skipping whole columns of k-space, never individual points.
Every mask here is therefore 1D over the phase-encode axis and broadcast across
readout.

Two families are provided, matching the fastMRI benchmark:

* **Random**  — low-frequency centre fully sampled, remaining lines drawn i.i.d.
  This is the harder and more commonly reported setting.
* **Equispaced** — centre fully sampled, remaining lines on a regular grid. This
  is what real accelerated sequences do (regular undersampling gives coherent,
  predictable aliasing that parallel imaging can unfold).

Acceleration accounting
-----------------------
The nominal acceleration ``R`` is the ratio of all lines to acquired lines,
*including* the fully sampled centre. Getting this wrong is the single most
common way to accidentally report an easier problem than claimed.

The previous equispaced implementation set ``mask[::R] = 1`` and then added the
centre on top, which acquires ``N/R + N*cf`` lines instead of ``N/R``. At
``R = 4, cf = 0.08`` that is an effective acceleration of **3.2x, not 4x** — a
20% denser acquisition than reported, and enough to inflate PSNR by several
tenths of a dB relative to a correctly simulated baseline. The implementation
below solves for the outer-region stride that makes the *total* line count come
out at ``N / R``, which is what fastMRI's own ``EquispacedMaskFunc`` does.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


def _center_lines(num_cols: int, center_fraction: float) -> tuple[int, int]:
    """Return ``(num_low_freqs, start_index)`` for the fully sampled centre."""
    num_low_freqs = int(round(num_cols * center_fraction))
    start = (num_cols - num_low_freqs + 1) // 2
    return num_low_freqs, start


def random_mask(
    shape: Sequence[int],
    center_fraction: float = 0.08,
    acceleration: int = 4,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    """
    Random Cartesian undersampling mask.

    The centre ``center_fraction`` of lines is always acquired; the rest are
    drawn independently with probability chosen so the *total* number of
    acquired lines is ``num_cols / acceleration``.

    Parameters
    ----------
    shape
        k-space shape; only the last entry (number of columns) is used.
    center_fraction
        Fraction of low-frequency lines always sampled.
    acceleration
        Nominal acceleration factor.
    seed
        Seed for a fresh generator. Mutually exclusive with ``rng``.
    rng
        An existing generator to draw from, for callers that manage their own
        RNG stream (e.g. per-epoch mask resampling).

    Returns
    -------
    ``(1, num_cols)`` float32 tensor of 0/1.
    """
    num_cols = int(shape[-1])
    num_low_freqs, start = _center_lines(num_cols, center_fraction)

    # Solve for p such that: num_low_freqs + p * (N - num_low_freqs) = N / R
    target = num_cols / acceleration
    remaining = num_cols - num_low_freqs
    prob = (target - num_low_freqs) / remaining if remaining > 0 else 0.0
    prob = float(np.clip(prob, 0.0, 1.0))

    generator = rng if rng is not None else np.random.default_rng(seed)
    mask = generator.uniform(size=num_cols) < prob
    mask[start : start + num_low_freqs] = True

    return torch.from_numpy(mask.reshape(1, num_cols).astype(np.float32))


def equispaced_mask(
    shape: Sequence[int],
    center_fraction: float = 0.08,
    acceleration: int = 4,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    randomize_offset: bool = True,
) -> torch.Tensor:
    """
    Equispaced Cartesian undersampling mask with correct acceleration accounting.

    The outer stride is chosen so that centre lines plus strided lines total
    ``num_cols / acceleration``.

    Parameters
    ----------
    randomize_offset
        Jitter the starting phase of the grid. Without this, every scan in the
        dataset is sampled at exactly the same k-space locations and the network
        can overfit to one fixed aliasing pattern rather than learning to invert
        undersampling in general.
    """
    num_cols = int(shape[-1])
    num_low_freqs, start = _center_lines(num_cols, center_fraction)

    target = num_cols / acceleration
    outer_budget = target - num_low_freqs
    mask = np.zeros(num_cols, dtype=np.float32)

    if outer_budget > 0:
        # Effective stride over the whole axis that yields `outer_budget` extra
        # lines once the centre (already counted) is excluded.
        adjusted = (num_cols - num_low_freqs) / outer_budget
        generator = rng if rng is not None else np.random.default_rng(seed)
        offset = generator.integers(0, max(1, int(round(adjusted)))) if randomize_offset else 0
        positions = np.arange(offset, num_cols - 1, adjusted)
        mask[np.round(positions).astype(int).clip(0, num_cols - 1)] = 1.0

    mask[start : start + num_low_freqs] = 1.0
    return torch.from_numpy(mask.reshape(1, num_cols))


def magic_mask(
    shape: Sequence[int],
    center_fraction: float = 0.08,
    acceleration: int = 4,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    """
    Golden-ratio ("magic") equispaced sampling.

    Offsets successive lines by the golden angle rather than a fixed stride,
    which spreads energy in the point-spread function more evenly than a regular
    grid and produces incoherent aliasing closer to the compressed-sensing
    ideal, while remaining a realisable Cartesian trajectory.
    """
    num_cols = int(shape[-1])
    num_low_freqs, start = _center_lines(num_cols, center_fraction)

    target = num_cols / acceleration
    outer_budget = int(max(0, round(target - num_low_freqs)))
    mask = np.zeros(num_cols, dtype=np.float32)

    if outer_budget > 0:
        generator = rng if rng is not None else np.random.default_rng(seed)
        phi = (1 + 5**0.5) / 2  # golden ratio
        offset = float(generator.uniform())
        idx = np.floor(((np.arange(outer_budget) * phi + offset) % 1.0) * num_cols)
        mask[idx.astype(int).clip(0, num_cols - 1)] = 1.0

    mask[start : start + num_low_freqs] = 1.0
    return torch.from_numpy(mask.reshape(1, num_cols))


_MASK_FUNCS = {
    "random": random_mask,
    "equispaced": equispaced_mask,
    "magic": magic_mask,
}


def build_mask(
    mask_type: str,
    shape: Sequence[int],
    center_fraction: float = 0.08,
    acceleration: int = 4,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    """Dispatch to a mask function by name."""
    fn = _MASK_FUNCS.get((mask_type or "random").lower())
    if fn is None:
        raise ValueError(
            f"Unknown mask_type {mask_type!r}. Choose from: {', '.join(_MASK_FUNCS)}"
        )
    return fn(shape, center_fraction, acceleration, seed=seed, rng=rng)


def effective_acceleration(mask: torch.Tensor) -> float:
    """
    Measured acceleration of a mask: total lines / acquired lines.

    Worth asserting in tests and logging at the start of training. A mask whose
    effective acceleration does not match the configured one means the reported
    problem difficulty is wrong.
    """
    total = mask.numel()
    acquired = float(mask.sum())
    return float("inf") if acquired == 0 else total / acquired
