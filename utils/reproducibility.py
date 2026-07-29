"""
reproducibility.py
------------------
Utilities for ensuring reproducible training runs.

Reproducibility in PyTorch requires more than ``torch.manual_seed``. This module
covers the three sources of run-to-run variation that actually matter in practice:

1. **Global RNG state** — ``random``, ``numpy``, ``torch`` (CPU + all CUDA devices).
2. **DataLoader worker state** — each worker process forks with its own RNG; without
   an explicit ``worker_init_fn`` every worker inherits the *same* numpy seed, so
   any numpy randomness in ``__getitem__`` (e.g. mask sampling) is duplicated
   across workers. This is a classic silent bug and it directly affects this
   project, whose dataset draws undersampling masks with numpy.
3. **Nondeterministic kernels** — cuDNN autotuning and atomics-based backward
   passes. Full determinism costs throughput, so it is opt-in.

Usage
-----
    from utils.reproducibility import seed_everything, seed_worker, make_generator

    seed_everything(42)
    loader = DataLoader(ds, worker_init_fn=seed_worker, generator=make_generator(42))
"""

from __future__ import annotations

import logging
import os
import random
from typing import Any

import numpy as np
import torch

log = logging.getLogger(__name__)


def seed_everything(seed: int = 42, deterministic: bool = False) -> int:
    """
    Set all random seeds for reproducibility.

    Parameters
    ----------
    seed
        Random seed applied to ``random``, ``numpy`` and ``torch``.
    deterministic
        If True, force deterministic algorithms and disable cuDNN autotuning.
        This makes runs bit-reproducible on identical hardware at the cost of
        throughput (typically 5-20% slower). If False, cuDNN benchmarking is
        enabled, which is faster whenever input shapes are constant — the usual
        case here, since every sample is cropped to the same size.

    Returns
    -------
    The seed that was applied (convenient for logging into a run manifest).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Required for deterministic CUBLAS GEMMs on CUDA >= 10.2.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as exc:  # pragma: no cover - depends on torch build
            log.warning("Could not enable deterministic algorithms: %s", exc)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    return seed


def seed_worker(worker_id: int) -> None:
    """
    ``worker_init_fn`` for DataLoader that gives every worker a distinct,
    run-reproducible RNG stream.

    PyTorch already offsets ``torch``'s seed per worker, but it does *not* touch
    ``numpy`` or ``random``. Without this, N workers each produce an identical
    stream of numpy random numbers, which silently correlates the undersampling
    masks drawn in ``FastMRIKneeDataset.__getitem__`` across the batch.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int = 42) -> torch.Generator:
    """Build a seeded ``torch.Generator`` for DataLoader shuffling."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def collect_environment() -> dict[str, Any]:
    """
    Capture the environment fingerprint of a run.

    Written into every run directory as ``manifest.json`` so that a reported
    number can always be traced back to the exact code and hardware that
    produced it — the minimum bar for a reproducible experimental claim.
    """
    import platform
    import subprocess

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=repo_root,
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:
            return None

    env: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
    }
    if torch.cuda.is_available():  # pragma: no cover - no GPU in CI
        env["cuda_version"] = torch.version.cuda
        env["gpu"] = torch.cuda.get_device_name(0)
        env["gpu_count"] = torch.cuda.device_count()
    return env
