"""
fourier.py
----------
Centred Fourier transforms — the single source of truth for k-space conventions
across the whole project.

Why this module exists
----------------------
Data loading, the data-consistency layer, and the frequency-domain loss each
need to move between image space and k-space. When each defines its own
transform they drift apart, and a mismatched ``fftshift`` is invisible in the
loss curve but silently destroys data consistency: the measured samples get
written back to the wrong frequencies. Every FFT in this project routes through
here.

Convention
----------
The fastMRI convention, which places the DC (zero-frequency) component at the
centre of the array in both domains::

    fft2c  : ifftshift -> fft2 (norm="ortho") -> fftshift
    ifft2c : ifftshift -> ifft2(norm="ortho") -> fftshift

``norm="ortho"`` makes the pair unitary, so Parseval's theorem holds and an
L2 penalty in k-space is exactly an L2 penalty in image space. Together the two
functions are exact inverses to floating-point precision, which is verified in
the test suite.

Representations
---------------
Two layouts for complex data are used, and both are supported:

* **last-dim** ``(..., H, W, 2)``  — used by the data pipeline, matching fastMRI.
* **channel-dim** ``(B, 2, H, W)`` — used by the networks, so that real and
  imaginary parts are ordinary convolution channels.

``chan_to_last`` / ``last_to_chan`` convert between them.
"""

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# Layout conversion
# ---------------------------------------------------------------------------


def chan_to_last(x: torch.Tensor) -> torch.Tensor:
    """``(B, 2, H, W)`` -> ``(B, H, W, 2)``."""
    if x.shape[1] != 2:
        raise ValueError(f"expected 2 channels (real, imag), got shape {tuple(x.shape)}")
    return x.permute(0, 2, 3, 1).contiguous()


def last_to_chan(x: torch.Tensor) -> torch.Tensor:
    """``(B, H, W, 2)`` -> ``(B, 2, H, W)``."""
    if x.shape[-1] != 2:
        raise ValueError(f"expected trailing dim 2 (real, imag), got shape {tuple(x.shape)}")
    return x.permute(0, 3, 1, 2).contiguous()


def to_complex(x: torch.Tensor) -> torch.Tensor:
    """View a real ``(..., 2)`` tensor as a native complex tensor."""
    # view_as_complex requires a contiguous, stride-1 final dimension.
    return torch.view_as_complex(x.contiguous().float())


def from_complex(x: torch.Tensor) -> torch.Tensor:
    """View a complex tensor as a real ``(..., 2)`` tensor."""
    return torch.view_as_real(x)


# ---------------------------------------------------------------------------
# Centred transforms, last-dim layout
# ---------------------------------------------------------------------------


def fft2c(x: torch.Tensor) -> torch.Tensor:
    """
    Centred 2D FFT. ``(..., H, W, 2)`` image -> ``(..., H, W, 2)`` k-space.

    Autocast is explicitly disabled: ``torch.fft`` has no half-precision CUDA
    kernel, and even where it runs, fp16 lacks the dynamic range for k-space,
    whose centre exceeds its periphery by several orders of magnitude.
    """
    with torch.autocast(device_type=x.device.type, enabled=False):
        z = to_complex(x)
        z = torch.fft.ifftshift(z, dim=(-2, -1))
        z = torch.fft.fft2(z, norm="ortho")
        z = torch.fft.fftshift(z, dim=(-2, -1))
        return from_complex(z)


def ifft2c(x: torch.Tensor) -> torch.Tensor:
    """Centred 2D inverse FFT. ``(..., H, W, 2)`` k-space -> image."""
    with torch.autocast(device_type=x.device.type, enabled=False):
        z = to_complex(x)
        z = torch.fft.ifftshift(z, dim=(-2, -1))
        z = torch.fft.ifft2(z, norm="ortho")
        z = torch.fft.fftshift(z, dim=(-2, -1))
        return from_complex(z)


# ---------------------------------------------------------------------------
# Centred transforms, channel layout (for use inside networks)
# ---------------------------------------------------------------------------


def fft2c_chan(x: torch.Tensor) -> torch.Tensor:
    """Centred 2D FFT on ``(B, 2, H, W)``."""
    return last_to_chan(fft2c(chan_to_last(x)))


def ifft2c_chan(x: torch.Tensor) -> torch.Tensor:
    """Centred 2D inverse FFT on ``(B, 2, H, W)``."""
    return last_to_chan(ifft2c(chan_to_last(x)))


# ---------------------------------------------------------------------------
# Magnitude / phase
# ---------------------------------------------------------------------------


def complex_abs(x: torch.Tensor) -> torch.Tensor:
    """
    Magnitude of a complex tensor stored as ``(..., 2)``.

    The ``+ eps`` under the square root is not cosmetic: ``d/dx sqrt(x)`` is
    unbounded at zero, and MRI backgrounds contain a great many near-zero
    pixels. Without it, a magnitude loss produces inf gradients on air.
    """
    return torch.sqrt((x**2).sum(dim=-1) + 1e-12)


def complex_abs_chan(x: torch.Tensor) -> torch.Tensor:
    """Magnitude of ``(B, 2, H, W)`` -> ``(B, 1, H, W)``."""
    return torch.sqrt((x**2).sum(dim=1, keepdim=True) + 1e-12)


def complex_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Complex multiply for ``(..., 2)`` tensors."""
    real = a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1]
    imag = a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0]
    return torch.stack([real, imag], dim=-1)


def phase(x: torch.Tensor) -> torch.Tensor:
    """Phase angle of a complex tensor stored as ``(..., 2)``."""
    return torch.atan2(x[..., 1], x[..., 0])
