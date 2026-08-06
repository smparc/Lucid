"""
data_consistency.py
-------------------
Data Consistency (DC) for physics-informed MRI reconstruction, plus the unrolled
cascade architecture built from it.

Theory
------
Accelerated MRI is not a generic image-restoration problem: we know *exactly*
which Fourier coefficients were measured. A purely learned network is free to
contradict those measurements, which is precisely how reconstruction networks
hallucinate anatomy. The DC operator removes that freedom::

    k_pred     = F(x_pred)
    k_dc[m=1]  = lambda * k_measured + (1 - lambda) * k_pred   # trust the scanner
    k_dc[m=0]  = k_pred                                        # network fills gaps
    x_dc       = F^-1(k_dc)

With ``lambda = 1`` the measured lines are restored exactly and the network can
only ever modify unobserved frequencies. Lower values allow the network to
denoise the measurements too, which helps at low SNR. ``lambda`` is learnable.

Cascading these with a denoiser gives an *unrolled* network — the standard
formulation for state-of-the-art MRI reconstruction (Schlemper et al. 2018;
Hammernik et al. 2018). Each stage is one iteration of a proximal-gradient
solver where the proximal operator has been replaced by a learned network.

What was wrong before
---------------------
The previous implementation could not work, for three independent reasons:

1. ``_kspace_to_image`` ended in ``x.abs()``, discarding phase. A DC layer that
   destroys phase cannot be cascaded: the next FFT sees a zero-phase image, so
   the measured coefficients it writes back are wrong.
2. ``_image_to_kspace`` took a *real* image, implicitly asserting zero phase,
   which is false for any MRI reconstruction.
3. Most decisively, nothing ever called it with data. The trainer invoked
   ``model(x)`` while ``ResidualDCWrapper.forward`` needed ``(x, k, mask)``;
   with ``k_measured=None`` the wrapper's guard silently skipped DC entirely.
   The config flag ``data_consistency.enabled`` therefore did nothing at all.

All three are fixed here: everything is complex end to end, the transforms come
from :mod:`models.fourier`, and the dataset now supplies k-space and the mask so
the trainer can actually route them through.

References
----------
Schlemper et al., "A Deep Cascade of CNNs for Dynamic MR Image Reconstruction",
IEEE TMI 2018.
Hammernik et al., "Learning a Variational Network for Reconstruction of
Accelerated MRI Data", MRM 2018.
Sriram et al., "End-to-End Variational Networks for Accelerated MRI
Reconstruction", MICCAI 2020.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from models.fourier import complex_abs_chan, fft2c_chan, ifft2c_chan


def _broadcast_mask(mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """
    Reshape a sampling mask to broadcast against ``(B, 2, H, W)`` k-space.

    Accepts the shapes the pipeline actually produces:
    ``(W,)``, ``(1, W)``, ``(B, W)``, ``(B, 1, W)``, ``(B, 1, 1, W)``,
    ``(B, 1, H, W)``. Cartesian undersampling masks whole phase-encode columns,
    so a 1D mask over the last axis is the normal case and is broadcast over
    rows.
    """
    B, _, H, W = like.shape

    if mask.dim() == 1:  # (W,)
        mask = mask.view(1, 1, 1, -1)
    elif mask.dim() == 2:  # (1, W) or (B, W)
        mask = mask.view(mask.shape[0], 1, 1, mask.shape[-1])
    elif mask.dim() == 3:  # (B, 1, W)
        mask = mask.unsqueeze(2)
    elif mask.dim() != 4:
        raise ValueError(f"Unsupported mask shape {tuple(mask.shape)}")

    if mask.shape[-1] != W:
        raise ValueError(
            f"Mask width {mask.shape[-1]} does not match k-space width {W}. "
            f"The mask must be generated at the same resolution the model "
            f"operates on."
        )
    if mask.shape[-2] not in (1, H):
        raise ValueError(f"Mask height {mask.shape[-2]} incompatible with {H}")

    return mask.to(dtype=like.dtype, device=like.device)


class DataConsistencyLayer(nn.Module):
    """
    Enforce fidelity to the measured k-space samples.

    Parameters
    ----------
    learnable_lambda
        If True, the mixing weight is trained.
    lambda_init
        Initial weight in ``[0, 1]``. ``1.0`` is hard data consistency.
    hard
        Skip the mixing entirely and always overwrite measured positions. Cheaper
        and strictly correct when the measurements are noise-free, which is the
        case for retrospectively undersampled data.
    """

    def __init__(
        self,
        learnable_lambda: bool = True,
        lambda_init: float = 1.0,
        hard: bool = False,
    ):
        super().__init__()
        self.hard = hard

        # Stored as a logit so that sigmoid() keeps lambda in [0, 1] without a
        # clamp. The original code applied sigmoid to lambda_init directly,
        # which turned a requested "hard DC" of 1.0 into an actual weight of
        # sigmoid(1.0) = 0.73 — a silent 27% leak of network output into
        # measured frequencies.
        eps = 1e-4
        p = min(max(lambda_init, eps), 1 - eps)
        logit = math.log(p / (1 - p))

        if learnable_lambda:
            self.lam_logit = nn.Parameter(torch.tensor(logit, dtype=torch.float32))
        else:
            self.register_buffer("lam_logit", torch.tensor(logit, dtype=torch.float32))

    @property
    def lam(self) -> torch.Tensor:
        """Current mixing weight in ``[0, 1]``."""
        return torch.sigmoid(self.lam_logit)

    def forward(
        self,
        x_pred: torch.Tensor,
        k_measured: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x_pred
            ``(B, 2, H, W)`` complex image-space prediction.
        k_measured
            ``(B, 2, H, W)`` measured (already masked) k-space.
        mask
            Binary sampling mask, broadcastable to the k-space shape.

        Returns
        -------
        ``(B, 2, H, W)`` data-consistent complex image.
        """
        if x_pred.shape[1] != 2:
            raise ValueError(
                f"DataConsistencyLayer operates on complex (B, 2, H, W) tensors; "
                f"got {tuple(x_pred.shape)}. Magnitude-only reconstruction cannot "
                f"be made data-consistent, because writing measured coefficients "
                f"back requires phase."
            )

        k_pred = fft2c_chan(x_pred)
        m = _broadcast_mask(mask, k_pred)

        if self.hard:
            k_dc = (1 - m) * k_pred + m * k_measured
        else:
            lam = self.lam.to(k_pred.dtype)
            k_dc = (1 - m) * k_pred + m * (lam * k_measured + (1 - lam) * k_pred)

        return ifft2c_chan(k_dc)


class CascadedNet(nn.Module):
    """
    Unrolled reconstruction: ``(denoiser -> DC)`` repeated ``n_cascades`` times.

    This is the architecture that actually closes the gap to state of the art.
    A single feed-forward pass has to solve the inverse problem in one shot; an
    unrolled network instead takes several small, physics-corrected steps, and
    each DC layer re-anchors the estimate to the measurements so errors cannot
    compound across stages.

    Parameters
    ----------
    denoiser_fn
        Zero-argument callable returning a fresh denoiser network. It must map
        ``(B, 2, H, W) -> (B, 2, H, W)``.
    n_cascades
        Number of unrolled iterations.
    share_weights
        Reuse one denoiser across all stages (recurrent-style). Cuts parameters
        by ``n_cascades`` at some cost in accuracy.
    hard_dc
        Use hard data consistency.
    output
        ``"complex"`` returns ``(B, 2, H, W)``; ``"magnitude"`` returns
        ``(B, 1, H, W)``, which is what the image-domain losses and metrics
        consume.
    require_kspace
        Raise if ``forward`` is called without measurements, instead of quietly
        degrading to a plain cascade of denoisers. Default True.

        This is the same defect the v1 audit found in ``ResidualDCWrapper``,
        which skipped DC whenever ``k_measured`` was None — so the trainer,
        which passed only the image, turned ``data_consistency.enabled: true``
        into a complete no-op while every test still passed and every loss curve
        still fell. Running a cascade without DC is a legitimate *ablation*, so
        the capability stays; it just has to be asked for, because a
        physics-informed model that silently drops its physics produces
        plausible images and a 2.4 dB deficit nobody attributes to a bug.
    """

    def __init__(
        self,
        denoiser_fn,
        n_cascades: int = 3,
        share_weights: bool = False,
        hard_dc: bool = False,
        output: str = "magnitude",
        require_kspace: bool = True,
    ):
        super().__init__()
        if n_cascades < 1:
            raise ValueError(f"n_cascades must be >= 1, got {n_cascades}")
        if output not in ("complex", "magnitude"):
            raise ValueError(f"output must be 'complex' or 'magnitude', got {output!r}")

        self.n_cascades = n_cascades
        self.share_weights = share_weights
        self.output = output
        self.require_kspace = require_kspace

        if share_weights:
            shared = denoiser_fn()
            # A plain `[shared] * n` registers the same module n times, so
            # parameter counting double-counts it. Holding one module and
            # indexing it keeps the count honest.
            self.networks = nn.ModuleList([shared])
        else:
            self.networks = nn.ModuleList([denoiser_fn() for _ in range(n_cascades)])

        self.dc_layers = nn.ModuleList(
            [
                DataConsistencyLayer(learnable_lambda=not hard_dc, hard=hard_dc)
                for _ in range(n_cascades)
            ]
        )

    def _net(self, i: int) -> nn.Module:
        return self.networks[0] if self.share_weights else self.networks[i]

    def forward(
        self,
        x_input: torch.Tensor,
        k_measured: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x_input
            ``(B, 2, H, W)`` zero-filled complex reconstruction.
        k_measured, mask
            Measured k-space and its sampling mask. If either is None, this
            raises unless the cascade was built with ``require_kspace=False``,
            in which case the DC steps are skipped and the model degrades to a
            plain cascade of denoisers.
        """
        x = x_input
        use_dc = k_measured is not None and mask is not None

        if not use_dc and self.require_kspace:
            missing = "k_measured" if k_measured is None else "mask"
            raise ValueError(
                f"CascadedNet requires measurements to enforce data consistency, but "
                f"{missing} was None. Set data.return_kspace=true in the config, or "
                f"build the cascade with require_kspace=False if you intend the "
                f"no-physics ablation. (Silently skipping DC here is what made the "
                f"original ResidualDCWrapper a no-op: it trains, the loss falls, the "
                f"images look plausible, and the model is 2.4 dB worse for no visible "
                f"reason.)"
            )

        for i in range(self.n_cascades):
            x = self._net(i)(x)
            if use_dc:
                x = self.dc_layers[i](x, k_measured, mask)

        return complex_abs_chan(x) if self.output == "magnitude" else x

    @property
    def lambdas(self) -> list[float]:
        """Current DC weights, one per stage — informative to log during training."""
        return [float(dc.lam.detach()) for dc in self.dc_layers]


class ResidualDCWrapper(nn.Module):
    """
    Wrap any reconstruction network with a single data-consistency step.

    ``x_out = |DC(Net(x_in), k, m)|``

    The residual behaviour that used to live here now belongs to the backbones
    themselves (they zero-initialise their output layer and add the input), so
    this wrapper is purely about physics.

    Parameters
    ----------
    model
        Backbone mapping ``(B, 2, H, W) -> (B, 2, H, W)``.
    use_dc
        Apply the DC step.
    output
        ``"complex"`` or ``"magnitude"``.
    """

    def __init__(self, model: nn.Module, use_dc: bool = True, output: str = "magnitude"):
        super().__init__()
        self.model = model
        self.use_dc = use_dc
        self.output = output
        self.dc = DataConsistencyLayer(learnable_lambda=True) if use_dc else None

    def forward(
        self,
        x_input: torch.Tensor,
        k_measured: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.model(x_input)

        if self.use_dc:
            if k_measured is None or mask is None:
                raise ValueError(
                    "ResidualDCWrapper was configured with use_dc=True but received "
                    "no k-space or mask. Set data.return_kspace=true in the config, "
                    "or set model.data_consistency.enabled=false. (Silently skipping "
                    "DC here is what made the original implementation a no-op.)"
                )
            x = self.dc(x, k_measured, mask)

        return complex_abs_chan(x) if self.output == "magnitude" else x
