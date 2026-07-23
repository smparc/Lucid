"""
data_consistency.py
-------------------
Data Consistency (DC) layer for physics-informed MRI reconstruction.


This is the single most impactful technique for MRI reconstruction quality.
It enforces that the reconstructed image is consistent with the actually
measured k-space data — the known measurements are replaced back after
the network's prediction, ensuring the network never hallucates in
regions where we have ground-truth frequency information.


Theory
------
Given:
    - x_pred: network's predicted image
    - k_measured: undersampled k-space data (what was actually acquired)
    - mask: binary mask showing which k-space lines were sampled


The DC layer performs:
    k_pred = FFT(x_pred)
    k_dc[mask] = lambda * k_measured[mask] + (1-lambda) * k_pred[mask]  # weighted replacement
    k_dc[~mask] = k_pred[~mask]                                        # network fills gaps
    x_dc = IFFT(k_dc)


Where lambda controls the strength of data consistency (1.0 = hard, <1 = soft).


References
----------
Schlemper et al., "A Deep Cascade of CNNs for Dynamic MR Image Reconstruction", TMI 2018
Hammernik et al., "Learning a Variational Network for Reconstruction of Accelerated MRI Data", MRM 2018
"""


import torch
import torch.nn as nn
import torch.nn.functional as F



class DataConsistencyLayer(nn.Module):
    """
    Data Consistency layer that enforces fidelity to measured k-space.


    This can be inserted after any reconstruction network or cascaded
    between multiple network stages.


    Parameters
    ----------
    learnable_lambda : bool — if True, lambda is a learnable parameter
    lambda_init      : float — initial data consistency weight (1.0 = hard DC)
    """


    def __init__(self, learnable_lambda: bool = True, lambda_init: float = 1.0):
        super().__init__()
        if learnable_lambda:
            self.lam = nn.Parameter(torch.tensor(lambda_init))
        else:
            self.register_buffer("lam", torch.tensor(lambda_init))


    def forward(
        self,
        x_pred: torch.Tensor,
        k_measured: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply data consistency.


        Parameters
        ----------
        x_pred      : (B, 1, H, W) — network's image-space prediction
        k_measured  : (B, 1, H, W, 2) — measured undersampled k-space [real, imag]
        mask        : (B, 1, 1, W) or (B, 1, H, W) — binary sampling mask


        Returns
        -------
        x_dc : (B, 1, H, W) — data-consistent reconstruction
        """
        # Convert prediction to k-space
        k_pred = self._image_to_kspace(x_pred)  # (B, 1, H, W, 2)


        # Expand mask to match k-space shape
        if mask.dim() == 3:
            mask = mask.unsqueeze(-1)  # (B, 1, W, 1) -> need (B, 1, H, W, 1)
        while mask.dim() < k_pred.dim():
            mask = mask.unsqueeze(-1)


        mask = mask.expand_as(k_pred)


        # Soft data consistency: blend measured and predicted where mask=1
        lam = torch.sigmoid(self.lam)  # Constrain lambda to [0, 1]
        k_dc = torch.where(
            mask > 0.5,
            lam * k_measured + (1 - lam) * k_pred,
            k_pred,
        )


        # Convert back to image space
        x_dc = self._kspace_to_image(k_dc)
        return x_dc


    @staticmethod
    def _image_to_kspace(x: torch.Tensor) -> torch.Tensor:
        """(B, 1, H, W) real image → (B, 1, H, W, 2) k-space."""
        # FFT
        k = torch.fft.fft2(x, norm="ortho")
        k = torch.fft.fftshift(k, dim=(-2, -1))
        return torch.view_as_real(k)


    @staticmethod
    def _kspace_to_image(k: torch.Tensor) -> torch.Tensor:
        """(B, 1, H, W, 2) k-space → (B, 1, H, W) real image."""
        k_complex = torch.view_as_complex(k)
        k_complex = torch.fft.ifftshift(k_complex, dim=(-2, -1))
        x = torch.fft.ifft2(k_complex, norm="ortho")
        return x.abs()



class CascadedDCNetwork(nn.Module):
    """
    Cascaded reconstruction: Network → DC → Network → DC → ... → Output


    This interleaves learned reconstruction with physics-based data consistency,
    which is the standard approach in SOTA MRI reconstruction (Hammernik et al. 2018).


    Each cascade stage refines the reconstruction while maintaining consistency
    with the acquired data.


    Parameters
    ----------
    base_model_fn : callable — function that returns a reconstruction network
    n_cascades    : int — number of cascade stages
    share_weights : bool — if True, all cascades share the same network weights
    """


    def __init__(self, base_model_fn, n_cascades: int = 3, share_weights: bool = False):
        super().__init__()
        self.n_cascades = n_cascades


        if share_weights:
            base_model = base_model_fn()
            self.networks = nn.ModuleList([base_model] * n_cascades)
        else:
            self.networks = nn.ModuleList([base_model_fn() for _ in range(n_cascades)])


        self.dc_layers = nn.ModuleList([
            DataConsistencyLayer(learnable_lambda=True) for _ in range(n_cascades)
        ])


    def forward(
        self,
        x_input: torch.Tensor,
        k_measured: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x_input     : (B, 1, H, W) — zero-filled reconstruction (initial input)
        k_measured  : (B, 1, H, W, 2) — measured k-space data
        mask        : (B, 1, 1, W) or similar — sampling mask


        Returns
        -------
        x : (B, 1, H, W) — final reconstruction after all cascades
        """
        x = x_input


        for network, dc in zip(self.networks, self.dc_layers):
            # Residual learning: network predicts the correction/artifact
            x_refined = network(x)
            # Data consistency ensures fidelity to measurements
            x = dc(x_refined, k_measured, mask)


        return x



class ResidualDCWrapper(nn.Module):
    """
    Wraps any reconstruction model with residual learning + data consistency.


    Instead of predicting the full image, the network predicts the residual
    (artifact pattern) which is subtracted from the input. This is combined
    with a DC layer for physics-informed refinement.


    x_output = DC(x_input + Network(x_input), k_measured, mask)
    """


    def __init__(self, model: nn.Module, use_dc: bool = True):
        super().__init__()
        self.model = model
        self.use_dc = use_dc
        if use_dc:
            self.dc = DataConsistencyLayer(learnable_lambda=True)


    def forward(
        self,
        x_input: torch.Tensor,
        k_measured: torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # Residual learning
        residual = self.model(x_input)
        x_refined = x_input + residual


        # Data consistency (only if k-space data is provided)
        if self.use_dc and k_measured is not None and mask is not None:
            x_refined = self.dc(x_refined, k_measured, mask)


        return x_refined