"""
test_physics.py
---------------
Tests for the Fourier conventions and the data-consistency operator.

These are the tests that matter most for scientific validity. A reconstruction
network with a broken data-consistency layer still trains, still produces a
falling loss curve and still generates plausible images — it simply is not doing
the thing the method claims to do. Only an exactness assertion catches that.
"""

from __future__ import annotations

import pytest
import torch

from models.data_consistency import (
    CascadedNet,
    DataConsistencyLayer,
    ResidualDCWrapper,
    _broadcast_mask,
)
from models.fourier import (
    chan_to_last,
    complex_abs,
    complex_abs_chan,
    complex_mul,
    fft2c,
    fft2c_chan,
    ifft2c,
    ifft2c_chan,
    last_to_chan,
    phase,
)
from models.registry import build_backbone

# ---------------------------------------------------------------------------
# Fourier transforms
# ---------------------------------------------------------------------------


class TestFourier:
    @pytest.mark.parametrize("shape", [(32, 32), (64, 48), (17, 23)])
    def test_roundtrip_is_exact(self, shape):
        x = torch.randn(2, *shape, 2)
        assert torch.allclose(ifft2c(fft2c(x)), x, atol=1e-5)
        assert torch.allclose(fft2c(ifft2c(x)), x, atol=1e-5)

    def test_channel_layout_roundtrip(self):
        x = torch.randn(2, 2, 32, 32)
        assert torch.allclose(ifft2c_chan(fft2c_chan(x)), x, atol=1e-5)

    def test_layout_conversions_are_inverse(self):
        x = torch.randn(3, 2, 16, 16)
        assert torch.equal(last_to_chan(chan_to_last(x)), x)

    def test_layout_conversion_validates_shape(self):
        with pytest.raises(ValueError, match="expected 2 channels"):
            chan_to_last(torch.randn(1, 3, 8, 8))
        with pytest.raises(ValueError, match="trailing dim 2"):
            last_to_chan(torch.randn(1, 8, 8, 3))

    def test_transform_is_unitary(self):
        """norm='ortho' means Parseval holds, so energy is preserved."""
        x = torch.randn(1, 32, 32, 2)
        assert torch.allclose((x**2).sum(), (fft2c(x) ** 2).sum(), rtol=1e-4)

    def test_dc_component_sits_at_the_centre(self):
        """A constant image must concentrate all energy at the array centre."""
        img = torch.zeros(1, 32, 32, 2)
        img[..., 0] = 1.0
        k = fft2c(img)
        mag = complex_abs(k)[0]
        peak = torch.argmax(mag)
        assert (int(peak // 32), int(peak % 32)) == (16, 16)

    def test_complex_abs_is_nonnegative_and_differentiable_at_zero(self):
        x = torch.zeros(1, 4, 4, 2, requires_grad=True)
        loss = complex_abs(x).sum()
        loss.backward()
        assert torch.isfinite(x.grad).all(), "magnitude gradient must be finite at zero"

    def test_complex_multiply_matches_native(self):
        a = torch.randn(4, 2)
        b = torch.randn(4, 2)
        expected = torch.view_as_complex(a) * torch.view_as_complex(b)
        assert torch.allclose(complex_mul(a, b), torch.view_as_real(expected), atol=1e-6)

    def test_phase_recovers_angle(self):
        x = torch.tensor([[0.0, 1.0]])
        assert torch.allclose(phase(x), torch.tensor([torch.pi / 2]), atol=1e-6)


# ---------------------------------------------------------------------------
# Data consistency
# ---------------------------------------------------------------------------


class TestDataConsistency:
    @staticmethod
    def _setup(H=32, W=32, keep=0.4, seed=0):
        torch.manual_seed(seed)
        truth = torch.randn(2, 2, H, W)
        k_full = fft2c_chan(truth)
        mask = (torch.rand(2, 1, 1, W) < keep).float()
        return truth, k_full * mask, mask

    def test_hard_dc_restores_measured_lines_exactly(self):
        """
        The defining property. Anything else means the network is free to
        contradict the scanner.
        """
        truth, k_meas, mask = self._setup()
        dc = DataConsistencyLayer(learnable_lambda=False, hard=True)
        out = dc(torch.randn_like(truth), k_meas, mask)
        residual = (fft2c_chan(out) - k_meas) * mask
        assert residual.abs().max() < 1e-4

    def test_lambda_init_means_what_it_says(self):
        """
        The original applied sigmoid() to lambda_init directly, so a requested
        hard consistency of 1.0 became sigmoid(1.0) = 0.73 -- a silent 27% leak
        of network output into measured frequencies.
        """
        for requested in (0.5, 0.9, 1.0):
            dc = DataConsistencyLayer(learnable_lambda=True, lambda_init=requested)
            assert abs(float(dc.lam.detach()) - requested) < 1e-3

    def test_unmeasured_frequencies_come_from_the_network(self):
        truth, k_meas, mask = self._setup()
        pred = torch.randn_like(truth)
        out = DataConsistencyLayer(learnable_lambda=False, hard=True)(pred, k_meas, mask)
        k_out, k_pred = fft2c_chan(out), fft2c_chan(pred)
        assert ((k_out - k_pred) * (1 - mask)).abs().max() < 1e-4

    def test_perfect_prediction_is_a_fixed_point(self):
        truth, k_meas, mask = self._setup()
        # Feed the fully sampled truth; DC must leave it unchanged.
        k_all = fft2c_chan(truth)
        out = DataConsistencyLayer(learnable_lambda=False, hard=True)(truth, k_all, mask)
        assert torch.allclose(out, truth, atol=1e-4)

    def test_preserves_phase(self):
        """
        The old implementation returned ``x.abs()``, destroying phase and making
        cascading impossible.
        """
        truth, k_meas, mask = self._setup()
        out = DataConsistencyLayer(learnable_lambda=False, hard=True)(truth, k_meas, mask)
        assert out.shape[1] == 2
        assert out[:, 1].abs().max() > 1e-3, "imaginary component was discarded"

    def test_rejects_magnitude_input(self):
        _, k_meas, mask = self._setup()
        with pytest.raises(ValueError, match="complex"):
            DataConsistencyLayer()(torch.randn(2, 1, 32, 32), k_meas, mask)

    @pytest.mark.parametrize(
        "shape", [(32,), (1, 32), (2, 32), (2, 1, 32), (2, 1, 1, 32), (2, 1, 32, 32)]
    )
    def test_mask_broadcasting_accepts_pipeline_shapes(self, shape):
        like = torch.randn(2, 2, 32, 32)
        out = _broadcast_mask(torch.ones(*shape), like)
        assert out.shape[-1] == 32

    def test_mask_width_mismatch_is_an_error(self):
        like = torch.randn(2, 2, 32, 32)
        with pytest.raises(ValueError, match="does not match k-space width"):
            _broadcast_mask(torch.ones(2, 1, 1, 16), like)


class TestCascade:
    @staticmethod
    def _factory():
        return build_backbone("unet", {"base_ch": 8, "n_levels": 2}, 2, 2)

    def test_forward_returns_magnitude(self):
        net = CascadedNet(self._factory, n_cascades=2, output="magnitude")
        x = torch.randn(1, 2, 32, 32)
        mask = (torch.rand(1, 1, 1, 32) < 0.4).float()
        k = fft2c_chan(x) * mask
        assert net(x, k, mask).shape == (1, 1, 32, 32)

    def test_complex_output_mode(self):
        net = CascadedNet(self._factory, n_cascades=2, output="complex")
        x = torch.randn(1, 2, 32, 32)
        mask = (torch.rand(1, 1, 1, 32) < 0.4).float()
        assert net(x, fft2c_chan(x) * mask, mask).shape == (1, 2, 32, 32)

    def test_final_output_is_data_consistent(self):
        """
        End-to-end: the cascade's final estimate must still agree with the
        measurements after every learned stage.
        """
        net = CascadedNet(self._factory, n_cascades=3, hard_dc=True, output="complex").eval()
        torch.manual_seed(0)
        truth = torch.randn(1, 2, 32, 32)
        mask = (torch.rand(1, 1, 1, 32) < 0.4).float()
        k_meas = fft2c_chan(truth) * mask
        with torch.no_grad():
            out = net(ifft2c_chan(k_meas), k_meas, mask)
        assert ((fft2c_chan(out) - k_meas) * mask).abs().max() < 1e-3

    def test_runs_without_kspace_for_ablation(self):
        net = CascadedNet(self._factory, n_cascades=2)
        assert net(torch.randn(1, 2, 32, 32)).shape == (1, 1, 32, 32)

    def test_gradients_flow_through_dc(self):
        net = CascadedNet(self._factory, n_cascades=2)
        x = torch.randn(1, 2, 32, 32)
        mask = (torch.rand(1, 1, 1, 32) < 0.4).float()
        net(x, fft2c_chan(x) * mask, mask).mean().backward()
        assert any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in net.parameters()
            if p.requires_grad
        )

    def test_lambdas_are_reported(self):
        net = CascadedNet(self._factory, n_cascades=3)
        assert len(net.lambdas) == 3
        assert all(0.0 <= v <= 1.0 for v in net.lambdas)

    def test_rejects_invalid_cascade_count(self):
        with pytest.raises(ValueError, match="n_cascades"):
            CascadedNet(self._factory, n_cascades=0)


class TestResidualDCWrapper:
    def test_missing_kspace_raises_instead_of_silently_skipping(self):
        """
        The original returned a plain prediction when k-space was absent, which
        is how ``data_consistency.enabled: true`` became a no-op for the entire
        project.
        """
        model = build_backbone("unet", {"base_ch": 8, "n_levels": 2}, 2, 2)
        wrapper = ResidualDCWrapper(model, use_dc=True)
        with pytest.raises(ValueError, match="no k-space or mask"):
            wrapper(torch.randn(1, 2, 32, 32))

    def test_applies_dc_when_given_data(self):
        model = build_backbone("unet", {"base_ch": 8, "n_levels": 2}, 2, 2)
        wrapper = ResidualDCWrapper(model, use_dc=True)
        x = torch.randn(1, 2, 32, 32)
        mask = (torch.rand(1, 1, 1, 32) < 0.4).float()
        assert wrapper(x, fft2c_chan(x) * mask, mask).shape == (1, 1, 32, 32)


def test_complex_abs_chan_matches_last_dim_version():
    x = torch.randn(2, 2, 16, 16)
    assert torch.allclose(
        complex_abs_chan(x).squeeze(1), complex_abs(chan_to_last(x)), atol=1e-6
    )
