"""
test_models.py
--------------
Architecture tests: construction, shapes, gradients, and the specific defects
that previously made models unusable.
"""

from __future__ import annotations

import pytest
import torch

from models.bt_unet import BTUNet
from models.layers import DropPath, count_parameters, drop_path
from models.registry import available_models, build_backbone, build_model, forward_model
from models.swinunet import (
    PatchExpanding,
    PatchMerging,
    SwinUNet,
    pad_for_windows,
    window_partition,
    window_reverse,
)
from models.unet import NormUNet, UNet


# ---------------------------------------------------------------------------
# Regression tests for previously fatal defects
# ---------------------------------------------------------------------------


class TestRegressions:
    """Each test here corresponds to a bug that shipped in the original code."""

    @pytest.mark.parametrize("n_levels", [1, 2, 3, 4])
    def test_swinunet_constructs_at_indivisible_resolutions(self, n_levels):
        """
        The documented config (320px, patch 4, 3 levels, window 8) reaches a
        20x20 stage, and 20 % 8 != 0. Construction previously raised
        RuntimeError inside window_partition, so no forward pass was possible.
        """
        model = SwinUNet(
            img_size=320, patch_size=4, embed_dim=32, ws=8, head_dim=8, n_levels=n_levels
        )
        with torch.no_grad():
            out = model(torch.randn(1, 1, 320, 320))
        assert out.shape == (1, 1, 320, 320)

    @pytest.mark.parametrize("size", [(64, 64), (97, 131), (200, 180), (256, 320)])
    def test_swinunet_accepts_arbitrary_sizes(self, size):
        """Resolution is a forward-time property, not baked into buffers."""
        model = SwinUNet(embed_dim=16, ws=4, head_dim=8, n_levels=2).eval()
        with torch.no_grad():
            out = model(torch.randn(1, 1, *size))
        assert out.shape[-2:] == size

    def test_bt_unet_positional_encoding_is_size_agnostic(self):
        """
        The old positional table was sized from a hard-coded 320, so any other
        input silently truncated or over-indexed it.
        """
        model = BTUNet(base_ch=8, n_levels=2, tf_heads=4, tf_layers=1, img_size=320).eval()
        with torch.no_grad():
            small = model(torch.randn(1, 1, 64, 64))
            large = model(torch.randn(1, 1, 128, 128))
        assert small.shape == (1, 1, 64, 64)
        assert large.shape == (1, 1, 128, 128)

    def test_residual_models_start_as_identity(self):
        """
        Output layers are zero-initialised, so a residual model reproduces its
        input exactly at step 0. Without this the first optimiser steps chase a
        large, meaningless loss.
        """
        x = torch.randn(2, 1, 64, 64)
        for model in (
            UNet(base_ch=8, n_levels=2, residual=True),
            SwinUNet(embed_dim=16, ws=4, n_levels=1, residual=True),
            BTUNet(base_ch=8, n_levels=2, tf_heads=4, tf_layers=1, residual=True),
        ):
            model.eval()
            with torch.no_grad():
                assert torch.allclose(model(x), x, atol=1e-5), type(model).__name__

    def test_residual_rejects_mismatched_channels(self):
        """A residual connection between 1 and 2 channels is not well defined."""
        with pytest.raises(ValueError, match="in_ch == out_ch"):
            SwinUNet(in_ch=1, out_ch=2, residual=True)
        with pytest.raises(ValueError, match="in_channels == out_channels"):
            UNet(in_channels=1, out_channels=2, residual=True)


# ---------------------------------------------------------------------------
# Window mechanics
# ---------------------------------------------------------------------------


class TestWindowOps:
    @pytest.mark.parametrize("H,W,ws", [(16, 16, 4), (20, 20, 8), (13, 7, 5), (8, 8, 8)])
    def test_partition_reverse_roundtrip(self, H, W, ws):
        x = torch.randn(2, H, W, 6)
        padded, pad_h, pad_w = pad_for_windows(x, ws)
        windows = window_partition(padded, ws)
        restored = window_reverse(windows, ws, padded.shape[1], padded.shape[2])
        assert torch.allclose(restored[:, :H, :W], x, atol=1e-6)

    def test_pad_for_windows_reaches_a_multiple(self):
        x = torch.randn(1, 20, 13, 3)
        padded, pad_h, pad_w = pad_for_windows(x, 8)
        assert padded.shape[1] % 8 == 0 and padded.shape[2] % 8 == 0
        assert (pad_h, pad_w) == (4, 3)

    def test_window_reverse_uses_integer_batch_arithmetic(self):
        """The original computed the batch size through float division."""
        windows = torch.randn(2 * 4, 4, 4, 5)  # B=2, 2x2 windows of size 4
        out = window_reverse(windows, 4, 8, 8)
        assert out.shape == (2, 8, 8, 5)


class TestPatchOps:
    def test_patch_merging_halves_resolution_doubles_channels(self):
        merge = PatchMerging(8)
        out = merge(torch.randn(2, 16, 16, 8))
        assert out.shape == (2, 8, 8, 16)

    def test_patch_merging_handles_odd_resolution(self):
        merge = PatchMerging(8)
        out = merge(torch.randn(1, 15, 13, 8))
        assert out.shape == (1, 8, 7, 16)

    def test_patch_expanding_doubles_resolution_keeps_channels(self):
        expand = PatchExpanding(8)
        out = expand(torch.randn(2, 8, 8, 8))
        assert out.shape == (2, 16, 16, 8)

    def test_expand_inverts_merge_resolution(self):
        x = torch.randn(1, 16, 16, 8)
        merged = PatchMerging(8)(x)
        expanded = PatchExpanding(16)(merged)
        assert expanded.shape[1:3] == x.shape[1:3]


# ---------------------------------------------------------------------------
# Per-architecture behaviour
# ---------------------------------------------------------------------------


@pytest.fixture
def unet():
    return UNet(in_channels=1, out_channels=1, base_ch=16, n_levels=3)


@pytest.fixture
def bt_unet():
    return BTUNet(base_ch=16, n_levels=3, tf_heads=4, tf_layers=1)


@pytest.fixture
def swinunet():
    return SwinUNet(embed_dim=16, ws=4, head_dim=8, n_levels=2)


class TestForwardAndGradients:
    @pytest.mark.parametrize("fixture", ["unet", "bt_unet", "swinunet"])
    def test_forward_shape(self, fixture, request):
        model = request.getfixturevalue(fixture)
        out = model(torch.randn(2, 1, 64, 64))
        assert out.shape == (2, 1, 64, 64)
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("fixture", ["unet", "bt_unet", "swinunet"])
    def test_every_parameter_receives_gradient(self, fixture, request):
        model = request.getfixturevalue(fixture)
        # A residual model is the identity at init, so its output-layer gradient
        # is well defined but the loss must depend on the network's own output;
        # a plain mean over an identity would give zero gradient everywhere.
        out = model(torch.randn(2, 1, 64, 64))
        (out**2).mean().backward()
        missing = [
            name
            for name, p in model.named_parameters()
            if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())
        ]
        assert not missing, f"no finite gradient for: {missing[:5]}"

    @pytest.mark.parametrize("fixture", ["unet", "bt_unet", "swinunet"])
    def test_eval_mode_is_deterministic(self, fixture, request):
        model = request.getfixturevalue(fixture).eval()
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            assert torch.allclose(model(x), model(x))

    def test_single_sample_batch(self, swinunet):
        assert swinunet(torch.randn(1, 1, 64, 64)).shape == (1, 1, 64, 64)

    def test_norm_unet_is_scale_equivariant(self):
        """
        Instance normalisation makes the model exactly equivariant to input
        scaling, which matters because MR intensity has no absolute units.
        """
        model = NormUNet(base_ch=8, n_levels=2, residual=False).eval()
        x = torch.rand(1, 1, 64, 64) + 0.5
        with torch.no_grad():
            a = model(x * 100.0) / 100.0
            b = model(x)
        assert torch.allclose(a, b, atol=1e-3)


class TestDropPath:
    def test_identity_in_eval(self):
        x = torch.randn(4, 8)
        assert torch.equal(drop_path(x, 0.5, training=False), x)

    def test_drops_whole_samples(self):
        torch.manual_seed(0)
        x = torch.ones(64, 4, 4)
        out = drop_path(x, 0.5, training=True)
        # Every sample is either fully dropped or fully kept and rescaled.
        per_sample = out.reshape(64, -1)
        assert all(
            torch.allclose(row, row[0].expand_as(row)) for row in per_sample
        )
        assert (per_sample[:, 0] == 0).any()

    def test_preserves_expectation(self):
        torch.manual_seed(0)
        x = torch.ones(4096, 2)
        out = drop_path(x, 0.3, training=True)
        assert abs(out.mean().item() - 1.0) < 0.05

    def test_module_repr(self):
        assert "0.100" in repr(DropPath(0.1))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_lists_all_backbones(self):
        assert set(available_models()) >= {"unet", "bt_unet", "swinunet", "norm_unet"}

    @pytest.mark.parametrize("name", ["unet", "norm_unet", "bt_unet", "swinunet"])
    def test_builds_every_backbone(self, name):
        model = build_backbone(name, {}, in_channels=1, out_channels=1)
        with torch.no_grad():
            assert model(torch.randn(1, 1, 64, 64)).shape == (1, 1, 64, 64)

    def test_rejects_unknown_model(self):
        with pytest.raises(ValueError, match="Unknown model"):
            build_backbone("resnet", {})

    def test_rejects_unknown_parameter(self):
        """
        A silently ignored parameter produces a model that is not the one that
        was requested, and nothing in the run would reveal it.
        """
        with pytest.raises(ValueError, match="Unknown parameter"):
            build_backbone("unet", {"embed_dim": 64})

    def test_complex_model_reports_two_channels(self, base_config):
        base_config["model"]["complex"] = True
        model = build_model(base_config)
        assert model.in_channels == 2
        assert model.expects_kspace is False
        with torch.no_grad():
            assert forward_model(model, torch.randn(1, 2, 64, 64)).shape == (1, 1, 64, 64)

    def test_data_consistency_forces_complex(self, base_config):
        base_config["model"]["complex"] = False
        base_config["model"]["data_consistency"] = {"enabled": True, "n_cascades": 2}
        model = build_model(base_config)
        assert model.in_channels == 2, "DC requires phase, so complex must be forced on"
        assert model.expects_kspace is True

    def test_shared_weights_counts_parameters_once(self):
        from models.data_consistency import CascadedNet

        def factory():
            return build_backbone("unet", {"base_ch": 8, "n_levels": 2}, 2, 2)

        shared = CascadedNet(factory, n_cascades=4, share_weights=True)
        separate = CascadedNet(factory, n_cascades=4, share_weights=False)
        assert count_parameters(shared) < count_parameters(separate) / 3


class TestParameterCounts:
    @pytest.mark.parametrize(
        "model,low,high",
        [
            (UNet(base_ch=32, n_levels=4), 5e6, 15e6),
            (SwinUNet(embed_dim=64, n_levels=3, ws=8, head_dim=8), 5e6, 40e6),
        ],
    )
    def test_within_expected_range(self, model, low, high):
        assert low < count_parameters(model) < high

    def test_deeper_swin_has_more_parameters(self):
        small = count_parameters(SwinUNet(embed_dim=32, n_levels=2))
        large = count_parameters(SwinUNet(embed_dim=32, n_levels=3))
        assert large > small
