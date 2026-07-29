"""
test_visualizations.py
----------------------
Tests for the figure-generation and attention-inspection utilities.

This module had zero coverage, and it is where the original ``AttentionExtractor``
silently returned an empty list forever: its hooks looked for attention weights
in the forward *output*, but ``WindowAttention`` returns a bare tensor and never
stored them. Every attention figure in the project was blank, and nothing said
so. The tests below assert that maps are actually captured, with the right
shape, and that the fused-attention fast path is restored afterwards.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest
import torch

matplotlib.use("Agg")

from models.swinunet import SwinUNet, WindowAttention  # noqa: E402
from utils.visualizations import (  # noqa: E402
    AttentionExtractor,
    plot_architecture_comparison,
    plot_attention_maps,
    plot_attention_rollout,
    plot_kspace_analysis,
    plot_reconstruction_comparison,
    plot_training_curves,
)


@pytest.fixture
def swin():
    return SwinUNet(embed_dim=16, ws=4, head_dim=8, n_levels=2).eval()


@pytest.fixture
def images():
    rng = np.random.default_rng(0)
    truth = rng.random((32, 32))
    return {
        "input": truth + 0.15 * rng.standard_normal((32, 32)),
        "pred": truth + 0.05 * rng.standard_normal((32, 32)),
        "truth": truth,
    }


# ---------------------------------------------------------------------------
# Static figures
# ---------------------------------------------------------------------------


class TestReconstructionComparison:
    def test_writes_a_full_panel(self, images, tmp_path):
        out = tmp_path / "cmp.png"
        plot_reconstruction_comparison(
            images["input"], images["pred"], images["truth"],
            psnr_val=31.2, ssim_val=0.87, save_path=str(out),
        )
        assert out.exists() and out.stat().st_size > 1000

    @pytest.mark.parametrize("error,freq", [(True, True), (True, False), (False, False)])
    def test_optional_panels(self, images, tmp_path, error, freq):
        out = tmp_path / f"cmp_{error}_{freq}.png"
        plot_reconstruction_comparison(
            images["input"], images["pred"], images["truth"],
            save_path=str(out), show_error=error, show_frequency=freq,
        )
        assert out.exists()

    def test_accepts_torch_tensors(self, images, tmp_path):
        out = tmp_path / "t.png"
        plot_reconstruction_comparison(
            torch.from_numpy(images["input"]).unsqueeze(0),
            torch.from_numpy(images["pred"]),
            torch.from_numpy(images["truth"]),
            save_path=str(out),
        )
        assert out.exists()

    def test_creates_parent_directories(self, images, tmp_path):
        out = tmp_path / "a" / "b" / "cmp.png"
        plot_reconstruction_comparison(
            images["input"], images["pred"], images["truth"], save_path=str(out)
        )
        assert out.exists()


class TestTrainingCurves:
    @staticmethod
    def _history(n=8):
        return {
            "epoch": list(range(1, n + 1)),
            "epoch/train_loss": [0.5 / (i + 1) for i in range(n)],
            "epoch/val_loss": [0.6 / (i + 1) for i in range(n)],
            "epoch/val_psnr": [25 + i * 0.5 for i in range(n)],
            "epoch/val_ssim": [0.7 + 0.01 * i for i in range(n)],
            "epoch/lr": [1e-3 * 0.9**i for i in range(n)],
        }

    def test_full_history(self, tmp_path):
        out = tmp_path / "curves.png"
        plot_training_curves(self._history(), save_path=str(out))
        assert out.exists()

    def test_legacy_key_names(self, tmp_path):
        """Histories written before the `epoch/` prefix must still plot."""
        out = tmp_path / "legacy.png"
        plot_training_curves(
            {"train_loss": [0.5, 0.4], "val_loss": [0.6, 0.5], "val_psnr": [25, 26]},
            save_path=str(out),
        )
        assert out.exists()

    def test_partial_history(self, tmp_path):
        out = tmp_path / "partial.png"
        plot_training_curves({"epoch/train_loss": [0.5, 0.4, 0.3]}, save_path=str(out))
        assert out.exists()

    def test_empty_history_raises(self):
        with pytest.raises(ValueError, match="expected metric series"):
            plot_training_curves({})


class TestKspaceAnalysis:
    def test_with_a_one_dimensional_mask(self, tmp_path):
        rng = np.random.default_rng(0)
        kspace = rng.standard_normal((32, 32)) + 1j * rng.standard_normal((32, 32))
        out = tmp_path / "k.png"
        plot_kspace_analysis(
            kspace, (rng.random(32) < 0.3).astype(float), rng.random((32, 32)),
            save_path=str(out),
        )
        assert out.exists()

    def test_with_a_two_dimensional_mask(self, tmp_path):
        rng = np.random.default_rng(0)
        kspace = rng.standard_normal((32, 32)) + 1j * rng.standard_normal((32, 32))
        out = tmp_path / "k2.png"
        plot_kspace_analysis(
            kspace, (rng.random((32, 32)) < 0.3).astype(float), rng.random((32, 32)),
            save_path=str(out),
        )
        assert out.exists()


class TestArchitectureComparison:
    @staticmethod
    def _results():
        return {
            "unet": {"psnr": 28.0, "ssim": 0.69, "params_m": 7.8},
            "swinunet": {"psnr": 33.1, "ssim": 0.73, "params_m": 12.9},
        }

    def test_bar_and_scatter(self, tmp_path):
        out = tmp_path / "arch.png"
        plot_architecture_comparison(self._results(), save_path=str(out))
        assert out.exists()

    def test_with_confidence_intervals(self, tmp_path):
        """Error bars are what stop a bar chart overstating a difference."""
        out = tmp_path / "arch_ci.png"
        plot_architecture_comparison(
            self._results(),
            save_path=str(out),
            metric_errors={
                "unet": {"psnr": (27.5, 28.5), "ssim": (0.68, 0.70)},
                "swinunet": {"psnr": (32.6, 33.6), "ssim": (0.72, 0.74)},
            },
        )
        assert out.exists()

    def test_without_parameter_counts(self, tmp_path):
        out = tmp_path / "arch_np.png"
        plot_architecture_comparison(
            {"a": {"psnr": 28.0, "ssim": 0.69}, "b": {"psnr": 30.0, "ssim": 0.71}},
            save_path=str(out),
        )
        assert out.exists()

    def test_empty_results_raise(self):
        with pytest.raises(ValueError, match="No results"):
            plot_architecture_comparison({})


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class TestAttentionExtractor:
    def test_actually_captures_maps(self, swin):
        """
        The regression that matters. The original hooked the forward output,
        which is a plain tensor, so this list was always empty and every
        attention figure silently rendered nothing.
        """
        with AttentionExtractor(swin) as extractor:
            with torch.no_grad():
                swin(torch.rand(1, 1, 32, 32))
            maps = extractor.get_attention_maps()

        assert maps, "no attention maps captured"
        # (windows, heads, tokens, tokens) with tokens = ws^2
        for attn in maps:
            assert attn.dim() == 4
            assert attn.shape[-1] == attn.shape[-2] == swin.encoder_stages[0].blocks[0].ws ** 2

    def test_rows_are_probability_distributions(self, swin):
        with AttentionExtractor(swin) as extractor:
            with torch.no_grad():
                swin(torch.rand(1, 1, 32, 32))
            attn = extractor.get_attention_maps()[0]
        # Softmax output: every row sums to 1 and is non-negative.
        assert torch.allclose(attn.sum(-1), torch.ones_like(attn.sum(-1)), atol=1e-4)
        assert (attn >= 0).all()

    def test_restores_the_fused_attention_path(self, swin):
        """
        Extraction disables the fused SDPA kernel. Leaving it off would slow
        every later forward pass with no indication why.
        """
        before = [m.use_sdpa for m in swin.modules() if isinstance(m, WindowAttention)]
        with AttentionExtractor(swin) as extractor:
            assert not any(
                m.use_sdpa for m in swin.modules() if isinstance(m, WindowAttention)
            )
            with torch.no_grad():
                swin(torch.rand(1, 1, 32, 32))
            extractor.get_attention_maps()
        after = [m.use_sdpa for m in swin.modules() if isinstance(m, WindowAttention)]
        assert before == after

    def test_buffer_is_cleared_after_read(self, swin):
        with AttentionExtractor(swin) as extractor:
            with torch.no_grad():
                swin(torch.rand(1, 1, 32, 32))
            assert extractor.get_attention_maps()
            assert extractor.get_attention_maps() == []

    def test_warns_on_a_model_without_window_attention(self, caplog):
        extractor = AttentionExtractor(torch.nn.Linear(4, 4))
        try:
            assert "WindowAttention" in caplog.text
        finally:
            extractor.remove_hooks()

    def test_explicit_hook_removal(self, swin):
        extractor = AttentionExtractor(swin)
        extractor.remove_hooks()
        assert all(m.use_sdpa for m in swin.modules() if isinstance(m, WindowAttention))


class TestAttentionPlots:
    @pytest.fixture
    def maps(self, swin):
        with AttentionExtractor(swin) as extractor:
            with torch.no_grad():
                swin(torch.rand(1, 1, 32, 32))
            return extractor.get_attention_maps()

    def test_plot_maps(self, maps, tmp_path):
        out = tmp_path / "attn.png"
        plot_attention_maps(np.random.rand(32, 32), maps, save_path=str(out))
        assert out.exists()

    def test_plot_maps_with_explicit_layers(self, maps, tmp_path):
        out = tmp_path / "attn_l.png"
        plot_attention_maps(
            np.random.rand(32, 32), maps, layer_indices=[0], save_path=str(out)
        )
        assert out.exists()

    def test_plot_rollout(self, maps, tmp_path):
        out = tmp_path / "rollout.png"
        plot_attention_rollout(np.random.rand(32, 32), maps, save_path=str(out))
        assert out.exists()

    def test_empty_maps_raise_rather_than_render_blank(self, tmp_path):
        """
        The original printed a message and returned, producing no file and no
        error -- indistinguishable from success to a calling script.
        """
        with pytest.raises(ValueError, match="No attention maps"):
            plot_attention_maps(np.random.rand(32, 32), [], save_path=str(tmp_path / "x.png"))
        with pytest.raises(ValueError, match="No attention maps"):
            plot_attention_rollout(np.random.rand(32, 32), [], save_path=str(tmp_path / "y.png"))
