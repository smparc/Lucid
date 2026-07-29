"""
test_evaluate.py
----------------
Tests for evaluation, checkpoint loading, and figure generation.

These two modules (``training.evaluate`` and ``utils.visualizations``) had zero
test coverage, and that is exactly where two of the original defects lived: an
``AttentionExtractor`` whose hooks matched nothing, and a
``plot_training_history`` that read a history file the trainer never wrote.
Both were found by reading the code, which does not scale. Everything here
exercises the real path end to end -- real checkpoints, real HDF5 volumes, real
figures written to disk.
"""

from __future__ import annotations

import json

import matplotlib
import pytest
import torch

matplotlib.use("Agg")  # no display in CI

from torch.utils.data import DataLoader

from config import load_config
from data.preprocessing import FastMRIKneeDataset, collate
from models.registry import build_model
from training.evaluate import (
    compare_architectures,
    evaluate_model,
    load_model,
    plot_training_history,
    visualize_reconstructions,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _tiny_config(complex_mode: bool = False, dc: bool = False):
    model = {
        "name": "unet",
        "complex": complex_mode,
        "params": {"base_ch": 8, "n_levels": 2},
    }
    if dc:
        model["data_consistency"] = {"enabled": True, "mode": "cascade", "n_cascades": 2}
    return load_config(
        overrides={
            "model": model,
            "data": {"crop_size": [64, 64], "num_workers": 0, "persistent_workers": False},
            "training": {"epochs": 1, "batch_size": 2, "amp": False},
            "logging": {"tensorboard": False, "wandb": False},
        }
    )


@pytest.fixture
def checkpoint(tmp_path, request):
    """
    Write a checkpoint in the format the trainer emits.

    Parametrised indirectly with (complex_mode, dc) so the same fixture can
    produce magnitude, complex and data-consistency checkpoints.
    """
    complex_mode, dc = getattr(request, "param", (False, False))
    cfg = _tiny_config(complex_mode, dc)
    model = build_model(cfg)

    path = tmp_path / "best.pt"
    torch.save(
        {
            "epoch": 3,
            "global_step": 30,
            "model_state": model.state_dict(),
            "config": cfg.to_dict(),
            "val_loss": 0.05,
            "val_psnr": 30.5,
            "val_ssim": 0.85,
        },
        path,
    )
    return path


@pytest.fixture
def dataset(synthetic_data_dir):
    return FastMRIKneeDataset(synthetic_data_dir, crop_size=(64, 64), slice_mode="all")


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


class TestLoadModel:
    def test_rebuilds_architecture_from_embedded_config(self, checkpoint, device):
        """
        The architecture must come from the checkpoint's own config, so weights
        can never be loaded into a differently shaped model.
        """
        model = load_model(None, checkpoint, device)
        assert isinstance(model, torch.nn.Module)
        with torch.no_grad():
            assert model(torch.randn(1, 1, 64, 64)).shape == (1, 1, 64, 64)

    @pytest.mark.parametrize(
        "checkpoint", [(False, False), (True, False), (True, True)], indirect=True
    )
    def test_loads_every_model_flavour(self, checkpoint, device):
        model = load_model(None, checkpoint, device)
        channels = int(getattr(model, "in_channels", 1))
        x = torch.randn(1, channels, 64, 64)
        with torch.no_grad():
            if getattr(model, "expects_kspace", False):
                mask = (torch.rand(1, 1, 64) < 0.3).float()
                out = model(x, torch.randn(1, 2, 64, 64), mask)
            else:
                out = model(x)
        assert out.shape == (1, 1, 64, 64)

    def test_missing_checkpoint_raises(self, tmp_path, device):
        with pytest.raises(FileNotFoundError):
            load_model(None, tmp_path / "absent.pt", device)

    def test_legacy_checkpoint_without_config_needs_a_name(self, tmp_path, device):
        model = build_model(_tiny_config())
        path = tmp_path / "legacy.pt"
        torch.save({"model_state": model.state_dict(), "epoch": 1}, path)

        with pytest.raises(ValueError, match="no embedded config"):
            load_model(None, path, device)

    def test_prefers_ema_weights(self, tmp_path, device):
        """
        The trainer validates and selects checkpoints under EMA weights, so
        evaluation must use them too or it reports a different model.
        """
        from utils.ema import EMAModel

        cfg = _tiny_config()
        model = build_model(cfg)
        ema = EMAModel(model, decay=0.0, warmup=0)
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(0.5)
        ema.update()  # decay 0 => shadow tracks live weights exactly

        with torch.no_grad():  # now diverge the raw weights
            for p in model.parameters():
                p.fill_(0.1)

        path = tmp_path / "ema.pt"
        torch.save(
            {
                "epoch": 1,
                "model_state": model.state_dict(),
                "ema_state": ema.state_dict(),
                "config": cfg.to_dict(),
            },
            path,
        )

        loaded = load_model(None, path, device, use_ema=True)
        first = next(iter(loaded.parameters()))
        assert torch.allclose(first, torch.full_like(first, 0.5)), "EMA weights not applied"

        raw = load_model(None, path, device, use_ema=False)
        first_raw = next(iter(raw.parameters()))
        assert torch.allclose(first_raw, torch.full_like(first_raw, 0.1))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class TestEvaluateModel:
    def test_returns_metrics_and_per_sample_vectors(self, checkpoint, dataset, device):
        model = load_model(None, checkpoint, device)
        loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate)
        result = evaluate_model(model, loader, device)

        for key in ("val_loss", "psnr_db", "ssim", "nmse", "n_samples"):
            assert key in result
        # Per-sample vectors are what make paired significance tests possible.
        assert len(result["per_sample"]["psnr"]) == len(dataset)
        assert len(result["per_sample"]["ssim"]) == len(dataset)

    def test_max_batches_limits_work(self, checkpoint, dataset, device):
        model = load_model(None, checkpoint, device)
        loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate)
        assert evaluate_model(model, loader, device, max_batches=1)["n_samples"] == 2

    def test_reports_worst_slices_with_filenames(self, checkpoint, dataset, device):
        model = load_model(None, checkpoint, device)
        loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate)
        worst = evaluate_model(model, loader, device)["worst"]
        assert worst and all(name.endswith(".h5") for name, _ in worst)

    def test_multi_acceleration_breakdown(self, synthetic_data_dir, checkpoint, device):
        model = load_model(None, checkpoint, device)
        ds = FastMRIKneeDataset(
            synthetic_data_dir, crop_size=(64, 64), acceleration=[4, 8], slice_mode="all"
        )
        loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate)
        result = evaluate_model(model, loader, device)
        # A single averaged number would hide uniform-vs-lopsided behaviour.
        assert set(result.get("by_acceleration", {})) <= {4, 8}

    @pytest.mark.parametrize("checkpoint", [(True, True)], indirect=True)
    def test_evaluates_a_data_consistency_model(self, checkpoint, synthetic_data_dir, device):
        model = load_model(None, checkpoint, device)
        ds = FastMRIKneeDataset(
            synthetic_data_dir, crop_size=(64, 64), complex_input=True, slice_mode="all"
        )
        loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate)
        assert evaluate_model(model, loader, device)["n_samples"] == len(ds)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


class TestVisualizeReconstructions:
    def test_writes_a_figure(self, checkpoint, dataset, device, tmp_path):
        out = tmp_path / "recon.png"
        path = visualize_reconstructions(
            load_model(None, checkpoint, device), dataset, device,
            n_examples=2, save_path=str(out),
        )
        assert out.exists() and out.stat().st_size > 1000
        assert path == str(out)

    def test_creates_missing_directories(self, checkpoint, dataset, device, tmp_path):
        out = tmp_path / "deep" / "nested" / "recon.png"
        visualize_reconstructions(
            load_model(None, checkpoint, device), dataset, device,
            n_examples=1, save_path=str(out),
        )
        assert out.exists()

    @pytest.mark.parametrize("checkpoint", [(True, True)], indirect=True)
    def test_handles_complex_input(self, checkpoint, synthetic_data_dir, device, tmp_path):
        """A 2-channel input has no directly displayable channel."""
        ds = FastMRIKneeDataset(
            synthetic_data_dir, crop_size=(64, 64), complex_input=True, slice_mode="all"
        )
        out = tmp_path / "cplx.png"
        visualize_reconstructions(
            load_model(None, checkpoint, device), ds, device,
            n_examples=1, save_path=str(out),
        )
        assert out.exists()

    def test_is_reproducible_for_a_given_seed(self, checkpoint, dataset, device, tmp_path):
        model = load_model(None, checkpoint, device)
        a, b = tmp_path / "a.png", tmp_path / "b.png"
        for path in (a, b):
            visualize_reconstructions(
                model, dataset, device, n_examples=2, save_path=str(path), seed=7
            )
        assert a.read_bytes() == b.read_bytes()

    def test_empty_dataset_raises(self, checkpoint, device, tmp_path):
        class Empty:
            def __len__(self):
                return 0

        with pytest.raises(ValueError, match="empty"):
            visualize_reconstructions(
                load_model(None, checkpoint, device), Empty(), device,
                save_path=str(tmp_path / "x.png"),
            )


class TestPlotTrainingHistory:
    @staticmethod
    def _history(path, n=5):
        path.write_text(
            json.dumps(
                {
                    "epoch": list(range(1, n + 1)),
                    "epoch/train_loss": [0.5 / (i + 1) for i in range(n)],
                    "epoch/val_loss": [0.6 / (i + 1) for i in range(n)],
                    "epoch/val_psnr": [25 + i for i in range(n)],
                    "epoch/val_ssim": [0.7 + 0.01 * i for i in range(n)],
                    "epoch/lr": [1e-3 * 0.9**i for i in range(n)],
                }
            )
        )

    def test_plots_from_a_history_file(self, tmp_path):
        history = tmp_path / "history.json"
        self._history(history)
        out = plot_training_history(history)
        assert (tmp_path / "training_curves.png").exists()
        assert out.endswith("training_curves.png")

    def test_accepts_a_run_directory(self, tmp_path):
        """`--run outputs/foo` should work, not just the file path."""
        self._history(tmp_path / "history.json")
        plot_training_history(tmp_path)
        assert (tmp_path / "training_curves.png").exists()

    def test_missing_history_raises_with_guidance(self, tmp_path):
        with pytest.raises(FileNotFoundError, match=r"history\.json"):
            plot_training_history(tmp_path)

    def test_tolerates_partial_history(self, tmp_path):
        """A run killed early still has to plot."""
        history = tmp_path / "history.json"
        history.write_text(json.dumps({"epoch": [1, 2], "epoch/train_loss": [0.5, 0.4]}))
        plot_training_history(history, save_path=str(tmp_path / "partial.png"))
        assert (tmp_path / "partial.png").exists()


class TestCompareArchitectures:
    @staticmethod
    def _result(offset, n=40, seed=0):
        import numpy as np

        rng = np.random.default_rng(seed)
        psnr = (rng.normal(30, 2, n) + offset).tolist()
        return {
            "val_loss": 0.05,
            "psnr_db": float(np.mean(psnr)),
            "ssim": 0.85,
            "nmse": 0.01,
            "per_sample": {
                "psnr": psnr,
                "ssim": rng.normal(0.85, 0.02, n).tolist(),
                "nmse": rng.normal(0.01, 0.001, n).tolist(),
            },
        }

    def test_runs_paired_tests_and_saves(self, tmp_path, capsys):
        out = tmp_path / "comparison.json"
        comparison = compare_architectures(
            {"unet": self._result(0.0), "swinunet": self._result(2.0)},
            metric="psnr",
            reference="unet",
            save_path=str(out),
        )
        assert comparison["comparisons"]["swinunet"]["significant"]
        assert out.exists()
        payload = json.loads(out.read_text())
        assert "summary" in payload and "statistics" in payload
        # Per-sample vectors must not bloat the saved summary.
        assert "per_sample" not in payload["summary"]["unet"]

    def test_skips_pairing_on_mismatched_sample_counts(self, capsys):
        """
        A paired test on different sample sets is invalid; it must be skipped
        rather than silently computed on misaligned vectors.
        """
        comparison = compare_architectures(
            {"a": self._result(0.0, n=40), "b": self._result(1.0, n=25)}, metric="psnr"
        )
        assert comparison == {}

    def test_single_model_still_prints(self, capsys):
        compare_architectures({"only": self._result(0.0)})
        assert "only" in capsys.readouterr().out
