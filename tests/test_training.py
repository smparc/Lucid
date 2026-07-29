"""
test_training.py
----------------
Tests for losses, metrics, config validation, EMA, schedulers and the
end-to-end training loop.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from config import Config, ConfigError, load_config, validate_config
from training.losses import (
    CharbonnierLoss,
    CombinedLoss,
    EdgeLoss,
    FrequencyLoss,
    SSIMLoss,
)
from training.metrics import SSIM, MetricAccumulator, nmse, psnr, ssim
from training.stats import (
    bootstrap_ci,
    compare_models,
    format_comparison_table,
    holm_bonferroni,
    paired_permutation_test,
)
from utils.ema import EMAModel, load_ema_weights_into
from utils.schedulers import WarmupCosineScheduler, build_scheduler

# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


class TestLossFunctions:
    def test_ssim_loss_of_identical_images_is_zero(self):
        x = torch.rand(2, 1, 64, 64)
        assert SSIMLoss()(x, x).item() < 0.01

    def test_ssim_loss_of_dissimilar_images_is_large(self):
        loss = SSIMLoss()(torch.zeros(2, 1, 64, 64), torch.ones(2, 1, 64, 64))
        assert loss.item() > 0.5

    def test_charbonnier_approximates_l1_away_from_zero(self):
        pred, target = torch.tensor([1.0]), torch.tensor([0.0])
        assert abs(CharbonnierLoss()(pred, target).item() - 1.0) < 1e-3

    def test_charbonnier_gradient_is_finite_at_zero(self):
        x = torch.zeros(4, requires_grad=True)
        CharbonnierLoss()(x, torch.zeros(4)).backward()
        assert torch.isfinite(x.grad).all()

    def test_frequency_loss_is_zero_for_identical_images(self):
        x = torch.rand(1, 1, 32, 32)
        assert FrequencyLoss()(x, x).item() < 1e-5

    def test_frequency_loss_penalises_high_frequency_error_more(self):
        """
        The whole point of the term. A high-frequency perturbation of the same
        energy must cost more than a low-frequency one.
        """
        base = torch.zeros(1, 1, 64, 64)
        yy, xx = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
        low = 0.1 * torch.sin(2 * math.pi * xx / 64).view(1, 1, 64, 64)
        high = 0.1 * torch.sin(2 * math.pi * 24 * xx / 64).view(1, 1, 64, 64)

        loss = FrequencyLoss(focus_high_freq=True)
        assert loss(base + high, base) > loss(base + low, base)

    def test_edge_loss_is_zero_for_identical_images(self):
        x = torch.rand(1, 1, 32, 32)
        assert EdgeLoss()(x, x).item() < 1e-5

    def test_edge_loss_detects_blur(self):
        import torch.nn.functional as F

        sharp = torch.zeros(1, 1, 32, 32)
        sharp[..., 16:, :] = 1.0
        blurred = F.avg_pool2d(sharp, 5, stride=1, padding=2)
        assert EdgeLoss()(blurred, sharp).item() > 0.01

    def test_combined_loss_is_nonnegative(self):
        loss = CombinedLoss()(torch.rand(2, 1, 64, 64), torch.rand(2, 1, 64, 64))
        assert loss.item() >= 0

    def test_combined_loss_reports_components(self):
        criterion = CombinedLoss(l1_weight=0.7, ssim_weight=0.3, edge_weight=0.1)
        criterion(torch.rand(2, 1, 32, 32), torch.rand(2, 1, 32, 32))
        assert set(criterion.last_components) == {"l1", "ssim", "edge"}

    def test_zero_weight_components_are_not_constructed(self):
        """Perceptual loss downloads VGG weights; it must stay off by default."""
        assert CombinedLoss().perceptual is None
        assert CombinedLoss().freq is None

    def test_legacy_lambda_kwargs_still_work(self):
        criterion = CombinedLoss(lambda1=0.9, lambda2=0.1)
        assert criterion.weights["l1"] == 0.9
        assert criterion.weights["ssim"] == 0.1

    def test_different_weights_give_different_losses(self):
        x, y = torch.rand(2, 1, 64, 64), torch.rand(2, 1, 64, 64)
        a = CombinedLoss(lambda1=0.9, lambda2=0.1)(x, y)
        b = CombinedLoss(lambda1=0.1, lambda2=0.9)(x, y)
        assert not torch.isclose(a, b, atol=1e-6)

    def test_all_zero_weights_is_rejected(self):
        with pytest.raises(ValueError, match="at least one positive"):
            CombinedLoss(l1_weight=0, ssim_weight=0)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_psnr_of_identical_images_is_finite_and_large(self):
        """
        Returning inf would poison any mean taken over a validation set, so the
        MSE is floored instead.
        """
        x = torch.rand(2, 1, 64, 64)
        value = float(psnr(x, x))
        assert math.isfinite(value) and value > 100

    def test_psnr_is_computed_per_image_not_per_batch(self):
        """
        The original averaged the MSE across the batch before the log. Because
        log is concave the two differ, and the gap depends on batch size, so
        models trained at different batch sizes were not comparable.
        """
        clean = torch.rand(1, 1, 32, 32)
        good = clean.clone()
        bad = torch.rand(1, 1, 32, 32)

        pred = torch.cat([good, bad])
        target = torch.cat([clean, clean])
        rng = torch.tensor([1.0, 1.0])

        per_image = psnr(pred, target, rng, reduce=False)
        assert float(psnr(pred, target, rng)) == pytest.approx(float(per_image.mean()), rel=1e-5)

        batch_mse = ((pred - target) ** 2).mean()
        batch_style = 10 * torch.log10(1.0 / batch_mse)
        assert float(psnr(pred, target, rng)) > float(batch_style)

    def test_psnr_respects_data_range(self):
        pred = torch.zeros(1, 1, 8, 8)
        target = torch.full((1, 1, 8, 8), 0.5)
        assert float(psnr(pred, target, data_range=1.0)) != float(
            psnr(pred, target, data_range=2.0)
        )

    def test_psnr_decreases_with_noise(self):
        clean = torch.rand(2, 1, 64, 64)
        low = float(psnr(clean + 0.01 * torch.randn_like(clean), clean, 1.0))
        high = float(psnr(clean + 0.10 * torch.randn_like(clean), clean, 1.0))
        assert low > high

    def test_ssim_of_identical_images_is_one(self):
        x = torch.rand(2, 1, 64, 64)
        assert float(ssim(x, x)) > 0.99

    def test_ssim_is_bounded(self):
        value = float(ssim(torch.rand(2, 1, 64, 64), torch.rand(2, 1, 64, 64)))
        assert -1.0 <= value <= 1.0

    def test_ssim_module_matches_functional(self):
        x, y = torch.rand(1, 1, 32, 32), torch.rand(1, 1, 32, 32)
        assert torch.allclose(SSIM()(x, y), ssim(x, y), atol=1e-6)

    def test_ssim_survives_constant_regions(self):
        """Variance can go slightly negative numerically; clamping prevents NaN."""
        value = ssim(torch.ones(1, 1, 32, 32), torch.ones(1, 1, 32, 32))
        assert torch.isfinite(value)

    def test_nmse_is_zero_for_identical_and_scale_invariant(self):
        x = torch.rand(2, 1, 32, 32)
        assert float(nmse(x, x)) < 1e-6
        y = torch.rand(2, 1, 32, 32)
        assert float(nmse(x, y)) == pytest.approx(float(nmse(10 * x, 10 * y)), rel=1e-4)

    def test_metrics_are_computed_in_float32(self):
        """fp16 cannot represent an MSE small enough for a high PSNR."""
        x = torch.rand(1, 1, 32, 32, dtype=torch.float16)
        y = x.clone()
        assert math.isfinite(float(psnr(x, y)))


class TestMetricAccumulator:
    def test_accumulates_per_sample_values(self):
        acc = MetricAccumulator()
        for _ in range(3):
            pred, target = torch.rand(2, 1, 32, 32), torch.rand(2, 1, 32, 32)
            acc.update(pred, target, data_range=torch.ones(2), loss=0.5)
        assert len(acc.psnr) == 6
        assert acc.compute()["n"] == 6

    def test_by_acceleration_breakdown(self):
        acc = MetricAccumulator()
        for R in (4, 8):
            acc.update(
                torch.rand(2, 1, 32, 32),
                torch.rand(2, 1, 32, 32),
                data_range=torch.ones(2),
                accelerations=torch.tensor([R, R]),
            )
        breakdown = acc.by_acceleration()
        assert set(breakdown) == {4, 8}
        assert breakdown[4]["n"] == 2

    def test_worst_returns_lowest_scoring_files(self):
        acc = MetricAccumulator()
        acc.update(
            torch.rand(2, 1, 32, 32),
            torch.rand(2, 1, 32, 32),
            data_range=torch.ones(2),
            fnames=["a.h5", "b.h5"],
        )
        worst = acc.worst(k=1)
        assert len(worst) == 1 and worst[0][0] in {"a.h5", "b.h5"}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


class TestStatistics:
    def test_bootstrap_ci_brackets_the_mean(self):
        rng = np.random.default_rng(0)
        values = rng.normal(30.0, 2.0, size=200)
        ci = bootstrap_ci(values)
        assert ci.low < ci.mean < ci.high
        assert abs(ci.mean - 30.0) < 0.6

    def test_bootstrap_ci_narrows_with_more_samples(self):
        rng = np.random.default_rng(0)
        small = bootstrap_ci(rng.normal(0, 1, 30))
        large = bootstrap_ci(rng.normal(0, 1, 3000))
        assert (large.high - large.low) < (small.high - small.low)

    def test_bootstrap_handles_degenerate_inputs(self):
        assert math.isnan(bootstrap_ci([]).mean)
        assert bootstrap_ci([5.0]).mean == 5.0

    def test_permutation_test_detects_a_real_difference(self):
        rng = np.random.default_rng(0)
        base = rng.normal(28.0, 2.0, 200)
        better = base + 2.0  # a consistent, paired improvement
        result = paired_permutation_test(better, base)
        assert result["p_value"] < 0.01
        assert result["mean_diff"] == pytest.approx(2.0, abs=0.1)

    def test_permutation_test_accepts_the_null(self):
        rng = np.random.default_rng(1)
        a = rng.normal(28.0, 2.0, 200)
        b = rng.normal(28.0, 2.0, 200)
        assert paired_permutation_test(a, b)["p_value"] > 0.05

    def test_p_value_is_never_exactly_zero(self):
        """A finite permutation count cannot support p = 0."""
        rng = np.random.default_rng(0)
        base = rng.normal(0, 0.01, 100)
        result = paired_permutation_test(base + 10.0, base, n_perm=100)
        assert result["p_value"] > 0

    def test_permutation_test_requires_paired_inputs(self):
        with pytest.raises(ValueError, match="equal-length"):
            paired_permutation_test([1, 2, 3], [1, 2])

    def test_holm_bonferroni_is_monotonic_and_conservative(self):
        adjusted = holm_bonferroni({"a": 0.01, "b": 0.02, "c": 0.04})
        assert adjusted["a"]["p_adjusted"] <= adjusted["b"]["p_adjusted"]
        assert adjusted["c"]["p_adjusted"] >= 0.04

    def test_compare_models_end_to_end(self):
        rng = np.random.default_rng(0)
        base = rng.normal(28.0, 2.0, 150)
        results = {
            "unet": {"psnr": base.tolist()},
            "swinunet": {"psnr": (base + 1.5).tolist()},
        }
        comparison = compare_models(results, metric="psnr", reference="unet")
        assert comparison["comparisons"]["swinunet"]["significant"]
        assert "swinunet" in format_comparison_table(comparison)


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------


class TestEMA:
    @staticmethod
    def _model():
        torch.manual_seed(0)
        return torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))

    def test_average_parameters_actually_swaps_weights(self):
        """
        Without @contextmanager the method returned a bare generator, the body
        never ran, and validation silently used raw training weights.
        """
        model = self._model()
        ema = EMAModel(model, decay=0.9, warmup=0)
        original = model[0].weight.detach().clone()

        with torch.no_grad():
            model[0].weight.add_(1.0)
        ema.update()

        with ema.average_parameters():
            inside = model[0].weight.detach().clone()
        after = model[0].weight.detach().clone()

        assert not torch.allclose(inside, after), "EMA weights were not swapped in"
        assert torch.allclose(after, original + 1.0), "original weights were not restored"

    def test_weights_are_restored_after_an_exception(self):
        model = self._model()
        ema = EMAModel(model, decay=0.9, warmup=0)
        with torch.no_grad():
            model[0].weight.add_(1.0)
        ema.update()
        before = model[0].weight.detach().clone()

        with pytest.raises(RuntimeError):
            with ema.average_parameters():
                raise RuntimeError("boom")

        assert torch.allclose(model[0].weight, before)

    def test_state_dict_round_trip(self):
        model = self._model()
        ema = EMAModel(model, decay=0.9, warmup=0)
        with torch.no_grad():
            model[0].weight.add_(2.0)
        ema.update()

        state = ema.state_dict()
        restored = EMAModel(self._model(), decay=0.9, warmup=0)
        restored.load_state_dict(state)

        assert restored.step_count == ema.step_count
        for key, value in ema.shadow.items():
            assert torch.allclose(restored.shadow[key], value)

    def test_state_dict_exposes_both_key_formats(self):
        """
        Inference and evaluation looked for ``shadow_params`` while the EMA wrote
        ``shadow``, so EMA weights were never loaded anywhere.
        """
        state = EMAModel(self._model(), warmup=0).state_dict()
        assert "shadow" in state and "shadow_params" in state

    def test_load_ema_weights_into_applies_by_name(self):
        source = self._model()
        ema = EMAModel(source, decay=0.0, warmup=0)
        with torch.no_grad():
            source[0].weight.fill_(3.0)
        ema.update()  # decay 0 => shadow tracks the live weights exactly

        target = self._model()
        assert load_ema_weights_into(target, ema.state_dict())
        assert torch.allclose(target[0].weight, torch.full_like(target[0].weight, 3.0))

    def test_load_ema_returns_false_without_state(self):
        assert not load_ema_weights_into(self._model(), None)
        assert not load_ema_weights_into(self._model(), {})

    def test_warmup_tracks_live_weights(self):
        model = self._model()
        ema = EMAModel(model, decay=0.999, warmup=5)
        with torch.no_grad():
            model[0].weight.fill_(7.0)
        ema.update()
        assert torch.allclose(ema.shadow["0.weight"], torch.full((4, 4), 7.0))

    def test_rejects_invalid_decay(self):
        with pytest.raises(ValueError, match="decay"):
            EMAModel(self._model(), decay=1.5)


# ---------------------------------------------------------------------------
# Schedulers
# ---------------------------------------------------------------------------


class TestSchedulers:
    @staticmethod
    def _optimizer(lr=1e-3):
        param = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.SGD([param], lr=lr)
        # PyTorch warns when a scheduler steps before the optimiser ever has;
        # one no-op step up front keeps the tests focused on schedule shape.
        param.grad = torch.zeros_like(param)
        opt.step()
        return opt

    def test_warmup_ramps_then_anneals(self):
        opt = self._optimizer(1e-3)
        sched = WarmupCosineScheduler(opt, warmup=10, total=100, eta_min=1e-6)
        lrs = []
        for _ in range(100):
            lrs.append(opt.param_groups[0]["lr"])
            sched.step()

        assert lrs[0] < lrs[5] < lrs[10]           # ramping up
        assert lrs[10] == pytest.approx(1e-3, rel=1e-3)  # peaks at base lr
        assert lrs[-1] < lrs[50] < lrs[10]          # annealing down
        assert lrs[-1] >= 1e-6

    def test_schedule_never_goes_negative(self):
        opt = self._optimizer()
        sched = WarmupCosineScheduler(opt, warmup=5, total=20)
        for _ in range(60):  # deliberately overrun the schedule
            sched.step()
            assert opt.param_groups[0]["lr"] >= 0

    def test_warmup_longer_than_total_is_clamped(self):
        opt = self._optimizer()
        with pytest.warns(UserWarning, match="warmup"):
            sched = WarmupCosineScheduler(opt, warmup=100, total=50)
        assert sched.warmup < sched.total

    @pytest.mark.parametrize(
        "name", ["warmup_cosine", "warmup_linear", "warmup_restart", "cosine", "step"]
    )
    def test_factory_builds_every_scheduler(self, name):
        sched = build_scheduler(self._optimizer(), name, total_steps=100)
        sched.step()
        assert sched is not None

    def test_factory_rejects_unknown_name(self):
        with pytest.raises(ValueError, match="Unknown scheduler"):
            build_scheduler(self._optimizer(), "magic", total_steps=10)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_loads_defaults(self):
        cfg = load_config()
        assert cfg.model.name == "swinunet"
        assert cfg.training.epochs == 50

    def test_programmatic_override(self):
        assert load_config(overrides={"training": {"epochs": 100}}).training.epochs == 100

    def test_cli_override_with_type_casting(self):
        cfg = load_config(cli_overrides=["training.lr=0.001", "model.name=unet"])
        assert cfg.training.lr == 0.001 and isinstance(cfg.training.lr, float)
        assert cfg.model.name == "unet"

    def test_cli_override_casts_bool_and_null(self):
        cfg = load_config(cli_overrides=["training.amp=false", "data.max_files=null"])
        assert cfg.training.amp is False
        assert cfg.data.max_files is None

    def test_cli_override_casts_lists(self):
        cfg = load_config(cli_overrides=["data.acceleration=[4,8]"])
        assert cfg.data.acceleration == [4, 8]

    def test_dot_access_is_nested(self):
        assert Config({"a": {"b": {"c": 42}}}).a.b.c == 42

    def test_missing_key_gives_a_helpful_error(self):
        with pytest.raises(AttributeError, match="Available"):
            Config({"a": 1}).nonexistent  # noqa: B018

    def test_to_dict_is_plain(self):
        plain = load_config().to_dict()
        assert isinstance(plain, dict) and not isinstance(plain, Config)

    def test_missing_config_file_is_an_error(self):
        """The original silently ignored an unreadable path and used defaults."""
        with pytest.raises(ConfigError, match="not found"):
            load_config("configs/does_not_exist.yaml")

    def test_malformed_override_is_rejected(self):
        with pytest.raises(ConfigError, match="Malformed override"):
            load_config(cli_overrides=["training.lr"])

    @pytest.mark.parametrize(
        "overrides,message",
        [
            ({"model": {"name": "resnet"}}, "unknown model"),
            ({"training": {"epochs": 0}}, "epochs"),
            ({"training": {"lr": 5.0}}, "lr"),
            ({"training": {"optimizer": "rmsprop"}}, "optimizer"),
            ({"data": {"center_fraction": 1.5}}, "center_fraction"),
            ({"data": {"mask_type": "spiral"}}, "mask_type"),
            ({"training": {"monitor_mode": "sideways"}}, "monitor_mode"),
        ],
    )
    def test_validation_rejects_bad_values(self, overrides, message):
        with pytest.raises(ConfigError, match=message):
            load_config(overrides=overrides)

    def test_unknown_key_is_rejected(self):
        """A typo'd key that silently keeps the default is undetectable later."""
        with pytest.raises(ConfigError, match="unknown key"):
            load_config(overrides={"training": {"learning_rate": 1e-4}})

    def test_dc_with_full_domain_is_rejected(self):
        """This combination makes data consistency mathematically invalid."""
        errors = validate_config(
            {
                "model": {"name": "unet", "data_consistency": {"enabled": True}},
                "data": {"undersample_domain": "full"},
            }
        )
        assert any("undersample_domain" in e for e in errors)

    def test_shipped_configs_are_valid(self):
        from pathlib import Path

        for path in sorted(Path("configs").glob("*.yaml")):
            if path.name == "default.yaml":
                continue
            load_config(str(path))  # raises on any problem


# ---------------------------------------------------------------------------
# Trainer integration
# ---------------------------------------------------------------------------


class TestTrainerIntegration:
    def _config(self, synthetic_data_dir, tmp_path, **extra):
        training = {
            "epochs": 2,
            "batch_size": 2,
            "amp": False,
            "warmup_epochs": 1,
            "val_split": 0.4,
            "patience": 5,
            "ema": {"enabled": True, "warmup_steps": 1},
        }
        training.update(extra.pop("training", {}))
        return load_config(
            overrides={
                "model": {
                    "name": "swinunet",
                    "params": {
                        "img_size": 32,
                        "patch_size": 2,
                        "embed_dim": 8,
                        "ws": 4,
                        "head_dim": 8,
                        "n_levels": 1,
                        "depths": [1, 1],
                        "drop_path_rate": 0.0,
                    },
                    **extra.pop("model", {}),
                },
                "data": {
                    "train_dir": synthetic_data_dir,
                    "val_dir": None,
                    "crop_size": [32, 32],
                    "num_workers": 0,
                    "persistent_workers": False,
                    "pin_memory": False,
                    "slice_mode": "all",
                    **extra.pop("data", {}),
                },
                "training": training,
                "logging": {
                    "output_dir": str(tmp_path),
                    "experiment_name": "test",
                    "tensorboard": False,
                    "wandb": False,
                    "save_top_k": 1,
                },
            }
        )

    def test_full_training_run(self, synthetic_data_dir, tmp_path):
        from training.train import Trainer

        summary = Trainer(self._config(synthetic_data_dir, tmp_path)).fit()

        assert summary["final_epoch"] == 2
        assert math.isfinite(summary["best_val_loss"])

        run_dir = tmp_path / "test"
        assert (run_dir / "history.json").exists()
        assert (run_dir / "config.yaml").exists()
        assert (run_dir / "manifest.json").exists()
        assert (run_dir / "summary.json").exists()
        assert (run_dir / "checkpoints" / "best.pt").exists()
        assert (run_dir / "checkpoints" / "last.pt").exists()

    def test_training_with_data_consistency(self, synthetic_data_dir, tmp_path):
        from training.train import Trainer

        cfg = self._config(
            synthetic_data_dir,
            tmp_path,
            model={
                "complex": True,
                "data_consistency": {"enabled": True, "mode": "cascade", "n_cascades": 2},
            },
        )
        trainer = Trainer(cfg)
        assert trainer.expects_kspace and trainer.model_in_channels == 2
        assert math.isfinite(trainer.fit()["best_val_loss"])

    def test_resume_restores_full_state(self, synthetic_data_dir, tmp_path):
        from training.train import Trainer

        cfg = self._config(synthetic_data_dir, tmp_path)
        first = Trainer(cfg)
        first.fit()
        checkpoint = tmp_path / "test" / "checkpoints" / "last.pt"

        cfg2 = self._config(synthetic_data_dir, tmp_path, training={"epochs": 4})
        second = Trainer(cfg2, resume_from=str(checkpoint))

        assert second.start_epoch == 3
        assert second.global_step == first.global_step
        assert second.best_score == pytest.approx(first.best_score)

        assert second.fit()["final_epoch"] == 4

    def test_start_epoch_beyond_total_returns_cleanly(self, synthetic_data_dir, tmp_path):
        """The original raised UnboundLocalError when the loop body never ran."""
        from training.train import Trainer

        trainer = Trainer(self._config(synthetic_data_dir, tmp_path))
        trainer.start_epoch = 99
        summary = trainer.fit()
        assert summary["final_epoch"] == 98

    def test_gradient_accumulation_runs(self, synthetic_data_dir, tmp_path):
        from training.train import Trainer

        cfg = self._config(
            synthetic_data_dir, tmp_path, training={"gradient_accumulation": 2, "epochs": 1}
        )
        assert math.isfinite(Trainer(cfg).fit()["best_val_loss"])

    def test_monitoring_a_maximised_metric(self, synthetic_data_dir, tmp_path):
        from training.train import Trainer

        cfg = self._config(
            synthetic_data_dir,
            tmp_path,
            training={"monitor": "val_ssim", "monitor_mode": "max", "epochs": 1},
        )
        trainer = Trainer(cfg)
        assert trainer.higher_is_better
        assert trainer.fit()["best_val_ssim"] > -math.inf
