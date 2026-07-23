"""
test_training.py
----------------
Unit tests for training utilities.


Tests:
- Loss functions correctness
- Metrics computation
- Augmentation consistency
- Config loading
"""


import pytest
import torch
import torch.nn.functional as F
import tempfile
import os
import yaml


from training.train import (
    CombinedLoss,
    SSIMLoss,
    psnr,
    ssim_metric,
    augment,
    build_model,
)
from config import load_config, Config



class TestLossFunctions:
    def test_ssim_loss_perfect(self):
        """SSIM loss of identical images should be ~0."""
        x = torch.rand(2, 1, 64, 64)
        loss_fn = SSIMLoss()
        loss = loss_fn(x, x)
        assert loss.item() < 0.01


    def test_ssim_loss_different(self):
        """SSIM loss of very different images should be high."""
        x = torch.zeros(2, 1, 64, 64)
        y = torch.ones(2, 1, 64, 64)
        loss_fn = SSIMLoss()
        loss = loss_fn(x, y)
        assert loss.item() > 0.5


    def test_combined_loss_range(self):
        """Combined loss should be non-negative."""
        x = torch.rand(2, 1, 64, 64)
        y = torch.rand(2, 1, 64, 64)
        loss_fn = CombinedLoss(lambda1=0.7, lambda2=0.3)
        loss = loss_fn(x, y)
        assert loss.item() >= 0


    def test_combined_loss_weights(self):
        """Different weights should give different losses."""
        x = torch.rand(2, 1, 64, 64)
        y = torch.rand(2, 1, 64, 64)
        loss1 = CombinedLoss(lambda1=0.9, lambda2=0.1)(x, y)
        loss2 = CombinedLoss(lambda1=0.1, lambda2=0.9)(x, y)
        # They should be different (except in degenerate cases)
        assert not torch.isclose(loss1, loss2, atol=1e-6)



class TestMetrics:
    def test_psnr_identical(self):
        x = torch.rand(1, 1, 64, 64)
        p = psnr(x, x)
        assert p == float("inf")


    def test_psnr_range(self):
        x = torch.rand(1, 1, 64, 64)
        y = x + 0.01 * torch.randn_like(x)
        p = psnr(x, y)
        assert 20 < p < 60  # Reasonable PSNR for slightly noisy


    def test_ssim_identical(self):
        x = torch.rand(1, 1, 64, 64)
        s = ssim_metric(x, x)
        assert s > 0.99


    def test_ssim_range(self):
        x = torch.rand(1, 1, 64, 64)
        y = torch.rand(1, 1, 64, 64)
        s = ssim_metric(x, y)
        assert 0 <= s <= 1



class TestAugmentation:
    def test_augment_shape(self):
        x = torch.randn(4, 1, 320, 320)
        y = torch.randn(4, 1, 320, 320)
        x_aug, y_aug = augment(x, y)
        assert x_aug.shape == x.shape
        assert y_aug.shape == y.shape


    def test_augment_consistency(self):
        """Augmentation should be applied identically to input and target."""
        torch.manual_seed(42)
        x = torch.randn(1, 1, 64, 64)
        y = x.clone()  # Same as input
        x_aug, y_aug = augment(x, y)
        # If x==y and same transform applied, x_aug should equal y_aug
        assert torch.equal(x_aug, y_aug)



class TestConfig:
    def test_load_default_config(self):
        cfg = load_config()
        assert cfg.model.name == "swinunet"
        assert cfg.training.epochs == 50


    def test_config_override(self):
        cfg = load_config(overrides={"training": {"epochs": 100}})
        assert cfg.training.epochs == 100


    def test_config_cli_override(self):
        cfg = load_config(cli_overrides=["training.lr=0.001", "model.name=unet"])
        assert cfg.training.lr == 0.001
        assert cfg.model.name == "unet"


    def test_config_dot_access(self):
        cfg = Config({"a": {"b": {"c": 42}}})
        assert cfg.a.b.c == 42


    def test_build_model_unet(self):
        cfg = load_config(overrides={"model": {"name": "unet"}})
        model = build_model(cfg)
        assert isinstance(model, torch.nn.Module)


    def test_build_model_swinunet(self):
        cfg = load_config(overrides={"model": {"name": "swinunet"}})
        model = build_model(cfg)
        assert isinstance(model, torch.nn.Module)


    def test_build_model_invalid(self):
        cfg = load_config(overrides={"model": {"name": "invalid_model"}})
        with pytest.raises(ValueError):
            build_model(cfg)