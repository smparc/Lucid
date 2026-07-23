"""
test_models.py
--------------
Unit tests for all model architectures.


Tests:
- Forward pass shape correctness
- Parameter count sanity
- Gradient flow
- Different input sizes
- Deterministic output with same input
"""


import pytest
import torch


from models.unet import UNet, count_parameters
from models.bt_unet import BTUNet
from models.swinunet import SwinUNet



# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def device():
    return torch.device("cpu")



@pytest.fixture
def dummy_input(device):
    return torch.randn(2, 1, 320, 320, device=device)



@pytest.fixture
def unet(device):
    return UNet(in_channels=1, out_channels=1, base_ch=32, n_levels=4).to(device)



@pytest.fixture
def bt_unet(device):
    return BTUNet(in_channels=1, out_channels=1, base_ch=32, n_levels=4,
                  tf_heads=8, tf_layers=2).to(device)



@pytest.fixture
def swinunet(device):
    return SwinUNet(img_size=320, patch_size=4, in_ch=1, out_ch=1,
                    embed_dim=64, ws=8, head_dim=8, n_levels=3).to(device)



# ---------------------------------------------------------------------------
# U-Net Tests
# ---------------------------------------------------------------------------


class TestUNet:
    def test_forward_shape(self, unet, dummy_input):
        out = unet(dummy_input)
        assert out.shape == (2, 1, 320, 320)


    def test_single_sample(self, unet, device):
        x = torch.randn(1, 1, 320, 320, device=device)
        out = unet(x)
        assert out.shape == (1, 1, 320, 320)


    def test_parameter_count(self, unet):
        n = count_parameters(unet)
        # U-Net with base_ch=32, n_levels=4 should be ~7-8M params
        assert 5_000_000 < n < 15_000_000


    def test_gradient_flow(self, unet, dummy_input):
        out = unet(dummy_input)
        loss = out.mean()
        loss.backward()
        # Check that all parameters have gradients
        for name, param in unet.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


    def test_deterministic(self, unet, dummy_input):
        unet.eval()
        with torch.no_grad():
            out1 = unet(dummy_input)
            out2 = unet(dummy_input)
        assert torch.allclose(out1, out2)


    def test_different_base_channels(self, device):
        for base_ch in [16, 32, 64]:
            model = UNet(base_ch=base_ch).to(device)
            x = torch.randn(1, 1, 320, 320, device=device)
            out = model(x)
            assert out.shape == (1, 1, 320, 320)



# ---------------------------------------------------------------------------
# BT-UNet Tests
# ---------------------------------------------------------------------------


class TestBTUNet:
    def test_forward_shape(self, bt_unet, dummy_input):
        out = bt_unet(dummy_input)
        assert out.shape == (2, 1, 320, 320)


    def test_parameter_count(self, bt_unet):
        n = count_parameters(bt_unet)
        # BT-UNet should have more params than base U-Net due to transformer
        assert n > 7_000_000


    def test_gradient_flow(self, bt_unet, dummy_input):
        out = bt_unet(dummy_input)
        loss = out.mean()
        loss.backward()
        for name, param in bt_unet.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


    def test_different_transformer_configs(self, device):
        for heads, layers in [(4, 2), (8, 4)]:
            model = BTUNet(tf_heads=heads, tf_layers=layers).to(device)
            x = torch.randn(1, 1, 320, 320, device=device)
            out = model(x)
            assert out.shape == (1, 1, 320, 320)



# ---------------------------------------------------------------------------
# SwinUNet Tests
# ---------------------------------------------------------------------------


class TestSwinUNet:
    def test_forward_shape(self, swinunet, dummy_input):
        out = swinunet(dummy_input)
        assert out.shape == (2, 1, 320, 320)


    def test_parameter_count(self, swinunet):
        n = count_parameters(swinunet)
        # SwinUNet-64 should be ~25-30M params
        assert 20_000_000 < n < 50_000_000


    def test_gradient_flow(self, swinunet, dummy_input):
        out = swinunet(dummy_input)
        loss = out.mean()
        loss.backward()
        has_grad = any(p.grad is not None for p in swinunet.parameters() if p.requires_grad)
        assert has_grad


    def test_different_embed_dims(self, device):
        for embed_dim in [32, 64]:
            model = SwinUNet(embed_dim=embed_dim, n_levels=2, ws=8, head_dim=8).to(device)
            x = torch.randn(1, 1, 320, 320, device=device)
            out = model(x)
            assert out.shape == (1, 1, 320, 320)


    def test_output_range(self, swinunet, device):
        """SwinUNet output should be reasonable for normalized inputs."""
        x = torch.rand(1, 1, 320, 320, device=device)  # Input in [0, 1]
        swinunet.eval()
        with torch.no_grad():
            out = swinunet(x)
        # Output should be finite
        assert torch.isfinite(out).all()