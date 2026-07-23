"""
test_data.py
------------
Unit tests for data preprocessing utilities.


Tests:
- Mask generation correctness
- FFT/IFFT roundtrip
- Normalization
- Center cropping edge cases
- Tensor conversion
"""


import pytest
import numpy as np
import torch


from data.preprocessing import (
    random_mask,
    equispaced_mask,
    ifft2c,
    complex_abs,
    to_tensor,
    center_crop,
    normalize,
)



class TestMaskGeneration:
    def test_random_mask_shape(self):
        mask = random_mask((256, 320), center_fraction=0.08, acceleration=4)
        assert mask.shape == (1, 320)


    def test_random_mask_values(self):
        mask = random_mask((256, 320), center_fraction=0.08, acceleration=4, seed=42)
        # Mask should be binary
        assert torch.all((mask == 0) | (mask == 1))


    def test_random_mask_center_sampled(self):
        mask = random_mask((256, 320), center_fraction=0.08, acceleration=4, seed=42)
        num_cols = 320
        num_center = int(round(num_cols * 0.08))
        pad = (num_cols - num_center + 1) // 2
        center_mask = mask[0, pad: pad + num_center]
        # All center frequencies should be sampled
        assert torch.all(center_mask == 1)


    def test_random_mask_reproducible(self):
        m1 = random_mask((256, 320), seed=42)
        m2 = random_mask((256, 320), seed=42)
        assert torch.equal(m1, m2)


    def test_random_mask_different_seeds(self):
        m1 = random_mask((256, 320), seed=42)
        m2 = random_mask((256, 320), seed=123)
        assert not torch.equal(m1, m2)


    def test_equispaced_mask_shape(self):
        mask = equispaced_mask((256, 320), center_fraction=0.08, acceleration=4)
        assert mask.shape == (1, 320)


    def test_equispaced_mask_acceleration(self):
        mask = equispaced_mask((256, 320), center_fraction=0.08, acceleration=4)
        # Every 4th line should be sampled (plus center)
        assert mask[0, 0] == 1
        assert mask[0, 4] == 1
        assert mask[0, 8] == 1



class TestFFTUtilities:
    def test_ifft2c_shape(self):
        kspace = torch.randn(256, 320, 2)
        image = ifft2c(kspace)
        assert image.shape == (256, 320, 2)


    def test_complex_abs_shape(self):
        x = torch.randn(256, 320, 2)
        mag = complex_abs(x)
        assert mag.shape == (256, 320)


    def test_complex_abs_nonnegative(self):
        x = torch.randn(100, 100, 2)
        mag = complex_abs(x)
        assert torch.all(mag >= 0)


    def test_to_tensor_complex(self):
        kspace = np.random.randn(256, 320) + 1j * np.random.randn(256, 320)
        t = to_tensor(kspace)
        assert t.shape == (256, 320, 2)
        assert t.dtype == torch.float32


    def test_to_tensor_real(self):
        data = np.random.randn(256, 320, 2).astype(np.float32)
        t = to_tensor(data)
        assert t.shape == (256, 320, 2)



class TestCenterCrop:
    def test_basic_crop(self):
        img = torch.randn(1, 400, 400)
        cropped = center_crop(img, (320, 320))
        assert cropped.shape == (1, 320, 320)


    def test_no_crop_needed(self):
        img = torch.randn(1, 320, 320)
        cropped = center_crop(img, (320, 320))
        assert cropped.shape == (1, 320, 320)


    def test_smaller_input(self):
        img = torch.randn(1, 200, 200)
        cropped = center_crop(img, (320, 320))
        # Should pad then crop
        assert cropped.shape == (1, 320, 320)


    def test_asymmetric_crop(self):
        img = torch.randn(1, 500, 400)
        cropped = center_crop(img, (320, 320))
        assert cropped.shape == (1, 320, 320)



class TestNormalization:
    def test_normalize_basic(self):
        img = torch.tensor([0.0, 5.0, 10.0])
        normed, max_val = normalize(img)
        assert max_val == 10.0
        assert torch.allclose(normed, torch.tensor([0.0, 0.5, 1.0]))


    def test_normalize_with_max(self):
        img = torch.tensor([0.0, 5.0, 10.0])
        normed, max_val = normalize(img, max_val=20.0)
        assert max_val == 20.0
        assert torch.allclose(normed, torch.tensor([0.0, 0.25, 0.5]))


    def test_normalize_zero_image(self):
        img = torch.zeros(10, 10)
        normed, max_val = normalize(img)
        assert max_val == 1.0
        assert torch.all(normed == 0)