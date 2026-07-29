"""
test_data.py
------------
Tests for mask generation, tensor utilities and the dataset.

The dataset tests run against real synthetic HDF5 volumes (see
``conftest.synthetic_data_dir``) rather than mocks, because the defects that
mattered most in this pipeline lived in the interaction between loading,
cropping, undersampling and normalisation — not in any one of them alone.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from data.masks import (
    build_mask,
    effective_acceleration,
    equispaced_mask,
    magic_mask,
    random_mask,
)
from data.preprocessing import (
    FastMRIKneeDataset,
    augment_complex,
    center_crop,
    center_crop_complex,
    collate,
    legacy_collate,
    normalize,
    to_tensor,
)
from models.fourier import complex_abs, fft2c_chan

# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------


class TestMaskGeneration:
    @pytest.mark.parametrize("fn", [random_mask, equispaced_mask, magic_mask])
    def test_shape_and_binary_values(self, fn):
        mask = fn((256, 320), center_fraction=0.08, acceleration=4, seed=0)
        assert mask.shape == (1, 320)
        assert torch.all((mask == 0) | (mask == 1))

    @pytest.mark.parametrize("fn", [random_mask, equispaced_mask, magic_mask])
    def test_centre_is_always_acquired(self, fn):
        num_cols, cf = 320, 0.08
        mask = fn((256, num_cols), center_fraction=cf, acceleration=4, seed=0)
        n_center = int(round(num_cols * cf))
        pad = (num_cols - n_center + 1) // 2
        assert torch.all(mask[0, pad : pad + n_center] == 1)

    @pytest.mark.parametrize("fn", [random_mask, equispaced_mask, magic_mask])
    @pytest.mark.parametrize("R", [2, 4, 8])
    def test_effective_acceleration_matches_nominal(self, fn, R):
        """
        The old equispaced mask set ``mask[::R] = 1`` and *then* added the
        centre, acquiring N/R + N*cf lines. At R=4, cf=0.08 that is an effective
        acceleration of 3.2x, not 4x -- a 20% denser acquisition than reported.
        """
        cf = 0.08 if R <= 4 else 0.04
        accs = [
            effective_acceleration(fn((640, 368), center_fraction=cf, acceleration=R, seed=s))
            for s in range(30)
        ]
        assert abs(float(np.mean(accs)) - R) < 0.35 * (R / 4), (
            f"{fn.__name__} at R={R} gave {np.mean(accs):.2f}"
        )

    def test_random_mask_is_reproducible_by_seed(self):
        assert torch.equal(random_mask((256, 320), seed=42), random_mask((256, 320), seed=42))

    def test_random_mask_varies_with_seed(self):
        assert not torch.equal(random_mask((256, 320), seed=1), random_mask((256, 320), seed=2))

    def test_equispaced_offset_varies(self):
        """A fixed sampling grid across the whole dataset invites overfitting."""
        masks = {
            tuple(equispaced_mask((256, 320), acceleration=4, seed=s)[0].tolist())
            for s in range(15)
        }
        assert len(masks) > 1

    def test_build_mask_dispatch(self):
        assert build_mask("random", (64, 64), seed=0).shape == (1, 64)
        with pytest.raises(ValueError, match="Unknown mask_type"):
            build_mask("spiral", (64, 64))

    def test_accepts_shared_generator(self):
        rng = np.random.default_rng(7)
        a = random_mask((64, 64), rng=rng)
        b = random_mask((64, 64), rng=rng)
        assert a.shape == b.shape  # consecutive draws, no seed collision


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------


class TestTensorUtilities:
    def test_to_tensor_from_complex(self):
        arr = np.random.randn(64, 80) + 1j * np.random.randn(64, 80)
        t = to_tensor(arr)
        assert t.shape == (64, 80, 2) and t.dtype == torch.float32
        assert np.allclose(t[..., 0].numpy(), arr.real, atol=1e-5)

    def test_to_tensor_from_real(self):
        arr = np.random.randn(64, 80, 2).astype(np.float32)
        assert to_tensor(arr).shape == (64, 80, 2)

    @pytest.mark.parametrize(
        "shape,crop", [((1, 400, 400), (320, 320)), ((1, 320, 320), (320, 320)),
                       ((1, 200, 200), (320, 320)), ((1, 500, 400), (320, 320))]
    )
    def test_center_crop_output_shape(self, shape, crop):
        assert center_crop(torch.randn(*shape), crop).shape == (1, *crop)

    def test_center_crop_is_centred(self):
        img = torch.zeros(1, 100, 100)
        img[0, 50, 50] = 1.0
        cropped = center_crop(img, (20, 20))
        assert cropped[0, 10, 10] == 1.0

    def test_center_crop_complex_preserves_pairs(self):
        img = torch.randn(64, 64, 2)
        out = center_crop_complex(img, (32, 32))
        assert out.shape == (32, 32, 2)

    def test_normalize_basic(self):
        normed, max_val = normalize(torch.tensor([0.0, 5.0, 10.0]))
        assert max_val == 10.0
        assert torch.allclose(normed, torch.tensor([0.0, 0.5, 1.0]))

    def test_normalize_with_explicit_max(self):
        normed, max_val = normalize(torch.tensor([0.0, 5.0, 10.0]), max_val=20.0)
        assert max_val == 20.0 and torch.allclose(normed[1], torch.tensor(0.25))

    def test_normalize_handles_zero_image(self):
        normed, max_val = normalize(torch.zeros(10, 10))
        assert max_val == 1.0 and torch.all(normed == 0)


class TestAugmentation:
    def test_preserves_shape_and_energy(self):
        img = torch.randn(32, 32, 2)
        out = augment_complex(img, np.random.default_rng(0))
        assert out.shape == img.shape
        # Flips and rotations are isometries.
        assert torch.allclose((out**2).sum(), (img**2).sum(), rtol=1e-5)

    def test_is_driven_by_the_supplied_rng(self):
        img = torch.randn(16, 16, 2)
        a = augment_complex(img, np.random.default_rng(3))
        b = augment_complex(img, np.random.default_rng(3))
        assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class TestDataset:
    def test_sample_contract(self, synthetic_data_dir):
        ds = FastMRIKneeDataset(synthetic_data_dir, crop_size=(64, 64), complex_input=True)
        sample = ds[0]
        assert sample["image"].shape == (2, 64, 64)
        assert sample["target"].shape == (1, 64, 64)
        assert sample["kspace"].shape == (2, 64, 64)
        assert sample["mask"].shape == (1, 64)
        assert sample["max_value"].ndim == 0
        assert isinstance(sample["fname"], str)

    def test_magnitude_mode_has_one_channel(self, synthetic_data_dir):
        ds = FastMRIKneeDataset(synthetic_data_dir, crop_size=(64, 64), complex_input=False)
        assert ds[0]["image"].shape == (1, 64, 64)

    def test_returned_kspace_matches_the_zero_filled_image(self, synthetic_data_dir):
        """
        The contract data consistency depends on: the measurements must be the
        exact Fourier transform of the supplied input at sampled locations, and
        zero elsewhere. Without this the DC layer writes back wrong values.
        """
        ds = FastMRIKneeDataset(
            synthetic_data_dir,
            crop_size=(64, 64),
            complex_input=True,
            undersample_domain="cropped",
        )
        s = ds[0]
        image, kspace = s["image"].unsqueeze(0), s["kspace"].unsqueeze(0)
        mask = s["mask"].view(1, 1, 1, -1)

        derived = fft2c_chan(image)
        assert ((derived - kspace) * mask).abs().max() < 1e-3
        assert (kspace * (1 - mask)).abs().max() < 1e-6

    def test_eval_masks_are_deterministic(self, synthetic_data_dir):
        """
        Validation must be identical every epoch, or "val loss improved" can
        just mean "this epoch drew easier masks".
        """
        ds = FastMRIKneeDataset(synthetic_data_dir, crop_size=(64, 64), train=False)
        assert torch.equal(ds[0]["mask"], ds[0]["mask"])
        assert torch.equal(ds[1]["image"], ds[1]["image"])

    def test_train_masks_are_resampled(self, synthetic_data_dir):
        ds = FastMRIKneeDataset(synthetic_data_dir, crop_size=(64, 64), train=True)
        masks = [ds[0]["mask"] for _ in range(8)]
        assert not all(torch.equal(masks[0], m) for m in masks[1:])

    def test_multi_acceleration_sampling(self, synthetic_data_dir):
        ds = FastMRIKneeDataset(
            synthetic_data_dir, crop_size=(64, 64), acceleration=[4, 8], train=True
        )
        seen = {int(ds[0]["acceleration"]) for _ in range(40)}
        assert seen == {4, 8}

    def test_slice_modes(self, synthetic_data_dir):
        middle = FastMRIKneeDataset(synthetic_data_dir, crop_size=(64, 64), slice_mode="middle")
        every = FastMRIKneeDataset(synthetic_data_dir, crop_size=(64, 64), slice_mode="all")
        assert len(every) > len(middle)
        assert len(middle) == 3  # one per volume

    def test_normalisation_modes(self, synthetic_data_dir):
        for mode in ("zf_max", "attr_max", "none"):
            ds = FastMRIKneeDataset(synthetic_data_dir, crop_size=(64, 64), normalization=mode)
            sample = ds[0]
            assert torch.isfinite(sample["image"]).all()
            assert float(sample["scale"]) > 0

    def test_zf_max_normalisation_bounds_the_input(self, synthetic_data_dir):
        ds = FastMRIKneeDataset(
            synthetic_data_dir, crop_size=(64, 64), complex_input=True, normalization="zf_max"
        )
        s = ds[0]
        assert float(complex_abs(s["image"].permute(1, 2, 0)).max()) <= 1.0 + 1e-5

    def test_full_domain_mode_still_works(self, synthetic_data_dir):
        ds = FastMRIKneeDataset(
            synthetic_data_dir, crop_size=(64, 64), undersample_domain="full"
        )
        assert ds[0]["image"].shape == (1, 64, 64)

    def test_cache_returns_equal_targets(self, synthetic_data_dir):
        ds = FastMRIKneeDataset(synthetic_data_dir, crop_size=(64, 64), cache=True)
        assert torch.allclose(ds[0]["target"], ds[0]["target"])

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            FastMRIKneeDataset(str(tmp_path / "nope"))

    def test_empty_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No readable"):
            FastMRIKneeDataset(str(tmp_path))

    def test_invalid_options_are_rejected(self, synthetic_data_dir):
        with pytest.raises(ValueError, match="undersample_domain"):
            FastMRIKneeDataset(synthetic_data_dir, undersample_domain="bogus")
        with pytest.raises(ValueError, match="normalization"):
            FastMRIKneeDataset(synthetic_data_dir, normalization="bogus")

    def test_metadata_and_describe(self, synthetic_data_dir):
        ds = FastMRIKneeDataset(synthetic_data_dir, crop_size=(64, 64))
        assert ds.get_metadata(0)["file"].endswith(".h5")
        assert ds.describe()["n_volumes"] == 3


class TestCollation:
    def test_collate_stacks_tensors_and_keeps_names(self, synthetic_data_dir):
        ds = FastMRIKneeDataset(synthetic_data_dir, crop_size=(64, 64))
        batch = collate([ds[i] for i in range(3)])
        assert batch["image"].shape[0] == 3
        assert isinstance(batch["fname"], list) and len(batch["fname"]) == 3

    def test_legacy_collate_returns_tuple(self, synthetic_data_dir):
        ds = FastMRIKneeDataset(synthetic_data_dir, crop_size=(64, 64))
        x, y = legacy_collate([ds[i] for i in range(2)])
        assert x.shape == (2, 1, 64, 64) and y.shape == (2, 1, 64, 64)
