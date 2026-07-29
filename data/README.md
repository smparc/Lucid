# Data

## Option A — synthetic data (no download, works immediately)

Every part of this repository can be exercised without fastMRI:

```bash
python scripts/make_synthetic_data.py --out data/synthetic --volumes 16
python main.py train --config configs/smoke.yaml data.train_dir=data/synthetic
```

The generator writes HDF5 volumes in the fastMRI single-coil layout, with
smooth elliptical phantoms and a non-trivial phase ramp. The phantoms are not
anatomically meaningful, but they reproduce the two properties that matter for
testing a reconstruction pipeline: rapid k-space decay (so undersampling
produces real coherent aliasing rather than noise) and complex-valued images
(so the phase handling in the data-consistency layer is genuinely exercised).

Use this for development, CI and debugging. Use fastMRI for any number you
intend to report.

## Option B — fastMRI single-coil knee

1. Register and accept the data-use agreement at <https://fastmri.med.nyu.edu/>.
2. Download **Knee MRI → single-coil**: `knee_singlecoil_train` (~89 GB) and
   `knee_singlecoil_val` (~19 GB).
3. Extract so the layout is:

```
data/
├── knee_singlecoil_train/    # 973 volumes
│   ├── file1000001.h5
│   └── ...
└── knee_singlecoil_val/      # 199 volumes
    └── ...
```

These directories are git-ignored.

### File format

Each `.h5` volume contains:

| Key / attribute      | Shape / type                 | Meaning                                   |
|----------------------|------------------------------|-------------------------------------------|
| `kspace`             | `(slices, H, W)` complex64   | Fully sampled k-space, DC-centred         |
| `reconstruction_esc` | `(slices, 320, 320)` float32 | Reference magnitude reconstruction        |
| `max`                | float attribute              | Maximum of the reference reconstruction   |
| `acquisition`        | string attribute             | `CORPD_FBK` (PD) or `CORPDFS_FBK` (PD-FS) |

`H` is the readout direction and `W` the phase-encode direction. Undersampling
skips whole `W` columns, which is why every mask in `data/masks.py` is 1D.

Only `kspace` is required — targets are derived from it, so the pipeline stays
self-consistent with whatever preprocessing is configured.

## What the pipeline does to a slice

1. Load complex k-space for one slice.
2. Inverse FFT to a **complex** image; centre-crop to `crop_size`.
3. (Training only) flip/rotate the complex image.
4. Forward FFT of the cropped image.
5. Apply the undersampling mask.
6. Inverse FFT to the zero-filled reconstruction; divide everything by one scalar.

### Why cropping happens before undersampling

The obvious ordering — mask the full-FOV k-space, then crop the image — is a
faithful simulation, but it makes the measured coefficients unusable for data
consistency: after cropping, they are no longer the Fourier transform of the
image the network outputs, so writing them back is mathematically wrong.

Cropping first puts the measurements and the model's estimate in the same
space, which makes data consistency exact. Set
`data.undersample_domain: full` to recover the original behaviour; config
validation will then refuse to enable data consistency.

### Why augmentation happens before undersampling

Flipping or rotating an *already undersampled* pair moves the aliasing artefact
away from the phase-encode direction that produced it, so the network sees
artefact orientations no Cartesian acquisition can generate. Augmenting the
fully sampled complex image and undersampling afterwards keeps every training
sample a physically realisable acquisition.

## Configuration

| Key | Default | Notes |
|---|---|---|
| `data.acceleration` | `4` | Int, or a list like `[4, 8]` to train one model across factors |
| `data.center_fraction` | `0.08` | Low-frequency lines always acquired (use `0.04` at R=8) |
| `data.mask_type` | `random` | `random`, `equispaced`, or `magic` (golden-ratio) |
| `data.slice_mode` | `middle` | `middle`, `all`, or `range:start:end` |
| `data.normalization` | `zf_max` | `zf_max`, `attr_max` (original behaviour), or `none` |
| `data.undersample_domain` | `cropped` | `cropped` (exact DC) or `full` (legacy) |
| `data.cache_dataset` | `false` | Cache decoded complex images in RAM |

Masks are drawn fresh on every access in training mode and are a deterministic
function of the sample index in evaluation mode — so validation is identical
across epochs and runs, and "the loss improved" cannot mean "this epoch drew
easier masks".

All three mask families acquire `num_cols / acceleration` lines *in total*,
including the fully sampled centre. Verify with:

```python
from data.masks import build_mask, effective_acceleration
effective_acceleration(build_mask("equispaced", (640, 368), 0.08, 4, seed=0))  # ~4.0
```

## Slice selection and leakage

`slice_mode: middle` takes one slice per volume, matching the original study
(973 training / 199 validation samples). `slice_mode: all` gives roughly 35x
more data and is what you want for a serious run.

If `data.val_dir` is missing the trainer splits `train_dir` at the **slice**
level and warns. With `slice_mode: all` that means adjacent slices from one
patient can land on both sides of the split, which inflates validation scores.
Always use a separate `val_dir` for reported numbers.
