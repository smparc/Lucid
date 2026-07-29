# Lucid: Accelerated MRI Reconstruction

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> Recovering high-quality MRI images from undersampled k-space using a
> hierarchical Swin Transformer, constrained by the physics of the acquisition.

MRI is diagnostically invaluable and slow to acquire. **Lucid** collects only a
fraction of the raw frequency-domain data (k-space) and reconstructs a
full-quality image from it, cutting scan time proportionally.

The distinguishing feature is that reconstruction is not treated as generic
image restoration. We know exactly which Fourier coefficients the scanner
measured, so an unrolled **data-consistency** cascade restores them after every
learned refinement. The network can only ever modify frequencies that were never
acquired — the constraint that separates reconstruction from plausible-looking
inpainting.

---

## Quick start

```bash
git clone <repository-url> && cd Lucid
python -m venv .venv && .venv\Scripts\activate      # Linux/macOS: source .venv/bin/activate
pip install -e ".[dev,tracking,export]"
```

Verify every architecture builds and runs — no data required:

```bash
python main.py test_models
```

Train end to end without downloading anything:

```bash
python scripts/make_synthetic_data.py --out data/synthetic --volumes 16
python main.py train --config configs/smoke.yaml data.train_dir=data/synthetic
```

For real experiments see [`data/README.md`](data/README.md) to obtain fastMRI,
then:

```bash
# Strongest configuration: SwinUNet + unrolled data consistency
python main.py train --config configs/swinunet_dc.yaml

# Image-domain baselines
python main.py train --config configs/swinunet.yaml
python main.py train --config configs/unet.yaml
python main.py train --config configs/bt_unet.yaml

# Override anything from the CLI (dot notation, validated)
python main.py train --config configs/swinunet.yaml training.lr=5e-5 data.acceleration=[4,8]

# Resume exactly where a run stopped
python main.py train --config configs/swinunet.yaml --resume outputs/swinunet/checkpoints/last.pt
```

Evaluate, compare, benchmark, export:

```bash
python main.py eval      --ckpt outputs/swinunet_dc/checkpoints/best.pt
python main.py compare   --ckpt_dir outputs          # with significance tests
python main.py benchmark --ckpt outputs/swinunet_dc/checkpoints/best.pt
python main.py export    --ckpt outputs/swinunet/checkpoints/best.pt --format onnx
python main.py curves    --run outputs/swinunet_dc
```

Every command returns a non-zero exit code on failure.

---

## Reproducing results

This repository does not ship trained weights, and **no benchmark numbers are
quoted here**. Earlier versions of this README carried a results table whose
headline configuration could not be constructed at all (see below), so numbers
now come only from runs you can reproduce.

Run the suite yourself:

```bash
for cfg in unet bt_unet swinunet swinunet_dc; do
    python main.py train --config configs/$cfg.yaml
done
python main.py compare --ckpt_dir outputs --metric psnr --reference unet_baseline
```

`compare` prints bootstrap confidence intervals and Holm-corrected paired
permutation tests, so the output states whether a difference is resolvable
rather than only which number is larger.

The full write-up — the audit of the previous implementation, the corrected
methodology, and a controlled ablation isolating the data-consistency
contribution — is in [`paper/paper.pdf`](paper/paper.pdf) (LaTeX source
alongside it; rebuild with `paper/build.sh`). Intended use, factors and
limitations are documented in [`MODEL_CARD.md`](MODEL_CARD.md).

---

## Architecture

### SwinUNet (primary)

A hierarchical Swin Transformer in a U-Net encoder–decoder:

- **Patch embedding** — non-overlapping patches projected to `embed_dim`.
- **Shifted-window attention** — attention within `ws × ws` windows, alternating
  regular and shifted partitioning. Cost is **O(n)** in image area rather than
  O(n²), which is what makes 320×320 feasible.
- **Patch merging / expanding** — hierarchical down- and upsampling.
- **Skip connections** — encoder features fused at each decoder scale.
- **Global residual** — predicts a correction to the zero-filled input, with a
  zero-initialised output layer so training begins from the exact identity.

Feature maps are padded to the window size, so any input resolution works and
one trained model runs at any size.

### Data consistency

```
k_pred     = F(x_pred)
k_dc[m=1]  = λ·k_measured + (1−λ)·k_pred     # trust the scanner
k_dc[m=0]  = k_pred                          # network fills the gaps
x_dc       = F⁻¹(k_dc)
```

`λ` is learnable and initialised at hard consistency. `CascadedNet` interleaves
`n_cascades` denoisers with this operator — one unrolled iteration of a
proximal-gradient solver whose proximal step has been learned.

Data consistency requires phase, so it forces complex (2-channel) I/O. Config
validation enforces this rather than silently degrading.

### Baselines

- **U-Net** — 4 levels, InstanceNorm + LeakyReLU. `NormUNet` adds the fastMRI
  instance-normalisation wrapper for exact scale equivariance.
- **BT-UNet** — U-Net with a Transformer encoder at the bottleneck. Positional
  embeddings interpolate to any bottleneck grid.

---

## What changed in v2

The original codebase could not run. Three independent defects made every
documented entry point fail:

| Defect | Consequence |
|---|---|
| `SwinUNet` reshaped windows without padding | At the documented config (320px, patch 4, 3 levels, window 8) stage 3 is 20×20 and 20 % 8 ≠ 0. Construction raised `RuntimeError`, so **the headline model never ran**. |
| `utils/reproductibility.py` misspelled | `utils/__init__.py` imported `utils.reproducibility`, so importing `training` raised `ModuleNotFoundError`. |
| `from_checkpoint` / `_build_model` missing decorators | Plain functions on the class; `from_checkpoint(path)` bound `path` to `cls`. Every inference, benchmark and export path raised immediately. |

And several that ran but produced wrong or meaningless results:

| Defect | Consequence |
|---|---|
| `EMAModel.average_parameters` lacked `@contextmanager` | The body never executed; validation silently used raw training weights. |
| EMA wrote `shadow`, loaders read `shadow_params` | EMA weights were never loaded anywhere, silently. |
| `ResidualDCWrapper` skipped DC when k-space was `None` | The trainer only ever passed the image, so `data_consistency.enabled: true` was a **complete no-op**. |
| DC returned `x.abs()` | Phase discarded, so cascading was impossible. |
| `sigmoid(lambda_init)` | A requested hard consistency of 1.0 became 0.73. |
| `equispaced_mask` added the centre on top of `mask[::R]` | Effective acceleration 3.2× at a nominal 4× — a 20% denser acquisition than reported. |
| PSNR averaged MSE across the batch before the log | Values depended on batch size, so the U-Net (batch 8) and SwinUNet (batch 6) were not comparable. |
| Metrics used a hard-coded data range of 1.0 | Wrong denominator under per-slice scaling. |
| Validation ran inside `autocast` | Metrics computed in fp16. |
| Augmentation applied after undersampling | Rotated the aliasing away from the phase-encode axis that produced it. |
| Epoch metrics logged at `step=epoch`, batch at `global_step` | W&B rejects out-of-order steps; epoch curves were dropped. |
| `resume` restored no patience or best score | Resuming reset early stopping and could overwrite a better checkpoint. |
| `save_top_k` retained by mtime | Kept the *newest* k, not the *best* k. |
| `.github/.workflows/ci.yml` | GitHub only reads `.github/workflows/`; CI had never run. |
| `test_models` caught all exceptions and exited 0 | The sanity job passed green while nothing worked. |
| `AttentionExtractor` read weights from the forward output | `WindowAttention` returns a tensor; the hook matched nothing and every attention figure was empty. |

Additions: unrolled DC cascade, complex I/O, per-image metrics with NMSE,
bootstrap CIs and paired permutation tests, DropPath, truncated-normal init,
fused attention, gradient checkpointing, multi-acceleration training, TTA,
MC-dropout uncertainty, config validation, a synthetic data generator, run
manifests with git SHA, a model card, and 267 tests at 88% coverage.

---

## Project structure

```
├── main.py                   # CLI with real exit codes
├── config.py                 # Layered YAML config with schema validation
├── inference.py              # Inference, TTA, uncertainty, ONNX/TorchScript
│
├── configs/
│   ├── default.yaml          # Every default, documented
│   ├── swinunet_dc.yaml      # Strongest: SwinUNet + DC cascade
│   ├── swinunet.yaml         # Image-domain SwinUNet
│   ├── unet.yaml             # U-Net baseline
│   ├── bt_unet.yaml          # Transformer bottleneck
│   └── smoke.yaml            # Minutes-long end-to-end test
│
├── models/
│   ├── swinunet.py           # Swin Transformer U-Net
│   ├── unet.py               # U-Net and NormUNet
│   ├── bt_unet.py            # Transformer bottleneck
│   ├── data_consistency.py   # DC layer and unrolled cascade
│   ├── fourier.py            # Single source of truth for FFT conventions
│   ├── layers.py             # DropPath, ConvBlock, init
│   └── registry.py           # One place that builds models
│
├── data/
│   ├── preprocessing.py      # Physics-aware dataset
│   ├── masks.py              # Random / equispaced / golden-ratio masks
│   └── README.md
│
├── training/
│   ├── train.py              # Trainer
│   ├── evaluate.py           # Evaluation and comparison
│   ├── losses.py             # L1, SSIM, frequency, edge, perceptual
│   ├── metrics.py            # Per-image PSNR / SSIM / NMSE
│   └── stats.py              # Bootstrap CIs, permutation tests, Holm
│
├── utils/
│   ├── logger.py             # TensorBoard + W&B + history.json
│   ├── ema.py                # Exponential moving average
│   ├── schedulers.py         # Warmup schedules
│   ├── reproducibility.py    # Seeding, worker seeding, run manifest
│   └── visualizations.py     # Figures and attention maps
│
├── MODEL_CARD.md              # Intended use, factors, limitations
├── paper/                     # LaTeX source and built PDF of the write-up
├── scripts/make_synthetic_data.py
└── tests/                    # 267 tests, ~36 s, 88% coverage
```

---

## Configuration

Precedence: `configs/default.yaml` → experiment YAML → programmatic → CLI.

```yaml
model:
  name: swinunet
  complex: true
  params: { embed_dim: 48, ws: 8, n_levels: 3 }
  data_consistency: { enabled: true, mode: cascade, n_cascades: 4 }

data:
  acceleration: [4, 8]          # one model across both factors
  undersample_domain: cropped   # required for exact DC

training:
  epochs: 50
  lr: 4.0e-5
  monitor: val_ssim
  monitor_mode: max
```

Configs are validated on load. Unknown keys are errors, not silent no-ops — a
typo'd `learning_rate` that quietly keeps the default is indistinguishable from
a successful override until you compare two runs and find them identical.

---

## Training features

| Feature | Notes |
|---|---|
| Data consistency | Unrolled cascade; measured k-space restored every stage |
| Mixed precision | bf16 preferred; FFTs and metrics forced to fp32 |
| EMA | Validated and checkpointed on the averaged weights |
| Per-step LR schedule | Warmup resolved in steps, not epochs |
| Multi-acceleration | Train once across R ∈ {4, 8}, reported per factor |
| Gradient accumulation | Decouples effective batch from memory |
| Checkpointing | Top-k by monitored metric, plus `best.pt` and `last.pt` |
| Resume | Optimiser, scheduler, scaler, EMA, patience and best score |
| Reproducibility | Per-worker seeding; run manifest records git SHA and hardware |
| Failure analysis | Worst-slice reporting and per-acceleration breakdown |

---

## Deployment

```python
from inference import MRIReconstructionPipeline

pipe = MRIReconstructionPipeline.from_checkpoint("outputs/swinunet/checkpoints/best.pt")

recon = pipe.reconstruct(zero_filled_image)
recon = pipe.reconstruct(zero_filled_image, tta=True)          # dihedral averaging
mean, std = pipe.reconstruct_with_uncertainty(zero_filled_image)  # MC dropout

pipe.export_onnx("exports/swinunet.onnx")   # verified against PyTorch output
```

Exports are numerically verified against the PyTorch model; an export that
produces a *different* model is more dangerous than one that fails outright.

---

## Testing

```bash
pytest tests/ -q                    # 267 tests, ~36 s, 88% coverage
pytest tests/ --cov --cov-report=term
pytest tests/test_physics.py -v     # Fourier and data-consistency exactness
```

`tests/test_physics.py` is the scientifically load-bearing file: it asserts that
the Fourier pair round-trips exactly, that hard data consistency restores
measured coefficients to floating-point precision, and that the cascade's final
output still agrees with the measurements. A broken DC layer still trains and
still produces a falling loss curve — only an exactness assertion catches it.

---

## Problem formulation

Given undersampled measurements

$$y = M \odot \mathcal{F}(x)$$

with $\mathcal{F}$ the Fourier transform, $M$ a binary sampling mask and $x$ the
fully sampled image, we learn $f_\theta$ such that

$$\hat{x} = f_\theta(y', M, y) \approx x$$

where $y' = \mathcal{F}^{-1}(y)$ is the zero-filled reconstruction. Passing $M$
and $y$ — not just $y'$ — is what makes the data-consistency constraint
expressible.

---

## Citation

```bibtex
@misc{lucid2026,
  title  = {Lucid: Physics-Informed Accelerated MRI Reconstruction with SwinUNet},
  author = {Park, Matthew},
  year   = {2026},
  url    = {https://github.com/lucid-mri}
}
```

## License

MIT — see [LICENSE](LICENSE). **Not for clinical use.**
