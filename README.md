# 🧠 Accelerated MRI Reconstruction — SwinUNet

> Recovering high-quality MRI images from undersampled k-space data using a hybrid Swin Transformer + U-Net architecture.

---

## Overview

MRI scans are invaluable for diagnosis but notoriously slow to acquire. **Accelerated MRI** addresses this by collecting only a fraction of the raw frequency-domain data (k-space), then using machine learning to reconstruct a full-quality image — dramatically reducing patient scan time.

This project benchmarks three deep learning architectures for this reconstruction task and demonstrates that a **SwinUNet** model achieves a **33.1 dB PSNR** and **0.72 SSIM** on the fastMRI knee dataset — approximately **4 dB better** than the U-Net baseline — while preserving fine anatomical details critical for clinical diagnosis.

---

## Results at a Glance

| Architecture | Val. Loss | PSNR (dB) | SSIM |
|---|---|---|---|
| U-Net Baseline | 0.0496 | 28.03 | 0.6935 |
| Transformer at Bottleneck (BT-UNet) | 0.0412 | 29.87 | 0.7102 |
| **SwinUNet (Optimized)** | **0.0352** | **33.10** | **0.7274** |

---

## Architectures

### Baseline U-Net
A standard encoder-decoder architecture with skip connections. The encoder uses 4 downsampling stages with doubling feature channels (starting at 32), and the decoder mirrors this with transposed convolutions. Uses InstanceNorm2D + LeakyReLU throughout.

### Transformer at Bottleneck (BT-UNet)
Extends the U-Net by injecting a standard Transformer encoder at the bottleneck. The bottleneck features are flattened into tokens, processed through multi-head self-attention layers, then reshaped and fed into the decoder — adding global context at the network's most abstract layer.

### SwinUNet *(Best Model)*
Replaces the CNN backbone with a hierarchical **Swin Transformer** in a U-Net-like structure:

- **Patch Embedding** — Input divided into non-overlapping patches, projected to feature space
- **Encoder** — Swin Transformer blocks with patch merging (progressively reduces spatial resolution)
- **Bottleneck** — Deep Swin Transformer processing of most abstract features
- **Decoder** — Swin Transformer blocks with patch expanding layers
- **Skip Connections** — Encoder feature maps concatenated with decoder at each scale

**Key advantage:** Shifted window attention achieves **linear complexity** (vs. quadratic for standard transformers) by computing self-attention within local windows and shifting window partitions between layers — making high-resolution medical image processing feasible.

---

## Problem Formulation

Given undersampled k-space measurements:

$$y = M \odot \mathcal{F}(x)$$

where $\mathcal{F}$ is the Fourier transform, $M$ is a binary sampling mask, and $x$ is the fully-sampled image, the goal is to learn:

$$\hat{x} = f_\theta(y')$$

where $y'$ is the zero-filled reconstruction from the inverse Fourier transform of $y$, such that $\hat{x} \approx x$.

---

## Dataset & Preprocessing

**Dataset:** [fastMRI Single-Coil Knee](https://fastmri.org/) — 973 training / 199 validation volumes

| Step | Details |
|---|---|
| Slice Extraction | Middle slice per volume |
| Undersampling | RandomMaskFunc, center fraction 0.08, acceleration factor 4× |
| Reconstruction | Inverse FFT → absolute value (zero-filled) |
| Normalization | Divide by per-scan maximum |
| Cropping | Center crop to 320 × 320 px |

The central 8% of k-space lines are fully sampled to preserve low-frequency contrast information; the remaining lines are randomly subsampled.

---

## Training Details

| Setting | Value |
|---|---|
| Loss Function | L1 + (1 − SSIM) combined loss |
| Optimizer | Adam |
| Learning Rate | 5e-5 to 8e-5 (cosine annealing) |
| Regularization | Weight decay (1e-4), dropout in transformer layers |
| Augmentation | Random flips + rotations |
| Batch Size | 4–6 |
| Epochs | 50 |

---

## Model Efficiency

| Model | Parameters (M) | Inference Time (ms) |
|---|---|---|
| U-Net Baseline | 7.8 | 18.5 |
| SwinUNet-64 | 27.3 | 42.7 |
| SwinUNet-80 | 42.6 | 56.3 |

Despite higher compute cost, SwinUNet inference remains practical for clinical workflows where reconstruction quality takes priority.

---

## Key Findings

- **Window size 8 > 7** — Slightly larger attention windows capture more contextual structure
- **Base dim 64 > 80** — Increasing to 80 raised compute costs without accuracy gains
- **Learning rate sensitivity** — SwinUNet performs best at 8e-5; more sensitive than U-Net baseline
- **Data augmentation is critical** — Removing it accelerated early convergence but hurt generalization

---

## Tech Stack

- **Language:** Python
- **Framework:** PyTorch
- **Hardware:** NVIDIA GPU
- **Libraries:** fastMRI utilities, h5py, NumPy

---

## Project Structure

```
mri-reconstruction/
├── data/
│   └── preprocessing.py          # K-space loading, masking, normalization
├── models/
│   ├── unet.py                   # Baseline U-Net
│   ├── bt_unet.py                # Transformer-at-bottleneck variant
│   └── swinunet.py               # SwinUNet architecture
├── training/
│   ├── train.py                  # Training loop
│   └── evaluate.py               # PSNR, SSIM, loss evaluation
├── notebooks/
│   └── results_visualization.ipynb
└── README.md
```

---

## References

1. Cao, H., et al. *Swin-UNet: UNet-like Pure Transformer for Medical Image Segmentation.* arXiv:2105.05537, 2021.
2. Liu, Z., et al. *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.* ICCV, 2021.
3. Zbontar, J., et al. *fastMRI: An Open Dataset and Benchmarks for Accelerated MRI.* arXiv:1811.08839, 2018.
4. Dosovitskiy, A., et al. *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale.* arXiv:2010.11929, 2020.
5. Hammernik, K., et al. *Learning a Variational Network for Reconstruction of Accelerated MRI Data.* MRM, 2018.
