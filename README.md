# Lucid: Accelerated MRI Reconstruction


[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)


> Recovering high-quality MRI images from undersampled k-space data using a hybrid Swin Transformer + U-Net architecture.


MRI scans are invaluable for diagnosis but notoriously slow to acquire. **Lucid** addresses this by collecting only a fraction of the raw frequency-domain data (k-space), then using deep learning to reconstruct a full-quality image — dramatically reducing patient scan time.


---


## Results


| Architecture | Val. Loss | PSNR (dB) | SSIM | Params (M) | Inference (ms) |
|---|---|---|---|---|---|
| U-Net Baseline | 0.0496 | 28.03 | 0.6935 | 7.8 | 18.5 |
| BT-UNet | 0.0412 | 29.87 | 0.7102 | — | — |
| **SwinUNet (Best)** | **0.0352** | **33.10** | **0.7274** | 27.3 | 42.7 |


The SwinUNet achieves **~5 dB improvement** over the baseline U-Net, with superior preservation of fine anatomical details (cartilage boundaries, meniscal structures, bone margins).


---


## Quick Start


### Installation


```bash
# Clone the repository
git clone <repository-url>
cd vba_agent2


# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac


# Install dependencies
pip install -r requirements.txt


# Or install as a package (editable mode)
pip install -e ".[dev]"
```


### Sanity Check (No Data Required)


```bash
python main.py test_models
```


This runs a forward pass through all three architectures and verifies output shapes and parameter counts.


### Download Data


1. Register at [fastMRI](https://fastmri.med.nyu.edu/)
2. Download: `Knee MRI → Single-coil`
3. Place `.h5` files:


```
data/
├── knee_singlecoil_train/    ← training volumes
└── knee_singlecoil_val/      ← validation volumes
```


### Train


```bash
# Train SwinUNet with optimized config
python main.py train --config configs/swinunet.yaml


# Train U-Net baseline
python main.py train --config configs/unet.yaml


# Override config values via CLI
python main.py train --config configs/swinunet.yaml training.lr=5e-5 training.batch_size=6


# Resume from checkpoint
python main.py train --config configs/swinunet.yaml --resume outputs/swinunet/checkpoints/best.pt
```


### Evaluate


```bash
python main.py eval --model swinunet --ckpt outputs/swinunet_optimized/checkpoints/best.pt
```


### Benchmark & Export


```bash
# Benchmark inference speed
python main.py benchmark --ckpt outputs/swinunet_optimized/checkpoints/best.pt


# Export to ONNX for deployment
python main.py export --ckpt outputs/swinunet_optimized/checkpoints/best.pt --format onnx


# Export to TorchScript
python main.py export --ckpt outputs/swinunet_optimized/checkpoints/best.pt --format torchscript
```


---


## Project Structure


```
├── main.py                  # Unified CLI entry point
├── config.py                # YAML config loader with CLI overrides
├── inference.py             # Production inference pipeline + ONNX export
├── pyproject.toml           # Package metadata & tool config
├── requirements.txt         # Python dependencies
├── Dockerfile               # Containerized training/inference
│
├── configs/                 # Experiment configurations (YAML)
│   ├── default.yaml         # Base config (all defaults)
│   ├── swinunet.yaml        # SwinUNet optimized
│   ├── unet.yaml            # U-Net baseline
│   └── bt_unet.yaml         # BT-UNet
│
├── models/                  # Architecture implementations
│   ├── unet.py              # Baseline U-Net
│   ├── bt_unet.py           # U-Net + Transformer bottleneck
│   └── swinunet.py          # SwinUNet (best model)
│
├── data/                    # Data loading & preprocessing
│   ├── preprocessing.py     # FastMRI dataset, masks, FFT utils
│   └── README.md            # Data download instructions
│
├── training/                # Training & evaluation
│   ├── train.py             # Trainer class (AMP, logging, checkpoints)
│   └── evaluate.py          # Metrics, visualization, comparison
│
├── utils/                   # Utilities
│   ├── logger.py            # TensorBoard + W&B unified logger
│   └── reproducibility.py   # Seed management
│
├── tests/                   # Unit tests (pytest)
│   ├── test_models.py       # Model forward pass, gradients, shapes
│   ├── test_data.py         # Preprocessing, masks, FFT
│   └── test_training.py     # Loss, metrics, config, augmentation
│
└── notebooks/               # Analysis & visualization
    └── results_visualization.ipynb
```


---


## Architectures


### U-Net Baseline
Standard encoder-decoder with skip connections. 4 downsampling stages (32→64→128→256→512 channels), InstanceNorm2D + LeakyReLU, ConvTranspose2d for upsampling.


### BT-UNet (Transformer at Bottleneck)
Extends U-Net by injecting a standard Transformer encoder at the bottleneck. Features are flattened into tokens, processed through multi-head self-attention, then reshaped for the decoder — adding global context at the most abstract layer.


### SwinUNet (Best Model)
Replaces the CNN backbone with a hierarchical **Swin Transformer** in a U-Net-like structure:


- **Patch Embedding** — Non-overlapping patches projected to feature space
- **Shifted Window Attention** — Linear complexity self-attention via local windows with shifted partitioning
- **Hierarchical Encoder/Decoder** — Patch merging/expanding for multi-scale features
- **Skip Connections** — Encoder features concatenated at each decoder scale


Key advantage: **O(n)** complexity vs O(n²) for standard attention, making high-resolution medical image processing feasible.


---


## Training Features


| Feature | Description |
|---|---|
| **Mixed Precision (AMP)** | FP16 training for 2x speedup on modern GPUs |
| **Config-Driven** | YAML configs with dot-notation CLI overrides |
| **Experiment Tracking** | TensorBoard + Weights & Biases |
| **Early Stopping** | Patience-based on validation loss |
| **Cosine Annealing** | LR schedule with warm restarts |
| **Gradient Clipping** | Stability for transformer training |
| **Checkpoint Management** | Best model + per-epoch saves |
| **Resume Training** | Continue from any checkpoint |
| **Augmentation** | Random flips + 90° rotations |


---


## Configuration


All hyperparameters are managed via YAML configs in `configs/`. The system supports:


1. **Base config** (`configs/default.yaml`) — all defaults
2. **Experiment config** — overrides base (e.g., `configs/swinunet.yaml`)
3. **CLI overrides** — highest priority, dot-notation: `training.lr=1e-4`


Example config:
```yaml
model:
  name: swinunet
  params:
    embed_dim: 64
    ws: 8


training:
  epochs: 50
  batch_size: 6
  lr: 5.0e-5
  amp: true


logging:
  tensorboard: true
  wandb: true
  wandb_project: lucid-mri
```


---


## Deployment


### ONNX Export
```python
from inference import MRIReconstructionPipeline


pipe = MRIReconstructionPipeline.from_checkpoint("checkpoints/best.pt")
pipe.export_onnx("exports/swinunet.onnx")
```


### Python Inference API
```python
from inference import MRIReconstructionPipeline


pipe = MRIReconstructionPipeline.from_checkpoint("checkpoints/best.pt")
reconstruction = pipe.reconstruct(undersampled_image)
reconstruction, time_ms = pipe.reconstruct(undersampled_image, return_time=True)
```


---


## Testing


```bash
# Run all tests
pytest tests/ -v


# Run with coverage
pytest tests/ --cov=models --cov=data --cov=training


# Run specific test file
pytest tests/test_models.py -v
```


---


## Docker


```bash
# Build
docker build -t lucid-mri .


# Train
docker run --gpus all -v ./data:/app/data -v ./outputs:/app/outputs \
    lucid-mri train --config configs/swinunet.yaml


# Inference
docker run --gpus all lucid-mri benchmark --ckpt /app/outputs/best.pt
```


---


## Problem Formulation


Given undersampled k-space measurements:


$$y = M \odot \mathcal{F}(x)$$


where $\mathcal{F}$ is the Fourier transform, $M$ is a binary sampling mask, and $x$ is the fully-sampled image, the goal is to learn:


$$\hat{x} = f_\theta(y')$$


where $y'$ is the zero-filled reconstruction from the inverse Fourier transform of $y$, such that $\hat{x} \approx x$.


---


## Citation


If you use this code in your research, please cite:


```bibtex
u/misc{lucid2024,
  title={Lucid: Accelerated MRI Reconstruction with SwinUNet},
  author={Lucid Team},
  year={2024},
  url={https://github.com/lucid-mri}
}
```


---


## License


MIT License. See [LICENSE](LICENSE) for details.
