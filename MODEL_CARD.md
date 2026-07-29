# Model Card — Lucid MRI Reconstruction

Following the model-card framework of Mitchell et al. (2019). This card covers
the architectures released in this repository, not any particular set of trained
weights: no weights are distributed, so the numbers a user obtains depend on the
data and configuration they train with.

## Model details

| | |
|---|---|
| **Task** | Reconstruct a magnitude MR image from retrospectively undersampled single-coil k-space |
| **Architectures** | U-Net, NormUNet, BT-UNet, SwinUNet; each optionally wrapped in an unrolled data-consistency cascade |
| **Input** | Zero-filled reconstruction (1-channel magnitude or 2-channel complex), plus measured k-space and sampling mask when data consistency is enabled |
| **Output** | Magnitude image, `(B, 1, H, W)` |
| **Version** | 2.0.0 |
| **License** | MIT (code). fastMRI data is governed separately by NYU Langone's data-use agreement. |
| **Contact** | See repository |

## Intended use

**Intended**: methods research on accelerated MRI reconstruction; benchmarking
architectures under a stated and verifiable protocol; teaching the relationship
between sampling, aliasing and reconstruction.

**Out of scope**: any clinical decision-making. These models have not been
evaluated by any regulatory body, have not been read by radiologists, and have
not been validated on prospectively undersampled acquisitions.

## Factors

Performance is expected to vary with:

- **Acceleration factor.** Difficulty rises sharply with R. Train and report per
  factor; `data.acceleration: [4, 8]` trains one model across both and the
  evaluator breaks results down by factor.
- **Contrast.** The fastMRI knee set contains proton-density weighted
  acquisitions with and without fat suppression, which have visibly different
  noise and texture.
- **Anatomy and field of view.** Models trained on knee data should not be
  assumed to transfer to brain, cardiac or abdominal imaging.
- **Sampling pattern.** A model trained only on random masks may degrade on
  equispaced ones, which produce coherent rather than incoherent aliasing.
- **Slice position.** `slice_mode: middle` selects the most anatomically
  informative slice per volume. Edge slices are harder and are excluded under
  that setting.

## Metrics

PSNR, SSIM and NMSE, computed **per image** against each target's own dynamic
range, then averaged. See `training/metrics.py` for why the per-image
formulation matters: a batch-averaged MSE inside the logarithm makes the
reported value depend on batch size.

Comparisons are reported with percentile bootstrap confidence intervals and
Holm-corrected paired permutation tests (`training/stats.py`). Point estimates
alone are not sufficient to establish that one architecture beats another.

## Training data

fastMRI single-coil knee: 973 training and 199 validation volumes
(Zbontar et al., 2018). Undersampling is **retrospective** — fully sampled
acquisitions are masked in software. Real accelerated acquisitions differ in
noise statistics and may exhibit inter-shot motion.

A synthetic phantom generator (`scripts/make_synthetic_data.py`) is provided so
the pipeline can be exercised without the dataset. Synthetic results are not
comparable to fastMRI benchmark numbers.

## Evaluation data

The held-out fastMRI validation split. If `data.val_dir` is absent the trainer
splits the training directory at the **slice** level and warns; with
`slice_mode: all` that allows adjacent slices from one patient on both sides of
the split, which inflates scores. Use a separate `val_dir` for anything you
intend to report.

## Ethical considerations

**Hallucination.** A learned reconstruction can synthesise structure that is
consistent with its training distribution but absent from the measurements. In
medical imaging this is the central risk: the output is plausible, sharp, and
wrong. The data-consistency layer removes that freedom on the measured
frequencies — the network cannot alter what the scanner recorded — but
unobserved frequencies remain a generative choice. Higher acceleration means
more of the image is inferred rather than measured.

**Distribution shift.** Reconstruction quality can degrade on scanners, coils,
sequences or pathologies unlike the training set, and degradation may be
visually subtle while being diagnostically significant. Rare pathology is by
construction under-represented in training data.

**Metric–diagnosis gap.** PSNR and SSIM correlate imperfectly with diagnostic
utility. A reconstruction can improve on both while losing a small lesion.

**Uncertainty.** Monte-Carlo dropout uncertainty maps are implemented
(`inference.reconstruct_with_uncertainty`) and indicate where the model is
interpolating rather than reconstructing. Their calibration has **not** been
evaluated; treat them as qualitative.

## Caveats and recommendations

- Single-coil only. Clinical scanners are multi-coil, where sensitivity encoding
  supplies information this pipeline does not model.
- Enabling data consistency requires `data.undersample_domain: cropped`, which
  restricts the field of view before undersampling. This makes the projection
  exact at the cost of a slightly idealised acquisition model.
- Always report the acceleration factor, centre fraction, mask type and slice
  mode alongside any metric. All four change the difficulty of the task, and all
  four are recorded in each run's `config.yaml` and `manifest.json`.
- Verify a checkpoint's provenance before trusting its numbers: every run
  records the git commit and whether the working tree was dirty.

## Quantitative analyses

No benchmark results are published with this card. The previously reported
figures for this project are not reproducible from the released code, and
corrections to the sampling mask and metric definitions mean results produced
under the old protocol are not comparable to results produced under the current
one. `paper/paper.pdf` documents the audit; its "Running without fastMRI"
section gives the commands to regenerate benchmark numbers.

## References

Mitchell et al., *Model Cards for Model Reporting*, FAT* 2019.
Zbontar et al., *fastMRI: An Open Dataset and Benchmarks for Accelerated MRI*, 2018.
