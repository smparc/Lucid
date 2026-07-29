#!/usr/bin/env python
"""
Regenerate per-slice metric vectors from the ablation's saved checkpoints.

The ablation runner writes aggregate results and the statistics computed against
the magnitude baseline, but it strips the per-slice vectors before serialising
(they would bloat results.json). Those vectors are needed for the
capacity-matched paired test -- cascade versus equal-parameter control -- which
is the comparison the paper's central claim actually rests on.

Rather than rerun training, this reloads each reported checkpoint and
re-evaluates it. Evaluation is cheap, and because validation masks are a
deterministic function of the sample index, the vectors are identical to those
produced during the run.

Which seed to reload: the runner reports the median-SSIM seed, so this reads
`seed_ssim` from results.json and reloads the matching checkpoint. Reloading a
different seed would silently misalign the tables.

    python paper/collect_per_sample.py --runs <dir>/runs --results <dir> --data <val_dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.preprocessing import FastMRIKneeDataset, collate  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from training.evaluate import evaluate_model, load_model  # noqa: E402
from training.metrics import MetricAccumulator  # noqa: E402


def zero_filled_vectors(data_dir: str, crop: int) -> dict[str, list[float]]:
    """Per-slice metrics for the un-reconstructed input."""
    ds = FastMRIKneeDataset(
        data_dir, crop_size=(crop, crop), slice_mode="all", acceleration=4
    )
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate)
    acc = MetricAccumulator()
    for batch in loader:
        acc.update(batch["image"], batch["target"], data_range=batch["max_value"])
    return {"psnr": acc.psnr, "ssim": acc.ssim, "nmse": acc.nmse}


def median_seed(seed_ssim: list[float], seeds: list[int]) -> int:
    """The seed the runner reported: median by SSIM, matching its own selection."""
    order = sorted(range(len(seed_ssim)), key=lambda i: seed_ssim[i])
    return seeds[order[len(order) // 2]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="Directory of <name>_s<seed> run dirs")
    ap.add_argument("--results", required=True, help="Directory holding results.json")
    ap.add_argument("--data", required=True, help="Validation data directory")
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = json.loads((Path(args.results) / "results.json").read_text())

    per_sample: dict[str, dict[str, list[float]]] = {}

    for name, entry in results.items():
        if name == "zero-filled":
            per_sample[name] = zero_filled_vectors(args.data, args.crop)
            print(f"{name:<12} zero-filled baseline  n={len(per_sample[name]['psnr'])}")
            continue

        seed_ssim = entry.get("seed_ssim")
        if not seed_ssim:
            print(f"{name:<12} SKIP (no seed_ssim recorded)")
            continue

        seed = median_seed(seed_ssim, args.seeds)
        ckpt = Path(args.runs) / f"{name}_s{seed}" / "checkpoints" / "best.pt"
        if not ckpt.exists():
            print(f"{name:<12} SKIP (missing {ckpt})")
            continue

        model = load_model(None, ckpt, device)
        ds = FastMRIKneeDataset(
            args.data,
            crop_size=(args.crop, args.crop),
            slice_mode="all",
            acceleration=4,
            complex_input=int(getattr(model, "in_channels", 1)) == 2,
        )
        loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate)
        metrics = evaluate_model(model, loader, device)
        per_sample[name] = metrics["per_sample"]

        reported = entry["psnr_db"]
        recomputed = metrics["psnr_db"]
        flag = "" if abs(reported - recomputed) < 0.05 else "  <-- MISMATCH"
        print(
            f"{name:<12} seed {seed}  reported {reported:6.2f}  "
            f"recomputed {recomputed:6.2f}{flag}"
        )

    out = Path(args.results) / "per_sample.json"
    out.write_text(json.dumps(per_sample, indent=2))
    print(f"\nWrote {out} ({len(per_sample)} configurations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
