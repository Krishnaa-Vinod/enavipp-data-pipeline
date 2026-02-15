#!/usr/bin/env python3
"""Inspect dataset with history stacking (P past frames).

Loads the dataset with history=P, prints shapes, and saves
a grid of voxel-sum heatmaps for each of the P frames plus
an optional RGB montage.

Usage:
    python scripts/inspect_h5_history.py \
        --h5 data/processed/DSEC/thun_00_a_bins5_50ms_480x640_rgb_imu.h5 \
        --sequence thun_00_a \
        --history 5 \
        --out artifacts/inspect_h5_history
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description="Inspect H5 with history stacking")
    parser.add_argument("--h5", required=True)
    parser.add_argument("--sequence", default=None)
    parser.add_argument("--history", type=int, default=5)
    parser.add_argument("--history_stride", type=int, default=1)
    parser.add_argument("--sample_idx", type=int, default=None,
                        help="Sample index to visualize (default: middle)")
    parser.add_argument("--out", default="artifacts/inspect_h5_history")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Import dataset
    src_root = Path(__file__).resolve().parent.parent / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from enavipp.data.datasets import EnavippH5Dataset, collate_enavipp
    from torch.utils.data import DataLoader

    seqs = [args.sequence] if args.sequence else None
    ds = EnavippH5Dataset(
        args.h5,
        sequences=seqs,
        load_rgb=True,
        load_imu=True,
        history=args.history,
        history_stride=args.history_stride,
        boundary="drop",
    )

    P = args.history
    print(f"Dataset len: {len(ds)} (history={P}, stride={args.history_stride})")

    # Pick a sample
    idx = args.sample_idx if args.sample_idx is not None else len(ds) // 2
    sample = ds[idx]

    voxel = sample["voxel"]
    print(f"Voxel shape: {voxel.shape}")  # (P, C, H, W) or (C, H, W)
    print(f"t_start_us: {sample['t_start_us']}")
    print(f"t_end_us: {sample['t_end_us']}")
    print(f"meta: {sample['meta']}")

    if "rgb_left" in sample:
        print(f"RGB shape: {sample['rgb_left'].shape}")
    if "imu_data" in sample:
        print(f"IMU data shape: {sample['imu_data'].shape}")

    # ─── Voxel heatmap grid ───────────────────────────────────────
    if voxel.ndim == 4:
        # (P, C, H, W) -> sum over C for each P
        fig, axes = plt.subplots(1, P, figsize=(4 * P, 4))
        if P == 1:
            axes = [axes]
        for p in range(P):
            hm = voxel[p].sum(dim=0).numpy()
            axes[p].imshow(hm, cmap="hot")
            axes[p].set_title(f"t-{(P-1-p)*args.history_stride}")
            axes[p].axis("off")
        fig.suptitle(f"Voxel sum heatmaps (history={P}, sample={idx})", fontsize=13)
        fig.tight_layout()
        path_vox = out / f"history_voxel_P{P}_sample{idx}.png"
        fig.savefig(path_vox, dpi=120)
        plt.close(fig)
        print(f"Saved: {path_vox}")

    # ─── RGB montage ──────────────────────────────────────────────
    if "rgb_left" in sample and sample["rgb_left"].ndim == 4:
        rgb = sample["rgb_left"]  # (P, 3, H, W)
        fig, axes = plt.subplots(1, P, figsize=(4 * P, 4))
        if P == 1:
            axes = [axes]
        for p in range(P):
            img = rgb[p].permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            axes[p].imshow(img)
            axes[p].set_title(f"t-{(P-1-p)*args.history_stride}")
            axes[p].axis("off")
        fig.suptitle(f"RGB left montage (history={P}, sample={idx})", fontsize=13)
        fig.tight_layout()
        path_rgb = out / f"history_rgb_P{P}_sample{idx}.png"
        fig.savefig(path_rgb, dpi=120)
        plt.close(fig)
        print(f"Saved: {path_rgb}")

    # ─── Quick batch test ─────────────────────────────────────────
    dl = DataLoader(ds, batch_size=4, collate_fn=collate_enavipp, num_workers=0)
    batch = next(iter(dl))
    print(f"\nBatch test: voxel={batch['voxel'].shape}", end="")
    if "rgb_left" in batch:
        print(f", rgb={batch['rgb_left'].shape}", end="")
    print()

    ds.close()
    print(f"\nAll outputs saved to {out}/")


if __name__ == "__main__":
    main()
