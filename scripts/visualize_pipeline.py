#!/usr/bin/env python3
"""
Comprehensive visualization of the ENavi++ preprocessed data pipeline.

Generates publication-quality figures showing:
1. Event voxel grids (single frame, per-bin slices)
2. RGB frames
3. Voxel + RGB overlay
4. History stacking (P past frames)
5. IMU traces (if available)
6. Dataset statistics (timestamp continuity, voxel activity)

Output saved to docs/images/ for inclusion in README.
"""

import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from enavipp.data.datasets import EnavippH5Dataset, collate_enavipp
from torch.utils.data import DataLoader


def parse_args():
    p = argparse.ArgumentParser(description="Visualize ENavi++ preprocessed data")
    p.add_argument("--h5", required=True, help="Path to preprocessed H5 file")
    p.add_argument("--out", default="docs/images", help="Output directory for images")
    p.add_argument("--sample_idx", type=int, default=None, help="Sample index (default: middle)")
    p.add_argument("--history", type=int, default=5, help="History depth P")
    p.add_argument("--dpi", type=int, default=150, help="Figure DPI")
    return p.parse_args()


def save_fig(fig, path, dpi=150):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------- Figure 1: Single voxel frame (sum + per-bin) ----------
def fig_voxel_single(ds, idx, out_dir, dpi):
    sample = ds[idx]
    voxel = sample["voxel"].numpy()  # (C, H, W)
    C = voxel.shape[0]

    fig, axes = plt.subplots(1, C + 1, figsize=(3 * (C + 1), 3.5))
    # Sum across bins
    vsum = np.abs(voxel).sum(axis=0)
    axes[0].imshow(vsum, cmap="inferno")
    axes[0].set_title("Event sum (|all bins|)", fontsize=9)
    axes[0].axis("off")
    # Per-bin
    for b in range(C):
        axes[b + 1].imshow(voxel[b], cmap="RdBu_r", vmin=-voxel.max(), vmax=voxel.max())
        axes[b + 1].set_title(f"Bin {b}", fontsize=9)
        axes[b + 1].axis("off")

    fig.suptitle(f"Voxel Grid — sample {idx}  [{C}×{voxel.shape[1]}×{voxel.shape[2]}]", fontsize=11, y=1.02)
    save_fig(fig, out_dir / "01_voxel_single.png", dpi)


# ---------- Figure 2: RGB frame ----------
def fig_rgb_single(ds, idx, out_dir, dpi):
    sample = ds[idx]
    if "rgb_left" not in sample:
        print("  (skipped — no RGB)")
        return
    rgb = sample["rgb_left"].permute(1, 2, 0).numpy()  # (H,W,3)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
    ax.imshow(rgb)
    ax.set_title(f"Left RGB — sample {idx}  [{rgb.shape[0]}×{rgb.shape[1]}]", fontsize=11)
    ax.axis("off")
    save_fig(fig, out_dir / "02_rgb_single.png", dpi)


# ---------- Figure 3: Voxel + RGB overlay ----------
def fig_overlay(ds, idx, out_dir, dpi):
    sample = ds[idx]
    if "rgb_left" not in sample:
        print("  (skipped — no RGB)")
        return
    voxel = sample["voxel"].numpy()
    rgb = sample["rgb_left"].permute(1, 2, 0).numpy()

    vsum = np.abs(voxel).sum(axis=0)
    vsum_norm = vsum / (vsum.max() + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(vsum, cmap="inferno")
    axes[1].set_title("Event voxel (sum)", fontsize=10)
    axes[1].axis("off")

    # Overlay: RGB with event heatmap
    overlay = rgb.copy()
    cmap_hot = plt.cm.hot(vsum_norm)[..., :3]
    mask = vsum_norm > 0.05
    overlay[mask] = 0.5 * overlay[mask] + 0.5 * cmap_hot[mask]
    axes[2].imshow(overlay)
    axes[2].set_title("RGB + Events overlay", fontsize=10)
    axes[2].axis("off")

    fig.suptitle(f"Multi-modal Alignment — sample {idx}", fontsize=12, y=1.02)
    save_fig(fig, out_dir / "03_overlay.png", dpi)


# ---------- Figure 4: History stacking (P frames) ----------
def fig_history(h5_path, idx_base, P, out_dir, dpi):
    ds_hist = EnavippH5Dataset(h5_path, load_rgb=True, load_imu=False,
                               history=P, history_stride=1, boundary="drop")
    # Use middle sample of history dataset
    hist_idx = min(idx_base, len(ds_hist) - 1)
    sample = ds_hist[hist_idx]
    voxel = sample["voxel"].numpy()  # (P, C, H, W)
    rgb = sample["rgb_left"].numpy()  # (P, 3, H, W)

    fig = plt.figure(figsize=(3.5 * P, 7))
    gs = gridspec.GridSpec(2, P, hspace=0.15, wspace=0.05)

    for p in range(P):
        # Voxel row
        ax_v = fig.add_subplot(gs[0, p])
        vsum = np.abs(voxel[p]).sum(axis=0)
        ax_v.imshow(vsum, cmap="inferno")
        ax_v.set_title(f"t-{P - 1 - p}" if p < P - 1 else "t (current)", fontsize=8)
        ax_v.axis("off")
        if p == 0:
            ax_v.set_ylabel("Voxel", fontsize=10)

        # RGB row
        ax_r = fig.add_subplot(gs[1, p])
        ax_r.imshow(np.moveaxis(rgb[p], 0, -1))
        ax_r.axis("off")
        if p == 0:
            ax_r.set_ylabel("RGB", fontsize=10)

    fig.suptitle(f"History Stacking (P={P}) — NoMaD-style past context", fontsize=12, y=1.02)
    save_fig(fig, out_dir / "04_history_stacking.png", dpi)


# ---------- Figure 5: IMU trace for one window ----------
def fig_imu_trace(h5_path, seq_name, idx, out_dir, dpi):
    with h5py.File(h5_path, "r") as f:
        if f"sequences/{seq_name}/imu/t_us" not in f:
            print("  (skipped — no IMU in H5)")
            return
        imu_t = f[f"sequences/{seq_name}/imu/t_us"][:]
        imu_d = f[f"sequences/{seq_name}/imu/data"][:]
        imu_ptr = f[f"sequences/{seq_name}/imu/ptr"][:]
        t_start = f[f"sequences/{seq_name}/t_start_us"][idx]
        t_end = f[f"sequences/{seq_name}/t_end_us"][idx]

    lo, hi = imu_ptr[idx], imu_ptr[idx + 1]
    if hi <= lo:
        print(f"  (skipped — no IMU samples in window {idx})")
        return

    t_ms = (imu_t[lo:hi] - t_start) / 1000.0  # ms relative to window start
    data = imu_d[lo:hi]  # (T, 6)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    for i, label in enumerate(["ax", "ay", "az"]):
        ax1.plot(t_ms, data[:, i], label=label, linewidth=0.8)
    ax1.set_ylabel("Accel (m/s²)")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.set_title(f"IMU — window {idx}  ({hi - lo} samples in 50ms)", fontsize=10)
    ax1.grid(alpha=0.3)

    for i, label in enumerate(["gx", "gy", "gz"]):
        ax2.plot(t_ms, data[:, 3 + i], label=label, linewidth=0.8)
    ax2.set_ylabel("Gyro (rad/s)")
    ax2.set_xlabel("Time within window (ms)")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(alpha=0.3)

    save_fig(fig, out_dir / "05_imu_trace.png", dpi)


# ---------- Figure 6: Dataset statistics ----------
def fig_statistics(h5_path, seq_name, out_dir, dpi):
    with h5py.File(h5_path, "r") as f:
        grp = f[f"sequences/{seq_name}"]
        N = grp.attrs["num_samples"]
        t_start = grp["t_start_us"][:]
        t_end = grp["t_end_us"][:]
        # Sample a few voxels for activity stats
        n_probe = min(N, 50)
        probe_idx = np.linspace(0, N - 1, n_probe, dtype=int)
        activities = []
        for i in probe_idx:
            v = grp["events/voxel"][i]  # (C,H,W)
            activities.append(np.abs(v).sum())

    windows_ms = (t_end - t_start) / 1000.0
    gaps_ms = (t_start[1:] - t_end[:-1]) / 1000.0

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Window durations
    axes[0].hist(windows_ms, bins=30, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("Window duration (ms)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Window durations (N={N})")
    axes[0].axvline(50.0, color="red", ls="--", label="50ms target")
    axes[0].legend(fontsize=8)

    # Inter-window gaps
    axes[1].hist(gaps_ms, bins=30, color="coral", edgecolor="white")
    axes[1].set_xlabel("Gap between windows (ms)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Inter-window gaps")

    # Voxel activity over time
    axes[2].plot(probe_idx, activities, ".-", color="purple", markersize=3)
    axes[2].set_xlabel("Sample index")
    axes[2].set_ylabel("Total |voxel| activity")
    axes[2].set_title("Event activity over sequence")

    fig.suptitle(f"Dataset Statistics — {seq_name}", fontsize=12, y=1.02)
    plt.tight_layout()
    save_fig(fig, out_dir / "06_statistics.png", dpi)


def main():
    args = parse_args()
    h5_path = args.h5
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine sequence name
    with h5py.File(h5_path, "r") as f:
        seq_names = list(f["sequences"].keys())
    seq_name = seq_names[0]
    print(f"Sequence: {seq_name}")

    # Single-step dataset (no history)
    ds = EnavippH5Dataset(h5_path, load_rgb=True, load_imu=False)
    idx = args.sample_idx if args.sample_idx is not None else len(ds) // 2
    print(f"Using sample index: {idx} / {len(ds)}")
    print()

    print("Generating figures ...")
    fig_voxel_single(ds, idx, out_dir, args.dpi)
    fig_rgb_single(ds, idx, out_dir, args.dpi)
    fig_overlay(ds, idx, out_dir, args.dpi)
    fig_history(h5_path, idx, args.history, out_dir, args.dpi)
    fig_imu_trace(h5_path, seq_name, idx, out_dir, args.dpi)
    fig_statistics(h5_path, seq_name, out_dir, args.dpi)

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
