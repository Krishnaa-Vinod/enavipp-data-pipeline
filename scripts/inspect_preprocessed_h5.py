#!/usr/bin/env python3
"""Inspect & visualize a preprocessed ENavi++ H5 file.

Usage
-----
python scripts/inspect_preprocessed_h5.py \
    --h5 data/processed/DSEC/thun_00_a_bins5_50ms.h5 \
    --num_samples 4 \
    --out artifacts/inspect_h5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Inspect a preprocessed H5 file")
    parser.add_argument("--h5", required=True, help="Path to H5 file")
    parser.add_argument("--num_samples", type=int, default=4, help="Samples to visualize")
    parser.add_argument("--out", default="artifacts/inspect_h5", help="Output directory")
    parser.add_argument("--sequence", default=None, help="Specific sequence (default: first)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.h5, "r") as f:
        # --- File-level info ---
        print("=" * 60)
        print(f"H5 File: {args.h5}")
        print(f"Attrs: {dict(f.attrs)}")
        print(f"Sequences: {list(f['sequences'].keys())}")
        print("=" * 60)

        # Pick sequence
        seqs = list(f["sequences"].keys())
        seq_name = args.sequence if args.sequence and args.sequence in seqs else seqs[0]
        grp = f[f"sequences/{seq_name}"]
        n = grp.attrs.get("num_samples", grp["t_start_us"].shape[0])
        print(f"\nSequence: {seq_name}  ({n} samples)")

        # Datasets summary
        def _show_ds(name, ds):
            if isinstance(ds, h5py.Dataset):
                info = f"  {name:50s} shape={ds.shape}  dtype={ds.dtype}"
                if ds.compression:
                    info += f"  compression={ds.compression}"
                print(info)

        grp.visititems(_show_ds)
        print()

        # --- Visualize samples ---
        has_voxel = "events/voxel" in grp
        has_rgb = "rgb/left/jpeg" in grp
        has_imu = "imu/data" in grp

        indices = np.linspace(0, n - 1, min(args.num_samples, n), dtype=int)

        for idx in indices:
            tag = f"{seq_name}_sample{idx:04d}"

            # Voxel grid
            if has_voxel:
                voxel = grp["events/voxel"][idx]  # (C, H, W)
                C, H, W = voxel.shape
                print(f"[{tag}] voxel shape=({C},{H},{W})  "
                      f"min={voxel.min():.4f}  max={voxel.max():.4f}  "
                      f"mean={voxel.mean():.4f}")

                # Sum heatmap
                heatmap = voxel.sum(axis=0)
                plt.figure(figsize=(10, 6))
                plt.imshow(heatmap, cmap="hot")
                plt.colorbar(label="Sum over bins")
                plt.title(f"{tag} — voxel sum heatmap")
                plt.tight_layout()
                plt.savefig(out_dir / f"{tag}_voxel_heatmap.png", dpi=120)
                plt.close()

                # Per-bin grid
                if C <= 10:
                    fig, axes = plt.subplots(1, C, figsize=(3 * C, 3))
                    if C == 1:
                        axes = [axes]
                    for b in range(C):
                        axes[b].imshow(voxel[b], cmap="coolwarm", vmin=-2, vmax=2)
                        axes[b].set_title(f"bin {b}")
                        axes[b].axis("off")
                    fig.suptitle(f"{tag} — per-bin voxel slices", fontsize=12)
                    fig.tight_layout()
                    fig.savefig(out_dir / f"{tag}_voxel_bins.png", dpi=100)
                    plt.close(fig)

            # RGB
            if has_rgb:
                jpeg_ds = grp["rgb/left/jpeg"]
                if idx < jpeg_ds.shape[0]:
                    buf = np.array(jpeg_ds[idx], dtype=np.uint8)
                    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                    if img is not None:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        plt.figure(figsize=(10, 6))
                        plt.imshow(img_rgb)
                        plt.title(f"{tag} — left RGB")
                        plt.axis("off")
                        plt.tight_layout()
                        plt.savefig(out_dir / f"{tag}_rgb_left.png", dpi=120)
                        plt.close()
                        print(f"[{tag}] rgb shape={img_rgb.shape}")

            # Timestamps
            t0 = grp["t_start_us"][idx]
            t1 = grp["t_end_us"][idx]
            print(f"[{tag}] t_start={t0}  t_end={t1}  "
                  f"delta={t1 - t0} us ({(t1 - t0) / 1000:.1f} ms)")

        # --- IMU global plot ---
        if has_imu:
            imu_t = grp["imu/t_us"][:]
            imu_d = grp["imu/data"][:]
            print(f"\nIMU: {imu_d.shape[0]} samples, {imu_d.shape[1]} channels")

            fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
            t_sec = (imu_t - imu_t[0]) / 1e6
            labels_accel = ["ax", "ay", "az"]
            labels_gyro = ["gx", "gy", "gz"]

            for i in range(min(3, imu_d.shape[1])):
                axes[0].plot(t_sec, imu_d[:, i], label=labels_accel[i], alpha=0.8)
            axes[0].set_ylabel("Accel (m/s²)")
            axes[0].legend()

            for i in range(3, min(6, imu_d.shape[1])):
                axes[1].plot(t_sec, imu_d[:, i], label=labels_gyro[i - 3], alpha=0.8)
            axes[1].set_ylabel("Gyro (rad/s)")
            axes[1].set_xlabel("Time (s)")
            axes[1].legend()

            fig.suptitle(f"{seq_name} — full IMU trace")
            fig.tight_layout()
            fig.savefig(out_dir / f"{seq_name}_imu_trace.png", dpi=120)
            plt.close(fig)

    print(f"\nSaved visualizations to {out_dir}/")


if __name__ == "__main__":
    main()
