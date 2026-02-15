#!/usr/bin/env python3
"""Validate time-sync invariants in a preprocessed ENavi++ H5 file.

Checks:
  1. All voxel windows are exactly 50ms (or whatever the config says)
  2. RGB timestamps align with voxel t_end_us
  3. IMU timestamps fall within [t_start, t_end) per window (+ tolerance)
  4. Prints stats and saves diagnostic plots

Usage:
    python scripts/validate_preprocessed_h5.py \
        --h5 data/processed/DSEC/thun_00_a_bins5_50ms_480x640_rgb_imu.h5 \
        --sequence thun_00_a \
        --max_samples 200 \
        --out_dir artifacts/validate_h5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Validate preprocessed H5 sync")
    parser.add_argument("--h5", required=True, help="Path to H5 file")
    parser.add_argument("--sequence", default=None, help="Sequence name (default: first)")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--out_dir", default="artifacts/validate_h5")
    parser.add_argument("--tolerance_us", type=int, default=0,
                        help="IMU tolerance for window boundary check")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_pass = True

    with h5py.File(args.h5, "r") as f:
        print("=" * 65)
        print(f"Validating: {args.h5}")
        print(f"Root attrs: {dict(f.attrs)}")
        seqs = list(f["sequences"].keys())
        print(f"Sequences: {seqs}")

        seq_name = args.sequence if args.sequence in seqs else seqs[0]
        grp = f[f"sequences/{seq_name}"]
        N = int(grp.attrs.get("num_samples", grp["t_start_us"].shape[0]))
        n = min(N, args.max_samples)
        print(f"\nSequence: {seq_name}  (N={N}, checking first {n})")
        print("=" * 65)

        t_start = grp["t_start_us"][:n]
        t_end = grp["t_end_us"][:n]
        voxel_shape = grp["events/voxel"].shape
        has_rgb = "rgb/left/t_us" in grp
        has_imu = "imu/t_us" in grp

        print(f"\nVoxel: shape={voxel_shape}, dtype={grp['events/voxel'].dtype}")
        print(f"RGB present: {has_rgb}")
        print(f"IMU present: {has_imu}")

        # ─── Check 1: Window length ────────────────────────────────
        deltas = t_end - t_start
        expected_us = int(f.attrs.get("voxel_window_ms", 50)) * 1000
        all_correct_len = np.all(deltas == expected_us)
        print(f"\n[CHECK 1] Window length == {expected_us} us ({expected_us/1000:.0f} ms)")
        print(f"  unique deltas: {np.unique(deltas)}")
        if all_correct_len:
            print("  PASS: all windows have correct length")
        else:
            print(f"  FAIL: {np.sum(deltas != expected_us)}/{n} windows have wrong length")
            all_pass = False

        # ─── Check 2: RGB alignment ────────────────────────────────
        if has_rgb:
            rgb_t = grp["rgb/left/t_us"][:n]
            rgb_match = (t_end == rgb_t)
            frac = rgb_match.sum() / n
            print(f"\n[CHECK 2] RGB t_us == t_end_us")
            print(f"  Alignment fraction: {frac:.4f} ({rgb_match.sum()}/{n})")
            if frac == 1.0:
                print("  PASS: all RGB timestamps match t_end_us")
            else:
                mismatches = np.where(~rgb_match)[0][:5]
                for mi in mismatches:
                    print(f"    sample {mi}: t_end={t_end[mi]}, rgb_t={rgb_t[mi]}, "
                          f"diff={t_end[mi]-rgb_t[mi]} us")
                all_pass = False
                print("  FAIL: some RGB timestamps don't match t_end_us")
        else:
            print("\n[CHECK 2] RGB: SKIPPED (not present)")

        # ─── Check 3: IMU sync ─────────────────────────────────────
        if has_imu:
            imu_t_all = grp["imu/t_us"][:]
            imu_ptr = grp["imu/ptr"][:]
            total_imu = len(imu_t_all)
            tol = args.tolerance_us

            print(f"\n[CHECK 3] IMU sync (tolerance={tol} us)")
            print(f"  Total IMU samples: {total_imu}")
            print(f"  ptr shape: {imu_ptr.shape}")

            violations = 0
            counts = []
            for i in range(min(n, len(imu_ptr) - 1)):
                lo, hi = int(imu_ptr[i]), int(imu_ptr[i + 1])
                cnt = hi - lo
                counts.append(cnt)
                if cnt > 0:
                    imu_t_win = imu_t_all[lo:hi]
                    below = np.sum(imu_t_win < t_start[i] - tol)
                    above = np.sum(imu_t_win >= t_end[i] + tol)
                    if below > 0 or above > 0:
                        violations += 1
                        if violations <= 3:
                            print(f"    VIOLATION sample {i}: {below} below t_start, "
                                  f"{above} above t_end")

            counts = np.array(counts)
            print(f"  IMU samples/window: avg={counts.mean():.1f}, "
                  f"min={counts.min()}, max={counts.max()}")
            zero_pct = (counts == 0).sum() / len(counts) * 100
            print(f"  Windows with 0 IMU: {(counts==0).sum()}/{len(counts)} ({zero_pct:.1f}%)")
            print(f"  Violations: {violations}/{len(counts)}")

            if violations == 0:
                print("  PASS: all IMU timestamps within window bounds")
            else:
                print("  FAIL: some IMU timestamps outside window bounds")
                all_pass = False

            # ─── Plot: IMU samples/window histogram ─────────────────
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(counts, bins=30, edgecolor="black", alpha=0.7)
            ax.set_xlabel("IMU samples per window")
            ax.set_ylabel("Count")
            ax.set_title(f"{seq_name}: IMU samples/window distribution")
            ax.axvline(counts.mean(), color="red", ls="--", label=f"mean={counts.mean():.1f}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out / f"{seq_name}_imu_per_window_hist.png", dpi=120)
            plt.close(fig)
            print(f"  Saved histogram: {out / f'{seq_name}_imu_per_window_hist.png'}")

            # ─── Plot: Example IMU for one window ───────────────────
            mid = len(counts) // 2
            while mid < len(imu_ptr) - 1 and int(imu_ptr[mid + 1]) - int(imu_ptr[mid]) == 0:
                mid += 1
            if mid < len(imu_ptr) - 1:
                lo, hi = int(imu_ptr[mid]), int(imu_ptr[mid + 1])
                if hi > lo:
                    imu_t_ex = imu_t_all[lo:hi]
                    imu_d_ex = grp["imu/data"][lo:hi]
                    t_rel = (imu_t_ex - imu_t_ex[0]) / 1000.0  # ms

                    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
                    for j, lab in enumerate(["ax", "ay", "az"]):
                        axes[0].plot(t_rel, imu_d_ex[:, j], label=lab, alpha=0.8)
                    axes[0].set_ylabel("Accel (m/s²)")
                    axes[0].legend()
                    for j, lab in enumerate(["gx", "gy", "gz"]):
                        axes[1].plot(t_rel, imu_d_ex[:, j + 3], label=lab, alpha=0.8)
                    axes[1].set_ylabel("Gyro (rad/s)")
                    axes[1].set_xlabel("Time within window (ms)")
                    axes[1].legend()
                    fig.suptitle(f"{seq_name} sample {mid}: IMU in one 50ms window")
                    fig.tight_layout()
                    fig.savefig(out / f"{seq_name}_imu_example_window.png", dpi=120)
                    plt.close(fig)
                    print(f"  Saved example: {out / f'{seq_name}_imu_example_window.png'}")

        else:
            print("\n[CHECK 3] IMU: SKIPPED (not present)")

        # ─── Summary ───────────────────────────────────────────────
        print("\n" + "=" * 65)
        if all_pass:
            print("RESULT: ALL CHECKS PASSED")
        else:
            print("RESULT: SOME CHECKS FAILED (see above)")
        print("=" * 65)


if __name__ == "__main__":
    main()
