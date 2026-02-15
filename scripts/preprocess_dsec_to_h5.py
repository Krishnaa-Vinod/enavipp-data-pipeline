"""CLI: preprocess a DSEC sequence into a single H5 file.

Usage:
    python scripts/preprocess_dsec_to_h5.py \
        --config configs/preprocess/dsec_bins5_50ms_480x640.yaml \
        --dsec_root data/raw/DSEC \
        --split train --sequence thun_00_a \
        --out data/processed/DSEC/thun_00_a_bins5_50ms.h5 \
        --include_rgb 1 --include_imu 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description="Preprocess DSEC sequence to H5")
    parser.add_argument("--config", type=Path, required=True, help="YAML config file")
    parser.add_argument("--dsec_root", type=Path, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--sequence", type=str, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Output H5 path")
    parser.add_argument("--include_rgb", type=int, default=1, choices=[0, 1])
    parser.add_argument("--include_imu", type=int, default=0, choices=[0, 1])
    parser.add_argument("--lidar_imu_root", type=Path, default=None)
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    vox = cfg.get("voxelization", {})
    rgb_cfg = cfg.get("rgb", {})

    seq_dir = args.dsec_root / args.split / args.sequence
    if not seq_dir.is_dir():
        print(f"ERROR: sequence directory not found: {seq_dir}")
        sys.exit(1)

    # Make enavipp importable
    src_root = Path(__file__).resolve().parent.parent / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from enavipp.data.preprocess import preprocess_sequence

    resize = rgb_cfg.get("resize")
    if resize:
        resize = tuple(resize)

    preprocess_sequence(
        seq_dir=seq_dir,
        out_path=args.out,
        num_bins=vox.get("num_bins", 5),
        window_ms=vox.get("window_ms", 50),
        height=vox.get("height", 480),
        width=vox.get("width", 640),
        normalize=vox.get("normalize", True),
        rectify=vox.get("rectify", True),
        include_rgb=bool(args.include_rgb),
        include_imu=bool(args.include_imu),
        lidar_imu_root=args.lidar_imu_root,
        rgb_jpeg_quality=rgb_cfg.get("jpeg_quality", 95),
        rgb_resize=resize,
    )


if __name__ == "__main__":
    main()
