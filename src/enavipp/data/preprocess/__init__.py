"""DSEC -> H5 preprocessing module.

Converts a raw DSEC sequence (events + optional RGB + optional IMU)
into a single HDF5 file with aligned voxel grids, JPEG-encoded RGB,
and ragged IMU arrays.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
from tqdm import tqdm

# Make vendored DSEC code importable
_THIRD_PARTY = Path(__file__).resolve().parents[4] / "third_party" / "dsec_example"
if str(_THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(_THIRD_PARTY))

from dsec_dataset.representations import VoxelGrid  # noqa: E402
from dsec_utils.eventslicer import EventSlicer  # noqa: E402

from ..io import H5Writer  # noqa: E402


# ── voxelization ────────────────────────────────────────────────────

def voxelize(
    events: Dict[str, np.ndarray],
    t0_us: int,
    t1_us: int,
    num_bins: int,
    height: int,
    width: int,
    rectify_map: Optional[np.ndarray] = None,
    normalize: bool = True,
) -> np.ndarray:
    """Convert raw events in [t0_us, t1_us) to a voxel grid [bins, H, W].

    Returns np.float16 array. Returns zeros if no events in window.
    """
    if events is None or events["t"].size == 0:
        return np.zeros((num_bins, height, width), dtype=np.float16)

    x = events["x"].astype(np.float32)
    y = events["y"].astype(np.float32)
    p = events["p"].astype(np.float32)
    t = events["t"].astype(np.float64)

    # Rectify
    if rectify_map is not None:
        xy = rectify_map[y.astype(int), x.astype(int)]
        x = xy[:, 0].astype(np.float32)
        y = xy[:, 1].astype(np.float32)
        # Filter out-of-bounds
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        x, y, p, t = x[valid], y[valid], p[valid], t[valid]

    if t.size == 0:
        return np.zeros((num_bins, height, width), dtype=np.float16)

    # Normalize time to [0, 1]
    t_f = (t - t[0]).astype(np.float32)
    denom = t_f[-1] if t_f[-1] > 0 else 1.0
    t_f = t_f / denom

    vg = VoxelGrid(num_bins, height, width, normalize=normalize)
    grid = vg.convert(
        torch.from_numpy(x),
        torch.from_numpy(y),
        torch.from_numpy(p),
        torch.from_numpy(t_f),
    )
    return grid.numpy().astype(np.float16)


# ── RGB helpers ─────────────────────────────────────────────────────

def load_and_encode_rgb(
    img_path: Path,
    jpeg_quality: int = 95,
    resize: Optional[Tuple[int, int]] = None,
) -> bytes:
    """Load a PNG, optionally resize, encode as JPEG bytes."""
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    if resize is not None:
        img = cv2.resize(img, tuple(resize), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    assert ok, f"JPEG encode failed for {img_path}"
    return buf.tobytes()


# ── main preprocessing function ─────────────────────────────────────

def preprocess_sequence(
    seq_dir: Path,
    out_path: Path,
    num_bins: int = 5,
    window_ms: int = 50,
    height: int = 480,
    width: int = 640,
    normalize: bool = True,
    rectify: bool = True,
    include_rgb: bool = True,
    include_imu: bool = False,
    lidar_imu_root: Optional[Path] = None,
    rgb_jpeg_quality: int = 95,
    rgb_resize: Optional[Tuple[int, int]] = None,
) -> Path:
    """Preprocess one DSEC sequence into a single H5 file."""

    seq_name = seq_dir.name
    window_us = window_ms * 1000

    # ── Discover anchor timestamps ──────────────────────────────
    # Prefer image_timestamps if available, else fall back to disparity timestamps
    img_ts_file = seq_dir / "images" / "image_timestamps.txt"
    disp_ts_file = seq_dir / "disparity" / "timestamps.txt"

    if img_ts_file.exists():
        anchor_timestamps = np.loadtxt(str(img_ts_file), dtype=np.int64)
        anchor_source = "image_timestamps"
    elif disp_ts_file.exists():
        anchor_timestamps = np.loadtxt(str(disp_ts_file), dtype=np.int64)
        anchor_source = "disparity_timestamps"
    else:
        raise FileNotFoundError(
            f"No anchor timestamps found in {seq_dir}. "
            "Need images/image_timestamps.txt or disparity/timestamps.txt"
        )

    N = anchor_timestamps.shape[0]
    print(f"[preprocess] Sequence: {seq_name}, anchor={anchor_source}, N={N}")

    # ── Open event H5 files ─────────────────────────────────────
    ev_left_file = seq_dir / "events" / "left" / "events.h5"
    assert ev_left_file.exists(), f"Missing: {ev_left_file}"
    h5_left = h5py.File(str(ev_left_file), "r")
    slicer_left = EventSlicer(h5_left)

    rect_map_left = None
    if rectify:
        rect_file = seq_dir / "events" / "left" / "rectify_map.h5"
        if rect_file.exists():
            with h5py.File(str(rect_file), "r") as rf:
                rect_map_left = rf["rectify_map"][()]

    # ── Discover RGB image paths ────────────────────────────────
    rgb_paths: List[Path] = []
    if include_rgb:
        rgb_dir = seq_dir / "images" / "left" / "rectified"
        if rgb_dir.exists():
            rgb_paths = sorted(rgb_dir.glob("*.png"))
            print(f"[preprocess] Found {len(rgb_paths)} RGB images")
        else:
            print(f"[preprocess] WARNING: --include_rgb but no images at {rgb_dir}")
            include_rgb = False

    # ── Discover disparity GT paths ─────────────────────────────
    disp_event_dir = seq_dir / "disparity" / "event"
    disp_paths: List[Path] = []
    disp_timestamps: Optional[np.ndarray] = None
    if disp_event_dir.exists():
        disp_paths = sorted(disp_event_dir.glob("*.png"))
        if disp_ts_file.exists():
            disp_timestamps = np.loadtxt(str(disp_ts_file), dtype=np.int64)

    # ── Write H5 ────────────────────────────────────────────────
    attrs = {
        "dataset": "DSEC",
        "created_by": "enavipp-data-pipeline",
        "voxel_num_bins": num_bins,
        "voxel_window_ms": window_ms,
        "voxel_height": height,
        "voxel_width": width,
        "events_rectified": rectify,
        "anchor": anchor_source,
        "rgb_store": "jpeg_bytes" if include_rgb else "none",
    }
    if rgb_resize:
        attrs["rgb_resize_w"] = rgb_resize[0]
        attrs["rgb_resize_h"] = rgb_resize[1]

    with H5Writer(out_path, attrs=attrs) as writer:
        grp = writer.create_sequence_group(seq_name)

        # Create expandable datasets
        ds_t_start = writer._make_expandable(grp, "t_start_us", (), np.int64)
        ds_t_end = writer._make_expandable(grp, "t_end_us", (), np.int64)
        ds_voxel = writer._make_expandable(
            grp, "events/voxel",
            (num_bins, height, width), np.float16,
        )

        ds_rgb_jpeg = None
        ds_rgb_t = None
        if include_rgb:
            ds_rgb_jpeg = writer.create_vlen_bytes(grp, "rgb/left/jpeg")
            ds_rgb_t = writer._make_expandable(grp, "rgb/left/t_us", (), np.int64)

        # Disparity index mapping (sparse — only where we have GT)
        ds_disp_idx = None
        if disp_paths:
            ds_disp_idx = writer._make_expandable(grp, "gt/disparity_frame_idx", (), np.int32)

        # ── Process each anchor timestamp ───────────────────────
        n_written = 0
        for i in tqdm(range(N), desc=f"Preprocessing {seq_name}"):
            t_end = int(anchor_timestamps[i])
            t_start = t_end - window_us

            # Skip if events don't cover this window
            if t_start < 0:
                continue

            # Voxelize events
            try:
                ev = slicer_left.get_events(t_start, t_end)
            except (AssertionError, Exception):
                ev = None

            voxel = voxelize(
                ev, t_start, t_end,
                num_bins, height, width,
                rect_map_left, normalize,
            )

            writer.append_row(ds_t_start, np.array(t_start, dtype=np.int64))
            writer.append_row(ds_t_end, np.array(t_end, dtype=np.int64))
            writer.append_row(ds_voxel, voxel)

            # RGB
            if include_rgb and i < len(rgb_paths):
                jpeg_bytes = load_and_encode_rgb(
                    rgb_paths[i],
                    jpeg_quality=rgb_jpeg_quality,
                    resize=rgb_resize,
                )
                writer.append_bytes(ds_rgb_jpeg, jpeg_bytes)
                writer.append_row(ds_rgb_t, np.array(t_end, dtype=np.int64))

            # Disparity GT index (for later retrieval)
            if ds_disp_idx is not None and disp_timestamps is not None:
                # Find closest disparity timestamp
                diffs = np.abs(disp_timestamps - t_end)
                closest_idx = int(np.argmin(diffs))
                if diffs[closest_idx] < window_us:
                    writer.append_row(ds_disp_idx, np.array(closest_idx, dtype=np.int32))
                else:
                    writer.append_row(ds_disp_idx, np.array(-1, dtype=np.int32))

            n_written += 1

        grp.attrs["num_samples"] = n_written
        print(f"[preprocess] Wrote {n_written} samples to {out_path}")

    h5_left.close()
    return out_path
