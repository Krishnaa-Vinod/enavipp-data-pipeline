"""PyTorch Dataset for preprocessed ENavi++ H5 files.

Loads voxel grids, optional JPEG-encoded RGB, optional IMU, and
metadata from an H5 file written by the preprocessing pipeline.

Supports history stacking (P past frames) for NoMaD-style context.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class EnavippH5Dataset(Dataset):
    """Read preprocessed H5 files produced by ``preprocess_dsec_to_h5.py``.

    Parameters
    ----------
    h5_path : path to the H5 file
    sequences : list of sequence names to include (None = all)
    load_rgb : whether to decode and return RGB images
    load_imu : whether to return IMU data
    history : number of frames to stack (1 = no stacking, 5 = NoMaD-style)
    history_stride : stride between stacked frames (default 1)
    boundary : 'drop' or 'pad'. If 'drop', first (history-1)*stride indices
               are removed. If 'pad', early frames are zero-padded.
    """

    def __init__(
        self,
        h5_path: Path | str,
        sequences: Optional[List[str]] = None,
        load_rgb: bool = True,
        load_imu: bool = False,
        history: int = 1,
        history_stride: int = 1,
        boundary: str = "drop",
    ):
        self.h5_path = Path(h5_path)
        self.load_rgb = load_rgb
        self.load_imu = load_imu
        self.history = max(1, history)
        self.history_stride = max(1, history_stride)
        self.boundary = boundary
        self._h5f: Optional[h5py.File] = None

        # Build index: (sequence_name, local_idx)
        with h5py.File(str(self.h5_path), "r") as f:
            self.attrs = dict(f.attrs)
            seq_grp = f["sequences"]
            avail = list(seq_grp.keys())
            if sequences is not None:
                avail = [s for s in avail if s in sequences]

            self._index: List[tuple] = []
            self._seq_lengths: Dict[str, int] = {}
            for sname in sorted(avail):
                n = seq_grp[sname].attrs.get("num_samples", 0)
                if n == 0:
                    n = seq_grp[sname]["t_start_us"].shape[0]
                self._seq_lengths[sname] = n

                # Determine valid start index for this sequence
                min_idx = 0
                if self.history > 1 and boundary == "drop":
                    min_idx = (self.history - 1) * self.history_stride
                for i in range(min_idx, n):
                    self._index.append((sname, i))

            # Check what modalities are available
            first_seq = avail[0] if avail else None
            self._has_rgb = first_seq is not None and "rgb/left/jpeg" in seq_grp[first_seq]
            self._has_imu = first_seq is not None and "imu/t_us" in seq_grp[first_seq]
            self._has_disp_idx = first_seq is not None and "gt/disparity_frame_idx" in seq_grp[first_seq]

    def __len__(self) -> int:
        return len(self._index)

    @property
    def h5f(self) -> h5py.File:
        """Lazy-open per worker (for safe multi-worker DataLoader)."""
        if self._h5f is None:
            self._h5f = h5py.File(str(self.h5_path), "r")
        return self._h5f

    def _get_history_indices(self, seq_name: str, local_i: int) -> List[int]:
        """Return list of frame indices for history stacking."""
        indices = []
        for p in range(self.history - 1, -1, -1):
            idx = local_i - p * self.history_stride
            if idx < 0:
                idx = 0  # pad boundary: repeat first frame
            indices.append(idx)
        return indices

    def _load_single_voxel(self, grp: h5py.Group, idx: int) -> np.ndarray:
        return grp["events/voxel"][idx].astype(np.float32)

    def _decode_rgb(self, grp: h5py.Group, idx: int) -> Optional[np.ndarray]:
        jpeg_ds = grp["rgb/left/jpeg"]
        if idx >= jpeg_ds.shape[0]:
            return None
        jpeg_buf = np.array(jpeg_ds[idx], dtype=np.uint8)
        img = cv2.imdecode(jpeg_buf, cv2.IMREAD_COLOR)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.transpose(2, 0, 1).astype(np.float32) / 255.0

    def __getitem__(self, idx: int) -> Dict:
        seq_name, local_i = self._index[idx]
        grp = self.h5f[f"sequences/{seq_name}"]

        # History stacking
        if self.history > 1:
            hist_indices = self._get_history_indices(seq_name, local_i)

            # Voxel: (P, C, H, W)
            voxels = [self._load_single_voxel(grp, hi) for hi in hist_indices]
            voxel_tensor = torch.from_numpy(np.stack(voxels, axis=0))

            sample = {
                "t_start_us": int(grp["t_start_us"][local_i]),
                "t_end_us": int(grp["t_end_us"][local_i]),
                "voxel": voxel_tensor,
                "meta": {
                    "dataset": "DSEC",
                    "sequence": seq_name,
                    "sample_idx": local_i,
                    "history_indices": hist_indices,
                },
            }

            # RGB: (P, 3, H, W)
            if self.load_rgb and self._has_rgb:
                rgbs = []
                for hi in hist_indices:
                    rgb = self._decode_rgb(grp, hi)
                    if rgb is not None:
                        rgbs.append(rgb)
                if len(rgbs) == self.history:
                    sample["rgb_left"] = torch.from_numpy(np.stack(rgbs, axis=0))

        else:
            # Single-step (original behavior)
            sample = {
                "t_start_us": int(grp["t_start_us"][local_i]),
                "t_end_us": int(grp["t_end_us"][local_i]),
                "voxel": torch.from_numpy(
                    grp["events/voxel"][local_i].astype(np.float32)
                ),
                "meta": {
                    "dataset": "DSEC",
                    "sequence": seq_name,
                    "sample_idx": local_i,
                },
            }

            # RGB
            if self.load_rgb and self._has_rgb:
                rgb = self._decode_rgb(grp, local_i)
                if rgb is not None:
                    sample["rgb_left"] = torch.from_numpy(rgb)

        # IMU (always for the current window, not stacked)
        if self.load_imu and self._has_imu:
            ptr = grp["imu/ptr"]
            if local_i < ptr.shape[0] - 1:
                start, end = int(ptr[local_i]), int(ptr[local_i + 1])
                sample["imu_t_us"] = torch.from_numpy(
                    grp["imu/t_us"][start:end].astype(np.int64)
                )
                sample["imu_data"] = torch.from_numpy(
                    grp["imu/data"][start:end].astype(np.float32)
                )

        # Disparity GT index
        if self._has_disp_idx:
            sample["gt_disp_idx"] = int(grp["gt/disparity_frame_idx"][local_i])

        return sample

    def close(self):
        if self._h5f is not None:
            self._h5f.close()
            self._h5f = None

    def __del__(self):
        self.close()
