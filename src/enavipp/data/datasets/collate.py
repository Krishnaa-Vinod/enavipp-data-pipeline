"""Collate function for EnavippH5Dataset.

Handles variable-length IMU sequences by zero-padding + mask.
Works with both single-step voxels [C,H,W] and history-stacked [P,C,H,W].
"""
from __future__ import annotations

from typing import Dict, List

import torch


def collate_enavipp(batch: List[Dict]) -> Dict:
    """Custom collate that pads variable-length IMU and stacks everything else.

    Handles voxel shapes:
        - Single-step: [C, H, W] -> batched to [B, C, H, W]
        - History:     [P, C, H, W] -> batched to [B, P, C, H, W]

    Returns
    -------
    dict with:
        voxel : (B, C, H, W) or (B, P, C, H, W)
        rgb_left : (B, 3, H, W) or (B, P, 3, H, W) or absent
        imu_data : (B, T_max, D) padded
        imu_t_us : (B, T_max) padded
        imu_mask : (B, T_max) bool — True where valid
        t_start_us : (B,)
        t_end_us : (B,)
        gt_disp_idx : (B,) int or absent
        meta : list[dict]
    """
    out: Dict = {}

    # Required: voxel grids
    out["voxel"] = torch.stack([s["voxel"] for s in batch])
    out["t_start_us"] = torch.tensor([s["t_start_us"] for s in batch], dtype=torch.int64)
    out["t_end_us"] = torch.tensor([s["t_end_us"] for s in batch], dtype=torch.int64)

    # RGB (optional)
    if "rgb_left" in batch[0]:
        out["rgb_left"] = torch.stack([s["rgb_left"] for s in batch])

    # GT disparity index (optional)
    if "gt_disp_idx" in batch[0]:
        out["gt_disp_idx"] = torch.tensor(
            [s["gt_disp_idx"] for s in batch], dtype=torch.long
        )

    # IMU — pad to max length in batch
    if "imu_data" in batch[0]:
        lengths = [s["imu_data"].shape[0] for s in batch]
        T_max = max(lengths)
        D = batch[0]["imu_data"].shape[1]
        B = len(batch)

        imu_data = torch.zeros(B, T_max, D)
        imu_t = torch.zeros(B, T_max, dtype=torch.int64)
        imu_mask = torch.zeros(B, T_max, dtype=torch.bool)

        for i, s in enumerate(batch):
            L = lengths[i]
            imu_data[i, :L] = s["imu_data"]
            imu_t[i, :L] = s["imu_t_us"]
            imu_mask[i, :L] = True

        out["imu_data"] = imu_data
        out["imu_t_us"] = imu_t
        out["imu_mask"] = imu_mask

    # Meta (list of dicts - not tensorizable)
    out["meta"] = [s["meta"] for s in batch]

    return out
