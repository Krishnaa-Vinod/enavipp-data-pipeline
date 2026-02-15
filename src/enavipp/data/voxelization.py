"""Voxelization utilities -- wraps the vendored DSEC VoxelGrid class.

This module provides a clean API for converting raw events to voxel grid
tensors, abstracting away the third_party implementation details.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Make vendored DSEC code importable
_THIRD_PARTY = Path(__file__).resolve().parents[3] / "third_party" / "dsec_example"
if str(_THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(_THIRD_PARTY))

from dsec_dataset.representations import VoxelGrid  # noqa: E402

from .types import VoxelizationConfig  # noqa: E402


def events_to_voxel_grid(
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    t: np.ndarray,
    config: Optional[VoxelizationConfig] = None,
) -> torch.Tensor:
    """Convert raw events to a voxel grid tensor.

    Parameters
    ----------
    x, y : array, pixel coordinates
    p : array, polarity (0 or 1)
    t : array, timestamps (microseconds, already offset-corrected)
    config : VoxelizationConfig, optional

    Returns
    -------
    torch.Tensor of shape [num_bins, height, width]
    """
    if config is None:
        config = VoxelizationConfig()

    vg = VoxelGrid(config.num_bins, config.height, config.width, normalize=config.normalize)

    # Normalize time to [0, 1]
    t_f = (t - t[0]).astype("float32")
    if t_f[-1] > 0:
        t_f = t_f / t_f[-1]

    return vg.convert(
        torch.from_numpy(x.astype("float32")),
        torch.from_numpy(y.astype("float32")),
        torch.from_numpy(p.astype("float32")),
        torch.from_numpy(t_f),
    )
