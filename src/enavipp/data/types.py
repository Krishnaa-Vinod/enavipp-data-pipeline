"""Core data types for the ENavi++ pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class VoxelizationConfig:
    """Configuration for event-to-voxel-grid conversion."""
    num_bins: int = 15
    window_ms: int = 50
    normalize: bool = True
    rectify: bool = True
    height: int = 480
    width: int = 640


@dataclass
class HistoryConfig:
    """Configuration for stacking past voxel grids."""
    enable: bool = False
    past_steps: int = 5
    stride_ms: int = 50


@dataclass
class Sample:
    """A single data sample produced by the pipeline.

    All fields are optional except ``t_us`` and ``meta``.
    """
    t_us: int = 0
    events_voxel: Optional[torch.Tensor] = None   # [C,H,W] or [P,C,H,W]
    rgb: Optional[torch.Tensor] = None             # [3,H,W] or [P,3,H,W]
    imu: Optional[torch.Tensor] = None             # [T, D]
    gt: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dict (for DataLoader collation)."""
        d: Dict[str, Any] = {"t_us": self.t_us, "meta": self.meta}
        if self.events_voxel is not None:
            d["events_voxel"] = self.events_voxel
        if self.rgb is not None:
            d["rgb"] = self.rgb
        if self.imu is not None:
            d["imu"] = self.imu
        if self.gt:
            d["gt"] = self.gt
        return d
