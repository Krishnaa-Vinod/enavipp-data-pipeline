"""DSEC dataset wrapper -- standardized access to DSEC event voxel grids.

This wraps the vendored third_party/dsec_example loader to return
standardized Sample dicts compatible with the ENavi++ pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

# Make vendored DSEC code importable
_THIRD_PARTY = Path(__file__).resolve().parents[4] / "third_party" / "dsec_example"
if str(_THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(_THIRD_PARTY))

from dsec_dataset.provider import DatasetProvider  # noqa: E402


class DSECVoxelDataset(Dataset):
    """Thin wrapper over the vendored DSEC DatasetProvider.

    Returns standardized dicts with keys matching ``docs/DATASET_FORMAT.md``.
    """

    def __init__(
        self,
        dsec_root: Path | str,
        delta_t_ms: int = 50,
        num_bins: int = 15,
        split: str = "train",
    ):
        dsec_root = Path(dsec_root)
        provider = DatasetProvider(dsec_root, delta_t_ms=delta_t_ms, num_bins=num_bins)

        if split == "train":
            self._ds = provider.get_train_dataset()
        else:
            raise NotImplementedError(f"Split '{split}' not yet supported for DSEC.")

        self.delta_t_ms = delta_t_ms
        self.num_bins = num_bins

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, index: int) -> dict:
        raw = self._ds[index]

        sample = {
            "events_voxel": {
                "left": raw["representation"]["left"],
                "right": raw["representation"]["right"],
            },
            "gt": {
                "disparity": raw["disparity_gt"],
            },
            "meta": {
                "dataset": "DSEC",
                "file_index": raw["file_index"],
            },
        }
        return sample
