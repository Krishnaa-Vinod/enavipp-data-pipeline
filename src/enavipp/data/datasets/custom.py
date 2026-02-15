"""Custom dataset placeholder.

TODO: Implement this when you have your own event camera data.
See docs/CUSTOM_DATASET_GUIDE.md for the recommended raw layout.
"""
from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset


class CustomEventDataset(Dataset):
    """Placeholder for a future custom event camera dataset.

    Raises NotImplementedError until implemented.
    """

    def __init__(self, data_root: Path | str, **kwargs):
        raise NotImplementedError(
            "CustomEventDataset is not yet implemented. "
            "See docs/CUSTOM_DATASET_GUIDE.md for guidance on adding your own dataset."
        )

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> dict:
        raise NotImplementedError
