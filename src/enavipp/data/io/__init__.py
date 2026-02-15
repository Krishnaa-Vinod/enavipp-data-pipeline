"""H5 writer utility for the ENavi++ preprocessing pipeline.

Writes preprocessed samples incrementally to an HDF5 file so the full
dataset never needs to be held in RAM.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import h5py
import numpy as np


class H5Writer:
    """Incrementally write samples to an HDF5 file under a sequence group."""

    def __init__(self, path: Path | str, attrs: Optional[Dict[str, Any]] = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.h5f = h5py.File(str(self.path), "w")
        if attrs:
            for k, v in attrs.items():
                self.h5f.attrs[k] = v

    # ── sequence-level helpers ──────────────────────────────────────

    def create_sequence_group(self, name: str) -> h5py.Group:
        grp = self.h5f.require_group(f"sequences/{name}")
        return grp

    # ── dataset creation (1-D expandable) ───────────────────────────

    @staticmethod
    def _make_expandable(grp: h5py.Group, name: str, shape_tail: tuple,
                         dtype, chunks: bool = True, compression: str | None = "gzip"):
        """Create an expandable dataset with shape (0, *shape_tail)."""
        maxshape = (None, *shape_tail)
        ds = grp.create_dataset(
            name,
            shape=(0, *shape_tail),
            maxshape=maxshape,
            dtype=dtype,
            chunks=True if chunks else None,
            compression=compression,
        )
        return ds

    @staticmethod
    def append_row(ds: h5py.Dataset, row: np.ndarray):
        """Append one row along axis-0."""
        n = ds.shape[0]
        ds.resize(n + 1, axis=0)
        ds[n] = row

    @staticmethod
    def append_rows(ds: h5py.Dataset, rows: np.ndarray):
        """Append multiple rows along axis-0."""
        n = ds.shape[0]
        m = rows.shape[0]
        ds.resize(n + m, axis=0)
        ds[n : n + m] = rows

    # ── vlen bytes (for JPEG) ──────────────────────────────────────

    @staticmethod
    def create_vlen_bytes(grp: h5py.Group, name: str):
        vlen_dt = h5py.special_dtype(vlen=np.uint8)
        ds = grp.create_dataset(
            name,
            shape=(0,),
            maxshape=(None,),
            dtype=vlen_dt,
        )
        return ds

    @staticmethod
    def append_bytes(ds: h5py.Dataset, data: bytes):
        n = ds.shape[0]
        ds.resize(n + 1, axis=0)
        ds[n] = np.frombuffer(data, dtype=np.uint8)

    # ── close ──────────────────────────────────────────────────────

    def close(self):
        self.h5f.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
