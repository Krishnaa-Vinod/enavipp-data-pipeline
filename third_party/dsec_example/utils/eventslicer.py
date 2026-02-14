import math
from typing import Dict, Tuple

import h5py
import hdf5plugin  # noqa: F401  — registers compression filters
import numpy as np
from numba import jit


class EventSlicer:
    def __init__(self, h5f: h5py.File):
        self.h5f = h5f
        self.events = {d: self.h5f[f"events/{d}"] for d in ["p", "x", "y", "t"]}
        self.ms_to_idx = np.asarray(self.h5f["ms_to_idx"], dtype="int64")
        self.t_offset = int(h5f["t_offset"][()]) if "t_offset" in list(h5f.keys()) else 0
        self.t_final = int(self.events["t"][-1]) + self.t_offset

    def get_events(self, t_start_us: int, t_end_us: int) -> Dict[str, np.ndarray]:
        assert t_start_us < t_end_us
        t_start_us -= self.t_offset
        t_end_us -= self.t_offset

        t_start_ms, t_end_ms = self.get_conservative_window_ms(t_start_us, t_end_us)
        t_start_ms_idx = self.ms2idx(t_start_ms)
        t_end_ms_idx = self.ms2idx(t_end_ms)

        if t_start_ms_idx is None or t_end_ms_idx is None:
            return None

        time_array_conservative = np.asarray(self.events["t"][t_start_ms_idx:t_end_ms_idx])
        idx_start_offset, idx_end_offset = self.get_time_indices_offsets(
            time_array_conservative, t_start_us, t_end_us
        )

        t_start_us_idx = t_start_ms_idx + idx_start_offset
        t_end_us_idx = t_start_ms_idx + idx_end_offset

        out = {}
        out["t"] = time_array_conservative[idx_start_offset:idx_end_offset].astype("int64") + self.t_offset
        for d in ["p", "x", "y"]:
            out[d] = np.asarray(self.events[d][t_start_us_idx:t_end_us_idx])
            assert out[d].size == out["t"].size
        return out

    @staticmethod
    def get_conservative_window_ms(ts_start_us: int, ts_end_us: int) -> Tuple[int, int]:
        assert ts_end_us > ts_start_us
        return math.floor(ts_start_us / 1000), math.ceil(ts_end_us / 1000)

    @staticmethod
    @jit(nopython=True)
    def get_time_indices_offsets(time_array: np.ndarray, time_start_us: int, time_end_us: int) -> Tuple[int, int]:
        assert time_array.ndim == 1

        idx_start = -1
        if time_array[-1] < time_start_us:
            return time_array.size, time_array.size

        for idx_from_start in range(0, time_array.size, 1):
            if time_array[idx_from_start] >= time_start_us:
                idx_start = idx_from_start
                break
        assert idx_start >= 0

        idx_end = time_array.size
        for idx_from_end in range(time_array.size - 1, -1, -1):
            if time_array[idx_from_end] >= time_end_us:
                idx_end = idx_from_end
            else:
                break

        return idx_start, idx_end

    def ms2idx(self, time_ms: int) -> int:
        assert time_ms >= 0
        if time_ms >= self.ms_to_idx.size:
            return None
        return self.ms_to_idx[time_ms]
