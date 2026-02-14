from pathlib import Path
import weakref

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.representations import VoxelGrid
from utils.eventslicer import EventSlicer


class Sequence(Dataset):
    def __init__(self, seq_path: Path, mode: str = "train", delta_t_ms: int = 50, num_bins: int = 15):
        assert num_bins >= 1
        assert delta_t_ms <= 100, "adapt this code if duration is higher than 100 ms"
        assert seq_path.is_dir()

        self.mode = mode
        self.height = 480
        self.width = 640
        self.num_bins = num_bins
        self.voxel_grid = VoxelGrid(self.num_bins, self.height, self.width, normalize=True)
        self.locations = ["left", "right"]
        self.delta_t_us = delta_t_ms * 1000

        disp_dir = seq_path / "disparity"
        assert disp_dir.is_dir()

        self.timestamps = np.loadtxt(disp_dir / "timestamps.txt", dtype="int64")

        ev_disp_dir = disp_dir / "event"
        assert ev_disp_dir.is_dir()

        disp_gt_pathstrings = [str(p) for p in ev_disp_dir.iterdir() if str(p.name).endswith(".png")]
        disp_gt_pathstrings.sort()
        self.disp_gt_pathstrings = disp_gt_pathstrings

        assert len(self.disp_gt_pathstrings) == self.timestamps.size

        # remove first: no events before first disparity
        assert int(Path(self.disp_gt_pathstrings[0]).stem) == 0
        self.disp_gt_pathstrings.pop(0)
        self.timestamps = self.timestamps[1:]

        self.h5f = {}
        self.rectify_ev_maps = {}
        self.event_slicers = {}

        ev_dir = seq_path / "events"
        for location in self.locations:
            ev_dir_location = ev_dir / location
            ev_data_file = ev_dir_location / "events.h5"
            ev_rect_file = ev_dir_location / "rectify_map.h5"

            h5f_location = h5py.File(str(ev_data_file), "r")
            self.h5f[location] = h5f_location
            self.event_slicers[location] = EventSlicer(h5f_location)

            with h5py.File(str(ev_rect_file), "r") as h5_rect:
                self.rectify_ev_maps[location] = h5_rect["rectify_map"][()]

        self._finalizer = weakref.finalize(self, self.close_callback, self.h5f)

    def __len__(self):
        return len(self.disp_gt_pathstrings)

    @staticmethod
    def get_disparity_map(filepath: Path):
        disp_16bit = cv2.imread(str(filepath), cv2.IMREAD_ANYDEPTH)
        return disp_16bit.astype("float32") / 256

    @staticmethod
    def close_callback(h5f_dict):
        for _, h5f in h5f_dict.items():
            h5f.close()

    def rectify_events(self, x: np.ndarray, y: np.ndarray, location: str):
        rectify_map = self.rectify_ev_maps[location]
        return rectify_map[y, x]

    def events_to_voxel_grid(self, x, y, p, t):
        t = (t - t[0]).astype("float32")
        t = (t / t[-1])
        return self.voxel_grid.convert(
            torch.from_numpy(x.astype("float32")),
            torch.from_numpy(y.astype("float32")),
            torch.from_numpy(p.astype("float32")),
            torch.from_numpy(t),
        )

    def __getitem__(self, index):
        ts_end = self.timestamps[index]
        ts_start = ts_end - self.delta_t_us

        disp_gt_path = Path(self.disp_gt_pathstrings[index])
        file_index = int(disp_gt_path.stem)

        output = {
            "disparity_gt": self.get_disparity_map(disp_gt_path),
            "file_index": file_index,
        }

        for location in self.locations:
            event_data = self.event_slicers[location].get_events(ts_start, ts_end)
            p = event_data["p"]
            t = event_data["t"]
            x = event_data["x"]
            y = event_data["y"]

            xy_rect = self.rectify_events(x, y, location)
            x_rect = xy_rect[:, 0]
            y_rect = xy_rect[:, 1]

            event_representation = self.events_to_voxel_grid(x_rect, y_rect, p, t)
            output.setdefault("representation", {})
            output["representation"][location] = event_representation

        return output
