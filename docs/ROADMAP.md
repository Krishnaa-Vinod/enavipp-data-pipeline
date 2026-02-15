# ENavi++ Data Pipeline — Roadmap

## Phase 1: Download + Visualize ✅
- [x] DSEC debug subset download/extract scripts (idempotent, resumable)
- [x] Support RGB images download (`--include_images`)
- [x] Support lidar/IMU download (`--include_lidar_imu`)
- [x] Vendored DSEC voxel grid loader (VoxelGrid + EventSlicer)
- [x] Inspect script: load voxel grids, save event-sum + disparity visualizations
- [x] Rename to ENavi++/enavipp, clean namespace

## Phase 2: Preprocessing Pipeline ✅
- [x] Configurable voxelization (bins, window, height, width, normalization, rectification)
- [x] YAML-based preprocessing config (`configs/preprocess/`)
- [x] H5Writer utility for incremental HDF5 writing with gzip compression
- [x] Preprocessing module: raw DSEC events + RGB → single H5 file
- [x] Anchored to image timestamps (or disparity timestamps as fallback)
- [x] JPEG-encoded RGB storage (saves ~80% vs raw PNG arrays)
- [x] Disparity GT index mapping per voxel window
- [x] CLI script: `scripts/preprocess_dsec_to_h5.py`
- [ ] IMU extraction from rosbags (requires `rosbags` package)
- [ ] History stacking: `[P, C, H, W]` tensors with P past voxel grids

## Phase 3: Dataloader ✅
- [x] `EnavippH5Dataset` (PyTorch Dataset) — lazy H5 open, multi-worker safe
- [x] JPEG decode on-the-fly for RGB (→ `[3,H,W]` float32 tensor)
- [x] Custom `collate_enavipp()` with IMU padding + mask
- [x] H5 inspection/visualization script (`scripts/inspect_preprocessed_h5.py`)
- [ ] Multi-H5 federated dataset (combine multiple sequences)
- [ ] Train/val/test split utilities

## Phase 4: Future Extensions
- [ ] Custom dataset support (non-DSEC event cameras)
- [ ] NoMaD-style tuple generation (past context + goal + future actions)
- [ ] Integration with navigation policy training repos
- [ ] Multi-resolution voxelization (coarse-to-fine)
- [ ] Online preprocessing mode (no disk write, stream events → voxels)

