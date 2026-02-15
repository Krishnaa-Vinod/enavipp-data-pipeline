# ENavi++ Data Pipeline -- Roadmap

## Phase 1: Download + Visualize (current)
- [x] DSEC debug subset download/extract scripts (idempotent, resumable)
- [x] Vendored DSEC voxel grid loader
- [x] Inspect script: load voxel grids, save event-sum + disparity visualizations
- [x] Rename to ENavi++/enavipp, clean namespace
- [ ] Document DSEC data format in detail (HDF5 schema, disparity, timestamps)
- [ ] Add more visualization: per-bin slices, event count histograms, timestamp distributions

## Phase 2: Data Preprocessing
- [ ] Design preprocessing pipeline: raw events -> aligned voxel grid sequences
- [ ] Support full DSEC download (events + images + lidar/imu)
- [ ] Timestamp alignment utilities (event-to-disparity sync, windowing)
- [ ] Configurable voxelization (bins, window size, normalization, downsampling)
- [ ] History stacking: produce `[P, C, H, W]` tensors with P past voxel grids
- [ ] Manifest-based dataset generation (JSON/CSV index of all samples)
- [ ] Save preprocessed tensors to disk (`.pt` or `.npz`)

## Phase 3: Dataloader
- [ ] PyTorch Dataset/DataLoader that reads preprocessed tensors
- [ ] Public API so anyone can: download preprocessed data -> load in their model
- [ ] Support for multiple modalities (events, RGB, IMU, GT)
- [ ] Train/val/test splits

## Phase 4: Future Extensions
- [ ] Custom dataset support (non-DSEC event cameras)
- [ ] NoMaD-style tuple generation (past context + goal + future actions)
- [ ] Integration with navigation policy training repos
