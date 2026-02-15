# ENavi++ Dataset Format (pipeline output)

## Preprocessed H5 file schema

Each H5 file contains one or more sequences. Created by `scripts/preprocess_dsec_to_h5.py`.

```
/ (root attrs)
  dataset           = "DSEC"
  created_by        = "enavipp-data-pipeline"
  voxel_num_bins    = 5
  voxel_window_ms   = 50
  voxel_height      = 480
  voxel_width       = 640
  events_rectified  = true
  anchor            = "image_timestamps"
  rgb_store         = "jpeg_bytes"
  rgb_resize_w      = 640
  rgb_resize_h      = 480

sequences/
  <sequence_name>/            # e.g., thun_00_a
    attrs: {num_samples: N}
    t_start_us           (N,)           int64      # voxel window start (us)
    t_end_us             (N,)           int64      # voxel window end = anchor ts
    events/voxel         (N,C,H,W)      float16    # C=num_bins, gzip compressed
    rgb/left/jpeg        (N,)           vlen bytes # JPEG encoded left RGB frame
    rgb/left/t_us        (N,)           int64      # RGB timestamp (us)
    gt/disparity_frame_idx (N,)         int32      # index into disp PNGs, -1=none
    imu/t_us             (M,)           int64      # (future) IMU timestamps
    imu/data             (M,D)          float32    # (future) accel+gyro
    imu/ptr              (N+1,)         int64      # (future) ragged array pointers
```

## PyTorch batch format

When loaded via `EnavippH5Dataset` + `collate_enavipp`:

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `voxel` | `(B, C, H, W)` | float32 | Voxel grids, C = num_bins |
| `rgb_left` | `(B, 3, H, W)` | float32 | RGB frames normalized to [0,1] |
| `t_start_us` | `(B,)` | int64 | Window start timestamps |
| `t_end_us` | `(B,)` | int64 | Window end timestamps (anchor) |
| `gt_disp_idx` | `(B,)` | long | Disparity PNG index (-1 = unavailable) |
| `imu_data` | `(B, T, D)` | float32 | Zero-padded IMU (when available) |
| `imu_t_us` | `(B, T)` | int64 | IMU timestamps (padded) |
| `imu_mask` | `(B, T)` | bool | True = valid IMU sample |
| `meta` | list[dict] | — | `{dataset, sequence, sample_idx}` |

## DSEC raw data notes

### Event HDF5 schema (`events.h5`, requires `hdf5plugin`)
```
events/x    -> uint16   # pixel x (0..639)
events/y    -> uint16   # pixel y (0..479)
events/p    -> uint8    # polarity (0=OFF, 1=ON)
events/t    -> uint32   # timestamp (us, relative to t_offset)
ms_to_idx   -> int64    # ms → event index (fast slicing)
t_offset    -> int64    # add to t for absolute timestamps
```

### Disparity GT
- 16-bit PNGs in `disparity/event/`
- Convert: `float_disp = imread(path, IMREAD_ANYDEPTH).astype('float32') / 256`
- Shape: `[480, 640]`

### RGB images
- PNG frames in `images/{left,right}/rectified/`
- Timestamps in `images/image_timestamps.txt` (one per line, microseconds)
- 20 Hz capture rate (~50 ms between frames)

### Sensor setup
- 2x Prophesee Gen3.1 event cameras (640×480, stereo)
- 2x FLIR Blackfly S RGB cameras (stereo)
- Velodyne VLP-16 LiDAR
- Bosch BMI085 IMU
- ~60 cm stereo baseline
