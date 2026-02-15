# ENavi++ Dataset Format (pipeline output)

A single training sample is a dictionary with optional modalities:

| Key | Type | Shape | Description |
|-----|------|-------|-------------|
| `t_us` | int | scalar | Anchor timestamp (microseconds) |
| `events_voxel` | float tensor | `[C,H,W]` or `[P,C,H,W]` | Voxel grid(s). `P` = past steps if history stacking is enabled |
| `rgb` | uint8/float tensor | `[3,H,W]` or `[P,3,H,W]` | Optional RGB frame(s) |
| `imu` | float tensor | `[T, D]` | Optional IMU (accel+gyro) |
| `gt` | dict | varies | Optional ground truth (pose, velocity, disparity, etc.) |
| `meta` | dict | varies | Sequence name, frame index, calibration IDs, etc. |

## DSEC-specific notes

The DSEC dataset stores events in HDF5 files with custom compression (requires `hdf5plugin`).

### Raw event HDF5 schema
```
events/
  x    -> uint16, pixel x-coordinate
  y    -> uint16, pixel y-coordinate
  p    -> uint8,  polarity (0 or 1)
  t    -> uint32, timestamp in microseconds (relative, add t_offset)
ms_to_idx  -> int64, maps millisecond -> event index for fast slicing
t_offset   -> int64, base timestamp offset
```

### Voxel grid representation
Events within a time window (default 50ms) are converted to a **voxel grid** tensor of shape `[num_bins, H, W]` (default `[15, 480, 640]`):
- Bins represent temporal slices within the window
- Each bin accumulates polarity-weighted event contributions using bilinear interpolation in (x, y, t)
- Normalized to zero mean, unit std over nonzero entries

### Disparity ground truth
- Stored as 16-bit PNGs in `disparity/event/`
- Divide by 256 to get disparity in pixels as float32
- Shape: `[480, 640]`

This repo focuses on producing consistent tensors and manifests; model training lives elsewhere.
