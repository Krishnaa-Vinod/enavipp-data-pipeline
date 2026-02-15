# Custom Dataset Guide (future)

## Recommended raw layout

```
data/raw/CUSTOM/<sequence_id>/
  events/left/events.h5
  events/left/rectify_map.h5            # optional
  rgb/left/timestamps.txt               # optional
  rgb/left/frames/000000.png            # optional
  imu/imu.csv                           # optional
  gt/pose.csv                           # optional
  calibration/*.yaml                    # optional
```

## Event HDF5 schema (recommended)
Match DSEC-style keys if possible:
- `events/x`, `events/y`, `events/p`, `events/t`
- `ms_to_idx`
- `t_offset`

This allows reuse of the same EventSlicer + voxelization code.

## Adding a new dataset
1. Place raw data under `data/raw/<DATASET_NAME>/`
2. Create a new dataset class in `src/enavipp/data/datasets/<name>.py`
3. Implement the `__getitem__` method returning a standardized sample dict (see `docs/DATASET_FORMAT.md`)
4. Add a download/extract script pair in `scripts/`
