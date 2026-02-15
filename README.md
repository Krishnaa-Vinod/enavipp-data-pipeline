# ENavi++ Data Pipeline

A modular data pipeline for **event-camera-based navigation** research. Currently focused on the [DSEC](https://dsec.ifi.uzh.ch/) stereo event camera dataset. Inspired by [NoMaD](https://general-navigation-models.github.io/)-style past-context setups, adapted to event voxel space.

> **This is a data pipeline repo** -- it handles downloading, visualizing, understanding, preprocessing, and loading event camera datasets. Model training code lives separately.

---

## Pipeline Stages

| Stage | Status | Description |
|-------|--------|-------------|
| **1. Download** | Done | Scripts to download DSEC (debug subset or full) |
| **2. Visualize & Understand** | **Current** | Load raw data, inspect formats, save visualizations |
| **3. Preprocess** | Planned | Convert raw events to aligned voxel grid sequences |
| **4. Dataloader** | Planned | PyTorch DataLoader for preprocessed tensors |

See [docs/ROADMAP.md](docs/ROADMAP.md) for the full roadmap.

---

## Stage 1: Download DSEC

The DSEC dataset is a large-scale stereo event camera dataset for driving scenarios, captured in Zurich. Download page: https://dsec.ifi.uzh.ch/dsec-datasets/download/

### Debug subset (single sequence, ~560 MB)

Downloads only `thun_00_a` -- one short driving sequence. Use this to validate the pipeline before committing to the full download (~50+ GB).

```bash
bash scripts/download_dsec.sh --mode debug
bash scripts/extract_dsec.sh --mode debug
```

### Full training set

```bash
bash scripts/download_dsec.sh --mode full
bash scripts/extract_dsec.sh --mode full
```

### What gets downloaded

**Debug mode** downloads 4 files for sequence `thun_00_a`:

| File | Size | Contents |
|------|------|----------|
| `thun_00_a_events_left.zip` | ~285 MB | Left camera events (HDF5) |
| `thun_00_a_events_right.zip` | ~261 MB | Right camera events (HDF5) |
| `thun_00_a_disparity_event.zip` | ~17 MB | Ground truth disparity maps (16-bit PNG) |
| `thun_00_a_disparity_timestamps.txt` | ~1 KB | Timestamps for each disparity frame |

Scripts are **idempotent** -- re-running skips already-downloaded files. Uses `aria2c` for parallel download if available, falls back to `curl` with resume support.

### Extracted directory structure

```
data/raw/DSEC/
  train/
    thun_00_a/
      events/
        left/
          events.h5           # Raw events: x, y, polarity, timestamp
          rectify_map.h5      # Undistortion map for event coordinates
        right/
          events.h5
          rectify_map.h5
      disparity/
        timestamps.txt        # One timestamp (microseconds) per line
        event/
          000000.png          # 16-bit PNG, divide by 256 -> disparity in pixels
          000001.png
          ...                 # 120 frames total for thun_00_a
```

---

## Stage 2: Visualize & Understand the Data (current step)

### How DSEC event data is stored

**Events** are stored in HDF5 files (`events.h5`) with this schema:
```
events/
  x    -> uint16   # pixel x-coordinate (0..639)
  y    -> uint16   # pixel y-coordinate (0..479)
  p    -> uint8    # polarity (0=OFF, 1=ON)
  t    -> uint32   # timestamp in microseconds (relative)
ms_to_idx  -> int64  # maps millisecond -> event array index (fast slicing)
t_offset   -> int64  # add to t values to get absolute timestamps
```

The HDF5 files use custom compression filters, so **`hdf5plugin` is required** to read them.

**Disparity ground truth** is stored as 16-bit PNGs:
- `disp_float32 = cv2.imread(path, cv2.IMREAD_ANYDEPTH).astype('float32') / 256`
- Shape: `[480, 640]`
- Resolution: 480x640 (same as event camera)

**Timestamps** (`timestamps.txt`): one integer per line, microseconds, one per disparity frame.

### Voxel grid representation

Raw events are converted to **voxel grids** for neural network consumption:
- Time window: 50ms (configurable via `--delta_t_ms`)
- Bins: 15 temporal slices within the window (configurable via `--num_bins`)
- Output tensor shape: `[15, 480, 640]`
- Bilinear interpolation in (x, y, t) space
- Normalized: zero mean, unit std over nonzero entries

### Running the inspection script

```bash
python scripts/inspect_dsec.py \
    --dsec_root data/raw/DSEC \
    --num_batches 3 \
    --save_dir artifacts/inspect
```

This loads voxel grids from the dataset, prints shapes and statistics, and saves visualizations:

```
Dataset length: 119
Batch 0: vox_left shape=(1, 15, 480, 640) disp shape=(1, 480, 640) nonzero_frac=0.265521
Batch 1: vox_left shape=(1, 15, 480, 640) disp shape=(1, 480, 640) nonzero_frac=0.258040
Saved visualizations to: artifacts/inspect
```

**Output images** saved to `artifacts/inspect/`:
- `event_sum_b*.png` -- Sum of all 15 voxel bins -> grayscale "event image" showing edges and motion
- `disparity_b*.png` -- Ground truth disparity map (inferno colormap)

### DSEC sensor setup

The DSEC rig includes:
- **2x Prophesee Gen3.1 event cameras** (stereo, 640x480, left + right)
- **2x FLIR Blackfly S RGB cameras** (stereo, not used in debug subset)
- **Velodyne VLP-16 LiDAR** (optional, available via `--mode full`)
- **Bosch BMI085 IMU** (available in lidar_imu.zip)
- Baseline: ~60 cm between stereo event cameras

For more details on the sensor layout and calibration, see the [DSEC paper](https://rpg.ifi.uzh.ch/docs/RAL21_DSEC.pdf).

---

## Stage 3: Data Preprocessing (planned)

Will include:
- Configurable voxelization (see `configs/voxelization/default.yaml`)
- History stacking: `[P, C, H, W]` tensors with P past voxel grids
- Timestamp alignment between events and ground truth
- Manifest generation (JSON index of all samples)
- Save preprocessed tensors to disk

---

## Stage 4: Dataloader (planned)

Will include:
- PyTorch Dataset/DataLoader for preprocessed tensors
- Public API: download preprocessed data -> plug into any model
- Multi-modality support (events, RGB, IMU, GT)
- Train/val/test splits

---

## Environment Setup

### On ASU Sol (recommended)

```bash
# Request a GPU node
interactive -G a100:1

# Create environment
module purge
module load mamba/latest
mamba create -n enavipp python=3.11 -y
source activate enavipp

# Install PyTorch (check Sol docs for latest compatible version)
pip install torch torchvision

# Install remaining deps
pip install -r requirements.txt
```

### Verify installation

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import h5py, hdf5plugin; print('hdf5 OK')"
python -c "import enavipp; print('enavipp', enavipp.__version__)"
```

### On any Linux/macOS machine

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

---

## Repository Structure

```
enavipp-data-pipeline/
  src/enavipp/                    # Python package
    data/
      types.py                    # VoxelizationConfig, Sample dataclasses
      voxelization.py             # Event-to-voxel-grid conversion API
      datasets/
        dsec.py                   # DSEC dataset wrapper
        custom.py                 # Placeholder for future datasets
  scripts/
    download_dsec.sh              # Download DSEC (--mode debug|full)
    extract_dsec.sh               # Extract downloaded zips
    inspect_dsec.py               # Load + visualize voxel grids
  third_party/dsec_example/       # Vendored DSEC loader (from uzh-rpg/DSEC)
    dsec_dataset/                 # Provider, Sequence, VoxelGrid classes
    dsec_utils/                   # EventSlicer (HDF5 event reader)
  configs/
    voxelization/default.yaml     # Voxelization parameters
    modalities/default.yaml       # Which modalities to enable
  docs/
    ROADMAP.md                    # Development roadmap
    DATASET_FORMAT.md             # Output sample format spec
    CUSTOM_DATASET_GUIDE.md       # How to add your own dataset
  data/                           # NOT committed (see .gitignore)
    raw/DSEC/                     # Extracted DSEC data lives here
    raw/_zips/dsec/               # Downloaded zip files
  artifacts/                      # NOT committed
```

---

## Quick Reference

```bash
# Full debug pipeline (download -> extract -> visualize)
bash scripts/download_dsec.sh --mode debug
bash scripts/extract_dsec.sh --mode debug
python scripts/inspect_dsec.py --dsec_root data/raw/DSEC --num_batches 3 --save_dir artifacts/inspect

# Check what was generated
ls artifacts/inspect/
```

---

## Policy

- **No data in git.** Raw data, zips, HDF5 files, processed tensors, and checkpoints are all in `.gitignore`.
- **Idempotent scripts.** Re-running download/extract won't destroy existing data.
- **Reproducible.** Anyone can clone this repo and run the scripts to get the same results.

## License

MIT
