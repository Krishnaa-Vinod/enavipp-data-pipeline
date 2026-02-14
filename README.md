# nomad-eventvoxels

NoMaD-style policy learning in **event voxel** space (DSEC as the first dataset).

## Quickstart (debug subset)
```bash
# Set up environment (on ASU Sol)
module purge
module load mamba/latest
mamba create -n nomad-eventvoxels --clone pytorch-gpu-2.2.1
source activate nomad-eventvoxels
mamba install -c conda-forge hdf5plugin opencv matplotlib tqdm pyyaml rich numba

# Download, extract, inspect
bash scripts/download_dsec.sh --mode debug
bash scripts/extract_dsec.sh --mode debug
python scripts/inspect_dsec.py --dsec_root data/raw/DSEC --num_batches 3 --save_dir artifacts/inspect
```

## Notes
- Raw DSEC data is **NOT** committed (see `.gitignore`).
- Start with `--mode debug` to validate the pipeline, then switch to `--mode full`.
