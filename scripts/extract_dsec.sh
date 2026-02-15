#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
# ENavi++ / DSEC extract script
# Extracts downloaded archives into a consistent directory layout.
# ──────────────────────────────────────────────────────────────────
set -euo pipefail

MODE="debug"
SPLIT="train"
SEQUENCE="thun_00_a"
ZIP_DIR="data/raw/_zips/dsec"
DSEC_ROOT="data/raw/DSEC"
INCLUDE_IMAGES=1
INCLUDE_LIDAR_IMU=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)              MODE="$2";              shift 2;;
    --split)             SPLIT="$2";             shift 2;;
    --sequence)          SEQUENCE="$2";          shift 2;;
    --zip_dir)           ZIP_DIR="$2";           shift 2;;
    --dsec_root)         DSEC_ROOT="$2";         shift 2;;
    --include_images)    INCLUDE_IMAGES="$2";    shift 2;;
    --include_lidar_imu) INCLUDE_LIDAR_IMU="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

mkdir -p "$DSEC_ROOT"

# Helper: extract zip if archive exists, skip if target marker exists
safe_unzip() {
  local archive="$1"
  local dest="$2"
  if [[ ! -f "$archive" ]]; then
    echo "[extract_dsec] SKIP (archive missing): $archive"
    return 0
  fi
  mkdir -p "$dest"
  echo "[extract_dsec] extracting: $archive -> $dest"
  unzip -n "$archive" -d "$dest" >/dev/null
}

# ─── DEBUG: single sequence ────────────────────────────────────────
if [[ "$MODE" == "debug" ]]; then
  SEQ="$SEQUENCE"
  SEQ_DIR="$DSEC_ROOT/$SPLIT/$SEQ"

  # --- Events ---
  mkdir -p "$SEQ_DIR/events/left" "$SEQ_DIR/events/right"
  safe_unzip "$ZIP_DIR/${SEQ}_events_left.zip"  "$SEQ_DIR/events/left"
  safe_unzip "$ZIP_DIR/${SEQ}_events_right.zip" "$SEQ_DIR/events/right"

  # --- Disparity GT ---
  mkdir -p "$SEQ_DIR/disparity/event"
  safe_unzip "$ZIP_DIR/${SEQ}_disparity_event.zip" "$SEQ_DIR/disparity"

  # Fix: PNGs may end up in disparity/ directly or in nested event/event/
  if [[ -d "$SEQ_DIR/disparity/event/event" ]]; then
    mv "$SEQ_DIR/disparity/event/event"/* "$SEQ_DIR/disparity/event/" 2>/dev/null || true
    rmdir "$SEQ_DIR/disparity/event/event" 2>/dev/null || true
  fi
  if compgen -G "$SEQ_DIR/disparity/*.png" > /dev/null 2>&1; then
    mv "$SEQ_DIR/disparity/"*.png "$SEQ_DIR/disparity/event/" 2>/dev/null || true
  fi

  # Timestamps
  if [[ -f "$ZIP_DIR/${SEQ}_disparity_timestamps.txt" ]]; then
    cp -f "$ZIP_DIR/${SEQ}_disparity_timestamps.txt" "$SEQ_DIR/disparity/timestamps.txt"
  fi

  # --- Calibration (optional) ---
  if [[ -f "$ZIP_DIR/${SEQ}_calibration.zip" ]]; then
    mkdir -p "$SEQ_DIR/calibration"
    safe_unzip "$ZIP_DIR/${SEQ}_calibration.zip" "$SEQ_DIR/calibration"
    # Some zips nest: calibration/calibration/cam_to_cam.yaml -> flatten
    if [[ -d "$SEQ_DIR/calibration/calibration" ]]; then
      mv "$SEQ_DIR/calibration/calibration"/* "$SEQ_DIR/calibration/" 2>/dev/null || true
      rmdir "$SEQ_DIR/calibration/calibration" 2>/dev/null || true
    fi
  fi

  # --- RGB images (if downloaded) ---
  if [[ "$INCLUDE_IMAGES" == "1" ]]; then
    mkdir -p "$SEQ_DIR/images/left/rectified" "$SEQ_DIR/images/right/rectified"
    safe_unzip "$ZIP_DIR/${SEQ}_images_rectified_left.zip" "$SEQ_DIR/images/left"
    safe_unzip "$ZIP_DIR/${SEQ}_images_rectified_right.zip" "$SEQ_DIR/images/right"

    # Fix nested extraction: some zips create rectified/rectified/
    for side in left right; do
      if [[ -d "$SEQ_DIR/images/$side/rectified/rectified" ]]; then
        mv "$SEQ_DIR/images/$side/rectified/rectified"/* "$SEQ_DIR/images/$side/rectified/" 2>/dev/null || true
        rmdir "$SEQ_DIR/images/$side/rectified/rectified" 2>/dev/null || true
      fi
      # Some zips extract PNGs directly into images/left/ instead of images/left/rectified/
      if compgen -G "$SEQ_DIR/images/$side/*.png" > /dev/null 2>&1; then
        mv "$SEQ_DIR/images/$side/"*.png "$SEQ_DIR/images/$side/rectified/" 2>/dev/null || true
      fi
    done

    # Image timestamps
    for f in image_timestamps image_exposure_timestamps_left image_exposure_timestamps_right; do
      if [[ -f "$ZIP_DIR/${SEQ}_${f}.txt" ]]; then
        cp -f "$ZIP_DIR/${SEQ}_${f}.txt" "$SEQ_DIR/images/${f}.txt"
      fi
    done
  fi

  # --- Lidar / IMU ---
  if [[ "$INCLUDE_LIDAR_IMU" == "1" && -f "$ZIP_DIR/lidar_imu.zip" ]]; then
    echo "[extract_dsec] extracting lidar_imu.zip (all sequences)"
    safe_unzip "$ZIP_DIR/lidar_imu.zip" "$DSEC_ROOT/lidar_imu"
  fi

  echo "[extract_dsec] debug subset ($SEQ) extracted to $SEQ_DIR"

# ─── FULL: bulk archives ──────────────────────────────────────────
elif [[ "$MODE" == "full" ]]; then
  echo "[extract_dsec] extracting full train zips into $DSEC_ROOT"
  safe_unzip "$ZIP_DIR/train_events.zip"       "$DSEC_ROOT"
  safe_unzip "$ZIP_DIR/train_disparity.zip"    "$DSEC_ROOT"
  safe_unzip "$ZIP_DIR/train_calibration.zip"  "$DSEC_ROOT"

  if [[ "$INCLUDE_IMAGES" == "1" && -f "$ZIP_DIR/train_images.zip" ]]; then
    safe_unzip "$ZIP_DIR/train_images.zip" "$DSEC_ROOT"
  fi

  if [[ "$INCLUDE_LIDAR_IMU" == "1" && -f "$ZIP_DIR/lidar_imu.zip" ]]; then
    safe_unzip "$ZIP_DIR/lidar_imu.zip" "$DSEC_ROOT/lidar_imu"
  fi

  echo "[extract_dsec] done."
else
  echo "MODE must be debug or full (got: $MODE)"; exit 1
fi
