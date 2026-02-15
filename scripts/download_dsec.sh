#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
# ENavi++ / DSEC download script
# Downloads events, disparity, (optional) RGB images, (optional) lidar/IMU
# ──────────────────────────────────────────────────────────────────
set -euo pipefail

MODE="debug"
SPLIT="train"
SEQUENCE="thun_00_a"
ZIP_DIR="data/raw/_zips/dsec"
INCLUDE_IMAGES=1
INCLUDE_LIDAR_IMU=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)              MODE="$2";              shift 2;;
    --split)             SPLIT="$2";             shift 2;;
    --sequence)          SEQUENCE="$2";          shift 2;;
    --zip_dir)           ZIP_DIR="$2";           shift 2;;
    --include_images)    INCLUDE_IMAGES="$2";    shift 2;;
    --include_lidar_imu) INCLUDE_LIDAR_IMU="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

mkdir -p "$ZIP_DIR"

have_aria2c() { command -v aria2c >/dev/null 2>&1; }

fetch() {
  local url="$1"
  local out="$2"
  local dst="$ZIP_DIR/$out"
  local optional="${3:-0}"

  if [[ -f "$dst" ]]; then
    echo "[download_dsec] exists: $dst"
    return 0
  fi

  echo "[download_dsec] downloading: $url -> $dst"

  if have_aria2c; then
    aria2c -x 8 -s 8 -k 1M -c -o "$out" -d "$ZIP_DIR" "$url" || {
      if [[ "$optional" == "1" ]]; then echo "[download_dsec] WARN: optional file not available: $url"; return 0; fi
      return 1
    }
  else
    curl -L -C - -o "$dst" --fail "$url" || {
      if [[ "$optional" == "1" ]]; then echo "[download_dsec] WARN: optional file not available: $url"; rm -f "$dst"; return 0; fi
      return 1
    }
  fi
}

BASE="https://download.ifi.uzh.ch/rpg/DSEC"

# ─── DEBUG: single sequence ────────────────────────────────────────
if [[ "$MODE" == "debug" ]]; then
  SEQ="$SEQUENCE"
  SEQ_URL="$BASE/$SPLIT/$SEQ"

  # Events (always)
  fetch "$SEQ_URL/${SEQ}_events_left.zip"        "${SEQ}_events_left.zip"
  fetch "$SEQ_URL/${SEQ}_events_right.zip"       "${SEQ}_events_right.zip"

  # Disparity GT
  fetch "$SEQ_URL/${SEQ}_disparity_event.zip"    "${SEQ}_disparity_event.zip"
  fetch "$SEQ_URL/${SEQ}_disparity_timestamps.txt" "${SEQ}_disparity_timestamps.txt"

  # Calibration (optional — not all sequences have it individually)
  fetch "$SEQ_URL/${SEQ}_calibration.zip"        "${SEQ}_calibration.zip" 1

  # RGB images (if requested)
  if [[ "$INCLUDE_IMAGES" == "1" ]]; then
    fetch "$SEQ_URL/${SEQ}_images_rectified_left.zip"  "${SEQ}_images_rectified_left.zip"
    fetch "$SEQ_URL/${SEQ}_images_rectified_right.zip" "${SEQ}_images_rectified_right.zip" 1
    fetch "$SEQ_URL/${SEQ}_image_timestamps.txt"       "${SEQ}_image_timestamps.txt"
    fetch "$SEQ_URL/${SEQ}_image_exposure_timestamps_left.txt"  "${SEQ}_image_exposure_timestamps_left.txt" 1
    fetch "$SEQ_URL/${SEQ}_image_exposure_timestamps_right.txt" "${SEQ}_image_exposure_timestamps_right.txt" 1
  fi

  # Lidar / IMU (single global archive)
  if [[ "$INCLUDE_LIDAR_IMU" == "1" ]]; then
    echo "[download_dsec] NOTE: lidar_imu.zip is a single ~3.8 GB archive covering ALL sequences."
    fetch "$BASE/lidar_imu.zip" "lidar_imu.zip"
  fi

  echo "[download_dsec] debug subset ($SEQ) downloaded."

# ─── FULL: bulk archives ──────────────────────────────────────────
elif [[ "$MODE" == "full" ]]; then
  echo "[download_dsec] WARNING: Full download is 50+ GB. Ctrl-C now if you don't have space."
  sleep 3

  fetch "$BASE/train_coarse/train_events.zip"       "train_events.zip"
  fetch "$BASE/train_coarse/train_disparity.zip"     "train_disparity.zip"
  fetch "$BASE/train_coarse/train_calibration.zip"   "train_calibration.zip"

  if [[ "$INCLUDE_IMAGES" == "1" ]]; then
    fetch "$BASE/train_coarse/train_images.zip"      "train_images.zip"
  fi

  if [[ "$INCLUDE_LIDAR_IMU" == "1" ]]; then
    fetch "$BASE/lidar_imu.zip"                      "lidar_imu.zip"
  fi

  echo "[download_dsec] full train downloads completed."
else
  echo "MODE must be debug or full (got: $MODE)"; exit 1
fi
