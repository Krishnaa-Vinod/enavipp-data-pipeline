#!/usr/bin/env bash
set -euo pipefail

MODE="debug"
ZIP_DIR="data/raw/_zips/dsec"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2;;
    --zip_dir) ZIP_DIR="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

mkdir -p "$ZIP_DIR"

have_aria2c() { command -v aria2c >/dev/null 2>&1; }

fetch() {
  local url="$1"
  local out="$2"
  local dst="$ZIP_DIR/$out"

  if [[ -f "$dst" ]]; then
    echo "[download_dsec] exists: $dst"
    return 0
  fi

  echo "[download_dsec] downloading: $url -> $dst"

  if have_aria2c; then
    aria2c -x 8 -s 8 -k 1M -c -o "$out" -d "$ZIP_DIR" "$url"
  else
    # -L follow redirects, -C - resume, -o output
    curl -L -C - -o "$dst" "$url"
  fi
}

if [[ "$MODE" == "debug" ]]; then
  # Small single-sequence subset (thun_00_a)
  fetch "https://download.ifi.uzh.ch/rpg/DSEC/train/thun_00_a/thun_00_a_events_left.zip" "thun_00_a_events_left.zip"
  fetch "https://download.ifi.uzh.ch/rpg/DSEC/train/thun_00_a/thun_00_a_events_right.zip" "thun_00_a_events_right.zip"
  fetch "https://download.ifi.uzh.ch/rpg/DSEC/train/thun_00_a/thun_00_a_disparity_event.zip" "thun_00_a_disparity_event.zip"
  fetch "https://download.ifi.uzh.ch/rpg/DSEC/train/thun_00_a/thun_00_a_disparity_timestamps.txt" "thun_00_a_disparity_timestamps.txt"

  echo "[download_dsec] debug subset downloaded."

elif [[ "$MODE" == "full" ]]; then
  fetch "https://download.ifi.uzh.ch/rpg/DSEC/train_coarse/train_events.zip" "train_events.zip"
  fetch "https://download.ifi.uzh.ch/rpg/DSEC/train_coarse/train_disparity.zip" "train_disparity.zip"
  fetch "https://download.ifi.uzh.ch/rpg/DSEC/train_coarse/train_calibration.zip" "train_calibration.zip"
  # optional but useful later
  fetch "https://download.ifi.uzh.ch/rpg/DSEC/lidar_imu.zip" "lidar_imu.zip"

  echo "[download_dsec] full train downloads completed."
else
  echo "MODE must be debug or full (got: $MODE)"; exit 1
fi
