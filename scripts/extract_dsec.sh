#!/usr/bin/env bash
set -euo pipefail

MODE="debug"
ZIP_DIR="data/raw/_zips/dsec"
DSEC_ROOT="data/raw/DSEC"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2;;
    --zip_dir) ZIP_DIR="$2"; shift 2;;
    --dsec_root) DSEC_ROOT="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

mkdir -p "$DSEC_ROOT"

if [[ "$MODE" == "debug" ]]; then
  SEQ="thun_00_a"
  SEQ_DIR="$DSEC_ROOT/train/$SEQ"

  mkdir -p "$SEQ_DIR/events/left" "$SEQ_DIR/events/right" "$SEQ_DIR/disparity/event"

  echo "[extract_dsec] extracting events left"
  unzip -n "$ZIP_DIR/${SEQ}_events_left.zip" -d "$SEQ_DIR/events/left" >/dev/null

  echo "[extract_dsec] extracting events right"
  unzip -n "$ZIP_DIR/${SEQ}_events_right.zip" -d "$SEQ_DIR/events/right" >/dev/null

  echo "[extract_dsec] extracting disparity_event"
  unzip -n "$ZIP_DIR/${SEQ}_disparity_event.zip" -d "$SEQ_DIR/disparity" >/dev/null

  # The disparity zip may extract PNGs into disparity/ directly or into disparity/event/.
  # Ensure final layout is disparity/event/*.png
  if [[ -d "$SEQ_DIR/disparity/event/event" ]]; then
    mv "$SEQ_DIR/disparity/event/event"/* "$SEQ_DIR/disparity/event/" || true
    rmdir "$SEQ_DIR/disparity/event/event" || true
  fi

  # Move any PNGs that ended up in disparity/ directly into disparity/event/
  if compgen -G "$SEQ_DIR/disparity/*.png" > /dev/null; then
    mv "$SEQ_DIR/disparity/"*.png "$SEQ_DIR/disparity/event/" || true
  fi

  echo "[extract_dsec] placing timestamps.txt"
  cp -f "$ZIP_DIR/${SEQ}_disparity_timestamps.txt" "$SEQ_DIR/disparity/timestamps.txt"

  echo "[extract_dsec] debug subset extracted to $SEQ_DIR"

elif [[ "$MODE" == "full" ]]; then
  echo "[extract_dsec] extracting full train zips into $DSEC_ROOT"
  unzip -n "$ZIP_DIR/train_events.zip" -d "$DSEC_ROOT" >/dev/null
  unzip -n "$ZIP_DIR/train_disparity.zip" -d "$DSEC_ROOT" >/dev/null
  unzip -n "$ZIP_DIR/train_calibration.zip" -d "$DSEC_ROOT" >/dev/null

  echo "[extract_dsec] done. (Optional) lidar_imu.zip can be extracted separately when needed."
else
  echo "MODE must be debug or full (got: $MODE)"; exit 1
fi
