"""DSEC IMU extraction from ROS1 bag files.

The DSEC lidar_imu archive contains one bag file per sequence at:
    <lidar_imu_root>/<sequence_name>.bag

Each bag contains sensor_msgs/Imu messages with:
    - header.stamp: ROS time (secs + nsecs)
    - linear_acceleration: (x, y, z) in m/s^2
    - angular_velocity: (x, y, z) in rad/s

This module extracts the IMU stream and slices it into windows aligned
with the preprocessed voxel grid anchor timestamps.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── bag discovery ───────────────────────────────────────────────────

def find_bag_for_sequence(lidar_imu_root: Path, sequence_name: str) -> Path:
    """Locate the bag file for a given DSEC sequence.

    DSEC bags cover full recordings — e.g., ``thun_00_a`` and ``thun_00_b``
    share the same bag ``thun_00/lidar_imu.bag``.  We strip the trailing
    segment letter (``_a``, ``_b``, …) to derive the base name.

    Search order:
        <root>/<sequence>.bag
        <root>/<sequence>/<sequence>.bag
        <root>/data/<base>/lidar_imu.bag      (DSEC zip layout)
        <root>/<base>/lidar_imu.bag
        <root>/<base>.bag
        <root>/**/<base>*.bag                 (recursive fallback)
    """
    import re
    # Derive base recording name: thun_00_a → thun_00
    base = re.sub(r'_[a-z]$', '', sequence_name)

    candidates = [
        lidar_imu_root / f"{sequence_name}.bag",
        lidar_imu_root / sequence_name / f"{sequence_name}.bag",
        lidar_imu_root / "data" / base / "lidar_imu.bag",
        lidar_imu_root / base / "lidar_imu.bag",
        lidar_imu_root / f"{base}.bag",
    ]
    for p in candidates:
        if p.is_file():
            logger.info("Found bag: %s", p)
            return p

    # Recursive fallback — search for base name
    found = list(lidar_imu_root.rglob(f"{base}*.bag"))
    if not found:
        found = list(lidar_imu_root.rglob(f"*{base}*/*.bag"))
    if found:
        logger.info("Found bag (recursive): %s", found[0])
        return found[0]

    raise FileNotFoundError(
        f"No bag file found for sequence '{sequence_name}' (base='{base}') "
        f"under {lidar_imu_root}. Searched: {[str(c) for c in candidates]} + recursive."
    )


# ── topic listing ───────────────────────────────────────────────────

def list_topics(bag_path: Path) -> List[Tuple[str, str, int]]:
    """List all topics in a bag: [(topic_name, msg_type, count)]."""
    from rosbags.rosbag1 import Reader

    with Reader(bag_path) as reader:
        result = []
        for conn in reader.connections:
            count = sum(1 for _ in reader.messages(connections=[conn]))
            result.append((conn.topic, conn.msgtype, count))
    return result


def _auto_detect_imu_topic(bag_path: Path, override: Optional[str] = None) -> str:
    """Auto-detect the IMU topic by message type, or use the override."""
    if override:
        return override

    from rosbags.rosbag1 import Reader

    with Reader(bag_path) as reader:
        for conn in reader.connections:
            if "Imu" in conn.msgtype or "imu" in conn.topic.lower():
                logger.info("Auto-detected IMU topic: %s (type: %s)", conn.topic, conn.msgtype)
                return conn.topic

    raise ValueError(
        f"No IMU topic found in {bag_path}. "
        f"Topics: {[(c.topic, c.msgtype) for c in Reader(bag_path).__enter__().connections]}"
    )


# ── IMU stream extraction ──────────────────────────────────────────

def extract_imu_stream(
    bag_path: Path,
    topic: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract full IMU stream from a bag file.

    Returns
    -------
    t_us : int64 array [M] — timestamps in microseconds
    data : float32 array [M, 6] — (ax, ay, az, gx, gy, gz)
    """
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS1_NOETIC)

    timestamps = []
    imu_vals = []

    with Reader(bag_path) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            raise ValueError(f"Topic '{topic}' not found in {bag_path}")

        for conn, timestamp, rawdata in reader.messages(connections=conns):
            msg = typestore.deserialize_ros1(rawdata, conn.msgtype)

            # Convert ROS stamp to microseconds
            t_us = int(msg.header.stamp.sec) * 1_000_000 + int(msg.header.stamp.nanosec) // 1_000

            ax = float(msg.linear_acceleration.x)
            ay = float(msg.linear_acceleration.y)
            az = float(msg.linear_acceleration.z)
            gx = float(msg.angular_velocity.x)
            gy = float(msg.angular_velocity.y)
            gz = float(msg.angular_velocity.z)

            timestamps.append(t_us)
            imu_vals.append([ax, ay, az, gx, gy, gz])

    t_us = np.array(timestamps, dtype=np.int64)
    data = np.array(imu_vals, dtype=np.float32)

    logger.info(
        "Extracted %d IMU samples from %s (topic=%s), t_range=[%d, %d] us",
        len(t_us), bag_path.name, topic,
        int(t_us[0]) if len(t_us) > 0 else 0,
        int(t_us[-1]) if len(t_us) > 0 else 0,
    )
    return t_us, data


# ── slicing to windows ─────────────────────────────────────────────

def slice_imu_to_windows(
    imu_t_us: np.ndarray,
    imu_data: np.ndarray,
    t_start_us: np.ndarray,
    t_end_us: np.ndarray,
    tolerance_us: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slice a full IMU stream into per-window ragged arrays.

    For each window i, selects IMU samples where:
        t_start_us[i] - tolerance_us <= imu_t < t_end_us[i] + tolerance_us

    Returns
    -------
    imu_t_concat : int64 [M_total] — concatenated IMU timestamps
    imu_data_concat : float32 [M_total, 6] — concatenated IMU data
    ptr : int64 [N+1] — pointer array; window i has samples ptr[i]:ptr[i+1]
    """
    N = len(t_start_us)
    assert len(t_end_us) == N

    # Sort IMU by time (should already be sorted, but ensure)
    order = np.argsort(imu_t_us)
    imu_t_sorted = imu_t_us[order]
    imu_d_sorted = imu_data[order]

    ptr = np.zeros(N + 1, dtype=np.int64)
    chunks_t = []
    chunks_d = []

    for i in range(N):
        lo = t_start_us[i] - tolerance_us
        hi = t_end_us[i] + tolerance_us
        idx_lo = np.searchsorted(imu_t_sorted, lo, side="left")
        idx_hi = np.searchsorted(imu_t_sorted, hi, side="left")
        chunk_t = imu_t_sorted[idx_lo:idx_hi]
        chunk_d = imu_d_sorted[idx_lo:idx_hi]
        chunks_t.append(chunk_t)
        chunks_d.append(chunk_d)
        ptr[i + 1] = ptr[i] + len(chunk_t)

    if chunks_t:
        imu_t_concat = np.concatenate(chunks_t).astype(np.int64)
        imu_data_concat = np.concatenate(chunks_d).astype(np.float32)
    else:
        imu_t_concat = np.zeros(0, dtype=np.int64)
        imu_data_concat = np.zeros((0, 6), dtype=np.float32)

    total = ptr[-1]
    logger.info(
        "Sliced IMU into %d windows: total=%d samples, avg=%.1f/window, "
        "min=%d, max=%d",
        N, total,
        total / N if N > 0 else 0,
        int(np.diff(ptr).min()) if N > 0 else 0,
        int(np.diff(ptr).max()) if N > 0 else 0,
    )
    return imu_t_concat, imu_data_concat, ptr
