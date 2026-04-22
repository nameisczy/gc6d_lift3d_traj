"""Convert EE pose + width to GC6D 17D grasp row (camera frame)."""

from __future__ import annotations

import numpy as np


def pose_width_to_grasp17d(
    center: np.ndarray,
    rotation: np.ndarray,
    width: float,
    score: float = 1.0,
    height: float = 0.02,
    depth: float = 0.02,
    object_id: float = -1.0,
) -> np.ndarray:
    row = np.zeros(17, dtype=np.float32)
    row[0] = score
    row[1] = width
    row[2] = height
    row[3] = depth
    R = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
    row[4:13] = R.reshape(9)
    row[13:16] = np.asarray(center, dtype=np.float32).reshape(3)
    row[16] = object_id
    return row
