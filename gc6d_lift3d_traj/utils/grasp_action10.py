"""10D Lift3D/GC6D grasp action: [t(3), R_col0(3), R_col1(3), width(1)]."""

from __future__ import annotations

import numpy as np


def grasp_matrix_width_to_action10(center: np.ndarray, R: np.ndarray, width: float) -> np.ndarray:
    center = np.asarray(center, dtype=np.float32).reshape(3)
    R = np.asarray(R, dtype=np.float32).reshape(3, 3)
    c1, c2 = R[:, 0], R[:, 1]
    w = np.array([float(width)], dtype=np.float32)
    return np.concatenate([center, c1, c2, w], axis=0).astype(np.float32)
