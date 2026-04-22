from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def interpolate_positions(p0: np.ndarray, p1: np.ndarray, steps: int) -> np.ndarray:
    alphas = np.linspace(0.0, 1.0, num=steps, dtype=np.float32)
    return (1.0 - alphas[:, None]) * p0[None, :] + alphas[:, None] * p1[None, :]


def interpolate_rotations(R0: np.ndarray, R1: np.ndarray, steps: int) -> np.ndarray:
    if steps <= 1:
        return np.asarray(R0, dtype=np.float32)[None, :, :]
    key_times = [0.0, 1.0]
    key_rots = Rotation.from_matrix(np.stack([R0, R1], axis=0))
    slerp = Slerp(key_times, key_rots)
    t = np.linspace(0.0, 1.0, num=steps)
    return slerp(t).as_matrix().astype(np.float32)

