from __future__ import annotations

import numpy as np


def make_pose(position: np.ndarray, rotation: np.ndarray) -> dict:
    return {
        "position": np.asarray(position, dtype=np.float32).reshape(3),
        "rotation": np.asarray(rotation, dtype=np.float32).reshape(3, 3),
    }


def lerp(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    return (1.0 - alpha) * np.asarray(a, dtype=np.float32) + alpha * np.asarray(
        b, dtype=np.float32
    )


def project_points_to_plane_distance(points: np.ndarray, plane_z: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    return points[:, 2] - float(plane_z)

