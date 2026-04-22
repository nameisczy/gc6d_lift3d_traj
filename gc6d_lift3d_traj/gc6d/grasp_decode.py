from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class GC6D17DLayout:
    score: int = 0
    width: int = 1
    height: int = 2
    depth: int = 3
    rotation_start: int = 4
    rotation_end: int = 13
    translation_start: int = 13
    translation_end: int = 16
    object_id: int = 16


LAYOUT = GC6D17DLayout()


def decode_gc6d_grasp(grasp_vec: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Decode GC6D/GraspNet 17D row:
    [score, width, height, depth, R(9), t(3), object_id]
    """
    g = np.asarray(grasp_vec, dtype=np.float32).reshape(-1)
    if g.shape[0] < 17:
        raise ValueError(f"Expected >=17 dims, got {g.shape}")
    R = g[LAYOUT.rotation_start : LAYOUT.rotation_end].reshape(3, 3)
    center = g[LAYOUT.translation_start : LAYOUT.translation_end]
    approach_dir = R[:, 0]  # verified from GraspNet viewpoint->matrix convention.
    return {
        "score": np.array([g[LAYOUT.score]], dtype=np.float32),
        "width": np.array([g[LAYOUT.width]], dtype=np.float32),
        "height": np.array([g[LAYOUT.height]], dtype=np.float32),
        "depth": np.array([g[LAYOUT.depth]], dtype=np.float32),
        "rotation": R.astype(np.float32),
        "center": center.astype(np.float32),
        "object_id": np.array([g[LAYOUT.object_id]], dtype=np.float32),
        "approach_dir": approach_dir.astype(np.float32),
    }


def extract_center_rotation_width(grasp_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    decoded = decode_gc6d_grasp(grasp_vec)
    return decoded["center"], decoded["rotation"], float(decoded["width"][0])

