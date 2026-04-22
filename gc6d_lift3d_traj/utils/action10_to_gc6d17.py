"""
Lift3D 10D grasp action -> GC6D / evaluator 17D row.

Layout is defined in graspclutter6dAPI/grasp.py (class Grasp / GraspGroup):
  [score, width, height, depth, rotation_matrix(9 row-major), translation(3), object_id]

Rotation 9: R.reshape(9) row-major (same as Grasp.rotation_matrix.flatten() order).
"""

from __future__ import annotations

import numpy as np

from gc6d_lift3d_traj.utils.rotations import lift3d_rotation_to_matrix

# Match common GC6D / offline dumps (see gc6d_grasp_pipeline action10_to_graspgroup)
DEFAULT_HEIGHT = 0.02
DEFAULT_DEPTH = 0.02


def action10_to_gc6d17(
    action10: np.ndarray,
    score: float = 1.0,
    height: float = DEFAULT_HEIGHT,
    depth: float = DEFAULT_DEPTH,
    object_id: float = 0.0,
) -> np.ndarray:
    """
    Parameters
    ----------
    action10 : (10,)  [t(3), R6d col0+col1(6), width(1)]
    score : default 1.0 when no model confidence
    height, depth : fixed constants (meters), documented for evaluator
    object_id : default 0.0 per request (API stores float in last column)
    """
    a = np.asarray(action10, dtype=np.float32).reshape(10)
    t = a[0:3]
    r6 = a[3:9]
    w = float(a[9])
    R = lift3d_rotation_to_matrix(r6)
    row = np.zeros(17, dtype=np.float32)
    row[0] = float(score)
    row[1] = w
    row[2] = float(height)
    row[3] = float(depth)
    row[4:13] = R.reshape(9)
    row[13:16] = t
    row[16] = float(object_id)
    return row


def batch_action10_to_gc6d17(actions: np.ndarray, **kwargs) -> np.ndarray:
    """(K, 10) -> (K, 17)"""
    a = np.asarray(actions, dtype=np.float32)
    if a.ndim != 2 or a.shape[1] != 10:
        raise ValueError(f"Expected (K, 10), got {a.shape}")
    return np.stack([action10_to_gc6d17(a[i], **kwargs) for i in range(a.shape[0])], axis=0)
