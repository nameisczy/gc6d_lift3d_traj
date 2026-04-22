from __future__ import annotations

import numpy as np


def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + eps)


def matrix_to_lift3d_rotation(R: np.ndarray) -> np.ndarray:
    """Lift3D GC6D offline uses 6D rotation: first two columns of R."""
    R = np.asarray(R, dtype=np.float32).reshape(3, 3)
    return np.concatenate([R[:, 0], R[:, 1]], axis=0).astype(np.float32)


def lift3d_rotation_to_matrix(rot: np.ndarray) -> np.ndarray:
    """6D -> SO(3) with Gram-Schmidt, matching Lift3D/GC6D style heads."""
    rot = np.asarray(rot, dtype=np.float32).reshape(-1)
    if rot.shape[0] != 6:
        raise ValueError(f"Expected 6D rotation, got shape {rot.shape}")
    c1 = _normalize(rot[0:3])
    c2 = rot[3:6] - c1 * np.dot(c1, rot[3:6])
    c2 = _normalize(c2)
    c3 = _normalize(np.cross(c1, c2))
    return np.stack([c1, c2, c3], axis=1).astype(np.float32)


def relative_rotation_matrix(R_current: np.ndarray, R_next: np.ndarray) -> np.ndarray:
    R_current = np.asarray(R_current, dtype=np.float32).reshape(3, 3)
    R_next = np.asarray(R_next, dtype=np.float32).reshape(3, 3)
    return (R_next @ R_current.T).astype(np.float32)


def pose_to_state_rotation(R: np.ndarray) -> np.ndarray:
    return matrix_to_lift3d_rotation(R)


def action_rotation_from_two_poses(R_current: np.ndarray, R_next: np.ndarray) -> np.ndarray:
    R_delta = relative_rotation_matrix(R_current, R_next)
    return matrix_to_lift3d_rotation(R_delta)

