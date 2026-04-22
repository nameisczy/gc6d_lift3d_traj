from __future__ import annotations

import numpy as np


def ensure_xyz(pc: np.ndarray) -> np.ndarray:
    pc = np.asarray(pc, dtype=np.float32)
    if pc.ndim != 2:
        raise ValueError(f"point cloud must be 2D, got {pc.shape}")
    if pc.shape[1] >= 3:
        return pc[:, :3].astype(np.float32)
    raise ValueError(f"point cloud must have >=3 channels, got {pc.shape}")


def estimate_table_z(pc_xyz: np.ndarray, q: float = 0.05) -> float:
    z = np.asarray(pc_xyz, dtype=np.float32)[:, 2]
    return float(np.quantile(z, q))

