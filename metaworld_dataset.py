"""
MetaWorld .npz from ``scripts/metaworld_collect_data.py`` (v2: real point clouds via LIFT3D).

``robot_states`` is 7D: hand(3) + goal_pos(3) + gripper(1). ``action`` is 4D.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from metaworld_state import metaworld_raw39_to_robot7_np, MW_ACTION_DIM, MW_OBS_FULL_DIM, MW_ROBOT7_DIM

_DATASET_V2 = 2


def _validate_pc_tensor(t: torch.Tensor) -> None:
    if t.shape != (1024, 3):
        raise ValueError(f"point_clouds must be (1024,3), got {tuple(t.shape)}")
    a = t.numpy()
    if not np.isfinite(a).all():
        raise ValueError("point_clouds has non-finite values")
    if np.allclose(a, 0.0, atol=1e-6):
        raise ValueError(
            "point_clouds is all zeros — re-collect with scripts/metaworld_collect_data.py (v2)."
        )
    if np.linalg.norm(a, axis=1).max() < 1e-5:
        raise ValueError("point_clouds is degenerate (near-zero norm everywhere)")


class MetaWorldPickPlaceDataset(Dataset):
    """
    Loads v2 format with ``all_point_clouds`` (T, 1024, 3).
    """

    def __init__(self, npz_path: Union[str, Path], action_dim: int = MW_ACTION_DIM, *, use_real_pointcloud: bool = True):
        path = Path(npz_path)
        d = np.load(path, allow_pickle=True)
        self._version = int(d["dataset_version"][0]) if "dataset_version" in d.files else 1
        self._obs = d["all_obs"]
        self._act = d["all_actions"]
        if self._obs.shape[0] != self._act.shape[0]:
            raise ValueError("all_obs and all_actions must have the same length")
        self._n = int(self._obs.shape[0])
        if self._n and int(self._obs.shape[1]) != MW_OBS_FULL_DIM:
            raise ValueError(
                f"expected all_obs last dim {MW_OBS_FULL_DIM} (pick-place-v3), got {self._obs.shape[1]}"
            )
        self._action_dim = int(action_dim)
        self._use_real = use_real_pointcloud
        if not self._use_real:
            raise ValueError(
                "use_real_pointcloud=False is forbidden for MetaWorld training dataset. "
                "Please collect real point clouds with scripts/metaworld_collect_data.py."
            )

        if self._use_real and self._version < _DATASET_V2 and "all_point_clouds" not in d.files:
            raise ValueError(
                f"Dataset {path} is v1 (no all_point_clouds). Re-collect with:\n"
                f"  LIFT3D_ROOT=/path/to/LIFT3D python scripts/metaworld_collect_data.py --out {path}\n"
                "and keep use_real_pointcloud=True."
            )
        if "all_point_clouds" in d.files:
            self._pc = np.asarray(d["all_point_clouds"], dtype=np.float32)
            if self._pc.shape[0] != self._n:
                raise ValueError("all_point_clouds length must match all_obs")
            if self._pc.ndim != 3 or self._pc.shape[1:] != (1024, 3):
                raise ValueError(f"all_point_clouds must be (N, 1024, 3), got {self._pc.shape}")
        else:
            raise ValueError("Missing all_point_clouds in npz; re-run metaworld_collect_data.py (v2).")

    def __len__(self) -> int:
        return self._n

    def _action_vec(self, a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        out = np.zeros(self._action_dim, dtype=np.float32)
        n = min(a.size, self._action_dim)
        out[:n] = a[:n]
        return out

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        full = self._obs[idx]
        act = self._action_vec(self._act[idx])
        robot7 = metaworld_raw39_to_robot7_np(full)
        assert robot7.shape[0] == MW_ROBOT7_DIM
        robot = torch.from_numpy(robot7)
        point_clouds = torch.from_numpy(self._pc[idx].copy())
        _validate_pc_tensor(point_clouds)
        action = torch.from_numpy(act)
        return {
            "point_clouds": point_clouds,
            "robot_states": robot,
            "raw_states": torch.from_numpy(np.asarray(full, dtype=np.float32).reshape(-1)),
            "action": action,
            "texts": "",
        }
