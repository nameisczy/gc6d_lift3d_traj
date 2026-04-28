"""
Minimal torch Dataset for MetaWorld .npz collected by ``scripts/metaworld_collect_data.py``.

``robot_states`` is a 7D vector: hand(3) + goal_pos(3) + gripper(1) from the full 39D obs
(see ``metaworld_state``). ``action`` is 4D (no padding to 10).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from metaworld_state import metaworld_raw39_to_robot7_np, MW_ACTION_DIM, MW_OBS_FULL_DIM, MW_ROBOT7_DIM


class MetaWorldPickPlaceDataset(Dataset):
    """
    Loads ``all_obs`` (39D) and ``all_actions`` (4D) from a compressed .npz.

    Exposes a 7D ``robot_states`` suitable for :func:`metaworld_state.robot7_to_trajectory_policy_inputs`.
    """

    def __init__(self, npz_path: Union[str, Path], action_dim: int = MW_ACTION_DIM):
        path = Path(npz_path)
        d = np.load(path, allow_pickle=True)
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
        point_clouds = torch.zeros(1024, 3, dtype=torch.float32)
        action = torch.from_numpy(act)
        return {
            "point_clouds": point_clouds,
            "robot_states": robot,
            "raw_states": torch.from_numpy(np.asarray(full, dtype=np.float32).reshape(-1)),
            "action": action,
            "texts": "",
        }
