"""Load/save consistency for episode dict keys."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from gc6d_lift3d_traj.dataset.dump_dataset import dump_episode_npz
from gc6d_lift3d_traj.dataset.episode_builder import poses_to_states_actions


def test_npz_roundtrip():
    T = 24
    pos = np.zeros((T, 3), dtype=np.float32)
    rot = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], T, axis=0)
    grip = np.ones((T, 1), dtype=np.float32)
    out = poses_to_states_actions(pos, rot, grip)
    out["point_cloud"] = np.random.randn(100, 3).astype(np.float32) * 0.01
    out["gt_grasp_center"] = np.zeros(3, dtype=np.float32)
    out["gt_grasp_rotation"] = np.eye(3, dtype=np.float32)
    out["gt_grasp_width"] = np.array([0.05], dtype=np.float32)
    out["metadata_json"] = np.array(["{}"], dtype=object)

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        path = f.name
    dump_episode_npz(Path(path), out)
    data = np.load(path, allow_pickle=True)
    assert data["actions_translation"].shape[0] == T - 1
    assert data["ee_rotations"].shape[1] == 6
    __import__("os").unlink(path)
