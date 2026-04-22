from __future__ import annotations

"""
Episode packing from a precomputed EE trajectory.

Trajectories are built in ``planner.trajectory_builder`` (simple interpolation or optional
cuRobo via ``TrajConfig.use_curobo``). This module only converts poses to 10D actions and
metadata; it does not plan motions.
"""

from typing import Dict

import numpy as np

from gc6d_lift3d_traj.gc6d.grasp_decode import decode_gc6d_grasp
from gc6d_lift3d_traj.utils.rotations import action_rotation_from_two_poses, matrix_to_lift3d_rotation


def poses_to_states_actions(positions: np.ndarray, rotations: np.ndarray, gripper: np.ndarray) -> Dict[str, np.ndarray]:
    T = positions.shape[0]
    ee_rot = np.stack([matrix_to_lift3d_rotation(R) for R in rotations], axis=0).astype(np.float32)
    dpos = positions[1:] - positions[:-1]
    drot = np.stack(
        [action_rotation_from_two_poses(rotations[t], rotations[t + 1]) for t in range(T - 1)],
        axis=0,
    ).astype(np.float32)
    dgrip = gripper[1:] - gripper[:-1]
    return {
        "ee_positions": positions.astype(np.float32),
        "ee_rotations": ee_rot,
        "gripper": gripper.astype(np.float32),
        "actions_translation": dpos.astype(np.float32),
        "actions_rotation": drot.astype(np.float32),
        "actions_gripper": dgrip.astype(np.float32),
    }


def build_episode(point_cloud: np.ndarray, grasp_17d: np.ndarray, trajectory: Dict[str, np.ndarray], metadata: Dict) -> Dict:
    dec = decode_gc6d_grasp(grasp_17d)
    # Keep lift phase generation unchanged, but enforce supervision endpoint at grasp pose.
    # This makes the last state/action target align with GT grasp as required.
    pos = np.asarray(trajectory["ee_positions"], dtype=np.float32).copy()
    rot = np.asarray(trajectory["ee_rotations_matrix"], dtype=np.float32).copy()
    grip = np.asarray(trajectory["gripper"], dtype=np.float32).copy()
    pos[-1] = dec["center"].astype(np.float32)
    rot[-1] = dec["rotation"].astype(np.float32)
    packed = poses_to_states_actions(
        pos,
        rot,
        grip,
    )
    packed.update(
        {
            "point_cloud": np.asarray(point_cloud, dtype=np.float32),
            "gt_grasp_center": dec["center"].astype(np.float32),
            "gt_grasp_rotation": dec["rotation"].astype(np.float32),
            "gt_grasp_width": dec["width"].astype(np.float32),
            "metadata_json": np.array([str(metadata)], dtype=object),
            # For Lift3D-style dataloading: load RGB from GC6D without parsing metadata_json.
            "scene_id": np.int32(metadata["scene_id"]),
            "ann_id": np.int32(metadata["ann_id"]),
            "camera": np.asarray(metadata.get("camera", "realsense-d415"), dtype="<U32"),
        }
    )
    return packed

