import numpy as np

from gc6d_lift3d_traj.planner.trajectory_builder import TrajConfig, build_trajectory_from_grasp


def _dummy_grasp():
    g = np.zeros(17, dtype=np.float32)
    g[1] = 0.04
    g[4:13] = np.eye(3, dtype=np.float32).reshape(-1)
    g[13:16] = np.array([0.0, 0.0, 0.3], dtype=np.float32)
    return g


def test_trajectory_length_and_phases():
    cfg = TrajConfig()
    traj = build_trajectory_from_grasp(_dummy_grasp(), cfg)
    assert traj["ee_positions"].shape[0] == sum(cfg.phase_steps)
    assert traj["gripper"].shape[0] == sum(cfg.phase_steps)
    assert traj["gripper"][0, 0] == cfg.gripper_open
    assert traj["gripper"][-1, 0] == cfg.gripper_closed
    assert traj["lift_pose"]["position"][2] > traj["final_grasp_pose"]["position"][2]

