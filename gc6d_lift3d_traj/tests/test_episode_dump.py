import numpy as np

from gc6d_lift3d_traj.dataset.episode_builder import poses_to_states_actions


def test_action_length():
    T = 24
    pos = np.zeros((T, 3), dtype=np.float32)
    rot = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], T, axis=0)
    grip = np.ones((T, 1), dtype=np.float32)
    out = poses_to_states_actions(pos, rot, grip)
    assert out["actions_translation"].shape[0] == T - 1
    assert out["actions_rotation"].shape[0] == T - 1
    assert out["actions_gripper"].shape[0] == T - 1

