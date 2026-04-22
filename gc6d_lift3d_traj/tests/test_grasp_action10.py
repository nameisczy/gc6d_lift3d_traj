import numpy as np

from gc6d_lift3d_traj.utils.grasp_action10 import grasp_matrix_width_to_action10


def test_action10_shape():
    R = np.eye(3, dtype=np.float32)
    a = grasp_matrix_width_to_action10(np.zeros(3), R, 0.05)
    assert a.shape == (10,)
