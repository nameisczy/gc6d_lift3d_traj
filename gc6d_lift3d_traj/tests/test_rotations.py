import numpy as np
from scipy.spatial.transform import Rotation

from gc6d_lift3d_traj.utils.rotations import (
    lift3d_rotation_to_matrix,
    matrix_to_lift3d_rotation,
    relative_rotation_matrix,
)


def test_rot6d_roundtrip():
    R = Rotation.from_euler("xyz", [0.2, -0.1, 0.5]).as_matrix().astype(np.float32)
    r6 = matrix_to_lift3d_rotation(R)
    R2 = lift3d_rotation_to_matrix(r6)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-5)
    assert np.allclose(R2 @ R2.T, np.eye(3), atol=1e-5)
    assert np.allclose(np.linalg.det(R2), 1.0, atol=1e-4)


def test_relative_rotation():
    R1 = Rotation.from_euler("xyz", [0.0, 0.0, 0.0]).as_matrix()
    R2 = Rotation.from_euler("xyz", [0.0, 0.0, 0.3]).as_matrix()
    Rd = relative_rotation_matrix(R1, R2)
    assert np.allclose(Rd, R2, atol=1e-6)

