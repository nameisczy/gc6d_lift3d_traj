import numpy as np

from gc6d_lift3d_traj.gc6d.grasp_decode import decode_gc6d_grasp, extract_center_rotation_width


def test_decode_mapping():
    row = np.zeros(17, dtype=np.float32)
    row[0] = 0.9
    row[1] = 0.04
    row[4:13] = np.eye(3, dtype=np.float32).reshape(-1)
    row[13:16] = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    d = decode_gc6d_grasp(row)
    assert np.allclose(d["center"], [0.1, -0.2, 0.3])
    assert np.allclose(d["rotation"], np.eye(3), atol=1e-6)
    assert np.isclose(float(d["width"][0]), 0.04)
    assert np.allclose(d["approach_dir"], [1.0, 0.0, 0.0])


def test_extract_triplet():
    row = np.zeros(17, dtype=np.float32)
    row[1] = 0.03
    row[4:13] = np.eye(3, dtype=np.float32).reshape(-1)
    row[13:16] = np.array([0.2, 0.1, 0.5], dtype=np.float32)
    c, R, w = extract_center_rotation_width(row)
    assert np.allclose(c, [0.2, 0.1, 0.5])
    assert np.allclose(R, np.eye(3))
    assert np.isclose(w, 0.03)

