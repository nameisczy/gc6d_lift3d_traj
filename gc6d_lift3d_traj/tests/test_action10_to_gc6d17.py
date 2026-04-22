"""17D layout must match graspclutter6dAPI.grasp.Grasp concatenation order."""
from __future__ import annotations

import numpy as np
import pytest

from gc6d_lift3d_traj.utils.action10_to_gc6d17 import DEFAULT_DEPTH, DEFAULT_HEIGHT, action10_to_gc6d17
from gc6d_lift3d_traj.utils.rotations import lift3d_rotation_to_matrix


def test_action10_to_gc6d17_matches_manual_concat():
    rng = np.random.default_rng(0)
    t = rng.standard_normal(3).astype(np.float32)
    r6 = rng.standard_normal(6).astype(np.float32)
    w = 0.08
    action10 = np.concatenate([t, r6, np.array([w], dtype=np.float32)])
    R = lift3d_rotation_to_matrix(r6)
    row = action10_to_gc6d17(action10, score=1.0, height=DEFAULT_HEIGHT, depth=DEFAULT_DEPTH, object_id=0.0)
    manual = np.concatenate(
        [
            np.array((1.0, w, DEFAULT_HEIGHT, DEFAULT_DEPTH), dtype=np.float64),
            R.reshape(-1).astype(np.float64),
            t.astype(np.float64),
            np.array((0.0,), dtype=np.float64),
        ]
    ).astype(np.float32)
    assert row.shape == (17,)
    np.testing.assert_allclose(row, manual, rtol=0, atol=1e-5)


def test_optional_grasp_api_roundtrip():
    try:
        from graspclutter6dAPI.grasp import Grasp
    except Exception:
        pytest.skip("graspclutter6dAPI not installed")
    rng = np.random.default_rng(1)
    t = rng.standard_normal(3).astype(np.float64)
    r6 = rng.standard_normal(6).astype(np.float64)
    w = 0.07
    R = lift3d_rotation_to_matrix(r6.astype(np.float32))
    action10 = np.concatenate([t.astype(np.float32), r6.astype(np.float32), np.array([w], dtype=np.float32)])
    row = action10_to_gc6d17(action10)
    g = Grasp(1.0, w, DEFAULT_HEIGHT, DEFAULT_DEPTH, R, t, 0.0)
    np.testing.assert_allclose(row, g.grasp_array.astype(np.float32), rtol=0, atol=1e-4)
