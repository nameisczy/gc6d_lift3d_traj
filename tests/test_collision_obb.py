from __future__ import annotations

import numpy as np

from gc6d_lift3d_traj.planner.collision import (
    CollisionConfig,
    _points_in_obb,
    check_pointcloud_box_collision,
)


def _rz(deg: float) -> np.ndarray:
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _mask_option_a(points: np.ndarray, center: np.ndarray, R: np.ndarray, size: np.ndarray) -> np.ndarray:
    local = (points - center[None, :]) @ R
    half = size[None, :] * 0.5
    return np.all(np.abs(local) <= half, axis=1)


def _mask_option_b(points: np.ndarray, center: np.ndarray, R: np.ndarray, size: np.ndarray) -> np.ndarray:
    local = (points - center[None, :]) @ R.T
    half = size[None, :] * 0.5
    return np.all(np.abs(local) <= half, axis=1)


def test_points_in_obb_identity_rotation():
    center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    R = np.eye(3, dtype=np.float32)
    size = np.array([2.0, 4.0, 6.0], dtype=np.float32)
    pts = np.array(
        [
            [0.0, 0.0, 0.0],   # inside
            [0.9, 1.9, 2.9],   # inside
            [1.1, 0.0, 0.0],   # outside x
            [0.0, 2.1, 0.0],   # outside y
        ],
        dtype=np.float32,
    )
    m = _points_in_obb(pts, center, R, size)
    assert m.tolist() == [True, True, False, False]


def test_points_in_obb_rotation_convention_z90_option_a_is_correct():
    # R columns are local axes in world frame. With this convention:
    # world = center + local @ R.T, inverse local = (world-center) @ R  (Option A).
    center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    R = _rz(90.0)
    size = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    local_pts = np.array(
        [
            [0.5, 0.0, 0.0],   # inside
            [0.0, 0.5, 0.0],   # inside
            [1.2, 0.0, 0.0],   # outside
            [0.0, -1.2, 0.0],  # outside
        ],
        dtype=np.float32,
    )
    world_pts = center[None, :] + local_pts @ R.T
    ma = _mask_option_a(world_pts, center, R, size)
    mb = _mask_option_b(world_pts, center, R, size)
    m = _points_in_obb(world_pts, center, R, size)
    expected = [True, True, False, False]
    assert ma.tolist() == expected
    assert m.tolist() == expected
    # Note: for exactly 90 deg, Option B may match because R.T@R.T = -I and abs() removes sign.
    assert mb.tolist() == expected


def test_points_in_obb_option_a_vs_b_distinguishable_non_right_angle():
    center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    R = _rz(30.0)
    size = np.array([2.0, 1.0, 2.0], dtype=np.float32)
    local_pts = np.array(
        [
            [0.9, 0.2, 0.0],    # inside under Option A
            [0.2, 0.6, 0.0],    # outside under Option A (y too large)
            [1.2, 0.0, 0.0],    # outside under Option A (x too large)
        ],
        dtype=np.float32,
    )
    world_pts = center[None, :] + local_pts @ R.T
    ma = _mask_option_a(world_pts, center, R, size)
    mb = _mask_option_b(world_pts, center, R, size)
    m = _points_in_obb(world_pts, center, R, size)
    assert m.tolist() == ma.tolist()
    assert ma.tolist() == [True, False, False]
    assert mb.tolist() != ma.tolist()


def test_points_in_obb_translated_box_known_inside_outside():
    center = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    R = _rz(90.0)
    size = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    local_inside = np.array([[0.5, 0.0, 0.2], [-0.9, 0.9, -0.9]], dtype=np.float32)
    local_outside = np.array([[1.1, 0.0, 0.0], [0.0, -1.1, 0.0]], dtype=np.float32)
    pts = np.concatenate([local_inside, local_outside], axis=0) @ R.T + center[None, :]
    m = _points_in_obb(pts, center, R, size)
    assert m.tolist() == [True, True, False, False]


def test_collision_ratio_threshold_triggers_with_dense_cloud():
    # Dense cloud near gripper OBB should trigger via ratio threshold even with high absolute limit.
    pc_hit = np.zeros((50, 3), dtype=np.float32)
    positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    rotations = np.eye(3, dtype=np.float32)[None, ...]
    cfg = CollisionConfig(
        max_points_in_boxes=100000,   # disable absolute threshold
        max_collision_ratio=0.2,      # 50/50 should exceed
    )
    assert check_pointcloud_box_collision(pc_hit, positions, rotations, grasp_width=0.04, cfg=cfg)


def test_collision_ratio_threshold_not_triggered_when_disabled_and_under_count():
    pc_hit = np.zeros((50, 3), dtype=np.float32)
    positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    rotations = np.eye(3, dtype=np.float32)[None, ...]
    cfg = CollisionConfig(
        max_points_in_boxes=100000,   # under threshold
        max_collision_ratio=None,     # disable ratio branch
    )
    assert not check_pointcloud_box_collision(pc_hit, positions, rotations, grasp_width=0.04, cfg=cfg)
