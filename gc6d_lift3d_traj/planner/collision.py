from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gc6d_lift3d_traj.planner.gripper_model import ParallelJawConfig, build_gripper_obbs


@dataclass
class CollisionConfig:
    table_z: float = 0.0
    table_tolerance: float = 0.005
    point_contact_threshold: float = 0.005
    max_points_in_boxes: int = 25


def _points_in_obb(points: np.ndarray, center: np.ndarray, R: np.ndarray, size: np.ndarray) -> np.ndarray:
    local = (points - center[None, :]) @ R
    half = size[None, :] * 0.5
    return np.all(np.abs(local) <= half, axis=1)


def check_table_collision(positions: np.ndarray, cfg: CollisionConfig) -> bool:
    return bool(np.any(positions[:, 2] < (cfg.table_z - cfg.table_tolerance)))


def check_pointcloud_box_collision(
    point_cloud: np.ndarray,
    positions: np.ndarray,
    rotations: np.ndarray,
    grasp_width: float,
    cfg: CollisionConfig,
    gripper_cfg: ParallelJawConfig | None = None,
) -> bool:
    pc = np.asarray(point_cloud, dtype=np.float32)[:, :3]
    gripper_cfg = gripper_cfg or ParallelJawConfig()
    for p, R in zip(positions, rotations):
        obbs = build_gripper_obbs(p, R, grasp_width=grasp_width, cfg=gripper_cfg)
        hit_count = 0
        for obb in obbs:
            mask = _points_in_obb(pc, obb.center, obb.rotation, obb.size)
            hit_count += int(mask.sum())
        if hit_count > cfg.max_points_in_boxes:
            return True
    return False


def trajectory_is_collision_free(
    point_cloud: np.ndarray,
    positions: np.ndarray,
    rotations: np.ndarray,
    grasp_width: float,
    cfg: CollisionConfig,
) -> bool:
    if check_table_collision(positions, cfg):
        return False
    if check_pointcloud_box_collision(point_cloud, positions, rotations, grasp_width, cfg):
        return False
    return True

