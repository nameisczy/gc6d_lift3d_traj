"""
Real MetaWorld point clouds matching ``lift3d.envs.metaworld_env.MetaWorldEnv`` (corner camera).

Requires ``LIFT3D_ROOT`` on ``PYTHONPATH`` and `open3d`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np


def ensure_lift3d_path() -> Path:
    root = Path(os.environ.get("LIFT3D_ROOT", "/home/ziyaochen/LIFT3D")).resolve()
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
    return root


# Same defaults as MetaWorldEnv for pick-place + corner (see LIFT3D lift3d/envs/metaworld_env.py)
_TASK_PC_BOUNDS_CORNER = [-1.66, 0.8, -0.6, -0.48, 1.38, 10.0]
_PC_TRANSFORM_CORNER = np.array(
    [
        [-0.66173422, -0.48809537, 0.56909642],
        [-0.31361979, 0.86966611, 0.38121317],
        [0.68099225, -0.0737819, 0.7285642],
    ],
    dtype=np.float64,
)


def apply_metaworld_lift3d_render_size(mujoco_env: object, image_size: int = 224) -> None:
    """Match ``MetaWorldEnv`` renderer resolution for consistent intrinsics / point clouds."""
    r = mujoco_env.mujoco_renderer
    w = h = int(image_size)
    r.width = w
    r.height = h
    r.model.vis.global_.offwidth = w
    r.model.vis.global_.offheight = h


def pinhole_intrinsics_from_mujoco(
    mujoco_model: object,
    camera_name: str,
    width: int,
    height: int,
) -> np.ndarray:
    import mujoco

    from lift3d.helpers.mujoco import camera_name_to_id

    cid = camera_name_to_id(mujoco_model, camera_name)
    aspect_ratio = width / height
    fovy = np.radians(mujoco_model.cam_fovy[cid])
    fovx = 2 * np.arctan(np.tan(fovy / 2) * aspect_ratio)
    fx = width / (2 * np.tan(fovx / 2))
    fy = height / (2 * np.tan(fovy / 2))
    cx, cy = width / 2, height / 2
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def point_cloud_from_mujoco_env(
    mujoco_env: object,
    *,
    camera_name: str = "corner",
    num_points: int = 1024,
) -> np.ndarray:
    """
    Same pipeline as ``MetaWorldEnv.get_point_cloud`` (world-frame xyz, fps, then xyz only).
    """
    if camera_name != "corner":
        raise NotImplementedError(
            "metaworld_pointcloud only ships crop+rotation for camera 'corner'. "
            "Use lift3d.envs.metaworld_env.MetaWorldEnv for other views."
        )
    ensure_lift3d_path()
    from lift3d.helpers.graphics import PointCloud
    from lift3d.helpers.mujoco import generate_point_cloud

    renderer = mujoco_env.mujoco_renderer
    point_cloud, _depth = generate_point_cloud(renderer, [camera_name])
    if point_cloud.shape[1] < 3:
        raise RuntimeError(f"Unexpected point_cloud shape: {point_cloud.shape}")
    pc = point_cloud.copy()
    pc[:, :3] = pc[:, :3] @ _PC_TRANSFORM_CORNER.T

    x_min, y_min, z_min, x_max, y_max, z_max = _TASK_PC_BOUNDS_CORNER
    m = (
        (pc[:, 0] > x_min)
        & (pc[:, 1] > y_min)
        & (pc[:, 2] > z_min)
        & (pc[:, 0] < x_max)
        & (pc[:, 1] < y_max)
        & (pc[:, 2] < z_max)
    )
    pc = pc[m]
    if pc.shape[0] < 8:
        raise RuntimeError(
            f"MetaWorld point cloud after crop is too small: {pc.shape[0]} points. "
            "Check MUJOCO_GL, camera, and image_size."
        )
    out = PointCloud.point_cloud_sampling(pc, num_points, "fps")
    out_xyz = np.asarray(out[:, :3], dtype=np.float32)
    # Remove LIFT3D zero-padding (when N < num_points) by resampling real points
    valid = np.linalg.norm(out_xyz, axis=1) > 1e-5
    if valid.sum() < num_points // 2:
        from .gc6d_pointcloud import sample_pointcloud

        out_xyz = sample_pointcloud(
            out_xyz[valid] if valid.any() else pc[:, :3].astype(np.float32),
            num_points,
            method="random",
        )
    return out_xyz


def render_rgb_depth_and_pc(
    mujoco_env: object,
    *,
    camera_name: str = "corner",
    num_points: int = 1024,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (rgb_uint8_HWC, depth_meters_HW, K_3x3, point_cloud_1024_3).
    """
    ensure_lift3d_path()
    from lift3d.helpers.mujoco import camera_name_to_id, depth_to_meters

    renderer = mujoco_env.mujoco_renderer
    w, h = renderer.width, renderer.height

    camera_id = camera_name_to_id(renderer.model, camera_name)
    viewer = renderer._get_viewer(render_mode="rgb_array")
    image = viewer.render(render_mode="rgb_array", camera_id=camera_id)
    image = np.flip(image, axis=0)
    depth_raw = viewer.render(render_mode="depth_array", camera_id=camera_id)
    depth_m = depth_to_meters(depth_raw, renderer.model)
    depth_m = np.flip(depth_m, axis=0)
    K = pinhole_intrinsics_from_mujoco(renderer.model, camera_name, w, h)
    pc = point_cloud_from_mujoco_env(mujoco_env, camera_name=camera_name, num_points=num_points)
    return image, depth_m, K.astype(np.float64), pc
