"""
Real point clouds for GC6D (GraspClutter6D) — backprojection aligned with graspclutter6dAPI.

Forbids all-zero or fake-random placeholder clouds in production paths; use
``GC6D_DEBUG_ALLOW_DUMMY_POINTCLOUD=1`` only for explicit debug scripts.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Literal, Optional, Tuple

import numpy as np

_DEBUG_DUMMY = os.environ.get("GC6D_DEBUG_ALLOW_DUMMY_POINTCLOUD", "").lower() in (
    "1",
    "true",
    "yes",
)


def validate_point_cloud(
    pc: np.ndarray,
    *,
    name: str = "point_cloud",
    allow_all_zero: bool = False,
    min_points: int = 32,
) -> None:
    """Raise if *pc* is empty, non-finite, or (unless allowed) all zeros / too few points."""
    p = np.asarray(pc, dtype=np.float32)
    if p.size == 0 or p.shape[-1] != 3:
        raise ValueError(f"{name} must be (N,3) with N>0, got {p.shape}")
    if not np.isfinite(p).all():
        raise ValueError(f"{name} contains non-finite values")
    n = p.shape[0]
    if n < min_points:
        raise ValueError(f"{name} has too few points: {n} (min {min_points})")
    if not allow_all_zero and _DEBUG_DUMMY is False:
        if np.allclose(p, 0.0, atol=1e-6):
            raise ValueError(
                f"{name} is all zeros — dummy placeholder is forbidden. "
                "Re-build episodes with real depth/point cloud or set GC6D_DEBUG_ALLOW_DUMMY_POINTCLOUD=1 for debug only."
            )


def depth_to_pointcloud_camera_frame(
    depth: np.ndarray,
    cam_k: np.ndarray,
    depth_scale: float,
    *,
    valid_min: float = 1e-6,
    valid_max: float = np.inf,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Back-project depth to 3D points in **camera frame** (Z forward, same as GraspClutter6D code).

    depth: (H, W) depth stored as uint16 in dataset; pass float or uint16, values are raw integers
        that are divided by depth_scale to get meters.
    cam_k: (3, 3) intrinsic matrix.
    """
    d = np.asarray(depth)
    if d.dtype == np.float32 and d.max() < 1.0 and d.max() > 0:
        # may already be meters
        z = d
    else:
        z = d.astype(np.float32) / float(depth_scale)
    K = np.asarray(cam_k, dtype=np.float64).reshape(3, 3)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    h, w = z.shape[:2]
    # Match graspclutter6dAPI: xmap=col, ymap=row
    xmap, ymap = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
    points_z = z.astype(np.float32)
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z
    pts = np.stack([points_x, points_y, points_z], axis=-1).reshape(-1, 3).astype(np.float32)
    valid = (pts[:, 2] > valid_min) & (pts[:, 2] < valid_max) & np.isfinite(pts).all(axis=1)
    return pts, valid


def _img_id_from_ann(graspclutter6d, ann_id: int, camera: str) -> int:
    return int(graspclutter6d.annId2ImgId(ann_id, camera))


def load_scene_camera_entry(gc6d_root: Path, scene_id: int, img_id: int) -> dict[str, Any]:
    path = gc6d_root / "scenes" / f"{int(scene_id):06d}" / "scene_camera.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing scene_camera.json: {path}")
    with open(path, "r", encoding="utf-8") as f:
        m = json.load(f)
    key = str(int(img_id))
    if key not in m:
        raise KeyError(f"imgId {key} not in {path}")
    return m[key]


def depth_scale_for_camera(camera: str) -> float:
    if camera in ("realsense-d415", "realsense-d435"):
        return 1000.0
    if camera in ("azure-kinect", "zivid"):
        return 10000.0
    raise ValueError(f"Unknown camera {camera!r} for depth scale")


def load_gc6d_pointcloud_from_api(
    scene_id: int,
    ann_id: int,
    camera: str,
    *,
    gc6d_root: str | Path,
    align: bool = False,
    split: str = "train",
) -> np.ndarray:
    """
    Load scene point cloud using official ``GraspClutter6D.loadScenePointCloud`` (numpy, xyz in camera frame).
    """
    root = Path(gc6d_root)
    ap_root = os.environ.get("GC6D_API_ROOT")
    if not ap_root:
        cand = root.parent / "graspclutter6dAPI"
        ap_root = str(cand) if cand.is_dir() else "/home/ziyaochen/graspclutter6dAPI"
    if str(ap_root) not in sys.path:
        sys.path.insert(0, str(ap_root))
    from graspclutter6dAPI.graspclutter6d import GraspClutter6D

    api = GraspClutter6D(root=str(root), camera=camera, split=split)
    raw = api.loadScenePointCloud(
        sceneId=scene_id,
        camera=camera,
        annId=ann_id,
        format="numpy",
        align=align,
        use_mask=True,
    )
    if isinstance(raw, tuple):
        pc = np.asarray(raw[0], dtype=np.float32)
    else:
        pc = np.asarray(raw, dtype=np.float32)
    validate_point_cloud(pc[:, :3], name="load_gc6d_pointcloud_from_api")
    return np.asarray(pc[:, :3], dtype=np.float32)


def pointcloud_from_depth_file(
    scene_id: int,
    ann_id: int,
    camera: str,
    gc6d_root: str | Path,
    *,
    split: str = "train",
) -> np.ndarray:
    """
    Rebuild point cloud from depth + scene_camera.json (same math as API's loadScenePointCloud body).
    """
    root = Path(gc6d_root)
    ap_root = os.environ.get("GC6D_API_ROOT")
    if not ap_root:
        cand = root.parent / "graspclutter6dAPI"
        ap_root = str(cand) if cand.is_dir() else "/home/ziyaochen/graspclutter6dAPI"
    if str(ap_root) not in sys.path:
        sys.path.insert(0, str(ap_root))
    from graspclutter6dAPI.graspclutter6d import GraspClutter6D

    api = GraspClutter6D(root=str(root), camera=camera, split=split)
    img_id = _img_id_from_ann(api, ann_id, camera)
    depths = np.asarray(
        api.loadDepth(sceneId=scene_id, camera=camera, annId=ann_id),
        dtype=np.uint16,
    )
    sc = load_scene_camera_entry(root, scene_id, img_id)
    K = np.array(sc["cam_K"], dtype=np.float64).reshape(3, 3)
    s = depth_scale_for_camera(camera)
    pts, valid = depth_to_pointcloud_camera_frame(depths, K, s)
    mask = valid & (depths.reshape(-1) > 0)
    pts = pts[mask]
    if pts.size == 0:
        raise RuntimeError("pointcloud_from_depth_file: no valid points after backprojection")
    validate_point_cloud(pts, name="rebuilt_pc")
    return pts.astype(np.float32)


def sample_pointcloud(
    pc: np.ndarray,
    num_points: int,
    method: Literal["random", "fps"] = "random",
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Resample to fixed *num_points* (for batching). Default ``random`` matches
    :func:`sample_point_cloud` in ``lift3d_dataset``; ``fps`` requires torch3d (optional).
    """
    p = np.asarray(pc, dtype=np.float32)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"pc must be (N,3), got {p.shape}")
    n = p.shape[0]
    r = rng or np.random.default_rng()
    if n >= num_points:
        if method == "random":
            idx = r.choice(n, num_points, replace=False)
        else:
            try:
                import torch
                from pytorch3d.ops import sample_farthest_points
            except ImportError as e:
                raise RuntimeError("fps sampling requires torch + pytorch3d") from e
            pts = torch.from_numpy(p).unsqueeze(0).float()
            out, _ = sample_farthest_points(pts, K=num_points)
            return out.squeeze(0).cpu().numpy().astype(np.float32)
        return p[idx].copy()
    idx = r.choice(n, num_points, replace=True)
    return p[idx].copy()
