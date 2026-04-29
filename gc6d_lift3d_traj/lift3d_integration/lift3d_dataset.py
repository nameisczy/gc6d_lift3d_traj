from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from gc6d_lift3d_traj.utils.gc6d_rgb import rgb_png_path
from gc6d_lift3d_traj.utils.grasp_action10 import grasp_matrix_width_to_action10
from gc6d_lift3d_traj.gc6d_pointcloud import (
    load_gc6d_pointcloud_from_api,
    sample_pointcloud as sample_point_cloud,
    validate_point_cloud,
)

FIXED_PC_POINTS = 1024


def _scene_ann_camera_from_npz_and_row(data: Any, row: Dict[str, Any], default_camera: str) -> Tuple[int, int, str]:
    """Resolve scene_id, ann_id, camera from episode .npz (preferred) or index row."""
    files = getattr(data, "files", None)
    if files is not None and "scene_id" in files and "ann_id" in files:
        scene_id = int(np.asarray(data["scene_id"]).reshape(-1)[0])
        ann_id = int(np.asarray(data["ann_id"]).reshape(-1)[0])
        if "camera" in files:
            cam = str(np.asarray(data["camera"]).reshape(-1)[0])
        else:
            cam = default_camera
        return scene_id, ann_id, cam
    scene_id = int(row["scene_id"])
    ann_id = int(row["ann_id"])
    cam = str(row.get("camera", default_camera))
    return scene_id, ann_id, cam


def load_rgb_tensor(path: Path, image_size: int) -> torch.Tensor:
    """Load BGR file from disk, return float tensor (3, H, W) in [0, 1], RGB order."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"RGB not found or unreadable: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(rgb).permute(2, 0, 1).to(dtype=torch.float32) / 255.0
    return t


class Lift3DTrajDataset(Dataset):
    """
    Per-step trajectory samples from generated episodes.
    Yields:
      - state: point_cloud (1024,3), ee_position (3,), ee_rotation (6,), gripper (1,)
      - action: delta vector (10,) = [dtranslation(3), drotation6d(6), dgripper(1)]
      - goal_action10: (10,) GT grasp in Lift3D 10D format (for goal loss)
    """

    def __init__(
        self,
        index_jsonl: str,
        *,
        use_real_pointcloud: bool = True,
        reload_pointcloud_from_api: bool = False,
        gc6d_root: Optional[str] = None,
        gc6d_api_root: Optional[str] = None,
        dataset_split: str = "train",
        default_camera: str = "realsense-d415",
    ):
        self.index_path = Path(index_jsonl)
        self.rows: List[dict] = []
        with self.index_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.rows.append(json.loads(line))

        self._use_real = use_real_pointcloud
        self._reload = reload_pointcloud_from_api
        self._gc6d_root = Path(gc6d_root).resolve() if gc6d_root else None
        if gc6d_api_root:
            os.environ["GC6D_API_ROOT"] = str(Path(gc6d_api_root).resolve())
        self._split = dataset_split
        self.default_camera = default_camera
        if self._reload and not self._gc6d_root:
            raise ValueError("reload_pointcloud_from_api=True requires gc6d_root")
        if not use_real_pointcloud:
            raise ValueError(
                "use_real_pointcloud=False is forbidden in this dataset class. "
                "Training must use real GC6D point clouds."
            )

        self.flat: List[Tuple[int, int]] = []
        self._cache: Dict[int, Any] = {}
        for epi_id, row in enumerate(self.rows):
            ep_path = Path(row["episode_path"])
            data = np.load(ep_path, allow_pickle=True)
            Tm1 = int(data["actions_translation"].shape[0])
            for t in range(Tm1):
                self.flat.append((epi_id, t))

    def episode_id_for_flat_index(self, idx: int) -> int:
        return self.flat[idx][0]

    def episode_row(self, epi_id: int) -> dict:
        return self.rows[epi_id]

    def _get_episode(self, epi_id: int) -> Any:
        if epi_id not in self._cache:
            self._cache[epi_id] = np.load(self.rows[epi_id]["episode_path"], allow_pickle=True)
        return self._cache[epi_id]

    def __len__(self) -> int:
        return len(self.flat)

    def __getitem__(self, idx: int):
        epi_id, t = self.flat[idx]
        data = self._get_episode(epi_id)
        row = self.episode_row(epi_id)
        scene_id, ann_id, camera = _scene_ann_camera_from_npz_and_row(data, row, self.default_camera)
        gt_w = float(np.asarray(data["gt_grasp_width"]).reshape(-1)[0])
        R = np.asarray(data["gt_grasp_rotation"], dtype=np.float32).reshape(3, 3)
        c = np.asarray(data["gt_grasp_center"], dtype=np.float32).reshape(3)
        goal10 = grasp_matrix_width_to_action10(c, R, gt_w)

        dpos = np.asarray(data["actions_translation"][t], dtype=np.float32).reshape(3)
        drot = np.asarray(data["actions_rotation"][t], dtype=np.float32).reshape(6)
        dg = np.asarray(data["actions_gripper"][t], dtype=np.float32).reshape(1)
        delta10 = np.concatenate([dpos, drot, dg], axis=0).astype(np.float32)

        if self._reload and self._gc6d_root is not None:
            pc_full = load_gc6d_pointcloud_from_api(
                scene_id, ann_id, camera, gc6d_root=self._gc6d_root, split=self._split
            )
        else:
            pc_full = np.asarray(data["point_cloud"], dtype=np.float32)
        validate_point_cloud(pc_full, name="episode[point_cloud]")
        pc = sample_point_cloud(pc_full, num_points=FIXED_PC_POINTS)
        assert pc.shape == (FIXED_PC_POINTS, 3), pc.shape

        state = {
            "point_cloud": torch.from_numpy(pc),
            "ee_position": torch.from_numpy(np.asarray(data["ee_positions"][t], dtype=np.float32)),
            "ee_rotation": torch.from_numpy(np.asarray(data["ee_rotations"][t], dtype=np.float32)),
            "gripper": torch.from_numpy(np.asarray(data["gripper"][t], dtype=np.float32)),
        }
        goal_action10 = torch.from_numpy(goal10)
        target_delta = torch.from_numpy(delta10)
        return state, target_delta, goal_action10


class Lift3DTrajDatasetLift3DStyle(Dataset):
    """
    Same as Lift3DTrajDataset but returns Lift3D-style sample dict with required fields:
    {
      images: (3, H, H) float RGB [0, 1] from GC6D scene rgb/{img_id}.png,
      point_clouds: (1024,3),
      robot_states: (10,) = [x,y,z,rot6d,gripper],
      raw_states: (10,) same as robot_states,
      action: (10,) trajectory delta,
      texts: "",
      goal: (10,) goal10 grasp target,
    }
    """

    def __init__(
        self,
        index_jsonl: str,
        gc6d_root: str,
        *,
        default_camera: str = "realsense-d415",
        image_size: int = 224,
        **kwargs: Any,
    ):
        self._inner = Lift3DTrajDataset(index_jsonl, default_camera=default_camera, **kwargs)
        self.gc6d_root = Path(gc6d_root)
        self.default_camera = default_camera
        self.image_size = int(image_size)
        self._rgb_cache: Dict[int, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self._inner)

    def _get_rgb_for_episode(self, epi_id: int) -> torch.Tensor:
        if epi_id in self._rgb_cache:
            return self._rgb_cache[epi_id]
        data = self._inner._get_episode(epi_id)
        row = self._inner.episode_row(epi_id)
        scene_id, ann_id, camera = _scene_ann_camera_from_npz_and_row(data, row, self.default_camera)
        path = rgb_png_path(self.gc6d_root, scene_id, ann_id, camera)
        img = load_rgb_tensor(path, self.image_size)
        self._rgb_cache[epi_id] = img
        return img

    def __getitem__(self, idx: int):
        state, target_delta, goal = self._inner[idx]
        epi_id = self._inner.episode_id_for_flat_index(idx)
        images = self._get_rgb_for_episode(epi_id)

        robot_states = torch.cat(
            [state["ee_position"], state["ee_rotation"], state["gripper"].reshape(1)], dim=0
        ).to(torch.float32)
        assert robot_states.shape == (10,), robot_states.shape
        return {
            "images": images,
            "point_clouds": state["point_cloud"].to(torch.float32),
            "robot_states": robot_states,
            "raw_states": robot_states.clone(),
            "action": target_delta.to(torch.float32),
            "texts": "",
            "goal": goal.to(torch.float32),
        }
