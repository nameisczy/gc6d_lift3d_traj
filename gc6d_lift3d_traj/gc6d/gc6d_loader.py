from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List

import numpy as np


@dataclass
class GC6DSample:
    scene_id: int
    ann_id: int
    split: str
    point_cloud: np.ndarray
    grasps_17d: np.ndarray


def _scene_id_from_name(scene_name: str) -> int:
    digits = "".join([c for c in scene_name if c.isdigit()])
    return int(digits)


class GC6DLoader:
    def __init__(self, gc6d_root: str, api_root: str, camera: str = "realsense-d415", split: str = "train"):
        self.gc6d_root = Path(gc6d_root)
        self.api_root = Path(api_root)
        self.camera = camera
        self.split = split

        if str(self.api_root) not in sys.path:
            sys.path.insert(0, str(self.api_root))
        from graspclutter6dAPI.graspclutter6d import GraspClutter6D

        self.api = GraspClutter6D(root=str(self.gc6d_root), camera=camera, split=split)

    def __len__(self) -> int:
        return len(self.api.sceneName)

    def iter_samples(self, max_samples: int | None = None) -> Iterator[GC6DSample]:
        n = len(self)
        limit = n if max_samples is None else min(n, int(max_samples))
        for idx in range(limit):
            scene_name = self.api.sceneName[idx]
            ann_id = int(self.api.annId[idx])
            scene_id = _scene_id_from_name(scene_name)
            pc = self.api.loadScenePointCloud(scene_id, self.camera, ann_id, align=False, format="numpy")
            # API returns (xyz, rgb) for format='numpy'
            if isinstance(pc, tuple):
                pc = np.asarray(pc[0], dtype=np.float32)
            else:
                pc = np.asarray(pc, dtype=np.float32)
            # loadGrasp(sceneId, annId, format=..., camera=..., ...)
            gg = self.api.loadGrasp(
                scene_id,
                ann_id,
                format="6d",
                camera=self.camera,
                fric_coef_thresh=1.0,
            )
            yield GC6DSample(
                scene_id=scene_id,
                ann_id=ann_id,
                split=self.split,
                point_cloud=np.asarray(pc, dtype=np.float32),
                grasps_17d=np.asarray(gg.grasp_group_array, dtype=np.float32),
            )

    def list_environment_ids(self) -> List[int]:
        env_ids = set()
        for name in self.api.sceneName:
            env_ids.add(_scene_id_from_name(name) // 1000)
        return sorted(list(env_ids))

