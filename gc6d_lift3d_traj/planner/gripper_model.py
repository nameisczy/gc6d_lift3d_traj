from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class OBB:
    center: np.ndarray
    rotation: np.ndarray
    size: np.ndarray


@dataclass
class ParallelJawConfig:
    finger_length: float = 0.06
    finger_width: float = 0.01
    finger_thickness: float = 0.01
    palm_width: float = 0.04
    palm_height: float = 0.02
    palm_thickness: float = 0.02


def build_gripper_obbs(center: np.ndarray, rotation: np.ndarray, grasp_width: float, cfg: ParallelJawConfig) -> List[OBB]:
    c = np.asarray(center, dtype=np.float32).reshape(3)
    R = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
    x_axis, y_axis, z_axis = R[:, 0], R[:, 1], R[:, 2]

    half_gap = float(grasp_width) * 0.5
    palm_center = c - x_axis * (cfg.palm_thickness * 0.5)
    left_center = c + y_axis * (half_gap + cfg.finger_width * 0.5) + x_axis * (cfg.finger_length * 0.5)
    right_center = c - y_axis * (half_gap + cfg.finger_width * 0.5) + x_axis * (cfg.finger_length * 0.5)

    palm = OBB(center=palm_center, rotation=R, size=np.array([cfg.palm_thickness, cfg.palm_width, cfg.palm_height], dtype=np.float32))
    left = OBB(center=left_center, rotation=R, size=np.array([cfg.finger_length, cfg.finger_width, cfg.finger_thickness], dtype=np.float32))
    right = OBB(center=right_center, rotation=R, size=np.array([cfg.finger_length, cfg.finger_width, cfg.finger_thickness], dtype=np.float32))
    return [palm, left, right]

