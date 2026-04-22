from __future__ import annotations

from typing import Dict

import numpy as np


def evaluate_step_errors(pred: Dict[str, np.ndarray], target: Dict[str, np.ndarray]) -> Dict[str, float]:
    dt = float(np.mean(np.linalg.norm(pred["delta_translation"] - target["delta_translation"], axis=-1)))
    dr = float(np.mean(np.linalg.norm(pred["delta_rotation"] - target["delta_rotation"], axis=-1)))
    dg = float(np.mean(np.abs(pred["delta_gripper"] - target["delta_gripper"])))
    return {"step_translation_l2": dt, "step_rotation_l2": dr, "step_gripper_l1": dg}


def evaluate_final_pose(pred_final_pos: np.ndarray, pred_final_rot: np.ndarray, gt_pos: np.ndarray, gt_rot: np.ndarray) -> Dict[str, float]:
    pos_err = float(np.linalg.norm(pred_final_pos - gt_pos))
    rot_err = float(np.linalg.norm(pred_final_rot - gt_rot))
    return {"final_position_error": pos_err, "final_rotation_error": rot_err}

