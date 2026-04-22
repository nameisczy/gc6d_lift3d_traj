from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


@dataclass
class LossWeights:
    imitation: float = 1.0
    goal: float = 1.0
    gripper: float = 0.2


def compute_losses(pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor], final_pred: Dict[str, torch.Tensor], final_target: Dict[str, torch.Tensor], w: LossWeights) -> Dict[str, torch.Tensor]:
    imitation = (
        F.mse_loss(pred["delta_translation"], target["delta_translation"])
        + F.mse_loss(pred["delta_rotation"], target["delta_rotation"])
    )
    gripper = F.mse_loss(pred["delta_gripper"], target["delta_gripper"])
    goal = (
        F.mse_loss(final_pred["ee_position"], final_target["ee_position"])
        + F.mse_loss(final_pred["ee_rotation"], final_target["ee_rotation"])
    )
    total = w.imitation * imitation + w.goal * goal + w.gripper * gripper
    return {"total": total, "imitation": imitation, "goal": goal, "gripper": gripper}


def compute_trajectory_losses(
    pred_delta: torch.Tensor,
    pred_goal10: torch.Tensor,
    target_delta: torch.Tensor,
    target_goal10: torch.Tensor,
    w: LossWeights,
) -> Dict[str, torch.Tensor]:
    """pred/target delta: (B, 10) = [dpos(3), drot6(6), dgrip(1)]; goal10: Lift3D grasp format."""
    imitation = F.mse_loss(pred_delta, target_delta)
    goal = F.mse_loss(pred_goal10, target_goal10)
    gripper = F.mse_loss(pred_delta[:, 9:10], target_delta[:, 9:10])
    total = w.imitation * imitation + w.goal * goal + w.gripper * gripper
    return {"total": total, "imitation": imitation, "goal": goal, "gripper": gripper}

