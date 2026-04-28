"""
MetaWorld pick-place-v3 state layout and mapping into ``TrajectoryPolicy`` inputs.

Observation layout (39) matches ``SawyerPickPlaceV3Policy._parse_obs`` in metaworld:

- hand_pos:  obs[0:3]
- gripper:   obs[3]  (gripper_distance_apart)
- puck/kinematics: obs[4:36]  (not used in our minimal 7D state)
- goal_pos:  obs[-3:]  (target location for the task)
"""

from __future__ import annotations

import numpy as np
import torch

MW_OBS_FULL_DIM = 39
MW_ROBOT7_DIM = 7  # hand(3) + goal(3) + gripper(1)
MW_ACTION_DIM = 4


def metaworld_raw39_to_robot7_np(obs39: np.ndarray) -> np.ndarray:
    """
    Build ``robot_states`` (7) = concat(hand_pos(3), goal_pos(3), gripper(1)).
    *obs39* is the full flat observation from the env.
    """
    o = np.asarray(obs39, dtype=np.float32).reshape(-1)
    if o.size < MW_OBS_FULL_DIM:
        raise ValueError(f"expected at least {MW_OBS_FULL_DIM} dims, got {o.size}")
    hand = o[0:3]
    goal = o[-3:]
    grip = o[3:4]
    return np.concatenate([hand, goal, grip], axis=0).astype(np.float32)


def metaworld_raw39_to_robot7_t(obs: torch.Tensor) -> torch.Tensor:
    """Batched: (B, 39) -> (B, 7)."""
    if obs.dim() == 1:
        obs = obs.unsqueeze(0)
    hand = obs[:, 0:3]
    goal = obs[:, -3:]
    grip = obs[:, 3:4]
    return torch.cat([hand, goal, grip], dim=1)


def robot7_to_trajectory_policy_inputs(robot7: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Map 7D ``robot_states`` to ``TrajectoryPolicy.forward`` (hand, no rot, grip, goal_10).

    * ee_position = hand (3)
    * ee_rotation = zeros (6)  — not observed in flat state; do not use arbitrary obs slices
    * goal (10) = [goal_x, goal_y, goal_z, 0, ..., 0] so the fusion net sees the task goal
    """
    b = robot7
    bsz = b.size(0)
    hand = b[:, 0:3]
    goal3 = b[:, 3:6]
    grip = b[:, 6:7]
    ee_rot = torch.zeros(bsz, 6, device=b.device, dtype=b.dtype)
    goal10 = torch.cat([goal3, torch.zeros(bsz, 7, device=b.device, dtype=b.dtype)], dim=1)
    return hand, ee_rot, grip, goal10
