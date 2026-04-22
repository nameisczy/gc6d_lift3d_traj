"""Point cloud + EE state -> delta action + optional grasp goal (10D), Lift3D-compatible 6D rotation."""

from __future__ import annotations

import torch
import torch.nn as nn


class TrajectoryPolicy(nn.Module):
    """
    Pools point cloud (mean + max xyz), concatenates EE pose (pos + rot6d + gripper),
    predicts per-step deltas (3 + 6 + 1) and auxiliary grasp prediction (10D) for goal loss.
    """

    def __init__(self, robot_state_dim: int = 1, hidden: int = 512):
        super().__init__()
        # mean(3)+max(3) + ee_pos(3) + rot6d(6) + gripper(robot_state_dim) + goal10(10)
        in_dim = 6 + 3 + 6 + robot_state_dim + 10
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
        )
        self.head_delta = nn.Linear(hidden, 10)  # 3 + 6 + 1
        self.head_goal = nn.Linear(hidden, 10)  # same as grasp 10D

    def forward(
        self,
        point_cloud: torch.Tensor,
        ee_position: torch.Tensor,
        ee_rotation: torch.Tensor,
        gripper: torch.Tensor,
        goal: torch.Tensor,
    ):
        # point_cloud: (B, N, 3)
        pc_mean = point_cloud.mean(dim=1)
        pc_max = point_cloud.max(dim=1)[0]
        if gripper.dim() == 1:
            gripper = gripper.unsqueeze(-1)
        if goal.dim() == 1:
            goal = goal.unsqueeze(0)
        x = torch.cat([pc_mean, pc_max, ee_position, ee_rotation, gripper, goal], dim=-1)
        h = self.encoder(x)
        return self.head_delta(h), self.head_goal(h)
