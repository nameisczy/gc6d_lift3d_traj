"""
Debug-only MLP baseline for MetaWorld pick-place: 7D state -> 4D action.

Not part of GC6D. Used with ``--policy-type mlp`` in MetaWorld train/eval scripts.
"""

from __future__ import annotations

import torch
import torch.nn as nn

STATE_DIM = 7
ACTION_DIM = 4
HIDDEN = 256


class MetaWorldMLPPolicy(nn.Module):
    """``MLP(7 -> 256 -> 256 -> 4)`` with ReLU."""

    def __init__(self, state_dim: int = STATE_DIM, hidden: int = HIDDEN, action_dim: int = ACTION_DIM):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, robot_states: torch.Tensor) -> torch.Tensor:
        """*robot_states*: (B, 7)  [hand(3), goal(3), grip(1)]"""
        return self.net(robot_states)
