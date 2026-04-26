"""Point cloud (Lift3dCLIP) + EE state -> delta action + grasp goal (10D), Lift3D-compatible 6D rotation."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn

# Official LIFT3D repo (override with env LIFT3D_ROOT).
_DEFAULT_LIFT3D_ROOT = Path("/home/ziyaochen/LIFT3D")
_DEFAULT_LIFT3D_CLIP_CKPT = Path("/home/ziyaochen/gc6d_lift3d_traj/lift3d_clip_base.pth")


def _ensure_lift3d_on_path(lift3d_root: Path) -> None:
    root = str(lift3d_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def _unpack_ckpt_state(loaded: Any) -> Dict[str, torch.Tensor]:
    if isinstance(loaded, dict):
        if "model" in loaded and isinstance(loaded["model"], dict):
            return loaded["model"]
        if "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
            return loaded["state_dict"]
        if loaded and all(isinstance(v, torch.Tensor) for v in loaded.values()):
            return loaded
    raise ValueError(f"Unsupported checkpoint structure: {type(loaded)}")


def _normalize_point_cloud_bnm3(pc: torch.Tensor) -> torch.Tensor:
    """Same centering / scale as ``PointCloud.normalize`` for (B, N, 3)."""
    coords = pc
    centroid = torch.mean(coords, dim=1, keepdim=True)
    coords = coords - centroid
    m = torch.max(torch.sqrt(torch.sum(coords**2, dim=-1)), dim=1, keepdim=True)[0]
    return coords / m.unsqueeze(-1).clamp(min=1e-8)


class TrajectoryPolicy(nn.Module):
    """
    Lift3dCLIP on point cloud (768-D) fused with EE pose + gripper + goal (20-D with robot_state_dim=1),
    then predicts per-step deltas (3 + 6 + 1) and auxiliary grasp prediction (10D).
    """

    def __init__(
        self,
        robot_state_dim: int = 1,
        hidden: int = 512,
        *,
        lift3d_root: Path | str | None = None,
        lift3d_clip_ckpt: Path | str | None = None,
    ):
        super().__init__()
        self.robot_state_dim = robot_state_dim
        self.hidden = hidden
        sg_dim = 3 + 6 + robot_state_dim + 10
        self._sg_dim = sg_dim

        root = Path(lift3d_root or os.environ.get("LIFT3D_ROOT", str(_DEFAULT_LIFT3D_ROOT)))
        ckpt_path = Path(
            lift3d_clip_ckpt
            or os.environ.get("LIFT3D_ENCODER_CKPT")
            or str(_DEFAULT_LIFT3D_CLIP_CKPT)
        )

        _ensure_lift3d_on_path(root)
        from lift3d.models.lift3d.backbone.lift3d_clip import Lift3dCLIP
        from lift3d.models.lift3d.model_utils.clip_loralib import apply_lora, merge_lora
        from lift3d.models.lift3d.model_utils.mv_utils import cfg_from_yaml_file

        yaml_path = root / "lift3d" / "models" / "lift3d" / "model_config" / "ViT-B-32.yaml"
        if not yaml_path.is_file():
            raise FileNotFoundError(f"ViT-B-32.yaml not found at {yaml_path}")
        config = cfg_from_yaml_file(str(yaml_path))
        self.pc_encoder = Lift3dCLIP(config=config.model)
        apply_lora(self.pc_encoder)

        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"Lift3D CLIP checkpoint required for TrajectoryPolicy: {ckpt_path} "
                "(set LIFT3D_ENCODER_CKPT or pass lift3d_clip_ckpt=)"
            )

        ckpt_raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        ckpt_sd = _unpack_ckpt_state(ckpt_raw)
        n_model_keys = len(self.pc_encoder.state_dict())
        incompatible = self.pc_encoder.load_state_dict(ckpt_sd, strict=False)
        missing_keys = list(incompatible.missing_keys)
        unexpected_keys = list(incompatible.unexpected_keys)
        n_loaded_tensors = n_model_keys - len(missing_keys)
        merge_lora(self.pc_encoder)
        file_found = True

        self._lift3d_encoder_load_report: Dict[str, Any] = {
            "ckpt_path": str(ckpt_path.resolve()),
            "file_found": file_found,
            "n_loaded": n_loaded_tensors,
            "n_loaded_tensors": n_loaded_tensors,
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
            "used_checkpoint_keys": [],
        }

        self.fusion_mlp = nn.Sequential(
            nn.Linear(768 + sg_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
        )
        self.head_delta = nn.Linear(hidden, 10)  # 3 + 6 + 1
        self.head_goal = nn.Linear(hidden, 10)

    def forward(
        self,
        point_cloud: torch.Tensor,
        ee_position: torch.Tensor,
        ee_rotation: torch.Tensor,
        gripper: torch.Tensor,
        goal: torch.Tensor,
    ):
        # point_cloud: (B, N, 3)
        pc = _normalize_point_cloud_bnm3(point_cloud.float())
        geom = self.pc_encoder(pc)
        if gripper.dim() == 1:
            gripper = gripper.unsqueeze(-1)
        if goal.dim() == 1:
            goal = goal.unsqueeze(0)
        sg = torch.cat([ee_position, ee_rotation, gripper, goal], dim=-1)
        h = torch.cat([geom, sg], dim=-1)
        h2 = self.fusion_mlp(h)
        return self.head_delta(h2), self.head_goal(h2)
