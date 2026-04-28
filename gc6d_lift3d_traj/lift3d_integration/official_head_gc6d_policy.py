"""Official Lift3D head + GC6D adapter baseline model."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


_DEFAULT_LIFT3D_ROOT = Path("/home/ziyaochen/LIFT3D")


def _ensure_lift3d_on_path(lift3d_root: Path) -> None:
    root = str(lift3d_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def _unwrap_state_dict(raw: Any) -> Dict[str, torch.Tensor]:
    if isinstance(raw, dict):
        if "model" in raw and isinstance(raw["model"], dict):
            return raw["model"]
        if "state_dict" in raw and isinstance(raw["state_dict"], dict):
            return raw["state_dict"]
        if raw and all(isinstance(v, torch.Tensor) for v in raw.values()):
            return raw
    raise ValueError(f"Unsupported checkpoint structure: {type(raw)}")


def _normalize_point_cloud_bnm3(pc: torch.Tensor) -> torch.Tensor:
    coords = pc.float()
    centroid = torch.mean(coords, dim=1, keepdim=True)
    coords = coords - centroid
    m = torch.max(torch.sqrt(torch.sum(coords**2, dim=-1)), dim=1, keepdim=True)[0]
    return coords / m.unsqueeze(-1).clamp(min=1e-8)


class OfficialHeadGC6DPolicy(nn.Module):
    """
    Reuse official MetaWorld-trained Lift3D `point_cloud_encoder + policy_head`.

    - MetaWorld mode: policy_head(cat(pc_feat, robot_states4))
    - GC6D mode: policy_head(adapter(cat(pc_feat, gc6d_state20))) where gc6d_state20 =
      [ee_pos(3), ee_rot6(6), gripper(1), goal10(10)].
    """

    def __init__(
        self,
        *,
        metaworld_ckpt: str | Path | None = None,
        lift3d_root: str | Path | None = None,
        adapter_hidden: int = 512,
        official_head_init: str = "metaworld",  # random | metaworld
        encoder_init: str = "lift3d_clip",       # lift3d_clip | random | metaworld
        head_init: str = "metaworld",            # random | metaworld
    ):
        super().__init__()
        self.metaworld_ckpt = str(Path(metaworld_ckpt)) if metaworld_ckpt else None
        self.official_head_init = official_head_init
        self.encoder_init = encoder_init
        self.head_init = head_init
        root = Path(lift3d_root or os.environ.get("LIFT3D_ROOT", str(_DEFAULT_LIFT3D_ROOT)))
        _ensure_lift3d_on_path(root)

        from lift3d.models.lift3d.backbone.lift3d_clip import Lift3dCLIP
        from lift3d.models.mlp.batchnorm_mlp import BatchNormMLP
        from lift3d.models.lift3d.model_utils.mv_utils import cfg_from_yaml_file

        ckpt_sd = None
        if self.metaworld_ckpt:
            ckpt_raw = torch.load(self.metaworld_ckpt, map_location="cpu", weights_only=False)
            ckpt_sd = _unwrap_state_dict(ckpt_raw)

        # 1) official point cloud encoder architecture
        yaml_path = root / "lift3d" / "models" / "lift3d" / "model_config" / "ViT-B-32.yaml"
        config = cfg_from_yaml_file(str(yaml_path))
        self.point_cloud_encoder = Lift3dCLIP(config=config.model)

        # 2) official policy head architecture (fixed by official MetaWorld actor)
        self.policy_head_input_dim = self.point_cloud_encoder.feature_dim + 4  # 772
        hidden1, hidden2, hidden3 = 256, 256, 256
        self.policy_head_output_dim = 4
        self.expected_robot_state_dim = 4
        self.policy_head = BatchNormMLP(
            input_dim=self.policy_head_input_dim,
            hidden_dims=[hidden1, hidden2, hidden3],
            output_dim=self.policy_head_output_dim,
            nonlinearity="relu",
            dropout_rate=0.0,
            debug_print_input=False,
        )

        # 3) adapter for GC6D mode: [pc(768), gc6d_state(20)] -> policy_head_input(772)
        self.adapter_in_dim = self.point_cloud_encoder.feature_dim + 20
        self.adapter = nn.Sequential(
            nn.Linear(self.adapter_in_dim, adapter_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(adapter_hidden, self.policy_head_input_dim),
        )

        if self.official_head_init == "random":
            self.head_init = "random"
            if self.encoder_init == "metaworld":
                self.encoder_init = "lift3d_clip"

        mapped = {}
        n_enc_loaded = 0
        n_head_loaded = 0
        if ckpt_sd is not None and (self.encoder_init == "metaworld" or self.head_init == "metaworld"):
            for k, v in ckpt_sd.items():
                if self.encoder_init == "metaworld" and k.startswith("point_cloud_encoder."):
                    tk = "point_cloud_encoder." + k[len("point_cloud_encoder.") :]
                    mapped[tk] = v
                    n_enc_loaded += 1
                elif self.head_init == "metaworld" and k.startswith("policy_head."):
                    mapped[k] = v
                    n_head_loaded += 1
        incompat = self.load_state_dict(mapped, strict=False)
        self._init_report = {
            "n_loaded_tensors": len(mapped),
            "encoder_loaded_tensors": n_enc_loaded,
            "policy_head_loaded_tensors": n_head_loaded,
            "adapter_loaded_tensors": 0,
            "n_missing_keys": len(incompat.missing_keys),
            "n_unexpected_keys": len(incompat.unexpected_keys),
            "missing_keys": list(incompat.missing_keys),
            "unexpected_keys": list(incompat.unexpected_keys),
            "official_head_init": self.official_head_init,
            "encoder_init": self.encoder_init,
            "head_init": self.head_init,
            "policy_head_first_layer_input_dim": self.policy_head_input_dim,
            "policy_head_output_dim": self.policy_head_output_dim,
            "encoder_output_dim": self.point_cloud_encoder.feature_dim,
            "expected_robot_state_dim": self.expected_robot_state_dim,
            "policy_head_arch": f"BatchNormMLP({self.policy_head_input_dim} -> {hidden1} -> {hidden2} -> {hidden3} -> {self.policy_head_output_dim})",
        }

    def gc6d_forward(
        self,
        point_clouds: torch.Tensor,
        ee_position: torch.Tensor,
        ee_rotation6: torch.Tensor,
        gripper: torch.Tensor,
        goal10: torch.Tensor,
    ) -> torch.Tensor:
        if gripper.dim() == 1:
            gripper = gripper.unsqueeze(-1)
        if goal10.dim() == 1:
            goal10 = goal10.unsqueeze(0)
        pc_feat = self.point_cloud_encoder(_normalize_point_cloud_bnm3(point_clouds))
        gc6d_state = torch.cat([ee_position, ee_rotation6, gripper, goal10], dim=-1)  # (B,20)
        adap_in = torch.cat([pc_feat, gc6d_state], dim=-1)
        head_in = self.adapter(adap_in)
        return self.policy_head(head_in)  # (B,4)

    def metaworld_forward(self, point_clouds: torch.Tensor, robot_states: torch.Tensor) -> torch.Tensor:
        pc_feat = self.point_cloud_encoder(_normalize_point_cloud_bnm3(point_clouds))
        x = torch.cat([pc_feat, robot_states], dim=-1)  # (B,772)
        return self.policy_head(x)  # (B,4)

    def inspect(self) -> Dict[str, Any]:
        return dict(self._init_report)

    def trainable_param_breakdown(self) -> Dict[str, int]:
        def count(module: nn.Module) -> int:
            return int(sum(p.numel() for p in module.parameters() if p.requires_grad))
        return {
            "encoder": count(self.point_cloud_encoder),
            "adapter": count(self.adapter),
            "policy_head": count(self.policy_head),
        }
