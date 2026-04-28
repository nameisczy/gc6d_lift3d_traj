"""Load official LIFT3D MetaWorld policy checkpoint into TrajectoryPolicy (partial map)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn


def _unwrap_state_dict(raw: Any) -> Dict[str, torch.Tensor]:
    if isinstance(raw, dict):
        if "model" in raw and isinstance(raw["model"], dict):
            return raw["model"]
        if "state_dict" in raw and isinstance(raw["state_dict"], dict):
            return raw["state_dict"]
        if raw and all(isinstance(v, torch.Tensor) for v in raw.values()):
            return raw
    raise ValueError(f"Unsupported checkpoint structure: {type(raw)}")


def inspect_checkpoint(ckpt_path: str | Path, max_keys: int = 30) -> Dict[str, Any]:
    path = Path(ckpt_path)
    raw = torch.load(str(path), map_location="cpu", weights_only=False)
    sd = _unwrap_state_dict(raw)
    keys = list(sd.keys())
    groups = {"point_cloud_encoder": 0, "policy_head": 0, "other": 0}
    for k in keys:
        if k.startswith("point_cloud_encoder."):
            groups["point_cloud_encoder"] += 1
        elif k.startswith("policy_head."):
            groups["policy_head"] += 1
        else:
            groups["other"] += 1
    return {
        "path": str(path.resolve()),
        "n_tensors": len(keys),
        "n_point_cloud_encoder": groups["point_cloud_encoder"],
        "n_policy_head": groups["policy_head"],
        "n_other": groups["other"],
        "first_keys": keys[:max_keys],
    }


def load_metaworld_policy_init(
    model: nn.Module,
    ckpt_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    """
    Map official policy params into current TrajectoryPolicy:
    - point_cloud_encoder.* -> pc_encoder.*
    - exact-name match fallback for any coincident keys
    """
    path = Path(ckpt_path)
    report: Dict[str, Any] = {
        "ckpt_path": str(path.resolve()),
        "file_found": path.is_file(),
        "n_source_tensors": 0,
        "n_loaded_tensors": 0,
        "n_mapped_encoder_tensors": 0,
        "n_policy_head_loaded_tensors": 0,
        "n_exact_match_tensors": 0,
        "n_policy_head_source_tensors": 0,
        "n_policy_head_shape_mismatch": 0,
        "missing_keys": [],
        "unexpected_keys": [],
    }
    if not path.is_file():
        report["missing_keys"] = list(model.state_dict().keys())
        return report

    raw = torch.load(str(path), map_location=map_location, weights_only=False)
    src = _unwrap_state_dict(raw)
    report["n_source_tensors"] = len(src)
    tgt = model.state_dict()

    mapped: Dict[str, torch.Tensor] = {}
    n_enc = 0
    n_exact = 0
    n_head_loaded = 0
    n_head_src = 0
    n_head_shape_mismatch = 0
    # Candidate target head keys in current TrajectoryPolicy.
    tgt_head_keys = [
        k for k in tgt.keys() if k.startswith("fusion_mlp.") or k.startswith("head_delta.") or k.startswith("head_goal.")
    ]
    used_tgt = set()
    for k, v in src.items():
        if k.startswith("point_cloud_encoder."):
            tk = "pc_encoder." + k[len("point_cloud_encoder.") :]
            if tk in tgt and tgt[tk].shape == v.shape:
                mapped[tk] = v.to(dtype=tgt[tk].dtype)
                n_enc += 1
            continue
        if k.startswith("policy_head."):
            n_head_src += 1
            # first try exact key match
            if k in tgt and tgt[k].shape == v.shape:
                mapped[k] = v.to(dtype=tgt[k].dtype)
                n_head_loaded += 1
                used_tgt.add(k)
                continue
            # then try shape-based partial transfer into current head params
            chosen = None
            for tk in tgt_head_keys:
                if tk in used_tgt:
                    continue
                if tgt[tk].shape == v.shape:
                    chosen = tk
                    break
            if chosen is not None:
                mapped[chosen] = v.to(dtype=tgt[chosen].dtype)
                used_tgt.add(chosen)
                n_head_loaded += 1
            else:
                n_head_shape_mismatch += 1
            continue
        if k in tgt and tgt[k].shape == v.shape:
            mapped[k] = v.to(dtype=tgt[k].dtype)
            n_exact += 1

    incompat = model.load_state_dict(mapped, strict=False)
    report["n_loaded_tensors"] = int(len(mapped) - len(incompat.unexpected_keys))
    report["n_mapped_encoder_tensors"] = n_enc
    report["n_policy_head_loaded_tensors"] = n_head_loaded
    report["n_exact_match_tensors"] = n_exact
    report["n_policy_head_source_tensors"] = n_head_src
    report["n_policy_head_shape_mismatch"] = n_head_shape_mismatch
    report["missing_keys"] = list(incompat.missing_keys)
    report["unexpected_keys"] = list(incompat.unexpected_keys)
    return report


def log_metaworld_init(tag: str, report: Dict[str, Any]) -> None:
    print(f"{tag} metaworld_init file_found={report['file_found']} path={report['ckpt_path']}")
    print(f"{tag} source_tensors={report['n_source_tensors']}")
    print(
        f"{tag} loaded_tensors={report['n_loaded_tensors']} "
        f"(encoder_mapped={report['n_mapped_encoder_tensors']}, "
        f"policy_head_loaded={report['n_policy_head_loaded_tensors']}, "
        f"exact_match={report['n_exact_match_tensors']})"
    )
    print(
        f"{tag} policy_head_source_tensors={report['n_policy_head_source_tensors']} "
        f"policy_head_shape_mismatch={report['n_policy_head_shape_mismatch']}"
    )
    print(f"{tag} n_missing_keys={len(report['missing_keys'])}")
    print(f"{tag} n_unexpected_keys={len(report['unexpected_keys'])}")
