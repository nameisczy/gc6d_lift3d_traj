"""Lift3D CLIP backbone weights are loaded inside ``TrajectoryPolicy``; this module logs the load report."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Set

import torch
import torch.nn as nn

# Default path for GC6D Lift3D integration (override with env LIFT3D_ENCODER_CKPT).
DEFAULT_LIFT3D_ENCODER_CKPT = Path("/home/ziyaochen/gc6d_lift3d_traj/lift3d_clip_base.pth")


def load_encoder_weights_partial(
    encoder: nn.Module,
    ckpt_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    """Legacy partial load for an ``nn.Module`` (kept for tests that still use a plain encoder)."""
    path = Path(ckpt_path)
    report: Dict[str, Any] = {
        "ckpt_path": str(path.resolve()),
        "file_found": path.is_file(),
        "n_loaded": 0,
        "n_loaded_tensors": 0,
        "missing_keys": [],
        "unexpected_keys": [],
        "used_checkpoint_keys": [],
    }
    if not path.is_file():
        report["missing_keys"] = list(encoder.state_dict().keys())
        return report

    raw = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
        src_sd = raw["model"]
    elif isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        src_sd = raw["state_dict"]
    elif isinstance(raw, dict) and raw and all(isinstance(v, torch.Tensor) for v in raw.values()):
        src_sd = raw
    else:
        raise ValueError(f"Unsupported checkpoint type: {type(raw)}")

    tgt_sd = encoder.state_dict()
    filtered: Dict[str, torch.Tensor] = {}
    used_src: Set[str] = set()

    for tname, tparam in tgt_sd.items():
        chosen_key: str | None = None
        chosen: torch.Tensor | None = None
        if tname in src_sd and src_sd[tname].shape == tparam.shape:
            chosen_key, chosen = tname, src_sd[tname]
        if chosen is None:
            for sk, sv in src_sd.items():
                if sk in used_src or sv.shape != tparam.shape:
                    continue
                for p in ("module.", "model.", "visual.", "vision_model.", "image_encoder."):
                    cand = sk[len(p) :] if sk.startswith(p) else sk
                    if cand == tname:
                        chosen_key, chosen = sk, sv
                        break
                if chosen is not None:
                    break
        if chosen_key is not None and chosen is not None:
            filtered[tname] = chosen.to(dtype=tparam.dtype, device=tparam.device)
            used_src.add(chosen_key)

    incompat = encoder.load_state_dict(filtered, strict=False)
    report["n_loaded"] = int(len(filtered) - len(incompat.unexpected_keys))
    report["n_loaded_tensors"] = report["n_loaded"]
    report["missing_keys"] = list(incompat.missing_keys)
    report["unexpected_keys"] = list(incompat.unexpected_keys)
    report["used_checkpoint_keys"] = sorted(used_src)
    return report


def apply_lift3d_encoder_checkpoint(
    model: nn.Module,
    *,
    ckpt_path: str | Path | None = None,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    """
    Return the Lift3dCLIP load report produced during ``TrajectoryPolicy`` construction.

    Weights are already applied to ``model.pc_encoder`` in ``TrajectoryPolicy.__init__``.
    """
    report = getattr(model, "_lift3d_encoder_load_report", None)
    if report is not None:
        return report
    if hasattr(model, "encoder") and isinstance(getattr(model, "encoder"), nn.Module):
        path = ckpt_path or os.environ.get("LIFT3D_ENCODER_CKPT") or str(DEFAULT_LIFT3D_ENCODER_CKPT)
        return load_encoder_weights_partial(getattr(model, "encoder"), path, map_location=map_location)
    raise AttributeError("model has no _lift3d_encoder_load_report or legacy .encoder")


def log_encoder_load(tag: str, report: Dict[str, Any]) -> None:
    """Print startup diagnostics for encoder weight loading."""
    path = report["ckpt_path"]
    found = report["file_found"]
    n_loaded = report.get("n_loaded_tensors", report.get("n_loaded", 0))
    miss: List[str] = report["missing_keys"]
    unexp: List[str] = report["unexpected_keys"]
    print(f"{tag} encoder checkpoint file_found={found} path={path}")
    print(f"{tag} Lift3dCLIP load: n_loaded_tensors={n_loaded}")
    print(f"{tag} encoder load: n_missing_keys={len(miss)}")
    if miss:
        max_show = 40
        shown = miss[:max_show]
        print(f"{tag} encoder missing_keys (first {len(shown)}): {shown}")
        if len(miss) > max_show:
            print(f"{tag} encoder missing_keys: ... and {len(miss) - max_show} more")
    print(f"{tag} encoder load: n_unexpected_keys={len(unexp)}")
    if unexp:
        max_show = 20
        shown = unexp[:max_show]
        print(f"{tag} encoder unexpected_keys (first {len(shown)}): {shown}")
        if len(unexp) > max_show:
            print(f"{tag} encoder unexpected_keys: ... and {len(unexp) - max_show} more")
    ok = bool(found and n_loaded > 0)
    print(f"{tag} encoder_pretrained_load_ok={ok}")
