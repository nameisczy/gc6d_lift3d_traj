#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from gc6d_lift3d_traj.lift3d_integration.lift3d_encoder_ckpt import (
    apply_lift3d_encoder_checkpoint,
    log_encoder_load,
)
from gc6d_lift3d_traj.lift3d_integration.official_head_gc6d_policy import (
    OfficialHeadGC6DPolicy,
)
from gc6d_lift3d_traj.lift3d_integration.trajectory_policy import TrajectoryPolicy
from gc6d_lift3d_traj.utils.action10_to_gc6d17 import action10_to_gc6d17
from gc6d_lift3d_traj.utils.rotations import (
    lift3d_rotation_to_matrix,
    matrix_to_lift3d_rotation,
)


def _ensure_numpy2_transforms3d_compat() -> None:
    """graspclutter6dAPI -> transforms3d may reference np.float / np.maximum_sctype (removed in NumPy 2)."""
    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]
    if not hasattr(np, "maximum_sctype"):

        def _maximum_sctype(t):
            dt = np.dtype(t)
            if np.issubdtype(dt, np.floating):
                return np.float64
            if np.issubdtype(dt, np.complexfloating):
                return np.complex128
            if np.issubdtype(dt, np.signedinteger):
                return np.int64
            if np.issubdtype(dt, np.unsignedinteger):
                return np.uint64
            if np.issubdtype(dt, np.bool_):
                return np.bool_
            return dt.type

        np.maximum_sctype = _maximum_sctype  # type: ignore[attr-defined]


def _load_rows(index_path: Path) -> List[dict]:
    rows = []
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _normalize_pc(pc: torch.Tensor) -> torch.Tensor:
    c = pc.float()
    centroid = c.mean(dim=1, keepdim=True)
    c = c - centroid
    m = torch.max(torch.sqrt(torch.sum(c**2, dim=-1)), dim=1, keepdim=True)[0]
    return c / m.unsqueeze(-1).clamp(min=1e-8)


def _to_t(x: np.ndarray, device: torch.device, add_batch: bool = True) -> torch.Tensor:
    t = torch.from_numpy(np.asarray(x, dtype=np.float32))
    if add_batch:
        t = t.unsqueeze(0)
    return t.to(device)


def _safe_cos(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _stats_1d(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return {"mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def _percentiles(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return {k: math.nan for k in ("p10", "p25", "p50", "p75", "p90")}
    return {
        "p10": float(np.percentile(x, 10)),
        "p25": float(np.percentile(x, 25)),
        "p50": float(np.percentile(x, 50)),
        "p75": float(np.percentile(x, 75)),
        "p90": float(np.percentile(x, 90)),
    }


def _pca_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 2:
        return np.zeros((X.shape[0], 2), dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    P = vt[:2].T
    return Xc @ P


def _plot_action_stats(pred4: np.ndarray, gt4: np.ndarray, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    names = ["dx", "dy", "dz", "dgrip"]
    for i, ax in enumerate(axes.reshape(-1)):
        ax.hist(gt4[:, i], bins=50, alpha=0.5, label="gt")
        ax.hist(pred4[:, i], bins=50, alpha=0.5, label="pred")
        ax.set_title(names[i])
        ax.grid(alpha=0.2)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _plot_rollout_curves(case: Dict[str, Any], out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    pos_err = np.asarray(case["pos_err_curve"], dtype=np.float64)
    dgoal_p = np.asarray(case["dist_goal_pred_curve"], dtype=np.float64)
    dgoal_g = np.asarray(case["dist_goal_gt_curve"], dtype=np.float64)
    grip_p = np.asarray(case["gripper_pred_curve"], dtype=np.float64)
    grip_g = np.asarray(case["gripper_gt_curve"], dtype=np.float64)
    t = np.arange(pos_err.shape[0])
    fig, axes = plt.subplots(3, 1, figsize=(8, 9))
    axes[0].plot(t, pos_err, label="|p_pred - p_gt|")
    axes[0].set_title("Per-step position error")
    axes[1].plot(t, dgoal_p, label="pred->goal")
    axes[1].plot(t, dgoal_g, label="gt->goal")
    axes[1].set_title("Distance-to-goal")
    axes[2].plot(np.arange(grip_p.shape[0]), grip_p, label="pred grip")
    axes[2].plot(np.arange(grip_g.shape[0]), grip_g, label="gt grip")
    axes[2].set_title("Gripper curve")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _draw_frame(ax, center: np.ndarray, R: np.ndarray, scale: float, label: str) -> None:
    c = center.reshape(3)
    cols = ["r", "g", "b"]
    for i in range(3):
        v = R[:, i] * scale
        ax.quiver(c[0], c[1], c[2], v[0], v[1], v[2], color=cols[i], linewidth=2)
    ax.text(c[0], c[1], c[2], label)


def _plot_grasp_case(case: Dict[str, Any], out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    pc = np.asarray(case["point_cloud"], dtype=np.float32)
    gt_c = np.asarray(case["gt_center"], dtype=np.float32)
    gt_R = np.asarray(case["gt_rot"], dtype=np.float32).reshape(3, 3)
    pr_c = np.asarray(case["pred_center"], dtype=np.float32)
    pr_R = np.asarray(case["pred_rot"], dtype=np.float32).reshape(3, 3)
    traj = np.asarray(case["traj_pred"], dtype=np.float32)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1, alpha=0.25)
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], "-m", linewidth=2, label="pred traj")
    _draw_frame(ax, gt_c, gt_R, 0.04, "GT")
    _draw_frame(ax, pr_c, pr_R, 0.04, "Pred")
    ax.set_title(
        f"scene={case['scene_id']} ann={case['ann_id']} final_center_err={case['final_center_err']:.4f}"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _plot_pca(points_a: np.ndarray, points_b: np.ndarray, label_a: str, label_b: str, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7, 6))
    if points_a.size > 0:
        plt.scatter(points_a[:, 0], points_a[:, 1], s=10, alpha=0.6, label=label_a)
    if points_b.size > 0:
        plt.scatter(points_b[:, 0], points_b[:, 1], s=10, alpha=0.6, label=label_b)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=170)
    plt.close(fig)


@dataclass
class Diagnostics:
    action_pred4: List[np.ndarray]
    action_gt4: List[np.ndarray]
    trans_cos: List[float]
    action_norm_pred: List[float]
    action_norm_gt: List[float]
    rollout_cases: List[Dict[str, Any]]
    gc6d_feats: List[np.ndarray]
    gc6d_feat_norms: List[float]
    mw_feats: List[np.ndarray]
    mw_feat_norms: List[float]
    top1_center_dist: List[float]
    top1_rot_trace: List[float]
    width: List[float]
    height: List[float]
    depth: List[float]


def _extract_feature(model: torch.nn.Module, pc_np: np.ndarray, device: torch.device, model_type: str) -> np.ndarray:
    pc = _to_t(pc_np, device, add_batch=True)
    with torch.no_grad():
        if model_type == "official_head_adapter":
            f = model.point_cloud_encoder(_normalize_pc(pc))
        else:
            f = model.pc_encoder(_normalize_pc(pc))
    return f[0].detach().cpu().numpy().astype(np.float32)


def _predict_delta10(
    model: torch.nn.Module,
    model_type: str,
    pc_np: np.ndarray,
    ee_pos: np.ndarray,
    ee_rot6: np.ndarray,
    grip: np.ndarray,
    goal10: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    pc = _to_t(pc_np, device, add_batch=True)
    ee = _to_t(ee_pos, device, add_batch=True)
    er = _to_t(ee_rot6, device, add_batch=True)
    g = _to_t(grip, device, add_batch=True)
    if g.dim() == 1:
        g = g.unsqueeze(-1)
    goal = _to_t(goal10, device, add_batch=True)
    with torch.no_grad():
        if model_type == "official_head_adapter":
            p4 = model.gc6d_forward(pc, ee, er, g, goal)[0].detach().cpu().numpy().astype(np.float32)
            return np.concatenate([p4[:3], np.zeros(6, np.float32), p4[3:4]], axis=0)
        pd, _pg = model(pc, ee, er, g, goal)
    return pd[0].detach().cpu().numpy().astype(np.float32)


def _build_model(args, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if args.model_type == "official_head_adapter":
        model = OfficialHeadGC6DPolicy(
            metaworld_ckpt=args.init_metaworld_ckpt,
            official_head_init=args.official_head_init,
            encoder_init=args.encoder_init,
            head_init=args.head_init,
        ).to(device)
        model.load_state_dict(sd, strict=True)
    else:
        model = TrajectoryPolicy(robot_state_dim=1).to(device)
        enc_report = apply_lift3d_encoder_checkpoint(model, map_location=device)
        log_encoder_load("[model]", enc_report)
        model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def _load_gt_grasps(api, scene_id: int, ann_id: int, camera: str) -> Tuple[np.ndarray, np.ndarray]:
    gg = api.loadGrasp(scene_id, ann_id, format="6d", camera=camera, fric_coef_thresh=1.0)
    arr = np.asarray(gg.grasp_group_array, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3, 3), np.float32)
    centers = arr[:, 13:16].astype(np.float32)
    rots = arr[:, 4:13].reshape(-1, 3, 3).astype(np.float32)
    return centers, rots


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--model-type", type=str, choices=("custom_action10", "official_head_adapter"), default="custom_action10")
    ap.add_argument("--init-metaworld-ckpt", type=str, default=None)
    ap.add_argument("--official-head-init", type=str, choices=("random", "metaworld"), default="metaworld")
    ap.add_argument("--encoder-init", type=str, choices=("lift3d_clip", "random", "metaworld"), default="lift3d_clip")
    ap.add_argument("--head-init", type=str, choices=("random", "metaworld"), default="metaworld")
    ap.add_argument("--gc6d-root", type=str, default="/mnt/ssd/ziyaochen/GraspClutter6D")
    ap.add_argument("--gc6d-api-root", type=str, default="/home/ziyaochen/graspclutter6dAPI")
    ap.add_argument("--camera", type=str, default="realsense-d415")
    ap.add_argument("--max-episodes", type=int, default=200)
    ap.add_argument("--metaworld-npz", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(args, device)
    rows = _load_rows(Path(args.index))[: args.max_episodes]

    if args.gc6d_api_root not in sys.path:
        sys.path.insert(0, args.gc6d_api_root)
    _ensure_numpy2_transforms3d_compat()
    from graspclutter6dAPI.graspclutter6d import GraspClutter6D

    gc6d_api = GraspClutter6D(root=args.gc6d_root, camera=args.camera, split="test")

    d = Diagnostics([], [], [], [], [], [], [], [], [], [], [], [], [], [], [])

    for row in rows:
        data = np.load(row["episode_path"], allow_pickle=True)
        pc = np.asarray(data["point_cloud"], dtype=np.float32)
        ee_pos_gt = np.asarray(data["ee_positions"], dtype=np.float32)
        ee_rot_gt = np.asarray(data["ee_rotations"], dtype=np.float32)
        grip_gt = np.asarray(data["gripper"], dtype=np.float32).reshape(-1)
        T = int(ee_pos_gt.shape[0])
        gt_center = np.asarray(data["gt_grasp_center"], dtype=np.float32).reshape(3)
        gt_rot = np.asarray(data["gt_grasp_rotation"], dtype=np.float32).reshape(3, 3)
        gt_width = float(np.asarray(data["gt_grasp_width"]).reshape(-1)[0])
        goal10 = np.concatenate(
            [gt_center, ee_rot_gt[-1], np.array([gt_width], dtype=np.float32)], axis=0
        ).astype(np.float32)

        feat = _extract_feature(model, pc, device, args.model_type)
        d.gc6d_feats.append(feat)
        d.gc6d_feat_norms.append(float(np.linalg.norm(feat)))

        pred_pos = [ee_pos_gt[0].copy()]
        pred_rot = [ee_rot_gt[0].copy()]
        pred_grip = [float(grip_gt[0])]

        for t in range(T - 1):
            pred10 = _predict_delta10(
                model,
                args.model_type,
                pc,
                pred_pos[-1],
                pred_rot[-1],
                np.array([pred_grip[-1]], np.float32),
                goal10,
                device,
            )
            gt10 = np.concatenate(
                [
                    np.asarray(data["actions_translation"][t], np.float32),
                    np.asarray(data["actions_rotation"][t], np.float32),
                    np.asarray(data["actions_gripper"][t], np.float32).reshape(1),
                ],
                axis=0,
            )
            pred4 = np.concatenate([pred10[:3], pred10[9:10]], axis=0)
            gt4 = np.concatenate([gt10[:3], gt10[9:10]], axis=0)
            d.action_pred4.append(pred4)
            d.action_gt4.append(gt4)
            d.trans_cos.append(_safe_cos(pred10[:3], gt10[:3]))
            d.action_norm_pred.append(float(np.linalg.norm(pred4)))
            d.action_norm_gt.append(float(np.linalg.norm(gt4)))

            nxt_pos = pred_pos[-1] + pred10[:3]
            if args.model_type == "official_head_adapter":
                nxt_rot = pred_rot[-1].copy()
            else:
                R_cur = lift3d_rotation_to_matrix(pred_rot[-1])
                R_del = lift3d_rotation_to_matrix(pred10[3:9])
                nxt_rot = matrix_to_lift3d_rotation(R_del @ R_cur).astype(np.float32)
            nxt_grip = float(pred_grip[-1] + pred10[9])
            pred_pos.append(nxt_pos.astype(np.float32))
            pred_rot.append(nxt_rot.astype(np.float32))
            pred_grip.append(nxt_grip)

        pred_pos_a = np.asarray(pred_pos, dtype=np.float32)
        pred_rot_a = np.asarray(pred_rot, dtype=np.float32)
        pred_grip_a = np.asarray(pred_grip, dtype=np.float32)
        pos_err = np.linalg.norm(pred_pos_a - ee_pos_gt, axis=1)
        dist_goal_pred = np.linalg.norm(pred_pos_a - gt_center[None, :], axis=1)
        dist_goal_gt = np.linalg.norm(ee_pos_gt - gt_center[None, :], axis=1)
        final_center_err = float(np.linalg.norm(pred_pos_a[-1] - gt_center))

        final_goal10 = np.concatenate(
            [pred_pos_a[-1], pred_rot_a[-1], np.array([np.clip(abs(pred_grip_a[-1]), 0.0, 0.14)], np.float32)],
            axis=0,
        ).astype(np.float32)
        row17 = action10_to_gc6d17(final_goal10, score=1.0)
        d.width.append(float(row17[1]))
        d.height.append(float(row17[2]))
        d.depth.append(float(row17[3]))

        centers_gt, rots_gt = _load_gt_grasps(
            gc6d_api, int(row["scene_id"]), int(row["ann_id"]), args.camera
        )
        if centers_gt.shape[0] > 0:
            cd = np.linalg.norm(centers_gt - final_goal10[:3][None, :], axis=1)
            nn = int(np.argmin(cd))
            d.top1_center_dist.append(float(cd[nn]))
            R_pred = lift3d_rotation_to_matrix(final_goal10[3:9])
            d.top1_rot_trace.append(float(np.trace(R_pred.T @ rots_gt[nn])))
        else:
            d.top1_center_dist.append(float("inf"))
            d.top1_rot_trace.append(-3.0)

        d.rollout_cases.append(
            {
                "scene_id": int(row["scene_id"]),
                "ann_id": int(row["ann_id"]),
                "point_cloud": pc,
                "gt_center": gt_center,
                "gt_rot": gt_rot,
                "pred_center": pred_pos_a[-1],
                "pred_rot": lift3d_rotation_to_matrix(pred_rot_a[-1]),
                "traj_pred": pred_pos_a,
                "pos_err_curve": pos_err,
                "dist_goal_pred_curve": dist_goal_pred,
                "dist_goal_gt_curve": dist_goal_gt,
                "final_center_err": final_center_err,
                "gripper_pred_curve": pred_grip_a,
                "gripper_gt_curve": grip_gt,
            }
        )

    if args.metaworld_npz:
        mw = np.load(args.metaworld_npz, allow_pickle=True)
        if "all_point_clouds" not in mw.files:
            print(
                "WARNING: metaworld npz has no 'all_point_clouds'; "
                "skip MetaWorld feature diagnostics. Re-collect with real point-cloud collector to enable this block."
            )
        else:
            pcs = np.asarray(mw["all_point_clouds"], dtype=np.float32)
            for i in range(min(len(pcs), len(rows))):
                f = _extract_feature(model, pcs[i], device, args.model_type)
                d.mw_feats.append(f)
                d.mw_feat_norms.append(float(np.linalg.norm(f)))

    pred4 = np.asarray(d.action_pred4, dtype=np.float32)
    gt4 = np.asarray(d.action_gt4, dtype=np.float32)
    mae4 = np.mean(np.abs(pred4 - gt4), axis=0) if pred4.size else np.zeros(4, np.float32)

    action_json = {
        "per_dim_mae_action4": {
            "dx": float(mae4[0]),
            "dy": float(mae4[1]),
            "dz": float(mae4[2]),
            "dgrip": float(mae4[3]),
        },
        "pred_action4_stats": {
            "dx": _stats_1d(pred4[:, 0] if pred4.size else np.array([])),
            "dy": _stats_1d(pred4[:, 1] if pred4.size else np.array([])),
            "dz": _stats_1d(pred4[:, 2] if pred4.size else np.array([])),
            "dgrip": _stats_1d(pred4[:, 3] if pred4.size else np.array([])),
        },
        "gt_action4_stats": {
            "dx": _stats_1d(gt4[:, 0] if gt4.size else np.array([])),
            "dy": _stats_1d(gt4[:, 1] if gt4.size else np.array([])),
            "dz": _stats_1d(gt4[:, 2] if gt4.size else np.array([])),
            "dgrip": _stats_1d(gt4[:, 3] if gt4.size else np.array([])),
        },
        "action_norm_pred": _stats_1d(np.asarray(d.action_norm_pred)),
        "action_norm_gt": _stats_1d(np.asarray(d.action_norm_gt)),
        "translation_cosine_similarity": _stats_1d(np.asarray(d.trans_cos)),
    }

    final_errs = np.asarray([c["final_center_err"] for c in d.rollout_cases], dtype=np.float64)
    rollout_json = {
        "final_center_error": _stats_1d(final_errs),
        "final_center_error_percentiles": _percentiles(final_errs),
        "distance_to_goal_last_pred": _stats_1d(
            np.asarray([c["dist_goal_pred_curve"][-1] for c in d.rollout_cases], dtype=np.float64)
        ),
    }

    feat_json = {
        "gc6d_feature_norm": _stats_1d(np.asarray(d.gc6d_feat_norms, dtype=np.float64)),
        "metaworld_feature_norm": _stats_1d(np.asarray(d.mw_feat_norms, dtype=np.float64)),
    }

    ap_top1_json = {
        "top1_center_distance_to_nearest_gt": _stats_1d(np.asarray(d.top1_center_dist)),
        "top1_rotation_trace_to_nearest_gt": _stats_1d(np.asarray(d.top1_rot_trace)),
        "width_stats": _stats_1d(np.asarray(d.width)),
        "height_stats": _stats_1d(np.asarray(d.height)),
        "depth_stats": _stats_1d(np.asarray(d.depth)),
        "center_distance_percentiles": _percentiles(np.asarray(d.top1_center_dist)),
        "rotation_trace_percentiles": _percentiles(np.asarray(d.top1_rot_trace)),
        "width_percentiles": _percentiles(np.asarray(d.width)),
        "height_percentiles": _percentiles(np.asarray(d.height)),
        "depth_percentiles": _percentiles(np.asarray(d.depth)),
    }

    # Plots
    _plot_action_stats(pred4, gt4, out_dir / "action_diagnostics.png")
    if d.rollout_cases:
        order = np.argsort([c["final_center_err"] for c in d.rollout_cases])
        best = d.rollout_cases[int(order[0])]
        worst = d.rollout_cases[int(order[-1])]
        rnd = d.rollout_cases[int(np.random.default_rng(args.seed).integers(0, len(d.rollout_cases)))]
        for tag, c in (("best", best), ("worst", worst), ("random", rnd)):
            _plot_rollout_curves(c, out_dir / f"rollout_curves_{tag}.png")
            _plot_grasp_case(c, out_dir / f"grasp_case_{tag}.png")

    if d.gc6d_feats:
        Fg = np.asarray(d.gc6d_feats, dtype=np.float64)
        if d.mw_feats:
            Fm = np.asarray(d.mw_feats, dtype=np.float64)
            X = np.concatenate([Fg, Fm], axis=0)
            Z = _pca_2d(X)
            zg, zm = Z[: len(Fg)], Z[len(Fg) :]
            _plot_pca(zg, zm, "GC6D", "MetaWorld", out_dir / "pca_metaworld_vs_gc6d.png")
        if len(Fg) >= 4:
            k = max(1, len(Fg) // 4)
            order = np.argsort(final_errs)
            lo = Fg[order[:k]]
            hi = Fg[order[-k:]]
            Z2 = _pca_2d(np.concatenate([lo, hi], axis=0))
            _plot_pca(Z2[: len(lo)], Z2[len(lo) :], "low-error", "high-error", out_dir / "pca_low_vs_high_error_gc6d.png")

    summary = {
        "action_diagnostics": action_json,
        "rollout_diagnostics": rollout_json,
        "feature_diagnostics": feat_json,
        "ap_top1_diagnostics": ap_top1_json,
        "num_episodes": len(rows),
    }
    (out_dir / "diagnostics_summary.json").write_text(
        json.dumps(summary, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x),
        encoding="utf-8",
    )

    # Markdown report
    fail_hi = int(np.sum(final_errs > 0.05))
    fail_med = int(np.sum((final_errs > 0.02) & (final_errs <= 0.05)))
    fail_lo = int(np.sum(final_errs <= 0.02))
    md = []
    md.append("# Diagnostics Report")
    md.append("")
    md.append("## Failure Mode Breakdown")
    md.append(f"- episodes: {len(rows)}")
    md.append(f"- final_center_err <= 2cm: {fail_lo}")
    md.append(f"- 2cm < final_center_err <= 5cm: {fail_med}")
    md.append(f"- final_center_err > 5cm: {fail_hi}")
    md.append("")
    md.append("## Key Metrics")
    md.append(f"- action MAE (dx,dy,dz,dgrip): {action_json['per_dim_mae_action4']}")
    md.append(f"- top1 nearest-GT center distance stats: {ap_top1_json['top1_center_distance_to_nearest_gt']}")
    md.append(f"- top1 nearest-GT rotation trace stats: {ap_top1_json['top1_rotation_trace_to_nearest_gt']}")
    md.append(f"- GC6D feature norm: {feat_json['gc6d_feature_norm']}")
    if d.mw_feat_norms:
        md.append(f"- MetaWorld feature norm: {feat_json['metaworld_feature_norm']}")
    md.append("")
    md.append("## Artifacts")
    md.append("- `diagnostics_summary.json`")
    md.append("- `action_diagnostics.png`")
    md.append("- `rollout_curves_best.png`, `rollout_curves_worst.png`, `rollout_curves_random.png`")
    md.append("- `grasp_case_best.png`, `grasp_case_worst.png`, `grasp_case_random.png`")
    md.append("- `pca_metaworld_vs_gc6d.png` (if `--metaworld-npz` provided)")
    md.append("- `pca_low_vs_high_error_gc6d.png`")
    (out_dir / "diagnostics_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Saved diagnostics to: {out_dir}")


if __name__ == "__main__":
    main()
