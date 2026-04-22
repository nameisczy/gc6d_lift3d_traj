#!/usr/bin/env python3
"""
Single-grasp success proxy for GC6D test split (NOT official AP).

Does NOT call GraspClutter6DEval or eval_all. Compares top-1 predicted 17D grasp
per test image against GT from GraspClutter6D.loadGrasp.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Repo imports (run with PYTHONPATH=<repo> or from installed package)
from gc6d_lift3d_traj.gc6d.grasp_decode import decode_gc6d_grasp
from gc6d_lift3d_traj.utils.gc6d_rgb import ann_id_to_img_id


def _iter_test_scene_ann_camera(gc6d_root: Path, camera: str) -> List[Tuple[int, int, int]]:
    split_file = gc6d_root / "split_info" / "grasp_test_scene_ids.json"
    scene_ids = [int(x) for x in json.loads(split_file.read_text(encoding="utf-8"))]
    out: List[Tuple[int, int, int]] = []
    for scene_id in scene_ids:
        for ann_id in range(13):
            img_id = ann_id_to_img_id(ann_id, camera)
            out.append((scene_id, ann_id, img_id))
    return out


def _pred_path(dump_root: Path, camera: str, scene_id: int, img_id: int) -> Path:
    return dump_root / f"{scene_id:06d}" / camera / f"{img_id:06d}.npy"


def load_top1_grasp17(path: Path) -> np.ndarray:
    arr = np.load(path, allow_pickle=False)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != 17:
        raise ValueError(f"Expected (K,17) grasp rows at {path}, got {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError(f"Empty prediction at {path}")
    order = np.argsort(-arr[:, 0].astype(np.float64))
    return arr[order[0]].astype(np.float32)


def gt_centers_rotations(gt_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """gt_arr (N,17) -> centers (N,3), R (N,3,3)."""
    if gt_arr.size == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3, 3), np.float32)
    c = gt_arr[:, 13:16].astype(np.float32)
    r = gt_arr[:, 4:13].reshape(-1, 3, 3).astype(np.float32)
    return c, r


def rotation_traces(pred_R: np.ndarray, gt_R: np.ndarray) -> np.ndarray:
    """trace(R_pred^T R_gt[j]) for each j, shape (N,)."""
    if gt_R.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    # trace(R_p^T R_g) = sum_ij R_p[i,j] * R_g[i,j] with same i,j indexing
    return np.sum(pred_R[None, :, :] * gt_R, axis=(1, 2)).astype(np.float32)


@dataclass
class SingleGraspEvalResult:
    success_rate: float
    mean_translation_error: float
    mean_rotation_trace: float
    num_images: int
    num_success: int
    min_dists: np.ndarray
    max_traces: np.ndarray
    traces_to_nn: np.ndarray  # trace w.r.t. nearest GT center (by translation)
    n_trans_only: int = 0
    n_rot_only: int = 0
    n_both: int = 0
    n_split_gt: int = 0  # trans & rot thresholds met but on different GT rows
    collision_note: str = ""
    friction_note: str = ""


def evaluate_single_grasp_dump(
    dump_root: Path,
    gc6d_root: Path,
    camera: str,
    split: str,
    dataset: Any,
    fric_coef_thresh: float,
    trans_thresh: float,
    trace_thresh: float,
    grasp_labels_all: Optional[Any] = None,
) -> SingleGraspEvalResult:
    assert split == "test", "This script is implemented for split=test only."

    triples = _iter_test_scene_ann_camera(gc6d_root, camera)
    num_test_images = len(triples)

    missing: List[Path] = []
    for scene_id, ann_id, img_id in triples:
        p = _pred_path(dump_root, camera, scene_id, img_id)
        if not p.is_file():
            missing.append(p)

    num_predictions = num_test_images - len(missing)
    print(f"num_test_images: {num_test_images}")
    print(f"num_predictions: {num_predictions}")
    if num_predictions != num_test_images:
        print(
            "FAIL: num_predictions != num_test_images "
            f"({num_predictions} != {num_test_images}). "
            "Use official_dump layout: <dump_root>/<scene:06d>/<camera>/<img_id:06d>.npy",
            file=sys.stderr,
        )
        for mp in missing[:20]:
            print(f"  missing: {mp}", file=sys.stderr)
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more", file=sys.stderr)
        sys.exit(1)

    collision_cache: Dict[int, dict] = {}
    if grasp_labels_all is None:
        grasp_labels_all = dataset.loadGraspLabels(objIds=dataset.objIds)

    min_dists: List[float] = []
    max_traces: List[float] = []
    traces_nn: List[float] = []
    successes: List[bool] = []
    trans_only = 0
    rot_only = 0
    both = 0
    split_gt = 0

    for scene_id, ann_id, img_id in triples:
        pred_path = _pred_path(dump_root, camera, scene_id, img_id)
        pred_row = load_top1_grasp17(pred_path)
        dec_p = decode_gc6d_grasp(pred_row)
        pred_c = dec_p["center"].astype(np.float32).reshape(3)
        pred_R = dec_p["rotation"].astype(np.float32).reshape(3, 3)

        if scene_id not in collision_cache:
            collision_cache[scene_id] = dataset.loadCollisionLabels(sceneIds=scene_id)
        collision_labels = collision_cache[scene_id]

        gg = dataset.loadGrasp(
            scene_id,
            ann_id,
            format="6d",
            camera=camera,
            grasp_labels=grasp_labels_all,
            collision_labels=collision_labels,
            fric_coef_thresh=fric_coef_thresh,
        )
        gt_arr = np.asarray(gg.grasp_group_array, dtype=np.float32)
        gt_c, gt_R = gt_centers_rotations(gt_arr)

        if gt_c.shape[0] == 0:
            dists = np.array([np.inf], dtype=np.float32)
            traces = np.array([-3.0], dtype=np.float32)
        else:
            dists = np.linalg.norm(gt_c - pred_c.reshape(1, 3), axis=1)
            traces = rotation_traces(pred_R, gt_R)

        min_dist = float(np.min(dists)) if dists.size else float("inf")
        max_trace = float(np.max(traces)) if traces.size else -3.0
        nn = int(np.argmin(dists)) if dists.size else 0
        trace_nn = float(traces[nn]) if traces.size else -3.0

        min_dists.append(min_dist)
        max_traces.append(max_trace)
        traces_nn.append(trace_nn)

        # ∃ j : translation & rotation (same GT)
        if gt_c.shape[0] == 0:
            ok = False
        else:
            ok = bool(np.any((dists < trans_thresh) & (traces > trace_thresh)))
        successes.append(ok)

        t_ok = min_dist < trans_thresh
        r_ok = max_trace > trace_thresh
        if ok:
            both += 1
        elif t_ok and r_ok and not ok:
            split_gt += 1
        elif t_ok and not r_ok:
            trans_only += 1
        elif r_ok and not t_ok:
            rot_only += 1

    min_dists_a = np.asarray(min_dists, dtype=np.float64)
    traces_nn_a = np.asarray(traces_nn, dtype=np.float64)
    max_tr_a = np.asarray(max_traces, dtype=np.float64)

    num_success = int(sum(successes))
    n_img = len(triples)
    sr = num_success / max(n_img, 1)

    return SingleGraspEvalResult(
        success_rate=sr,
        mean_translation_error=float(np.mean(min_dists_a)),
        mean_rotation_trace=float(np.mean(traces_nn_a)),
        num_images=n_img,
        num_success=num_success,
        min_dists=min_dists_a.astype(np.float32),
        max_traces=max_tr_a.astype(np.float32),
        traces_to_nn=traces_nn_a.astype(np.float32),
        n_trans_only=trans_only,
        n_rot_only=rot_only,
        n_both=both,
        n_split_gt=split_gt,
        collision_note="prediction_collision: skipped (no per-prediction collision bit in npz; GT from loadGrasp is collision-filtered)",
        friction_note="prediction_force_closure: skipped (17D row has no friction coef; GT loadGrasp uses fric_coef_thresh)",
    )


def _print_summary(tag: str, r: SingleGraspEvalResult) -> None:
    print(f"{tag}")
    print("SINGLE_GRASP_EVAL:")
    print(f"  success_rate: {r.success_rate:.6f}")
    print(f"  mean_translation_error: {r.mean_translation_error:.6f}")
    print(f"  mean_rotation_trace: {r.mean_rotation_trace:.6f}")
    print(f"  num_images: {r.num_images}")
    print(f"  num_success: {r.num_success}")
    print(f"  breakdown_trans_only: {r.n_trans_only}")
    print(f"  breakdown_rot_only: {r.n_rot_only}")
    print(f"  breakdown_both: {r.n_both}")
    print(f"  breakdown_trans_and_rot_different_gt: {r.n_split_gt}")
    print(f"  note_collision: {r.collision_note}")
    print(f"  note_friction: {r.friction_note}")


def _print_diagnostics(r: SingleGraspEvalResult) -> None:
    if r.success_rate > 0:
        return
    print("\nDIAGNOSTICS (success_rate == 0):")
    d = r.min_dists.astype(np.float64)
    t_nn = r.traces_to_nn.astype(np.float64)
    t_max = r.max_traces.astype(np.float64)
    for name, arr in ("min_center_distance", d), ("trace_to_nn_gt", t_nn), ("max_trace_any_gt", t_max):
        if arr.size == 0:
            continue
        print(f"  {name}: min={arr.min():.6f} p10={np.percentile(arr,10):.6f} p50={np.percentile(arr,50):.6f} "
              f"p90={np.percentile(arr,90):.6f} max={arr.max():.6f}")

    bins = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, float("inf")]
    hist, _ = np.histogram(d, bins=bins)
    print("  min_distance_histogram (counts per bin):")
    for i in range(len(bins) - 1):
        print(f"    [{bins[i]:.3f}, {bins[i+1]:.3f}): {int(hist[i])}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Single-grasp GC6D success proxy (not AP).")
    ap.add_argument("--gc6d-root", type=str, required=True)
    ap.add_argument("--camera", type=str, default="realsense-d415")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument(
        "--dump-root",
        type=str,
        default=None,
        help="Official-style dump: <scene:06d>/<camera>/<img:06d>.npy (one evaluation).",
    )
    ap.add_argument(
        "--dump-baseline",
        type=str,
        default=None,
        help="Baseline Lift3D predictions (official layout). Used with --dump-finetune.",
    )
    ap.add_argument(
        "--dump-finetune",
        type=str,
        default=None,
        help="GC6D-finetuned predictions (official layout). Used with --dump-baseline.",
    )
    ap.add_argument("--fric-coef-thresh", type=float, default=0.2, help="loadGrasp(..., fric_coef_thresh=...)")
    ap.add_argument("--trans-thresh", type=float, default=0.02)
    ap.add_argument("--trace-thresh", type=float, default=2.0)
    args = ap.parse_args()

    if args.split != "test":
        raise SystemExit("Only split=test is supported for canonical image list.")

    from graspclutter6dAPI import GraspClutter6D

    gc6d_root = Path(args.gc6d_root)
    dataset = GraspClutter6D(root=str(gc6d_root), camera=args.camera, split=args.split)

    if args.dump_baseline and args.dump_finetune:
        grasp_labels_all = dataset.loadGraspLabels(objIds=dataset.objIds)
        r_base = evaluate_single_grasp_dump(
            Path(args.dump_baseline),
            gc6d_root,
            args.camera,
            args.split,
            dataset,
            args.fric_coef_thresh,
            args.trans_thresh,
            args.trace_thresh,
            grasp_labels_all=grasp_labels_all,
        )
        print("\n=== BASELINE ===\n")
        _print_summary("[BASELINE]", r_base)
        _print_diagnostics(r_base)

        r_ft = evaluate_single_grasp_dump(
            Path(args.dump_finetune),
            gc6d_root,
            args.camera,
            args.split,
            dataset,
            args.fric_coef_thresh,
            args.trans_thresh,
            args.trace_thresh,
            grasp_labels_all=grasp_labels_all,
        )
        print("\n=== FINETUNE ===\n")
        _print_summary("[FINETUNE]", r_ft)
        _print_diagnostics(r_ft)

        print("\n=== COMPARISON ===")
        print(f"BASELINE_SUCCESS = {r_base.success_rate:.6f}")
        print(f"FINETUNE_SUCCESS = {r_ft.success_rate:.6f}")
        return

    if not args.dump_root:
        raise SystemExit("Provide --dump-root, or both --dump-baseline and --dump-finetune.")

    r = evaluate_single_grasp_dump(
        Path(args.dump_root),
        gc6d_root,
        args.camera,
        args.split,
        dataset,
        args.fric_coef_thresh,
        args.trans_thresh,
        args.trace_thresh,
    )
    _print_summary("[SINGLE]", r)
    _print_diagnostics(r)


if __name__ == "__main__":
    main()
