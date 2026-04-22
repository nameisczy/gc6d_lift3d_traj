#!/usr/bin/env python3
"""
Numerical evaluation: rollout / goal head vs GT, 10D->17D via action10_to_gc6d17,
optional GC6D eval_grasp (DexNet force-closure pipeline) with fixed camera/split/TOP_K.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from gc6d_lift3d_traj.lift3d_integration.lift3d_dataset import Lift3DTrajDataset
from gc6d_lift3d_traj.lift3d_integration.lift3d_eval_adapter import evaluate_step_errors
from gc6d_lift3d_traj.lift3d_integration.trajectory_policy import TrajectoryPolicy
from gc6d_lift3d_traj.utils.action10_to_gc6d17 import DEFAULT_DEPTH, DEFAULT_HEIGHT, action10_to_gc6d17
from gc6d_lift3d_traj.utils.grasp_action10 import grasp_matrix_width_to_action10
from gc6d_lift3d_traj.utils.rotations import lift3d_rotation_to_matrix, matrix_to_lift3d_rotation

# --- GC6D evaluator (optional; heavy deps: open3d, dexnet, etc.) ---
def _try_import_gc6d_eval():
    try:
        from graspclutter6dAPI.grasp import GraspGroup
        from graspclutter6dAPI.graspclutter6d_eval import GraspClutter6DEval
        from graspclutter6dAPI.utils.config import get_config
        from graspclutter6dAPI.utils.eval_utils import eval_grasp, voxel_sample_points

        return GraspGroup, GraspClutter6DEval, get_config, eval_grasp, voxel_sample_points, None
    except Exception as e:
        return None, None, None, None, None, str(e)


def rotation_similarity(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    """trace(R_pred^T R_gt), identical rotations -> 3 for proper SO(3)."""
    Rp = np.asarray(R_pred, dtype=np.float64).reshape(3, 3)
    Rg = np.asarray(R_gt, dtype=np.float64).reshape(3, 3)
    return float(np.trace(Rp.T @ Rg))


def summarize_success(res: Tuple, top_k: int = 50, num_settings: int = 16):
    """Same as graspclutter6dAPI/tools/run_benchmark_from_offline_policy.summarize_success."""
    succ = res[2]
    rates = []
    if not isinstance(succ, (list, tuple)):
        succ = [succ]
    for s in succ:
        if isinstance(s, list) and len(s) == 0:
            rates.append(np.nan)
            continue
        s = np.asarray(s).astype(bool)
        if s.size == 0:
            rates.append(np.nan)
            continue
        rates.append(float(s[: min(top_k, s.size)].mean()))
    rates = np.asarray(rates, dtype=float)
    if rates.size < num_settings:
        rates = np.pad(rates, (0, num_settings - rates.size), constant_values=np.nan)
    elif rates.size > num_settings:
        rates = rates[:num_settings]
    mean_rate = float(np.nanmean(rates)) if np.isfinite(rates).any() else float("nan")
    return rates, mean_rate


def resolve_paths(args) -> Tuple[Path, Path, Path]:
    root = Path(args.data_root)
    if args.index:
        index_path = Path(args.index)
    else:
        index_path = root / "index" / args.index_name
    ckpt = Path(args.ckpt) if args.ckpt else root / "metadata" / "traj_policy.pt"
    return root, index_path, ckpt


def ann_id_to_img_id(ann_id: int, camera: str) -> int:
    if camera == "realsense-d415":
        return ann_id * 4 + 1
    if camera == "realsense-d435":
        return ann_id * 4 + 2
    if camera == "azure-kinect":
        return ann_id * 4 + 3
    if camera == "zivid":
        return ann_id * 4 + 4
    raise ValueError(f"Unsupported camera: {camera}")


def load_last_timestep_batches(
    index_rows: List[dict], model: torch.nn.Module, device: torch.device
) -> Tuple[List[float], List[float], List[np.ndarray], List[np.ndarray]]:
    """One forward per episode at last timestep t = T-1."""
    center_errs = []
    rot_sims = []
    pred_goals = []
    gt_goals = []
    model.eval()
    with torch.no_grad():
        for row in index_rows:
            data = np.load(row["episode_path"], allow_pickle=True)
            T = int(data["ee_positions"].shape[0])
            t = T - 1
            pc = torch.from_numpy(data["point_cloud"].astype(np.float32)).unsqueeze(0).to(device)
            ee = torch.from_numpy(data["ee_positions"][t].astype(np.float32)).unsqueeze(0).to(device)
            er = torch.from_numpy(data["ee_rotations"][t].astype(np.float32)).unsqueeze(0).to(device)
            g = torch.from_numpy(data["gripper"][t].astype(np.float32)).unsqueeze(0).to(device)
            if g.dim() == 1:
                g = g.unsqueeze(-1)
            gt_w = float(np.asarray(data["gt_grasp_width"]).reshape(-1)[0])
            R_gt = np.asarray(data["gt_grasp_rotation"], dtype=np.float32).reshape(3, 3)
            c_gt = np.asarray(data["gt_grasp_center"], dtype=np.float32).reshape(3)
            gt10 = grasp_matrix_width_to_action10(c_gt, R_gt, gt_w)
            goal = torch.from_numpy(gt10.astype(np.float32)).unsqueeze(0).to(device)
            _, pred_g = model(pc, ee, er, g, goal)
            pg = pred_g.detach().cpu().numpy()[0]

            c_p = pg[:3]
            R_p = lift3d_rotation_to_matrix(pg[3:9])
            c_g = gt10[:3]
            R_g = lift3d_rotation_to_matrix(gt10[3:9])

            center_errs.append(float(np.linalg.norm(c_p - c_g)))
            rot_sims.append(rotation_similarity(R_p, R_g))
            pred_goals.append(pg)
            gt_goals.append(gt10)
    return center_errs, rot_sims, pred_goals, gt_goals


def run_gc6d_eval_grasp(
    index_rows: List[dict],
    pred_goals: List[np.ndarray],
    gc6d_root: str,
    camera: str,
    split: str,
    top_k: int,
) -> Dict[str, Any]:
    GraspGroup, GraspClutter6DEval, get_config, eval_grasp, voxel_sample_points, imp_err = _try_import_gc6d_eval()
    if GraspGroup is None:
        return {"ok": False, "error": imp_err or "import failed"}

    ev = GraspClutter6DEval(root=gc6d_root, camera=camera, split=split)
    cfg = get_config()

    # group by (scene_id, ann_id), cap predictions per group at top_k
    groups: Dict[Tuple[int, int], List[Tuple[int, np.ndarray]]] = defaultdict(list)
    for i, row in enumerate(index_rows):
        if i >= len(pred_goals):
            break
        key = (int(row["scene_id"]), int(row["ann_id"]))
        groups[key].append((i, pred_goals[i]))

    mean_rates = []
    all_ok = True
    for (scene_id, ann_id), items in groups.items():
        items = items[: min(top_k, len(items))]
        rows17 = np.stack([action10_to_gc6d17(a[1], score=1.0) for a in items], axis=0)
        gg = GraspGroup(rows17.astype(np.float32))

        try:
            models, dexnet_models, _ = ev.get_scene_models(scene_id, ann_id)
            _obj_list, pose_list, _ = ev.get_model_poses(scene_id, ann_id)
            models_sampled = [voxel_sample_points(m, 0.008) for m in models]
            res = eval_grasp(
                gg,
                models_sampled,
                dexnet_models,
                pose_list,
                cfg,
                table=None,
                voxel_size=0.008,
                TOP_K=top_k,
            )
            rates, mean_rate = summarize_success(res, top_k=top_k)
            mean_rates.append(mean_rate)
        except Exception as e:
            all_ok = False
            mean_rates.append(float("nan"))
            return {"ok": False, "error": str(e), "scene": scene_id, "ann": ann_id}

    return {
        "ok": all_ok,
        "mean_collision_free_rate": float(np.nanmean(mean_rates)) if mean_rates else float("nan"),
        "per_group_rates": mean_rates,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="/mnt/ssd/ziyaochen/gc6d_lift3d_traj_data")
    p.add_argument("--index", type=str, default=None, help="Override index jsonl path")
    p.add_argument("--index-name", type=str, default="index_train.jsonl", help="relative to data-root/index/")
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--dump-17d-dir", type=str, default=None)
    p.add_argument("--gc6d-root", type=str, default="/mnt/ssd/ziyaochen/GraspClutter6D")
    p.add_argument("--camera", type=str, default="realsense-d415")
    p.add_argument("--split", type=str, default="test", help="Logged + passed to GraspClutter6DEval (dataset split)")
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--run-gc6d-evaluator", action="store_true", help="Run eval_grasp (slow, needs full dataset assets)")
    p.add_argument("--run-official-eval-all", action="store_true", help="Run official GraspClutter6DEval.eval_all on dump")
    p.add_argument("--full-test-inference", action="store_true", help="Run full test split inference (no index/max-samples)")
    p.add_argument("--rollout-steps", type=int, default=24, help="Rollout length for full-test inference")
    p.add_argument("--dump-official-dir", type=str, default=None, help="Per-image dump root: <scene>/<camera>/<img>.npy")
    p.add_argument("--gc6d-api-root", type=str, default="/home/ziyaochen/graspclutter6dAPI")
    p.add_argument("--assert-full-coverage", action="store_true", help="Assert num_prediction_files == num_test_images")
    p.add_argument("--verbose", action="store_true", help="Print per-episode metrics")
    p.add_argument("--json-summary", action="store_true", help="Print EVAL_SUMMARY_JSON line for pipeline_validate.py")
    args = p.parse_args()

    _, index_path, ckpt_path = resolve_paths(args)
    if (not args.full_test_inference) and (not index_path.is_file()):
        print(f"ERROR: index not found: {index_path}")
        sys.exit(1)
    if not ckpt_path.is_file():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print("=== GC6D / Lift3D eval alignment ===")
    print(f"camera: {args.camera}")
    print(f"split: {args.split}")
    print(f"TOP_K: {args.top_k}")
    print(f"height/depth constants (10D->17D): {DEFAULT_HEIGHT}, {DEFAULT_DEPTH}")
    print(f"index: {index_path} (used only when --full-test-inference is off)")
    print(f"ckpt: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model = TrajectoryPolicy(robot_state_dim=1).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()

    if args.full_test_inference:
        if not args.dump_official_dir:
            print("ERROR: --full-test-inference requires --dump-official-dir")
            sys.exit(1)
        out_root = Path(args.dump_official_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        # IMPORTANT: full-test inference must iterate test split directly (no index/max-samples).
        # Do NOT call loadGrasp here (expensive and unnecessary); only load per-image point cloud.
        if args.gc6d_api_root not in sys.path:
            sys.path.insert(0, args.gc6d_api_root)
        from graspclutter6dAPI.graspclutter6d import GraspClutter6D

        api = GraspClutter6D(root=args.gc6d_root, camera=args.camera, split=args.split)
        split_file = Path(args.gc6d_root) / "split_info" / "grasp_test_scene_ids.json"
        scene_ids = [int(x) for x in json.loads(split_file.read_text(encoding="utf-8"))]
        n_files = 0
        with torch.no_grad():
            for scene_id in scene_ids:
                for ann_id in range(13):
                    pc_raw = api.loadScenePointCloud(scene_id, args.camera, ann_id, align=False, format="numpy")
                    if isinstance(pc_raw, tuple):
                        pc_np = np.asarray(pc_raw[0], dtype=np.float32)
                    else:
                        pc_np = np.asarray(pc_raw, dtype=np.float32)
                    if pc_np.shape[0] > 1024:
                        idx = np.random.choice(pc_np.shape[0], 1024, replace=False)
                        pc_np = pc_np[idx]
                    elif pc_np.shape[0] < 1024:
                        idx = np.random.choice(pc_np.shape[0], 1024, replace=True)
                        pc_np = pc_np[idx]
                    pc = torch.from_numpy(pc_np).unsqueeze(0).to(device)

                    # rollout state0 -> stateT
                    ee_pos = torch.from_numpy(pc_np.mean(axis=0).astype(np.float32)).unsqueeze(0).to(device)
                    ee_rot = torch.from_numpy(np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)).unsqueeze(0).to(device)
                    grip = torch.ones((1, 1), dtype=torch.float32, device=device)
                    goal = torch.zeros((1, 10), dtype=torch.float32, device=device)
                    for _ in range(args.rollout_steps):
                        pred_d, pred_g = model(pc, ee_pos, ee_rot, grip, goal)
                        d = pred_d[0]
                        ee_pos = ee_pos + d[:3].unsqueeze(0)
                        R_cur = lift3d_rotation_to_matrix(ee_rot[0].detach().cpu().numpy())
                        R_del = lift3d_rotation_to_matrix(d[3:9].detach().cpu().numpy())
                        R_nxt = R_del @ R_cur
                        ee_rot = torch.from_numpy(matrix_to_lift3d_rotation(R_nxt)).unsqueeze(0).to(device)
                        grip = grip + d[9:10].reshape(1, 1)
                        goal = pred_g

                    final_goal = goal[0].detach().cpu().numpy().astype(np.float32)
                    final_goal[:3] = ee_pos[0].detach().cpu().numpy().astype(np.float32)
                    final_goal[3:9] = ee_rot[0].detach().cpu().numpy().astype(np.float32)
                    final_goal[9] = float(np.clip(abs(final_goal[9]), 0.0, 0.14))
                    row17 = action10_to_gc6d17(final_goal, score=1.0).reshape(1, 17).astype(np.float32)

                    img_id = ann_id_to_img_id(int(ann_id), args.camera)
                    scene_dir = out_root / f"{int(scene_id):06d}" / args.camera
                    scene_dir.mkdir(parents=True, exist_ok=True)
                    np.save(scene_dir / f"{img_id:06d}.npy", row17)
                    n_files += 1
                    if n_files % 200 == 0:
                        print(f"inference progress: {n_files} images", flush=True)

        n_test_scenes = len(scene_ids)
        n_test_images = n_test_scenes * 13
        print(f"total test images: {n_test_images}")
        print(f"number of prediction files: {n_files}")
        if args.assert_full_coverage:
            assert n_files == n_test_images, f"coverage mismatch: {n_files} != {n_test_images}"

        gc6d_result = {"ok": False, "skipped": True}
        if args.run_official_eval_all:
            try:
                from graspclutter6dAPI.graspclutter6d_eval import GraspClutter6DEval
            except Exception as e:
                print(f"official eval import failed: {e}")
                sys.exit(1)
            ge = GraspClutter6DEval(root=args.gc6d_root, camera=args.camera, split=args.split)
            _res, ap = ge.eval_all(str(out_root), proc=4)
            print(f"AP = {float(ap[0]):.6f}")
            print(f"AP0.4 = {float(ap[1]):.6f}")
            print(f"AP0.8 = {float(ap[2]):.6f}")
            gc6d_result = {
                "ok": True,
                "ap": float(ap[0]),
                "ap04": float(ap[1]),
                "ap08": float(ap[2]),
                "n_pred_files": n_files,
                "n_test_images": n_test_images,
            }
        summary = {
            "camera": args.camera,
            "split": args.split,
            "top_k": args.top_k,
            "n_pred_files": n_files,
            "n_test_images": n_test_images,
            "full_coverage": bool(n_files == n_test_images),
            "gc6d_eval": gc6d_result,
        }
        if args.json_summary:
            print("EVAL_SUMMARY_JSON:", json.dumps(summary, default=str))
        return summary

    index_rows = []
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                index_rows.append(json.loads(line))

    # --- Last-timestep goal vs GT (primary geometry check) ---
    center_errs, rot_sims, pred_goals, gt_goals = load_last_timestep_batches(index_rows, model, device)
    if args.verbose:
        for i, row in enumerate(index_rows):
            if i < len(center_errs):
                print(
                    f"  episode {i} scene={row.get('scene_id')} ann={row.get('ann_id')} "
                    f"||c_err||={center_errs[i]:.6f} trace(Rp^T Rg)={rot_sims[i]:.6f}"
                )

    print("\n=== Aggregate (last EE state, goal head vs GT grasp) ===")
    ce = np.asarray(center_errs, dtype=np.float64)
    rs = np.asarray(rot_sims, dtype=np.float64)
    print(f"mean center error: {float(np.mean(ce)):.6f}")
    print(f"std center error:  {float(np.std(ce)):.6f}")
    print(f"mean rotation similarity trace(Rp^T Rg): {float(np.mean(rs)):.6f}")
    print(f"min rotation similarity: {float(np.min(rs)):.6f}")
    print(f"max rotation similarity: {float(np.max(rs)):.6f}")

    warn = False
    if float(np.mean(ce)) > 0.1:
        warn = True
    if float(np.mean(rs)) < 2.0:
        warn = True
    if warn:
        print(
            "\nWARNING: rollout -> grasp conversion may be incorrect "
            "(mean center error > 0.1m OR mean trace(Rp^T Rg) < 2.0)"
        )

    # --- dataloader step errors (imitation) ---
    ds = Lift3DTrajDataset(str(index_path))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    step_errs = []
    with torch.no_grad():
        for state, target_delta, goal10 in dl:
            pc = state["point_cloud"].to(device)
            ee_pos = state["ee_position"].to(device)
            ee_rot = state["ee_rotation"].to(device)
            grip = state["gripper"].to(device)
            if grip.dim() == 1:
                grip = grip.unsqueeze(-1)
            pred_d, pred_g = model(pc, ee_pos, ee_rot, grip, goal10.to(device))
            pd = pred_d.cpu().numpy()
            tg = target_delta.cpu().numpy()
            step_errs.append(
                evaluate_step_errors(
                    {
                        "delta_translation": pd[:, :3],
                        "delta_rotation": pd[:, 3:9],
                        "delta_gripper": pd[:, 9:10],
                    },
                    {
                        "delta_translation": tg[:, :3],
                        "delta_rotation": tg[:, 3:9],
                        "delta_gripper": tg[:, 9:10],
                    },
                )
            )

    def mean_key(dlist, key):
        vals = [d[key] for d in dlist if key in d]
        return float(np.mean(vals)) if vals else float("nan")

    print("\n=== Step-wise (delta imitation) ===")
    if step_errs:
        for k in step_errs[0].keys():
            print(f"  {k}: {mean_key(step_errs, k):.6f}")

    # --- Dump 17D (verified conversion) ---
    if args.dump_17d_dir:
        out_dir = Path(args.dump_17d_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        rows17 = np.stack([action10_to_gc6d17(g) for g in pred_goals], axis=0).astype(np.float32)
        np.save(out_dir / "pred_grasps_concat.npy", rows17)
        print(f"\nWrote 17D rows (N,17) from goal 10D: {out_dir / 'pred_grasps_concat.npy'} (N={len(pred_goals)})")

    gc6d_result = {"ok": False, "skipped": True}
    if args.run_gc6d_evaluator:
        print("\n=== GC6D eval_grasp (DexNet / collision) ===")
        gc6d_result = run_gc6d_eval_grasp(
            index_rows,
            pred_goals,
            args.gc6d_root,
            args.camera,
            args.split,
            args.top_k,
        )
        if gc6d_result.get("ok"):
            print(
                f"eval_grasp completed. mean collision-mask success proxy (see benchmark tools): "
                f"{gc6d_result.get('mean_collision_free_rate', float('nan')):.6f}"
            )
        else:
            print(f"eval_grasp failed: {gc6d_result.get('error', 'unknown')}")

    # Summary for pipeline_validate.py
    summary = {
        "center_err_mean": float(np.mean(ce)),
        "rot_trace_mean": float(np.mean(rs)),
        "center_err_std": float(np.std(ce)),
        "rot_trace_min": float(np.min(rs)),
        "rot_trace_max": float(np.max(rs)),
        "warn_geometry": warn,
        "gc6d_eval": gc6d_result,
        "camera": args.camera,
        "split": args.split,
        "top_k": args.top_k,
    }
    if args.json_summary:
        print("EVAL_SUMMARY_JSON:", json.dumps(summary, default=str))
    return summary


if __name__ == "__main__":
    main()
