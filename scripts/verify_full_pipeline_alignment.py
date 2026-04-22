#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from gc6d_lift3d_traj.lift3d_integration.lift3d_dataset import (
    FIXED_PC_POINTS,
    Lift3DTrajDataset,
    Lift3DTrajDatasetLift3DStyle,
)
from gc6d_lift3d_traj.lift3d_integration.trajectory_policy import TrajectoryPolicy
from gc6d_lift3d_traj.utils.grasp_action10 import grasp_matrix_width_to_action10
from gc6d_lift3d_traj.utils.rotations import action_rotation_from_two_poses, lift3d_rotation_to_matrix


def _ok(cond: bool, msg: str) -> Tuple[bool, str]:
    return bool(cond), msg


def _load_index(index_path: Path) -> List[dict]:
    rows: List[dict] = []
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _check_dataset_conversion(data_root: Path, index_name: str) -> Tuple[bool, List[str]]:
    msgs: List[str] = []
    ok = True
    cfg_path = Path("/home/ziyaochen/gc6d_lift3d_traj/configs/default.yaml")
    import yaml

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    z_off = float(cfg["trajectory"]["start_height_offset"])
    g_open = float(cfg["trajectory"]["gripper_open"])

    index_path = data_root / "index" / index_name
    rows = _load_index(index_path)
    if not rows:
        return False, [f"index empty: {index_path}"]
    ep = np.load(rows[0]["episode_path"], allow_pickle=True)

    c = np.asarray(ep["gt_grasp_center"], dtype=np.float32).reshape(3)
    Rg = np.asarray(ep["gt_grasp_rotation"], dtype=np.float32).reshape(3, 3)
    p0 = np.asarray(ep["ee_positions"][0], dtype=np.float32).reshape(3)
    R0 = lift3d_rotation_to_matrix(np.asarray(ep["ee_rotations"][0], dtype=np.float32))
    g0 = float(np.asarray(ep["gripper"][0]).reshape(-1)[0])

    c1, m1 = _ok(np.allclose(p0, c + np.array([0.0, 0.0, z_off], np.float32), atol=1e-4), "init position = center + [0,0,start_height_offset]")
    c2, m2 = _ok(np.allclose(R0, Rg, atol=1e-4), "init rotation = GT grasp rotation")
    c3, m3 = _ok(abs(g0 - g_open) < 1e-6, "init gripper = open")
    for cnd, m in ((c1, m1), (c2, m2), (c3, m3)):
        ok &= cnd
        msgs.append(("PASS: " if cnd else "FAIL: ") + m)

    ds = Lift3DTrajDataset(str(index_path))
    state, action10, goal10 = ds[0]
    gt10 = grasp_matrix_width_to_action10(c, Rg, float(np.asarray(ep["gt_grasp_width"]).reshape(-1)[0]))
    c4, m4 = _ok(np.allclose(goal10.numpy(), gt10, atol=1e-5), "goal10 equals GT grasp [t(3),R6(6),width(1)]")
    ok &= c4
    msgs.append(("PASS: " if c4 else "FAIL: ") + m4)

    # phase + final grasp pose check
    p_final = np.asarray(ep["ee_positions"][-1], dtype=np.float32).reshape(3)
    c5, m5 = _ok(np.allclose(p_final, c, atol=1e-4), "final state EXACTLY equals GT grasp center")
    ok &= c5
    msgs.append(("PASS: " if c5 else "FAIL: ") + m5 + f" (actual final={p_final}, gt={c})")

    # delta rotation formula check
    ee_rot6 = np.asarray(ep["ee_rotations"], dtype=np.float32)
    act_rot = np.asarray(ep["actions_rotation"], dtype=np.float32)
    calc = []
    for t in range(ee_rot6.shape[0] - 1):
        Rc = lift3d_rotation_to_matrix(ee_rot6[t])
        Rn = lift3d_rotation_to_matrix(ee_rot6[t + 1])
        calc.append(action_rotation_from_two_poses(Rc, Rn))
    calc = np.asarray(calc, dtype=np.float32)
    c6, m6 = _ok(np.allclose(calc, act_rot, atol=1e-5), "delta_rotation uses R_next @ R_current^T (not subtraction)")
    ok &= c6
    msgs.append(("PASS: " if c6 else "FAIL: ") + m6)

    pc = state["point_cloud"].numpy()
    c7, m7 = _ok(pc.shape == (FIXED_PC_POINTS, 3), f"point_cloud fixed size = ({FIXED_PC_POINTS},3)")
    ok &= c7
    msgs.append(("PASS: " if c7 else "FAIL: ") + m7)
    return ok, msgs


def _check_lift3d_format(data_root: Path, index_name: str, gc6d_root: Path) -> Tuple[bool, List[str]]:
    msgs: List[str] = []
    ok = True
    index_path = data_root / "index" / index_name
    ds_style = Lift3DTrajDatasetLift3DStyle(str(index_path), str(gc6d_root))
    sample = ds_style[0]
    if isinstance(sample, dict):
        images = sample.get("images")
        point_clouds = sample.get("point_clouds")
        robot_states = sample.get("robot_states")
        raw_states = sample.get("raw_states")
        action = sample.get("action")
        texts = sample.get("texts")
        goal = sample.get("goal")
    else:
        images, point_clouds, robot_states, raw_states, action, texts = sample
        goal = None

    print("=== One-sample debug (Lift3D format) ===")
    print("images:", "None" if images is None else tuple(images.shape))
    print("point_clouds shape:", tuple(point_clouds.shape))
    print("robot_states shape:", tuple(robot_states.shape))
    print("raw_states shape:", tuple(raw_states.shape))
    print("action shape:", tuple(action.shape))
    print("texts:", repr(texts))
    print("goal shape:", tuple(goal.shape) if goal is not None else "MISSING")

    img_ok = images is not None and hasattr(images, "shape") and tuple(images.shape)[:1] == (3,)
    checks = [
        (img_ok, "images is (3,H,W) float tensor loaded from GC6D RGB"),
        (tuple(point_clouds.shape) == (FIXED_PC_POINTS, 3), f"point_clouds shape == ({FIXED_PC_POINTS},3)"),
        (tuple(robot_states.shape) == (10,), "robot_states shape == (10,)"),
        (tuple(raw_states.shape) == (10,), "raw_states shape == (10,)"),
        (tuple(action.shape) == (10,), "action shape == (10,)"),
        ((texts == "") or (texts is None), "texts is '' or None"),
        (goal is not None and tuple(goal.shape) == (10,), "goal exists in sample (10,) (required extension)"),
    ]
    for cond, msg in checks:
        ok &= bool(cond)
        msgs.append(("PASS: " if cond else "FAIL: ") + msg)
    return ok, msgs


def _check_training(data_root: Path, index_name: str) -> Tuple[bool, List[str]]:
    msgs: List[str] = []
    ok = True
    index_path = data_root / "index" / index_name
    ds = Lift3DTrajDataset(str(index_path))
    s, a, g = ds[0]
    c1 = isinstance(s, dict) and {"point_cloud", "ee_position", "ee_rotation", "gripper"} <= set(s.keys())
    c2 = tuple(a.shape) == (10,)
    c3 = tuple(g.shape) == (10,)
    for cond, msg in (
        (c1, "dataset returns state dict"),
        (c2, "dataset returns action(10,)"),
        (c3, "dataset returns goal(10,)"),
    ):
        ok &= cond
        msgs.append(("PASS: " if cond else "FAIL: ") + msg)

    # strict requirement: model.forward(state, goal) usage
    import inspect

    sig = inspect.signature(TrajectoryPolicy.forward)
    param_names = list(sig.parameters.keys())
    uses_goal_input = "goal" in param_names
    ok &= uses_goal_input
    msgs.append(("PASS: " if uses_goal_input else "FAIL: ") + "model.forward includes goal argument")

    ckpt = data_root / "metadata" / "traj_policy.pt"
    try:
        try:
            _ = torch.load(ckpt, map_location="cpu", weights_only=False)
        except TypeError:
            _ = torch.load(ckpt, map_location="cpu")
        c4 = True
    except Exception:
        c4 = False
    ok &= c4
    msgs.append(("PASS: " if c4 else "FAIL: ") + f"checkpoint reloadable: {ckpt}")
    return ok, msgs


def _count_test_images(gc6d_root: Path) -> int:
    split_file = gc6d_root / "split_info" / "grasp_test_scene_ids.json"
    scene_ids = [int(x) for x in json.loads(split_file.read_text(encoding="utf-8"))]
    # GC6D eval_scene iterates annId in range(13) for each test scene.
    return len(scene_ids) * 13


def _collect_pred_files(dump_root: Path, camera: str) -> List[Path]:
    return sorted(dump_root.glob(f"*/{camera}/*.npy"))


def _check_inference_and_dump_and_coverage(
    gc6d_root: Path,
    dump_root: Path,
    camera: str,
    top_k: int,
) -> Tuple[bool, bool, bool, List[str]]:
    msgs: List[str] = []
    inf_ok = True
    dump_ok = True
    cov_ok = True

    pred_files = _collect_pred_files(dump_root, camera)
    total_test_images = _count_test_images(gc6d_root)
    num_pred = len(pred_files)
    msgs.append(f"total test images: {total_test_images}")
    msgs.append(f"number of prediction files: {num_pred}")

    # Inference check: full test required, no subset
    inf_ok &= num_pred == total_test_images
    msgs.append(("PASS: " if inf_ok else "FAIL: ") + "full test inference coverage matches test split")

    # Dump format check
    for p in pred_files[: min(20, len(pred_files))]:
        arr = np.asarray(np.load(p, allow_pickle=False))
        if arr.ndim != 2 or arr.shape[1] != 17:
            dump_ok = False
            msgs.append(f"FAIL: bad dump shape at {p}: {arr.shape}")
            break
        if arr.shape[0] > top_k:
            dump_ok = False
            msgs.append(f"FAIL: K>{top_k} at {p}: {arr.shape[0]}")
            break
        if arr.shape[0] >= 2 and np.any(arr[:-1, 0] < arr[1:, 0]):
            dump_ok = False
            msgs.append(f"FAIL: not sorted by score desc at {p}")
            break
    if dump_ok:
        msgs.append(f"PASS: each checked dump file is (K,17), K<=TOP_K({top_k}), score-sorted")

    # Coverage check strict assert
    cov_ok &= num_pred == total_test_images
    msgs.append(("PASS: " if cov_ok else "FAIL: ") + "num_prediction_files == num_test_images")
    return inf_ok, dump_ok, cov_ok, msgs


def _check_eval(gc6d_root: Path, dump_root: Path, camera: str) -> Tuple[bool, List[str], Tuple[float, float, float]]:
    msgs: List[str] = []
    ok = True
    from graspclutter6dAPI.graspclutter6d_eval import GraspClutter6DEval

    ge = GraspClutter6DEval(root=str(gc6d_root), camera=camera, split="test")
    try:
        _res, ap = ge.eval_all(str(dump_root), proc=4)
        ap_all, ap04, ap08 = float(ap[0]), float(ap[1]), float(ap[2])
        msgs.append(f"AP = {ap_all:.6f}")
        msgs.append(f"AP0.4 = {ap04:.6f}")
        msgs.append(f"AP0.8 = {ap08:.6f}")
        nz = (ap_all > 0.0) or (ap04 > 0.0) or (ap08 > 0.0)
        ok &= nz
        msgs.append(("PASS: " if nz else "FAIL: ") + "official AP values non-zero")
        return ok, msgs, (ap_all, ap04, ap08)
    except Exception as e:
        ok = False
        msgs.append(f"FAIL: official eval_all crashed: {e}")
        return ok, msgs, (0.0, 0.0, 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="/mnt/ssd/ziyaochen/gc6d_lift3d_traj_data")
    ap.add_argument("--index-name", type=str, default="index_train.jsonl")
    ap.add_argument("--gc6d-root", type=str, default="/mnt/ssd/ziyaochen/GraspClutter6D")
    ap.add_argument(
        "--dump-root",
        type=str,
        default="/mnt/ssd/ziyaochen/gc6d_lift3d_traj_data/pred_17d/official_dump",
        help="Official dump folder for eval_all: <scene>/<camera>/<img>.npy",
    )
    ap.add_argument("--camera", type=str, default="realsense-d415")
    ap.add_argument("--top-k", type=int, default=50)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    gc6d_root = Path(args.gc6d_root)
    dump_root = Path(args.dump_root)

    dataset_ok, dataset_msgs = _check_dataset_conversion(data_root, args.index_name)
    fmt_ok, fmt_msgs = _check_lift3d_format(data_root, args.index_name, gc6d_root)
    train_ok, train_msgs = _check_training(data_root, args.index_name)
    inf_ok, dump_ok, cov_ok, idc_msgs = _check_inference_and_dump_and_coverage(
        gc6d_root=gc6d_root,
        dump_root=dump_root,
        camera=args.camera,
        top_k=args.top_k,
    )
    eval_ok, eval_msgs, _aps = _check_eval(gc6d_root, dump_root, args.camera)

    print("\n=== DETAILS ===")
    for section, msgs in (
        ("DATASET", dataset_msgs),
        ("LIFT3D_FORMAT", fmt_msgs),
        ("TRAINING", train_msgs),
        ("INFERENCE/DUMP/COVERAGE", idc_msgs),
        ("EVAL", eval_msgs),
    ):
        print(f"\n[{section}]")
        for m in msgs:
            print("-", m)

    print("\n=== FINAL STATUS ===")
    print(f"DATASET_CHECK: {'PASS' if dataset_ok else 'FAIL'}")
    print(f"LIFT3D_FORMAT_CHECK: {'PASS' if fmt_ok else 'FAIL'}")
    print(f"TRAINING_CHECK: {'PASS' if train_ok else 'FAIL'}")
    print(f"INFERENCE_CHECK: {'PASS' if inf_ok else 'FAIL'}")
    print(f"DUMP_FORMAT_CHECK: {'PASS' if dump_ok else 'FAIL'}")
    print(f"COVERAGE_CHECK: {'PASS' if cov_ok else 'FAIL'}")
    print(f"EVAL_CHECK: {'PASS' if eval_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
