#!/usr/bin/env python3
"""
Full official GC6D AP benchmark evaluation from predicted 17D grasps.

This script converts flat predicted rows (N,17) into the official dump layout:
  <dump_dir>/<scene_id:06d>/<camera>/<img_id:06d>.npy
and then runs:
  GraspClutter6DEval(...).eval_all(dump_dir, proc=...)
to report AP / AP0.4 / AP0.8.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


CAMERA_OFFSET = {
    "realsense-d415": 1,
    "realsense-d435": 2,
    "azure-kinect": 3,
    "zivid": 4,
}


def ann_id_to_img_id(ann_id: int, camera: str) -> int:
    if camera not in CAMERA_OFFSET:
        raise ValueError(f"Unsupported camera: {camera}")
    return int(ann_id) * 4 + CAMERA_OFFSET[camera]


def load_index(index_path: Path) -> List[dict]:
    rows: List[dict] = []
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_official_dump(
    pred_rows17: np.ndarray,
    index_rows: List[dict],
    out_dump_dir: Path,
    camera: str,
    top_k: int,
) -> Tuple[int, int]:
    """
    Write per-image npy files in official format expected by GC6D evaluator.
    Returns:
      (num_images_written, num_groups_total)
    """
    if pred_rows17.ndim != 2 or pred_rows17.shape[1] != 17:
        raise ValueError(f"Expected predicted rows shape (N,17), got {pred_rows17.shape}")
    if len(index_rows) != int(pred_rows17.shape[0]):
        raise ValueError(
            f"index rows ({len(index_rows)}) != predicted rows ({pred_rows17.shape[0]})"
        )

    groups: Dict[Tuple[int, int], List[np.ndarray]] = defaultdict(list)
    for i, row in enumerate(index_rows):
        scene_id = int(row["scene_id"])
        ann_id = int(row["ann_id"])
        groups[(scene_id, ann_id)].append(pred_rows17[i])

    num_written = 0
    for (scene_id, ann_id), grasp_rows in groups.items():
        arr = np.asarray(grasp_rows, dtype=np.float32)
        # keep highest-score grasps first
        if arr.shape[0] > 1:
            idx = np.argsort(-arr[:, 0])
            arr = arr[idx]
        arr = arr[: min(top_k, arr.shape[0])]

        img_id = ann_id_to_img_id(ann_id, camera)
        out_scene = out_dump_dir / f"{scene_id:06d}" / camera
        out_scene.mkdir(parents=True, exist_ok=True)
        np.save(out_scene / f"{img_id:06d}.npy", arr)
        num_written += 1

    return num_written, len(groups)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True, help="GC6D-lift3d data root")
    ap.add_argument("--index", type=str, default=None, help="Override index jsonl path")
    ap.add_argument("--index-name", type=str, default="index_train.jsonl")
    ap.add_argument("--dump-17d-dir", type=str, required=True, help="Directory containing pred_grasps_concat.npy")
    ap.add_argument(
        "--pred-file",
        type=str,
        default="pred_grasps_concat.npy",
        help="Pred file name inside dump-17d-dir",
    )
    ap.add_argument("--gc6d-root", type=str, default="/mnt/ssd/ziyaochen/GraspClutter6D")
    ap.add_argument("--camera", type=str, default="realsense-d415")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--proc", type=int, default=4)
    ap.add_argument(
        "--require-nonzero",
        action="store_true",
        help="Exit with code 2 if AP/AP0.4/AP0.8 are all zero",
    )
    ap.add_argument(
        "--official-dump-dir",
        type=str,
        default=None,
        help="Output folder for official per-scene dumps (default: <dump-17d-dir>/official_dump)",
    )
    ap.add_argument(
        "--clean-official-dump",
        action="store_true",
        help="Delete official-dump-dir before writing",
    )
    args = ap.parse_args()

    if args.top_k != 50:
        print("WARNING: official GC6D benchmark default is TOP_K=50. You set:", args.top_k)

    data_root = Path(args.data_root)
    index_path = Path(args.index) if args.index else data_root / "index" / args.index_name
    dump_17d_dir = Path(args.dump_17d_dir)
    pred_path = dump_17d_dir / args.pred_file
    official_dump_dir = (
        Path(args.official_dump_dir)
        if args.official_dump_dir
        else dump_17d_dir / "official_dump"
    )

    if not index_path.is_file():
        print(f"ERROR: index not found: {index_path}")
        sys.exit(1)
    if not pred_path.is_file():
        print(f"ERROR: prediction file not found: {pred_path}")
        sys.exit(1)

    if args.clean_official_dump and official_dump_dir.exists():
        shutil.rmtree(official_dump_dir)
    official_dump_dir.mkdir(parents=True, exist_ok=True)

    print("=== GC6D AP evaluation config ===")
    print(f"gc6d_root: {args.gc6d_root}")
    print(f"camera: {args.camera}")
    print(f"split: {args.split}")
    print(f"TOP_K: {args.top_k}")
    print(f"index: {index_path}")
    print(f"pred_17d: {pred_path}")
    print(f"official_dump_dir: {official_dump_dir}")

    index_rows = load_index(index_path)
    pred_rows17 = np.asarray(np.load(pred_path, allow_pickle=False), dtype=np.float32)
    num_written, num_groups = build_official_dump(
        pred_rows17=pred_rows17,
        index_rows=index_rows,
        out_dump_dir=official_dump_dir,
        camera=args.camera,
        top_k=args.top_k,
    )
    print(f"wrote official dump files: {num_written} image files from {num_groups} (scene,ann) groups")

    # Official evaluator pipeline
    from graspclutter6dAPI.graspclutter6d_eval import GraspClutter6DEval

    ge = GraspClutter6DEval(root=args.gc6d_root, camera=args.camera, split=args.split)
    res, ap = ge.eval_all(str(official_dump_dir), proc=args.proc)

    ap_all = float(ap[0])
    ap04 = float(ap[1])
    ap08 = float(ap[2])
    print("\n=== Official GC6D AP ===")
    print(f"AP = {ap_all:.6f}")
    print(f"AP0.4 = {ap04:.6f}")
    print(f"AP0.8 = {ap08:.6f}")

    # Basic sanity: no crash + not all-zero
    if (ap_all <= 0.0) and (ap04 <= 0.0) and (ap08 <= 0.0):
        print("WARNING: AP/AP0.4/AP0.8 are all zero.")
        if args.require_nonzero:
            sys.exit(2)
    else:
        print("AP sanity check: non-zero values observed.")

    _ = res  # keep reference for debugging if needed


if __name__ == "__main__":
    main()
