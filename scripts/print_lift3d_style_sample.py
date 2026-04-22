#!/usr/bin/env python3
"""Load one Lift3D-style GC6D sample and print tensor shapes (dataset conversion sanity check)."""
from __future__ import annotations

import argparse
from pathlib import Path

from gc6d_lift3d_traj.lift3d_integration.lift3d_dataset import Lift3DTrajDatasetLift3DStyle
from gc6d_lift3d_traj.utils.io import read_yaml


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None, help="YAML with paths.gc6d_root and dataset index path")
    p.add_argument("--index", type=str, default=None, help="index_train.jsonl (overrides config)")
    p.add_argument("--gc6d-root", type=str, default=None, help="GraspClutter6D root (overrides config)")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--sample-index", type=int, default=0)
    args = p.parse_args()

    gc6d_root = args.gc6d_root
    index_path = args.index
    if args.config:
        cfg = read_yaml(Path(args.config))
        paths = cfg["paths"]
        if gc6d_root is None:
            gc6d_root = paths["gc6d_root"]
        if index_path is None:
            index_path = str(Path(paths["output_root"]) / "index" / cfg["dataset"].get("index_filename", "index_train.jsonl"))
    if not gc6d_root or not index_path:
        raise SystemExit("Provide --config or both --index and --gc6d-root")

    ds = Lift3DTrajDatasetLift3DStyle(
        index_path,
        gc6d_root,
        default_camera="realsense-d415",
        image_size=args.image_size,
    )
    sample = ds[args.sample_index]

    print("Lift3D-style sample fields (one timestep):")
    for k, v in sample.items():
        if hasattr(v, "shape"):
            print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")
        else:
            print(f"  {k}: {type(v).__name__} = {repr(v)}")

    # expected shapes
    exp = {
        "images": (3, args.image_size, args.image_size),
        "point_clouds": (1024, 3),
        "robot_states": (10,),
        "raw_states": (10,),
        "action": (10,),
        "goal": (10,),
    }
    texts = sample["texts"]
    assert texts == "", f"texts must be empty string, got {texts!r}"
    ok = True
    for key, shape in exp.items():
        got = tuple(sample[key].shape)
        if got != shape:
            print(f"  mismatch {key}: expected {shape}, got {got}")
            ok = False
    if ok:
        print("All shapes match expected layout.")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
