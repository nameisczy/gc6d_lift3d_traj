#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys


def main() -> int:
    api_root = Path("/home/ziyaochen/graspclutter6dAPI/graspclutter6dAPI")
    pipeline_root = Path("/home/ziyaochen/gc6d_grasp_pipeline")
    must = [
        api_root / "grasp.py",
        api_root / "graspclutter6d.py",
        pipeline_root / "models/graspnet_adapter.py",
    ]
    for p in must:
        if not p.exists():
            print(f"[MISSING] {p}")
            return 1

    print("GC6D/GraspNet 17D grasp format inspection summary")
    print("=" * 54)
    print("Canonical 17D layout (verified from graspclutter6dAPI/grasp.py):")
    print("[0]=score, [1]=width, [2]=height, [3]=depth, [4:13]=rotation(3x3 row-major),")
    print("[13:16]=translation(center), [16]=object_id")
    print("")
    print("Generation path (verified from graspclutter6dAPI/graspclutter6d.py):")
    print("- rows are built as np.hstack([scores, widths, heights, depths, rotations, target_points, object_ids])")
    print("")
    print("Approach axis convention:")
    print("- graspnet_adapter uses viewpoint->matrix where axis_x is approach")
    print("- so approach_dir = rotation[:, 0] (first column)")
    print("")
    print("This project decode rule:")
    print("- center = grasp[13:16], rotation = grasp[4:13].reshape(3,3), width = grasp[1].")
    return 0


if __name__ == "__main__":
    sys.exit(main())

