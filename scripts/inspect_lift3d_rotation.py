#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import sys


def main() -> int:
    lift3d_root = Path("/home/ziyaochen/LIFT3D")
    files = [
        lift3d_root / "lift3d/dataset/gc6d_offline_npz.py",
        lift3d_root / "lift3d/models/grasp_head.py",
        lift3d_root / "lift3d/config/benchmark/gc6d_offline.yaml",
        lift3d_root / "lift3d/helpers/graphics.py",
    ]
    for p in files:
        if not p.exists():
            print(f"[MISSING] {p}")
            return 1

    print("Lift3D rotation inspection summary")
    print("=" * 48)
    print("1) GC6D offline action format uses 10D action:")
    print("   [translation(3), rotation_6d(6), width(1)]")
    print("   Evidence: lift3d/dataset/gc6d_offline_npz.py, lift3d/models/grasp_head.py")
    print("2) rotation_6d is represented as first two columns of rotation matrix.")
    print("3) Existing GC6D offline training loss is plain MSE over action vector.")
    print("4) For RLBench branch, Lift3D also has quaternion/euler utilities.")
    print("   Evidence: lift3d/helpers/graphics.py")
    print("")
    print("Conclusion for this project:")
    print("- Use Lift3D-compatible 6D rotation representation everywhere.")
    print("- State rotation: absolute 6D from EE pose.")
    print("- Action rotation: relative rotation matrix converted to 6D.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

