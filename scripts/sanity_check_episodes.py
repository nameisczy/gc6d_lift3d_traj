#!/usr/bin/env python3
"""Sanity checks on generated episode .npz files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gc6d_lift3d_traj.utils.rotations import lift3d_rotation_to_matrix


def check_episode(path: Path) -> list:
    issues = []
    data = np.load(path, allow_pickle=True)
    T = int(data["ee_positions"].shape[0])
    if data["actions_translation"].shape[0] != T - 1:
        issues.append("actions_translation length != T-1")
    if np.any(~np.isfinite(data["point_cloud"])):
        issues.append("non-finite point_cloud")
    for t in range(T):
        R = lift3d_rotation_to_matrix(data["ee_rotations"][t])
        if abs(np.linalg.det(R) - 1.0) > 0.05:
            issues.append(f"bad det(R) at t={t}")
    # lift segment: last phase should increase z
    phase = [8, 8, 2, 6]
    i0 = sum(phase[:-1])
    if T == sum(phase):
        if float(data["ee_positions"][-1, 2]) <= float(data["ee_positions"][i0, 2]) + 1e-5:
            issues.append("lift segment does not go +z")
    return issues


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=str, default=None, help="Path to index jsonl")
    p.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Dataset root; uses <data-root>/index/index_train.jsonl index_name",
    )
    p.add_argument("--index-name", type=str, default="index_train.jsonl")
    args = p.parse_args()
    if args.index:
        index_path = Path(args.index)
    elif args.data_root:
        index_path = Path(args.data_root) / "index" / args.index_name
    else:
        raise SystemExit("Provide --index or --data-root.")
    n_ok, n_bad = 0, 0
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            issues = check_episode(Path(rec["episode_path"]))
            if issues:
                print(rec["episode_path"], issues)
                n_bad += 1
            else:
                n_ok += 1
    print(f"ok={n_ok} bad={n_bad}")


if __name__ == "__main__":
    main()
