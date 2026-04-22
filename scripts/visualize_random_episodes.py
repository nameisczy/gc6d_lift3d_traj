#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gc6d_lift3d_traj.utils.viz import plot_episode_3d, random_indices


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True, type=str)
    p.add_argument("--num", type=int, default=10)
    p.add_argument("--out-dir", type=str, default="/mnt/ssd/ziyaochen/gc6d_lift3d_traj_data/visualizations")
    args = p.parse_args()

    rows = []
    with Path(args.index).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, idx in enumerate(random_indices(len(rows), args.num, seed=42)):
        ep = np.load(rows[idx]["episode_path"], allow_pickle=True)
        plot_episode_3d(ep["point_cloud"], ep["ee_positions"], out_dir / f"episode_{i:02d}.png")
    print(f"Saved {min(args.num, len(rows))} figures to {out_dir}")


if __name__ == "__main__":
    main()

