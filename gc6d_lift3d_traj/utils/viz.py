from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def plot_episode_3d(
    point_cloud: np.ndarray,
    positions: np.ndarray,
    save_path: Path,
    title: str = "GC6D Lift3D Trajectory",
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    pc = np.asarray(point_cloud, dtype=np.float32)
    pos = np.asarray(positions, dtype=np.float32)
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1, alpha=0.3)
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], "-o", markersize=2, linewidth=1.5, color="red")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def random_indices(n: int, k: int, seed: int = 0) -> Iterable[int]:
    rng = np.random.default_rng(seed)
    return rng.choice(np.arange(n), size=min(k, n), replace=False).tolist()

