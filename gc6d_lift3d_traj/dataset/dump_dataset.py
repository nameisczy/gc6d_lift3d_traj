from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import numpy as np

from gc6d_lift3d_traj.utils.io import append_jsonl


def dump_episode_npz(path: Path, episode: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **episode)


def append_index(path: Path, rows: Iterable[dict]) -> None:
    append_jsonl(path, rows)

