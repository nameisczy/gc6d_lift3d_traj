from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

"""
GC6D `loadGrasp(..., format='6d')` 已在 API 内用
  (fric_coefs <= fric_coef_thresh) & (fric_coefs > 0) & ~collision
筛掉无效抓取（见 graspclutter6dAPI/graspclutter6d.py）。
本模块在 **已解码的 17D 数组** 上再做 top_k / 分数阈值等后处理。
"""


@dataclass
class GraspFilterConfig:
    top_k: int = 50
    collision_free_only: bool = True
    force_closure_only: bool = True
    friction_valid_only: bool = True
    min_score: Optional[float] = None


def filter_valid_grasps(grasps_17d: np.ndarray, cfg: GraspFilterConfig) -> np.ndarray:
    g = np.asarray(grasps_17d, dtype=np.float32)
    if g.size == 0:
        return g.reshape(0, 17)
    valid = np.isfinite(g).all(axis=1)
    valid &= g[:, 1] > 0.0
    # score = 1.1 - mu * friction in dataset builder; valid grasps typically score > 0
    valid &= g[:, 0] > 0.0
    if cfg.min_score is not None:
        valid &= g[:, 0] >= float(cfg.min_score)
    g = g[valid]
    if g.shape[0] == 0:
        return g.reshape(0, 17)
    order = np.argsort(g[:, 0])[::-1]
    g = g[order]
    # collision_free / friction / force_closure: enforced when loading via API; flags kept for config compatibility
    _ = (cfg.collision_free_only, cfg.force_closure_only, cfg.friction_valid_only)
    return g[: int(cfg.top_k)]
