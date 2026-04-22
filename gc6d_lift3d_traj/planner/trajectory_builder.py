from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from gc6d_lift3d_traj.gc6d.grasp_decode import decode_gc6d_grasp
from gc6d_lift3d_traj.planner.interpolation import interpolate_positions, interpolate_rotations


@dataclass
class TrajConfig:
    start_height_offset: float = 0.15
    pregrasp_offset: float = 0.08
    lift_distance: float = 0.05
    phase_steps: tuple[int, int, int, int] = (8, 8, 2, 6)
    gripper_open: float = 1.0
    gripper_closed: float = 0.0
    # cuRobo (optional): replaces move+approach segments with GPU motion generation; requires CUDA + cuRobo install.
    use_curobo: bool = False
    curobo_robot: str = "franka.yml"
    curobo_scene: str = "collision_table.yml"
    curobo_verbose: bool = True


def _simple_trajectory(dec: Dict[str, Any], cfg: TrajConfig) -> Dict[str, Any]:
    """Original piecewise-linear trajectory (camera frame)."""
    c = dec["center"]
    Rg = dec["rotation"]
    approach = dec["approach_dir"] / (np.linalg.norm(dec["approach_dir"]) + 1e-8)

    p_init = c + np.array([0.0, 0.0, cfg.start_height_offset], dtype=np.float32)
    p_pre = c - cfg.pregrasp_offset * approach
    p_grasp = c.copy()
    p_lift = c + np.array([0.0, 0.0, cfg.lift_distance], dtype=np.float32)

    n_move, n_approach, n_close, n_lift = cfg.phase_steps
    pos_move = interpolate_positions(p_init, p_pre, n_move)
    pos_approach = interpolate_positions(p_pre, p_grasp, n_approach)
    pos_close = np.repeat(p_grasp[None, :], n_close, axis=0)
    pos_lift = interpolate_positions(p_grasp, p_lift, n_lift)
    positions = np.concatenate([pos_move, pos_approach, pos_close, pos_lift], axis=0).astype(np.float32)

    rot_move = interpolate_rotations(Rg, Rg, n_move)
    rot_approach = interpolate_rotations(Rg, Rg, n_approach)
    rot_close = np.repeat(Rg[None, :, :], n_close, axis=0)
    rot_lift = interpolate_rotations(Rg, Rg, n_lift)
    rotations = np.concatenate([rot_move, rot_approach, rot_close, rot_lift], axis=0).astype(np.float32)

    g_move = np.full((n_move, 1), cfg.gripper_open, dtype=np.float32)
    g_approach = np.full((n_approach, 1), cfg.gripper_open, dtype=np.float32)
    g_close = np.linspace(cfg.gripper_open, cfg.gripper_closed, n_close, dtype=np.float32).reshape(-1, 1)
    g_lift = np.full((n_lift, 1), cfg.gripper_closed, dtype=np.float32)
    gripper = np.concatenate([g_move, g_approach, g_close, g_lift], axis=0)

    return {
        "ee_positions": positions,
        "ee_rotations_matrix": rotations,
        "gripper": gripper,
        "initial_pose": {"position": p_init, "rotation": Rg},
        "pregrasp_pose": {"position": p_pre, "rotation": Rg},
        "final_grasp_pose": {"position": p_grasp, "rotation": Rg},
        "lift_pose": {"position": p_lift, "rotation": Rg},
        "planner": "simple",
    }


def _stack_curobo_then_close_lift(
    dec: Dict[str, Any],
    cfg: TrajConfig,
    curobo_waypoints: List[Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, Any]:
    """Append close + lift (simple interpolation) after a cuRobo approach segment."""
    c = dec["center"]
    Rg = dec["rotation"]
    p_grasp = c.copy()
    p_lift = c + np.array([0.0, 0.0, cfg.lift_distance], dtype=np.float32)
    _n_move, _n_approach, n_close, n_lift = cfg.phase_steps

    pos_main = np.stack([p for p, _ in curobo_waypoints], axis=0).astype(np.float32)
    rot_main = np.stack([r for _, r in curobo_waypoints], axis=0).astype(np.float32)

    pos_close = np.repeat(p_grasp[None, :], n_close, axis=0)
    rot_close = np.repeat(Rg[None, :, :], n_close, axis=0)
    pos_lift = interpolate_positions(p_grasp, p_lift, n_lift)
    rot_lift = interpolate_rotations(Rg, Rg, n_lift)

    positions = np.concatenate([pos_main, pos_close, pos_lift], axis=0).astype(np.float32)
    rotations = np.concatenate([rot_main, rot_close, rot_lift], axis=0).astype(np.float32)

    g_main = np.full((pos_main.shape[0], 1), cfg.gripper_open, dtype=np.float32)
    g_close = np.linspace(cfg.gripper_open, cfg.gripper_closed, n_close, dtype=np.float32).reshape(-1, 1)
    g_lift = np.full((n_lift, 1), cfg.gripper_closed, dtype=np.float32)
    gripper = np.concatenate([g_main, g_close, g_lift], axis=0)

    p_init = pos_main[0].copy()
    approach = dec["approach_dir"] / (np.linalg.norm(dec["approach_dir"]) + 1e-8)
    p_pre = c - cfg.pregrasp_offset * approach

    return {
        "ee_positions": positions,
        "ee_rotations_matrix": rotations,
        "gripper": gripper,
        "initial_pose": {"position": p_init, "rotation": Rg},
        "pregrasp_pose": {"position": p_pre, "rotation": Rg},
        "final_grasp_pose": {"position": p_grasp, "rotation": Rg},
        "lift_pose": {"position": p_lift, "rotation": Rg},
        "planner": "curobo",
    }


def build_trajectory_from_grasp(grasp_17d: np.ndarray, cfg: TrajConfig) -> dict:
    dec = decode_gc6d_grasp(grasp_17d)
    c = dec["center"]
    Rg = dec["rotation"]

    p_init = c + np.array([0.0, 0.0, cfg.start_height_offset], dtype=np.float32)

    if cfg.use_curobo:
        from gc6d_lift3d_traj.utils import curobo_planner

        cr = curobo_planner.plan_trajectory(
            (p_init, Rg),
            (c.astype(np.float32), Rg.astype(np.float32)),
            robot=cfg.curobo_robot,
            scene=cfg.curobo_scene,
            verbose=cfg.curobo_verbose,
        )
        n_wp = len(cr.poses)
        if cr.success and n_wp >= 2:
            if cfg.curobo_verbose:
                print(f"[trajectory_builder] cuRobo OK: trajectory_length={n_wp}")
            return _stack_curobo_then_close_lift(dec, cfg, cr.poses)
        if cfg.curobo_verbose:
            print(
                "[trajectory_builder] cuRobo failed or too short "
                f"(success={cr.success}, len={n_wp}, info={cr.info}); fallback to simple planner"
            )

    return _simple_trajectory(dec, cfg)

