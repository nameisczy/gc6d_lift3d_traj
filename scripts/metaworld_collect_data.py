#!/usr/bin/env python3
"""
Collect pick-and-place expert trajectories with MetaWorld's built-in policy.

Uses ``lift3d.envs.metaworld_env.MetaWorldEnv`` so RGB / depth / point clouds match
the official LIFT3D MetaWorld training pipeline (no zero point clouds).

Requires: LIFT3D on PYTHONPATH (set LIFT3D_ROOT), open3d, headless EGL if no display:
  export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_DEFAULT_OUT = _REPO_ROOT / "data" / "metaworld_pickplace_dataset.npz"


def _ensure_lift3d() -> Path:
    root = Path(os.environ.get("LIFT3D_ROOT", "/home/ziyaochen/LIFT3D")).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5, help="Number of expert episodes to collect")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode")
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
        help=f"Output .npz (default: {_DEFAULT_OUT})",
    )
    parser.add_argument("--task-name", type=str, default="pick-place", help="MetaWorld task name (LIFT3D style, no -v3 suffix)")
    parser.add_argument("--camera-name", type=str, default="corner")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-points", type=int, default=1024)
    args = parser.parse_args()
    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    _ensure_lift3d()
    from metaworld.policies import SawyerPickPlaceV3Policy

    from lift3d.envs.metaworld_env import MetaWorldEnv

    from gc6d_lift3d_traj.metaworld_pointcloud import apply_metaworld_lift3d_render_size, pinhole_intrinsics_from_mujoco
    from gc6d_lift3d_traj.gc6d_pointcloud import validate_point_cloud

    mwe = MetaWorldEnv(
        task_name=args.task_name,
        max_episode_length=args.max_steps + 10,
        image_size=args.image_size,
        camera_name=args.camera_name,
        use_point_crop=True,
        num_points=args.num_points,
        point_cloud_camera_names=[args.camera_name],
    )
    apply_metaworld_lift3d_render_size(mwe.env, args.image_size)
    pol = SawyerPickPlaceV3Policy()

    all_obs: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_pc: list[np.ndarray] = []
    all_rgb: list[np.ndarray] = []
    all_depth: list[np.ndarray] = []
    ep_lens: list[int] = []
    cam_k: np.ndarray | None = None

    for _ep in range(args.episodes):
        obs_dict = mwe.reset()
        if cam_k is None:
            r = mwe.env.mujoco_renderer
            cam_k = pinhole_intrinsics_from_mujoco(r.model, args.camera_name, r.width, r.height)
        steps = 0
        for _t in range(args.max_steps):
            o = np.asarray(obs_dict["raw_state"], dtype=np.float32)
            a = pol.get_action(o)
            if isinstance(a, np.ndarray) and a.dtype != np.float32:
                a = a.astype(np.float32)
            pc_raw = np.asarray(obs_dict["point_cloud"], dtype=np.float32)
            if pc_raw.ndim == 2 and pc_raw.shape[1] >= 3:
                pc = np.ascontiguousarray(pc_raw[:, :3])
            else:
                raise RuntimeError(f"Unexpected point_cloud shape: {pc_raw.shape}")
            if _ep == 0 and _t == 0:
                validate_point_cloud(pc, name="metaworld_point_cloud")
            rgb = np.asarray(obs_dict["image"], dtype=np.uint8)
            depth = np.asarray(obs_dict["depth"], dtype=np.float32)

            all_obs.append(o)
            all_actions.append(np.asarray(a, dtype=np.float32))
            all_pc.append(pc)
            all_rgb.append(rgb)
            all_depth.append(depth)

            obs_dict, _r, term, trunc, _i = mwe.step(a)
            steps += 1
            if term or trunc:
                break
        ep_lens.append(steps)

    if not all_obs:
        print("ERROR: no transitions collected", file=sys.stderr)
        sys.exit(1)

    obs_stacked = np.stack(all_obs, axis=0)
    act_stacked = np.stack(all_actions, axis=0)
    pc_stacked = np.stack(all_pc, axis=0)
    rgb_stacked = np.stack(all_rgb, axis=0)
    depth_stacked = np.stack(all_depth, axis=0)
    ep_starts = np.cumsum([0] + ep_lens[:-1], dtype=np.int64) if ep_lens else np.empty((0,), np.int64)

    np.savez_compressed(
        out_path,
        dataset_version=np.array([2], dtype=np.int32),
        all_obs=obs_stacked,
        all_actions=act_stacked,
        all_point_clouds=pc_stacked,
        all_rgb=rgb_stacked,
        all_depth=depth_stacked,
        cam_K=np.asarray(cam_k, dtype=np.float64),
        ep_lens=np.asarray(ep_lens, dtype=np.int64),
        ep_starts=ep_starts,
        task_name=np.array([args.task_name], dtype=object),
        camera_name=np.array([args.camera_name], dtype=object),
        image_size=np.int32(args.image_size),
    )
    n = int(obs_stacked.shape[0])
    print(
        f"Wrote {n} transitions from {args.episodes} episode(s) "
        f"(task={args.task_name}, v2 with real point clouds) to {out_path}"
    )


if __name__ == "__main__":
    main()
