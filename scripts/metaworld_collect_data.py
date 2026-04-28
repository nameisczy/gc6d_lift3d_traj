#!/usr/bin/env python3
"""
Collect pick-and-place expert trajectories with MetaWorld's built-in policy.

If your metaworld install is V3-only (common), this uses task ``pick-place-v3`` and
``SawyerPickPlaceV3Policy``. Older V2 task/policy can be added when available
(see _make_env_and_policy).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Project root: .../gc6d_lift3d_traj
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUT = _REPO_ROOT / "data" / "metaworld_pickplace_dataset.npz"


def _make_env_and_policy():
    from metaworld import MT1

    try:
        from metaworld.policies import SawyerPickPlaceV2Policy

        name = "pick-place-v2"
        mt1 = MT1(name, seed=0)
        env_cls = mt1.train_classes[name]
        pol = SawyerPickPlaceV2Policy()
    except (ImportError, ValueError, KeyError):
        from metaworld.policies import SawyerPickPlaceV3Policy

        name = "pick-place-v3"
        mt1 = MT1(name, seed=0)
        env_cls = mt1.train_classes[name]
        pol = SawyerPickPlaceV3Policy()
    return mt1, name, env_cls, pol


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
    args = parser.parse_args()
    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mt1, name, env_cls, policy = _make_env_and_policy()
    # Match train dataset format: flat transition list, store as np arrays
    all_obs: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    ep_lens: list[int] = []
    rng = np.random.default_rng(0)
    n_tasks = len(mt1.train_tasks)
    for ep in range(args.episodes):
        task = mt1.train_tasks[int(rng.integers(0, n_tasks))]
        env = env_cls()
        env.set_task(task)
        o, _ = env.reset()
        steps = 0
        for _t in range(args.max_steps):
            a = policy.get_action(o)
            if isinstance(a, np.ndarray) and a.dtype != np.float32:
                a = a.astype(np.float32)
            all_obs.append(np.asarray(o, dtype=np.float32))
            all_actions.append(np.asarray(a, dtype=np.float32))
            o, _r, term, trunc, _i = env.step(a)
            steps += 1
            if term or trunc:
                break
        ep_lens.append(steps)

    obs_stacked = np.stack(all_obs, axis=0) if all_obs else np.empty((0, 0), dtype=np.float32)
    act_stacked = np.stack(all_actions, axis=0) if all_actions else np.empty((0, 0), dtype=np.float32)
    ep_starts = np.cumsum([0] + ep_lens[:-1], dtype=np.int64) if ep_lens else np.empty((0,), np.int64)

    np.savez_compressed(
        out_path,
        all_obs=obs_stacked,
        all_actions=act_stacked,
        ep_lens=np.asarray(ep_lens, dtype=np.int64),
        ep_starts=ep_starts,
        task_name=np.array([name], dtype=object),
    )
    n = int(obs_stacked.shape[0])
    print(
        f"Wrote {n} transitions from {args.episodes} episode(s) "
        f"(task={name}) to {out_path} (all_obs shape {obs_stacked.shape})"
    )
    if n < 1:
        sys.exit(1)


if __name__ == "__main__":
    main()
