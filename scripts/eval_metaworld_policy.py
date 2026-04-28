#!/usr/bin/env python3
"""
Rollout evaluation for MetaWorld pick-place-v3: learned policy vs ``SawyerPickPlaceV3Policy``.

``--policy-type trajectory`` loads ``TrajectoryPolicy``; ``mlp`` loads :class:`metaworld_mlp_policy.MetaWorldMLPPolicy`.

For the first 3 episodes, follows the **expert** trajectory (actions from expert) and at each
observation prints pooled stats of learned vs expert actions (same obs).

Example:
  python scripts/eval_metaworld_policy.py --policy-type mlp --ckpt data/metaworld_mlp_policy.pt
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Sequence, Union

import numpy as np
import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from metaworld_mlp_policy import MetaWorldMLPPolicy
from metaworld_state import metaworld_raw39_to_robot7_t, robot7_to_trajectory_policy_inputs
from gc6d_lift3d_traj.lift3d_integration.trajectory_policy import TrajectoryPolicy

TASK_NAME = "pick-place-v3"


@dataclass
class RolloutStats:
    success_count: int
    total_reward: float
    lengths: List[int]


def _default_ckpt(policy_type: str) -> Path:
    if policy_type == "mlp":
        return _REPO_ROOT / "data" / "metaworld_mlp_policy.pt"
    return _REPO_ROOT / "data" / "metaworld_policy.pt"


def _load_state_dict(model: nn.Module, path: Path, device: torch.device) -> None:
    raw = torch.load(path, map_location=device, weights_only=False)
    if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        raw = raw["state_dict"]
    if isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
        raw = raw["model"]
    if not isinstance(raw, dict):
        raise TypeError(f"Unknown checkpoint at {path}: {type(raw)}")
    incomp = model.load_state_dict(raw, strict=False)
    if incomp.missing_keys:
        m = incomp.missing_keys
        print(f"warning: load_state_dict missing_keys ({len(m)}): {m[:8]} ...")
    if incomp.unexpected_keys:
        u = incomp.unexpected_keys
        print(f"warning: load_state_dict unexpected_keys ({len(u)}): {u[:8]} ...")


def _build_metaworld_env():
    from metaworld import MT1

    mt1 = MT1(TASK_NAME, seed=0)
    env_cls = mt1.train_classes[TASK_NAME]
    return mt1, env_cls


def _rollout(
    get_action4: Callable[[np.ndarray], np.ndarray],
    tasks: Sequence[object],
    max_steps: int,
) -> RolloutStats:
    _, env_cls = _build_metaworld_env()
    rewards: List[float] = []
    lengths: List[int] = []
    successes: List[bool] = []

    for task in tasks:
        env = env_cls()
        env.set_task(task)
        obs, _ = env.reset()
        ep_r = 0.0
        length = 0
        success = False
        for _t in range(max_steps):
            a4 = get_action4(obs)
            a4 = np.asarray(a4, dtype=np.float32).reshape(-1)[:4]
            obs, r, term, trunc, info = env.step(a4)
            ep_r += float(r)
            length += 1
            if info.get("success", False):
                success = True
            if term or trunc:
                break
        successes.append(success)
        rewards.append(ep_r)
        lengths.append(length)

    return RolloutStats(
        success_count=sum(1 for s in successes if s),
        total_reward=float(np.sum(rewards)),
        lengths=lengths,
    )


def _print_block(title: str, stats: RolloutStats, n_ep: int) -> None:
    rate = stats.success_count / n_ep
    mean_r = stats.total_reward / n_ep
    mean_len = float(np.mean(stats.lengths)) if stats.lengths else 0.0
    print(f"--- {title} ---")
    print(f"  success_rate:         {rate:.4f}  ({stats.success_count}/{n_ep})")
    print(f"  mean_reward:          {mean_r:.4f}")
    print(f"  mean_episode_length:  {mean_len:.2f}")


def _print_action_dim_table(label: str, x: np.ndarray) -> None:
    """*x* shape (T, 4): one line per channel with mean/std/min/max."""
    for d in range(x.shape[1]):
        c = x[:, d]
        print(
            f"  {label} dim{d}: mean={c.mean():.4f}  std={c.std():.4f}  "
            f"min={c.min():.4f}  max={c.max():.4f}"
        )


def _debug_first_episodes(
    n_debug_ep: int,
    tasks: Sequence[object],
    max_steps: int,
    get_pred4: Callable[[np.ndarray], np.ndarray],
    expert: object,
) -> None:
    """
    Roll out by applying **expert** actions so the observation path matches the expert; at each
    step compare pred (learned) vs expert action at the same obs.
    """
    _, env_cls = _build_metaworld_env()
    preds: List[np.ndarray] = []
    exps: List[np.ndarray] = []
    n_run = min(n_debug_ep, len(tasks))
    for ep in range(n_run):
        env = env_cls()
        env.set_task(tasks[ep])
        obs, _ = env.reset()
        for _t in range(max_steps):
            p = get_pred4(obs)
            p = np.asarray(p, dtype=np.float32).reshape(-1)[:4]
            e = np.asarray(expert.get_action(obs), dtype=np.float32).reshape(-1)[:4]
            preds.append(p)
            exps.append(e)
            obs, _r, term, trunc, _i = env.step(e)
            if term or trunc:
                break
    P = np.stack(preds, axis=0) if preds else np.zeros((0, 4), dtype=np.float32)
    E = np.stack(exps, axis=0) if exps else np.zeros((0, 4), dtype=np.float32)
    mae = np.abs(P - E).mean(axis=0) if len(preds) else np.zeros(4, dtype=np.float32)
    print("--- [debug] first 3 episodes: pred vs expert at same obs (rollout steered by expert) ---")
    if P.shape[0] == 0:
        print("  (no steps collected)")
        return
    print("  learned prediction:")
    _print_action_dim_table("pred", P)
    print("  expert (same obs):")
    _print_action_dim_table("expert", E)
    print("  per-dim MAE (mean |pred-expert| over time): " + "  ".join(f"dim{i}={mae[i]:.6f}" for i in range(4)))


def _make_learned_fn(
    policy_type: str,
    model: Union[MetaWorldMLPPolicy, TrajectoryPolicy],
    device: torch.device,
) -> Callable[[np.ndarray], np.ndarray]:

    @torch.inference_mode()
    def learned_action4_mlp(obs: np.ndarray) -> np.ndarray:
        o1 = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        r7 = metaworld_raw39_to_robot7_t(o1)
        pred4 = model(r7)  # type: ignore[operator, misc]
        return pred4.squeeze(0).float().cpu().numpy()

    @torch.inference_mode()
    def learned_action4_traj(obs: np.ndarray) -> np.ndarray:
        o1 = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        r7 = metaworld_raw39_to_robot7_t(o1)
        pc = torch.zeros(1, 1024, 3, device=device, dtype=torch.float32)
        ee_p, ee_r, grip, g10 = robot7_to_trajectory_policy_inputs(r7)
        pred4, _ = model(pc, ee_p, ee_r, grip, g10)  # type: ignore[operator, misc]
        return pred4.squeeze(0).float().cpu().numpy()

    if policy_type == "mlp":
        return learned_action4_mlp
    return learned_action4_traj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="State dict (default: data/metaworld_policy.pt or metaworld_mlp_policy.pt by type).",
    )
    ap.add_argument(
        "--policy-type",
        type=str,
        choices=("trajectory", "mlp"),
        default="trajectory",
    )
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument(
        "--seed", type=int, default=0, help="Seeds the task list so expert vs learned see the same tasks"
    )
    args = ap.parse_args()
    ckpt_path = args.ckpt or _default_ckpt(args.policy_type)

    if not ckpt_path.is_file():
        raise SystemExit(
            f"Missing checkpoint: {ckpt_path}\nTrain with matching --policy-type, e.g.:\n"
            f"  python scripts/train_metaworld_policy.py --policy-type {args.policy_type} --out-ckpt {ckpt_path}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.policy_type == "mlp":
        model: Union[MetaWorldMLPPolicy, TrajectoryPolicy] = MetaWorldMLPPolicy().to(device)
    else:
        model = TrajectoryPolicy(
            robot_state_dim=1,
            hidden=512,
            action_out_dim=4,
            with_goal_head=False,
        ).to(device)
    _load_state_dict(model, ckpt_path, device)
    model.eval()

    learned_action4 = _make_learned_fn(args.policy_type, model, device)

    from metaworld.policies import SawyerPickPlaceV3Policy

    expert_pol = SawyerPickPlaceV3Policy()

    def expert_action4(obs: np.ndarray) -> np.ndarray:
        return np.asarray(expert_pol.get_action(obs), dtype=np.float32)

    mt1, _ = _build_metaworld_env()
    rng = np.random.default_rng(args.seed)
    n_tasks = len(mt1.train_tasks)
    tasks: List[object] = [mt1.train_tasks[int(rng.integers(0, n_tasks))] for _ in range(args.episodes)]

    # Debug: first 3 episodes, pred vs expert at same obs (expert-steered rollout)
    _debug_first_episodes(3, tasks, args.max_steps, learned_action4, expert_pol)

    stats_expert = _rollout(expert_action4, tasks, args.max_steps)
    stats_learned = _rollout(learned_action4, tasks, args.max_steps)

    n = args.episodes
    _print_block(f"Expert: {SawyerPickPlaceV3Policy.__name__}", stats_expert, n)
    _print_block(f"Learned: {args.policy_type} ({type(model).__name__})", stats_learned, n)
    re = stats_expert.success_count / n
    rl = stats_learned.success_count / n
    print("--- Comparison ---")
    print(f"  expert_success_rate:   {re:.4f}")
    print(f"  learned_success_rate:  {rl:.4f}")
    print(f"  delta (learned - exp):  {rl - re:+.4f}")


if __name__ == "__main__":
    main()
