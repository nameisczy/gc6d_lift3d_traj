#!/usr/bin/env python3
"""MetaWorld evaluation for OfficialHeadGC6DPolicy (official head reuse check)."""

from __future__ import annotations

import argparse
import os

import torch

from gc6d_lift3d_traj.lift3d_integration.official_head_gc6d_policy import OfficialHeadGC6DPolicy


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--init-metaworld-ckpt", type=str, default=None)
    p.add_argument("--official-head-init", type=str, choices=("random", "metaworld"), default="metaworld")
    p.add_argument("--encoder-init", type=str, choices=("lift3d_clip", "random", "metaworld"), default="lift3d_clip")
    p.add_argument("--head-init", type=str, choices=("random", "metaworld"), default="metaworld")
    p.add_argument("--model-ckpt", type=str, default=None, help="Optional finetuned adapter checkpoint to load.")
    p.add_argument("--task-name", type=str, default="pick-place")
    p.add_argument("--num-episodes", type=int, default=20)
    p.add_argument("--camera-name", type=str, default="corner")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OfficialHeadGC6DPolicy(
        metaworld_ckpt=args.init_metaworld_ckpt,
        official_head_init=args.official_head_init,
        encoder_init=args.encoder_init,
        head_init=args.head_init,
    ).to(device)
    if args.model_ckpt:
        raw = torch.load(args.model_ckpt, map_location="cpu", weights_only=False)
        sd = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
        model.load_state_dict(sd, strict=False)
    model.eval()

    # Reuse official evaluator from LIFT3D
    os.environ.setdefault("LIFT3D_ROOT", "/home/ziyaochen/LIFT3D")
    from lift3d.envs.metaworld_env import MetaWorldEvaluator

    evaluator = MetaWorldEvaluator(task_name=args.task_name, camera_name=args.camera_name)

    class _PolicyWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, images, point_clouds, robot_states, texts):
            return self.m.metaworld_forward(point_clouds, robot_states)

    wrapped = _PolicyWrapper(model).to(device)
    success_rate, avg_rewards = evaluator.evaluate(num_episodes=args.num_episodes, policy=wrapped)
    print(f"success_rate={float(success_rate):.6f}")
    print(f"avg_rewards={float(avg_rewards):.6f}")


if __name__ == "__main__":
    main()
