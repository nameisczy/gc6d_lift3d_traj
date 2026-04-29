#!/usr/bin/env python3
"""
MetaWorld pick-place-v3 imitation: 7D state (hand + goal + grip) and 4D action.

State is parsed per MetaWorld spec (see ``metaworld_state``).

- ``--policy-type trajectory`` (default): ``TrajectoryPolicy`` + real point cloud from dataset, frozen ``pc_encoder``.
- ``--policy-type mlp``: :class:`metaworld_mlp_policy.MetaWorldMLPPolicy` (debug BC baseline).

Run from repo root, with ``gc6d`` (or similar) env that has torch; trajectory mode also needs LIFT3D checkpoint.

Example:
  python scripts/train_metaworld_policy.py --npz data/metaworld_pickplace_dataset.npz
  python scripts/train_metaworld_policy.py --policy-type mlp --out-ckpt data/metaworld_mlp_policy.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import project package
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from metaworld_state import robot7_to_trajectory_policy_inputs

from metaworld_dataset import MetaWorldPickPlaceDataset
from metaworld_mlp_policy import MetaWorldMLPPolicy
from gc6d_lift3d_traj.lift3d_integration.trajectory_policy import TrajectoryPolicy


def _default_out_ckpt(policy_type: str) -> Path:
    if policy_type == "mlp":
        return _REPO_ROOT / "data" / "metaworld_mlp_policy.pt"
    return _REPO_ROOT / "data" / "metaworld_policy.pt"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--npz",
        type=Path,
        default=_REPO_ROOT / "data" / "metaworld_pickplace_dataset.npz",
    )
    p.add_argument("--epochs", type=int, default=20, help="More epochs help BC on small datasets")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--policy-type",
        type=str,
        choices=("trajectory", "mlp"),
        default="trajectory",
        help="trajectory: Lift3d-based TrajectoryPolicy; mlp: 7D->4D MLP (debug).",
    )
    p.add_argument(
        "--freeze-encoder",
        action="store_true",
        default=True,
        help="(trajectory only) Freeze Lift3dCLIP. Default: on.",
    )
    p.add_argument(
        "--no-freeze-encoder",
        action="store_false",
        dest="freeze_encoder",
        help="(trajectory only) Unfreeze pc_encoder.",
    )
    p.add_argument(
        "--out-ckpt",
        type=Path,
        default=None,
        help="State dict path (default depends on --policy-type).",
    )
    args = p.parse_args()
    if os.environ.get("GC6D_DEBUG_ALLOW_DUMMY_POINTCLOUD", "").lower() in ("1", "true", "yes"):
        raise SystemExit(
            "GC6D_DEBUG_ALLOW_DUMMY_POINTCLOUD is debug-only and forbidden in training scripts."
        )
    out_path = args.out_ckpt or _default_out_ckpt(args.policy_type)

    if not args.npz.is_file():
        raise SystemExit(f"Dataset not found: {args.npz} (run scripts/metaworld_collect_data.py first)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = MetaWorldPickPlaceDataset(args.npz, action_dim=4, use_real_pointcloud=True)
    if len(ds) < 1:
        raise SystemExit("Empty dataset")
    loader = DataLoader(
        ds,
        batch_size=min(args.batch_size, len(ds)),
        shuffle=True,
        drop_last=False,
    )

    if args.policy_type == "mlp":
        model = MetaWorldMLPPolicy().to(device)
    else:
        model = TrajectoryPolicy(
            robot_state_dim=1,
            hidden=512,
            action_out_dim=4,
            with_goal_head=False,
        ).to(device)
        if args.freeze_encoder:
            for _n, param in model.pc_encoder.named_parameters():
                param.requires_grad = False

    opt = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
    )

    loss_curve: list[float] = []
    mse = nn.MSELoss()

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        n = 0
        for batch in loader:
            rob7 = batch["robot_states"].to(device)  # (B, 7) hand(3) goal(3) grip(1)
            y = batch["action"].to(device)  # (B, 4)
            B = rob7.size(0)
            if args.policy_type == "mlp":
                pred4 = model(rob7)
            else:
                pc = batch["point_clouds"].to(device)
                ee_p, ee_r, grip, g10 = robot7_to_trajectory_policy_inputs(rob7)
                pred4, _pred_goal = model(pc, ee_p, ee_r, grip, g10)
            loss = mse(pred4, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.item()) * B
            n += B
        epoch_loss = running / max(n, 1)
        loss_curve.append(epoch_loss)
        print(f"epoch {epoch+1}/{args.epochs}  mse={epoch_loss:.6f}  policy={args.policy_type}")

    print("---")
    print("loss_curve:", [round(x, 6) for x in loss_curve])
    if len(loss_curve) >= 2:
        print(
            f"first_loss={loss_curve[0]:.6f}  last_loss={loss_curve[-1]:.6f}  "
            f"decreased={loss_curve[-1] < loss_curve[0]}"
        )
    else:
        print(f"first vs last: {loss_curve!r}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"saved checkpoint: {out_path}  (policy_type={args.policy_type})")


if __name__ == "__main__":
    main()
