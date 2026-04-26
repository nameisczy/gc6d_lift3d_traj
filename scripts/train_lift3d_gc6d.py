#!/usr/bin/env python3
"""Train trajectory policy on generated GC6D-camera-frame episodes (imitation + goal + gripper)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

from gc6d_lift3d_traj.lift3d_integration.lift3d_dataset import Lift3DTrajDataset
from gc6d_lift3d_traj.lift3d_integration.lift3d_encoder_ckpt import apply_lift3d_encoder_checkpoint, log_encoder_load
from gc6d_lift3d_traj.lift3d_integration.lift3d_train_adapter import LossWeights, compute_trajectory_losses
from gc6d_lift3d_traj.lift3d_integration.trajectory_policy import TrajectoryPolicy


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as e:
        raise RuntimeError("PyYAML required for --config") from e
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None, help="YAML (epochs, batch_size, lr, weights, data_root, index_name, ...)")
    p.add_argument("--data-root", type=str, default=None, help="Dataset root; index = <data-root>/index/<index_name>")
    p.add_argument("--index", type=str, default=None, help="Explicit index_train.jsonl path (overrides data-root)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--out", type=str, default=None, help="Checkpoint path (default: data-root/metadata/traj_policy.pt)")
    p.add_argument("--w-imit", type=float, default=None)
    p.add_argument("--w-goal", type=float, default=None)
    p.add_argument("--w-grip", type=float, default=None)
    p.add_argument("--print-json-summary", action="store_true", help="Print TRAIN_SUMMARY_JSON line for pipeline_validate.py")
    args = p.parse_args()

    cfg: Dict[str, Any] = {}
    if args.config:
        cfg = _load_yaml(Path(args.config))

    data_root = args.data_root or cfg.get("data_root")
    index_name = cfg.get("index_name", "index_train.jsonl")
    if args.index:
        index_path = args.index
    elif data_root:
        index_path = str(Path(data_root) / "index" / index_name)
    else:
        raise SystemExit("Provide --index or --data-root (or data_root in --config YAML).")

    epochs = args.epochs if args.epochs is not None else int(cfg.get("epochs", 20))
    batch_size = args.batch_size if args.batch_size is not None else int(cfg.get("batch_size", 32))
    lr = args.lr if args.lr is not None else float(cfg.get("lr", 1e-3))
    wcfg = cfg.get("weights") or {}
    w_imit = args.w_imit if args.w_imit is not None else float(wcfg.get("imitation", 1.0))
    w_goal = args.w_goal if args.w_goal is not None else float(wcfg.get("goal", 0.5))
    w_grip = args.w_grip if args.w_grip is not None else float(wcfg.get("gripper", 0.2))

    if args.out:
        out_path = args.out
    elif cfg.get("output_ckpt"):
        out_path = str(cfg["output_ckpt"])
    elif data_root:
        out_path = str(Path(data_root) / "metadata" / "traj_policy.pt")
    else:
        out_path = str(Path(index_path).parent.parent / "metadata" / "traj_policy.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = Lift3DTrajDataset(index_path)
    if len(ds) == 0:
        raise RuntimeError(f"No samples in index: {index_path}")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)

    model = TrajectoryPolicy(robot_state_dim=1).to(device)
    enc_report = apply_lift3d_encoder_checkpoint(model, map_location=device)
    log_encoder_load("[model]", enc_report)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    weights = LossWeights(imitation=w_imit, goal=w_goal, gripper=w_grip)

    epoch_totals: List[float] = []
    for ep in range(epochs):
        model.train()
        totals = {"total": 0.0, "imitation": 0.0, "goal": 0.0, "gripper": 0.0}
        n = 0
        for state, target_delta, goal10 in dl:
            pc = state["point_cloud"].to(device)
            ee_pos = state["ee_position"].to(device)
            ee_rot = state["ee_rotation"].to(device)
            grip = state["gripper"].to(device)
            if grip.dim() == 1:
                grip = grip.unsqueeze(-1)
            tgt_d = target_delta.to(device)
            g10 = goal10.to(device)

            pred_d, pred_g = model(pc, ee_pos, ee_rot, grip, g10)
            losses = compute_trajectory_losses(pred_d, pred_g, tgt_d, g10, weights)
            opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            opt.step()
            for k in totals:
                totals[k] += float(losses[k].item())
            n += 1
        if n:
            avg_total = totals["total"] / n
            epoch_totals.append(avg_total)
            print(
                f"epoch={ep} total={avg_total:.6f} imit={totals['imitation']/n:.6f} "
                f"goal={totals['goal']/n:.6f} grip={totals['gripper']/n:.6f}"
            )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "weights": weights}, out)
    print(f"saved {out}")

    loss_decreased = False
    if len(epoch_totals) >= 2:
        loss_decreased = epoch_totals[-1] < epoch_totals[0]

    if args.print_json_summary:
        summary = {
            "index": index_path,
            "out": str(out),
            "epochs": epochs,
            "epoch_mean_total_loss": epoch_totals,
            "loss_decreased_first_vs_last": loss_decreased,
        }
        print("TRAIN_SUMMARY_JSON:", json.dumps(summary))


if __name__ == "__main__":
    main()
