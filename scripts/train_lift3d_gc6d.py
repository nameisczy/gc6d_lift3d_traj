#!/usr/bin/env python3
"""Train trajectory policy on generated GC6D-camera-frame episodes (imitation + goal + gripper)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gc6d_lift3d_traj.lift3d_integration.lift3d_dataset import Lift3DTrajDataset
from gc6d_lift3d_traj.lift3d_integration.lift3d_encoder_ckpt import apply_lift3d_encoder_checkpoint, log_encoder_load
from gc6d_lift3d_traj.lift3d_integration.metaworld_init_ckpt import (
    inspect_checkpoint,
    load_metaworld_policy_init,
    log_metaworld_init,
)
from gc6d_lift3d_traj.lift3d_integration.lift3d_train_adapter import LossWeights, compute_trajectory_losses
from gc6d_lift3d_traj.lift3d_integration.official_head_gc6d_policy import OfficialHeadGC6DPolicy
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
    p.add_argument("--model-type", type=str, choices=("custom_action10", "official_head_adapter"), default="custom_action10")
    p.add_argument("--official-head-init", type=str, choices=("random", "metaworld"), default="metaworld")
    p.add_argument("--encoder-init", type=str, choices=("lift3d_clip", "random", "metaworld"), default="lift3d_clip")
    p.add_argument("--head-init", type=str, choices=("random", "metaworld"), default="metaworld")
    p.add_argument("--init-metaworld-ckpt", type=str, default=None, help="Official LIFT3D policy checkpoint for initialization")
    p.add_argument("--stage", type=str, choices=("adapter", "adapter_head", "adapter_head_encoder", "joint"), default=None)
    p.add_argument("--phase", type=str, choices=("A", "B", "C"), default="A", help="Deprecated alias; use --stage")
    p.add_argument("--phase-a-min-epochs", type=int, default=5, help="Recommended minimum epochs for Phase A before Phase B")
    p.add_argument("--phase-b-ack", action="store_true", help="Required ack to run Phase B (ensure Phase A converged)")
    p.add_argument("--max-batches-per-epoch", type=int, default=0, help="0=no limit; >0 for quick pilot finetune")
    p.add_argument("--save-init-only", action="store_true", help="Load init checkpoint and save without optimizer steps")
    p.add_argument("--print-json-summary", action="store_true", help="Print TRAIN_SUMMARY_JSON line for pipeline_validate.py")
    p.add_argument(
        "--load-ckpt",
        type=str,
        default=None,
        help="Load {'model': state_dict} (or raw state_dict) after arch init, before training (staged finetune).",
    )
    args = p.parse_args()
    if args.stage is None:
        phase_to_stage = {"A": "adapter", "B": "adapter_head", "C": "adapter_head_encoder"}
        args.stage = phase_to_stage.get(args.phase, "adapter")

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
    if args.phase == "A" and epochs < args.phase_a_min_epochs:
        print(
            f"WARNING: Phase A epochs={epochs} < recommended {args.phase_a_min_epochs}. "
            "For stable transfer, run >=5 epochs before Phase B."
        )
    if args.stage in ("adapter_head", "adapter_head_encoder", "joint") and not args.phase_b_ack:
        raise SystemExit("Phase B/C requires --phase-b-ack (start only after Phase A converges).")

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
    paths = cfg.get("paths") or {}
    ds_block = cfg.get("dataset") or {}
    gc6d_root = paths.get("gc6d_root")
    gc6d_api = paths.get("gc6d_api_root")
    ds = Lift3DTrajDataset(
        index_path,
        use_real_pointcloud=bool(ds_block.get("use_real_pointcloud", True)),
        reload_pointcloud_from_api=bool(ds_block.get("reload_pointcloud_from_api", False)),
        gc6d_root=gc6d_root,
        gc6d_api_root=gc6d_api,
        dataset_split=str(ds_block.get("split", "train")),
        default_camera=str(ds_block.get("camera", "realsense-d415")),
    )
    if len(ds) == 0:
        raise RuntimeError(f"No samples in index: {index_path}")
    n_episode_rows = len(ds.rows)
    print(f"index_train_rows={n_episode_rows}")
    print(f"training_samples={len(ds)}")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)

    if args.model_type == "official_head_adapter":
        model = OfficialHeadGC6DPolicy(
            metaworld_ckpt=args.init_metaworld_ckpt,
            official_head_init=args.official_head_init,
            encoder_init=args.encoder_init,
            head_init=args.head_init,
        ).to(device)
        info = model.inspect()
        print(f"[official_model] policy_head_arch={info['policy_head_arch']}")
        print(
            f"[official_model] encoder_output_dim={info['encoder_output_dim']} "
            f"policy_head_first_layer_input_dim={info['policy_head_first_layer_input_dim']} "
            f"policy_head_output_dim={info['policy_head_output_dim']} "
            f"expected_robot_state_dim={info['expected_robot_state_dim']}"
        )
        print(
            "[official_model] "
            f"encoder_loaded_tensors={info.get('encoder_loaded_tensors', 0)} "
            f"policy_head_loaded_tensors={info.get('policy_head_loaded_tensors', 0)} "
            f"adapter_loaded_tensors={info.get('adapter_loaded_tensors', 0)}"
        )
        if args.load_ckpt:
            load_raw = torch.load(args.load_ckpt, map_location="cpu", weights_only=False)
            load_sd = load_raw["model"] if isinstance(load_raw, dict) and "model" in load_raw else load_raw
            incompat2 = model.load_state_dict(load_sd, strict=False)
            print(
                f"[load_ckpt] {args.load_ckpt} "
                f"missing={len(incompat2.missing_keys)} unexpected={len(incompat2.unexpected_keys)}"
            )
    else:
        model = TrajectoryPolicy(robot_state_dim=1).to(device)
        enc_report = apply_lift3d_encoder_checkpoint(model, map_location=device)
        log_encoder_load("[model]", enc_report)
    meta_init_report = None
    if args.init_metaworld_ckpt and args.model_type == "custom_action10":
        ckpt_info = inspect_checkpoint(args.init_metaworld_ckpt)
        print(
            "[init_ckpt] "
            f"n_tensors={ckpt_info['n_tensors']} "
            f"encoder={ckpt_info['n_point_cloud_encoder']} "
            f"policy_head={ckpt_info['n_policy_head']} "
            f"other={ckpt_info['n_other']}"
        )
        meta_init_report = load_metaworld_policy_init(model, args.init_metaworld_ckpt, map_location=device)
        log_metaworld_init("[model]", meta_init_report)

    if args.model_type == "official_head_adapter":
        train_encoder = args.stage in ("adapter_head_encoder", "joint")
        train_head = args.stage in ("adapter_head", "adapter_head_encoder", "joint")
        for p_ in model.point_cloud_encoder.parameters():
            p_.requires_grad = train_encoder
        for p_ in model.policy_head.parameters():
            p_.requires_grad = train_head
        for p_ in model.adapter.parameters():
            p_.requires_grad = True
        tb = model.trainable_param_breakdown()
        print(
            f"stage={args.stage} trainable_params "
            f"encoder={tb['encoder']} adapter={tb['adapter']} policy_head={tb['policy_head']}"
        )
    else:
        freeze_encoder = args.phase == "A"
        for p_enc in model.pc_encoder.parameters():
            p_enc.requires_grad = not freeze_encoder
        print(f"phase={args.phase} freeze_encoder={freeze_encoder}")

    def _normalize_pc(pc: torch.Tensor) -> torch.Tensor:
        c = pc.float()
        centroid = c.mean(dim=1, keepdim=True)
        c = c - centroid
        m = torch.max(torch.sqrt(torch.sum(c**2, dim=-1)), dim=1, keepdim=True)[0]
        return c / m.unsqueeze(-1).clamp(min=1e-8)

    @torch.no_grad()
    def _encoder_feature_stats(max_batches: int = 10) -> Dict[str, float]:
        model.eval()
        feats = []
        for bidx, (state, _target_delta, _goal10) in enumerate(dl):
            pc = state["point_cloud"].to(device)
            if args.model_type == "official_head_adapter":
                f = model.point_cloud_encoder(_normalize_pc(pc))
            else:
                f = model.pc_encoder(_normalize_pc(pc))
            feats.append(f.detach().cpu())
            if (bidx + 1) >= max_batches:
                break
        if not feats:
            return {"feat_mean": float("nan"), "feat_std": float("nan"), "norm_mean": float("nan"), "norm_std": float("nan")}
        F = torch.cat(feats, dim=0)
        norms = torch.norm(F, dim=1)
        return {
            "feat_mean": float(F.mean().item()),
            "feat_std": float(F.std(unbiased=False).item()),
            "norm_mean": float(norms.mean().item()),
            "norm_std": float(norms.std(unbiased=False).item()),
        }

    feat_stats_before = _encoder_feature_stats()
    print(
        "[encoder_stats][before] "
        f"feat_mean={feat_stats_before['feat_mean']:.6f} "
        f"feat_std={feat_stats_before['feat_std']:.6f} "
        f"norm_mean={feat_stats_before['norm_mean']:.6f} "
        f"norm_std={feat_stats_before['norm_std']:.6f}"
    )

    if args.model_type == "official_head_adapter" and args.stage in ("adapter_head_encoder", "joint"):
        enc_params = [p for p in model.point_cloud_encoder.parameters() if p.requires_grad]
        other_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("point_cloud_encoder.")]
        opt = torch.optim.AdamW(
            [{"params": enc_params, "lr": lr * 0.1}, {"params": other_params, "lr": lr}],
            lr=lr,
        )
    else:
        opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    weights = LossWeights(imitation=w_imit, goal=w_goal, gripper=w_grip)

    if args.save_init_only:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "weights": weights,
                "init_metaworld_ckpt": args.init_metaworld_ckpt,
                "phase": args.phase,
                "index_rows": n_episode_rows,
                "training_samples": len(ds),
            },
            out,
        )
        print(f"saved init-only checkpoint {out}")
        return

    epoch_totals: List[float] = []
    for ep in range(epochs):
        model.train()
        totals = {"total": 0.0, "imitation": 0.0, "goal": 0.0, "gripper": 0.0}
        n = 0
        for bidx, (state, target_delta, goal10) in enumerate(dl):
            pc = state["point_cloud"].to(device)
            ee_pos = state["ee_position"].to(device)
            ee_rot = state["ee_rotation"].to(device)
            grip = state["gripper"].to(device)
            if grip.dim() == 1:
                grip = grip.unsqueeze(-1)
            tgt_d = target_delta.to(device)
            g10 = goal10.to(device)

            if args.model_type == "official_head_adapter":
                pred4 = model.gc6d_forward(pc, ee_pos, ee_rot, grip, g10)
                tgt4 = torch.cat([tgt_d[:, :3], tgt_d[:, 9:10]], dim=-1)
                mse = F.mse_loss(pred4, tgt4)
                losses = {"total": mse, "imitation": mse, "goal": torch.zeros_like(mse), "gripper": mse}
            else:
                pred_d, pred_g = model(pc, ee_pos, ee_rot, grip, g10)
                losses = compute_trajectory_losses(pred_d, pred_g, tgt_d, g10, weights)
            opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            opt.step()
            for k in totals:
                totals[k] += float(losses[k].item())
            n += 1
            if args.max_batches_per_epoch > 0 and (bidx + 1) >= args.max_batches_per_epoch:
                break
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

    feat_stats_after = _encoder_feature_stats()
    print(
        "[encoder_stats][after] "
        f"feat_mean={feat_stats_after['feat_mean']:.6f} "
        f"feat_std={feat_stats_after['feat_std']:.6f} "
        f"norm_mean={feat_stats_after['norm_mean']:.6f} "
        f"norm_std={feat_stats_after['norm_std']:.6f}"
    )

    loss_decreased = False
    if len(epoch_totals) >= 2:
        loss_decreased = epoch_totals[-1] < epoch_totals[0]

    if args.print_json_summary:
        summary = {
            "index": index_path,
            "out": str(out),
            "epochs": epochs,
            "model_type": args.model_type,
            "official_head_init": args.official_head_init,
            "encoder_init": args.encoder_init,
            "head_init": args.head_init,
            "stage": args.stage,
            "phase": args.phase,
            "init_metaworld_ckpt": args.init_metaworld_ckpt,
            "load_ckpt": args.load_ckpt,
            "index_rows": n_episode_rows,
            "training_samples": len(ds),
            "max_batches_per_epoch": args.max_batches_per_epoch,
            "encoder_feat_stats_before": feat_stats_before,
            "encoder_feat_stats_after": feat_stats_after,
            "epoch_mean_total_loss": epoch_totals,
            "loss_decreased_first_vs_last": loss_decreased,
        }
        print("TRAIN_SUMMARY_JSON:", json.dumps(summary))


if __name__ == "__main__":
    main()
