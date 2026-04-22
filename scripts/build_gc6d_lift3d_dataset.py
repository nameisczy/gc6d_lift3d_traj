#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from gc6d_lift3d_traj.dataset.dataset_format import DatasetFormat
from gc6d_lift3d_traj.dataset.dump_dataset import append_index, dump_episode_npz
from gc6d_lift3d_traj.dataset.episode_builder import build_episode
from gc6d_lift3d_traj.gc6d.gc6d_filter import GraspFilterConfig, filter_valid_grasps
from gc6d_lift3d_traj.gc6d.gc6d_loader import GC6DLoader
from gc6d_lift3d_traj.gc6d.pointcloud_utils import estimate_table_z
from gc6d_lift3d_traj.planner.collision import CollisionConfig, trajectory_is_collision_free
from gc6d_lift3d_traj.planner.trajectory_builder import TrajConfig, build_trajectory_from_grasp
from gc6d_lift3d_traj.utils.io import read_yaml, write_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="/home/ziyaochen/gc6d_lift3d_traj/configs/default.yaml")
    p.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Override paths.output_root from config (large SSD path).",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap number of frames to process (debug). Default: use dataset.max_samples from YAML or full split.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip episode .npz that already exist on disk (same path).",
    )
    p.add_argument(
        "--no-collision-check",
        action="store_true",
        help="Accept all trajectories that pass grasp filtering (debug only).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = read_yaml(Path(args.config))
    paths = cfg["paths"]
    out_root = Path(args.output_root) if args.output_root else Path(paths["output_root"])
    ds_fmt = DatasetFormat(root=out_root)
    ds_fmt.ensure()
    index_name = cfg["dataset"].get("index_filename", "index_train.jsonl")
    index_path = ds_fmt.index_dir / index_name

    loader = GC6DLoader(
        gc6d_root=paths["gc6d_root"],
        api_root=paths["gc6d_api_root"],
        camera=cfg["dataset"]["camera"],
        split=cfg["dataset"]["split"],
    )
    fcfg = GraspFilterConfig(
        top_k=cfg["dataset"]["top_k"],
        collision_free_only=cfg["dataset"].get("filtering", {}).get("collision_free_only", True),
        force_closure_only=cfg["dataset"].get("filtering", {}).get("force_closure_only", True),
        friction_valid_only=cfg["dataset"].get("filtering", {}).get("friction_valid_only", True),
        min_score=cfg["dataset"].get("filtering", {}).get("min_score", None),
    )
    traj = cfg["trajectory"]
    tcfg = TrajConfig(
        start_height_offset=traj["start_height_offset"],
        pregrasp_offset=traj["pregrasp_offset"],
        lift_distance=traj["lift_distance"],
        phase_steps=tuple(traj["phase_steps"]),
        use_curobo=bool(traj.get("use_curobo", False)),
        curobo_robot=str(traj.get("curobo_robot", "franka.yml")),
        curobo_scene=str(traj.get("curobo_scene", "collision_table.yml")),
        curobo_verbose=bool(traj.get("curobo_verbose", True)),
    )
    base_collision = CollisionConfig(
        table_z=cfg["collision"]["table_z"],
        table_tolerance=cfg["collision"]["table_tolerance"],
        point_contact_threshold=cfg["collision"]["point_contact_threshold"],
        max_points_in_boxes=cfg["collision"]["max_points_in_boxes"],
    )

    index_rows = []
    ep_count = 0
    skipped_no_grasp = 0
    skipped_collision = 0
    max_samples = args.max_samples
    if max_samples is None:
        max_samples = cfg["dataset"].get("max_samples")
    for sample in tqdm(loader.iter_samples(max_samples=max_samples), desc="building"):
        table_z = float(estimate_table_z(sample.point_cloud, q=0.05))
        ccfg = CollisionConfig(
            table_z=table_z,
            table_tolerance=base_collision.table_tolerance,
            point_contact_threshold=base_collision.point_contact_threshold,
            max_points_in_boxes=base_collision.max_points_in_boxes,
        )
        valid = filter_valid_grasps(sample.grasps_17d, fcfg)
        if valid.shape[0] == 0:
            skipped_no_grasp += 1
            continue
        for gid, grasp in enumerate(valid):
            traj = build_trajectory_from_grasp(grasp, tcfg)
            is_ok = args.no_collision_check or trajectory_is_collision_free(
                sample.point_cloud,
                traj["ee_positions"],
                traj["ee_rotations_matrix"],
                float(grasp[1]),
                ccfg,
            )
            if not is_ok:
                skipped_collision += 1
                continue
            meta = {
                "scene_id": sample.scene_id,
                "ann_id": sample.ann_id,
                "grasp_id": gid,
                "split": sample.split,
                "table_z_est": table_z,
                "camera": cfg["dataset"]["camera"],
            }
            ep = build_episode(sample.point_cloud, grasp, traj, meta)
            ep_path = ds_fmt.episodes_dir / f"scene{sample.scene_id:04d}_ann{sample.ann_id:04d}_g{gid:03d}.npz"
            if args.resume and ep_path.exists():
                continue
            dump_episode_npz(ep_path, ep)
            index_rows.append(
                {
                    "episode_path": str(ep_path),
                    "scene_id": sample.scene_id,
                    "ann_id": sample.ann_id,
                    "grasp_id": gid,
                    "environment_id": sample.scene_id // 1000,
                    "split": sample.split,
                    "valid_collision": True,
                }
            )
            ep_count += 1

            if ep_count % 100 == 0:
                append_index(index_path, index_rows)
                index_rows.clear()

    if index_rows:
        append_index(index_path, index_rows)
    write_json(
        ds_fmt.metadata_dir / "build_run.json",
        {"episodes_written": ep_count, "index": str(index_path), "config": str(args.config)},
    )
    print(
        f"Done. episodes={ep_count} skipped_no_valid_grasp={skipped_no_grasp} "
        f"skipped_collision={skipped_collision} index={index_path}"
    )


if __name__ == "__main__":
    main()

