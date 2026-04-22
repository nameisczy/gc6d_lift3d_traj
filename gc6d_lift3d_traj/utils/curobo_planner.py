"""
cuRobo-based SE(3) trajectory generation for GC6D episodes.

Planning frame matches the dataset convention used elsewhere: poses are expressed
in the same camera / task frame as ``trajectory_builder`` (treat as cuRobo world).

Requires: NVIDIA cuRobo (``pip install -e`` from https://github.com/NVlabs/curobo),
CUDA-capable PyTorch, and a GPU. If import or planning fails, callers should fall
back to the simple interpolating planner.

Install (typical):
  git clone https://github.com/NVlabs/curobo.git && cd curobo
  pip install -e . --no-build-isolation
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# (position (3,), rotation (3,3))
PosePair = Tuple[np.ndarray, np.ndarray]


def _quat_wxyz_to_R(q: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    wxyz = np.asarray(q, dtype=np.float64).reshape(4)
    q_xyzw = np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])
    return Rotation.from_quat(q_xyzw).as_matrix().astype(np.float32)


@dataclass
class CuroboPlanResult:
    success: bool
    poses: List[PosePair]
    info: Dict[str, Any]


_planner: Any = None
_planner_key: Optional[Tuple[str, str]] = None


def curobo_runtime_available() -> bool:
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        import curobo  # noqa: F401

        return True
    except Exception:
        return False


def _get_motion_planner(robot_yaml: str, scene_yaml: str) -> Any:
    global _planner, _planner_key
    key = (robot_yaml, scene_yaml)
    if _planner is not None and _planner_key == key:
        return _planner

    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg

    cfg = MotionPlannerCfg.create(robot=robot_yaml, scene_model=scene_yaml)
    mp = MotionPlanner(cfg)
    mp.warmup(enable_graph=True, num_warmup_iterations=3)
    _planner = mp
    _planner_key = key
    return mp


def _goal_tool_pose_from_Rt(planner: Any, position: np.ndarray, R: np.ndarray) -> Any:
    from curobo.types import GoalToolPose, Pose
    import torch

    device = planner.default_joint_state.position.device
    dtype = planner.default_joint_state.position.dtype
    T = torch.eye(4, device=device, dtype=dtype)
    T[:3, :3] = torch.as_tensor(R, device=device, dtype=dtype)
    T[:3, 3] = torch.as_tensor(position, device=device, dtype=dtype)
    ee_pose = Pose.from_matrix(T.unsqueeze(0))
    frames = list(planner.kinematics.tool_frames)
    pose_dict = {frames[0]: ee_pose}
    return GoalToolPose.from_poses(pose_dict, ordered_tool_frames=frames, num_goalset=1)


def _joint_state_at_horizon(interp: Any, index: int) -> Any:
    from curobo._src.state.state_joint_trajectory_ops import get_joint_state_at_horizon_index

    return get_joint_state_at_horizon_index(interp, index)


def plan_trajectory(
    start_pose: PosePair,
    goal_pose: PosePair,
    *,
    robot: str = "franka.yml",
    scene: str = "collision_table.yml",
    verbose: bool = False,
) -> CuroboPlanResult:
    """
    Plan a collision-aware trajectory in joint space and return EE poses via FK.

    Parameters
    ----------
    start_pose :
        ``(position (3,), rotation (3x3))`` in the planning frame (camera frame).
    goal_pose :
        GT grasp ``(position (3,), rotation (3x3))``.

    Returns
    -------
    CuroboPlanResult
        ``poses[k]`` is ``(position (3,), rotation (3x3))`` for each waypoint.
    """
    info: Dict[str, Any] = {"backend": "curobo", "robot": robot, "scene": scene}
    empty = CuroboPlanResult(success=False, poses=[], info=info)

    if not curobo_runtime_available():
        info["reason"] = "curobo_unavailable_or_no_cuda"
        if verbose:
            print("[curobo_planner] skip: cuRobo not importable or CUDA not available")
        return empty

    p0, R0 = np.asarray(start_pose[0], dtype=np.float32).reshape(3), np.asarray(start_pose[1], dtype=np.float32).reshape(3, 3)
    p1, R1 = np.asarray(goal_pose[0], dtype=np.float32).reshape(3), np.asarray(goal_pose[1], dtype=np.float32).reshape(3, 3)

    try:
        import torch
        from curobo.types import JointState

        planner = _get_motion_planner(robot, scene)
        device = planner.default_joint_state.position.device
        dtype = planner.default_joint_state.position.dtype

        goal_start = _goal_tool_pose_from_Rt(planner, p0, R0)
        goal_end = _goal_tool_pose_from_Rt(planner, p1, R1)

        ik_start = planner.ik_solver.solve_pose(
            goal_start,
            current_state=planner.default_joint_state.unsqueeze(0),
            return_seeds=planner.trajopt_solver.config.num_seeds,
        )
        if ik_start.success is None or torch.count_nonzero(ik_start.success) == 0:
            info["reason"] = "ik_start_failed"
            if verbose:
                print("[curobo_planner] IK failed for start pose")
            return empty

        sol_ok = ik_start.solution[ik_start.success]
        if sol_ok.shape[0] == 0:
            info["reason"] = "ik_start_empty_solution"
            return empty
        q_start_pos = sol_ok[0].reshape(-1)

        js_start = JointState.from_position(q_start_pos.unsqueeze(0), joint_names=planner.joint_names)

        result = planner.plan_pose(goal_end, js_start, max_attempts=5, enable_graph_attempt=1)
        if result is None or not bool(result.success.any()):
            info["reason"] = "plan_pose_failed"
            if verbose:
                print("[curobo_planner] plan_pose failed")
            return empty

        interp = result.get_interpolated_plan()
        if interp is None or interp.position is None:
            info["reason"] = "no_interpolated_plan"
            return empty

        H = int(interp.position.shape[-2])
        frame = planner.tool_frames[0]
        out: List[PosePair] = []
        for i in range(H):
            js_i = _joint_state_at_horizon(interp, i)
            q = js_i.position
            if q.ndim == 1:
                q = q.unsqueeze(0)
            js_i = JointState.from_position(q, joint_names=js_i.joint_names)
            kin = planner.compute_kinematics(js_i)
            tp = kin.tool_poses.get_link_pose(frame)
            pos = tp.position[0].detach().cpu().numpy().astype(np.float32).reshape(3)
            quat = tp.quaternion[0].detach().cpu().numpy().astype(np.float32).reshape(4)
            R_out = _quat_wxyz_to_R(quat)
            out.append((pos, R_out))

        info["trajectory_length"] = len(out)
        info["reason"] = "ok"
        if verbose:
            print(f"[curobo_planner] success: {len(out)} waypoints")
        return CuroboPlanResult(success=True, poses=out, info=info)
    except Exception as e:
        info["reason"] = f"exception:{type(e).__name__}"
        info["error"] = str(e)
        if verbose:
            print(f"[curobo_planner] exception: {e}")
        return empty
