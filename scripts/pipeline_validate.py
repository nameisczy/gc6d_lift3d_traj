#!/usr/bin/env python3
"""
End-to-end numerical pipeline check: dataset index → sanity → train → eval → PASS/FAIL line.

Does not start OMPL/MoveIt. Optional GC6D eval_grasp via --run-gc6d-evaluator.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _torch_import_ok(exe: str) -> bool:
    """
    Test `import torch` in a subprocess — never import torch in this script's interpreter.
    (A broken ~/.local torch can crash or raise AttributeError on import.)
    """
    env = {**os.environ, "PYTHONNOUSERSITE": "1"}
    try:
        r = subprocess.run(
            [exe, "-c", "import torch; print(torch.__version__)"],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def _python_with_torch() -> str:
    """Pick an interpreter where `import torch` succeeds (train/eval need torch)."""
    candidates: List[str] = []
    envp = os.environ.get("TORCH_PYTHON") or os.environ.get("GC6D_LIFT3D_PYTHON")
    if envp:
        candidates.append(envp)
    for cand in (
        Path.home() / "miniconda3/envs/gc6d/bin/python",
        Path.home() / "anaconda3/envs/gc6d/bin/python",
    ):
        if cand.is_file():
            candidates.append(str(cand))
    candidates.append(sys.executable)

    seen: set[str] = set()
    for exe in candidates:
        if not exe or exe in seen:
            continue
        seen.add(exe)
        if not Path(exe).is_file():
            continue
        if _torch_import_ok(exe):
            return exe

    print(
        "ERROR: No Python found where `import torch` works. Fix broken user-site torch "
        "(often ~/.local) or set TORCH_PYTHON=/path/to/conda/env/bin/python.",
        file=sys.stderr,
    )
    sys.exit(2)


def _count_index_episodes(index_path: Path) -> int:
    n = 0
    if not index_path.is_file():
        return 0
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _run_capture(cmd: List[str], cwd: Path, env: Dict[str, str] | None = None) -> Tuple[int, str]:
    merged = {**os.environ}
    if env:
        merged.update(env)
    # Avoid ~/.local shadowing conda torch (same issue as _torch_import_ok).
    merged["PYTHONNOUSERSITE"] = "1"
    pp0 = merged.get("PYTHONPATH", "")
    merged["PYTHONPATH"] = str(cwd) if not pp0 else str(cwd) + os.pathsep + pp0
    p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, env=merged)
    out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
    return p.returncode, out


def _parse_json_line(prefix: str, text: str) -> Dict[str, Any]:
    for line in text.splitlines():
        if line.startswith(prefix):
            return json.loads(line[len(prefix) :].strip())
    raise ValueError(f"missing {prefix} in command output")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--config", type=str, default="configs/train_traj.yaml")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument(
        "--skip-loss-decrease-check",
        action="store_true",
        help="Do not require mean(total loss) last epoch < first (default: enforced when epochs>=2)",
    )
    ap.add_argument("--max-samples", type=int, default=20, help="Only used when --run-build")
    ap.add_argument("--run-build", action="store_true", help="Run build_gc6d_lift3d_dataset.py first")
    ap.add_argument("--run-gc6d-evaluator", action="store_true", help="Slow; needs GraspClutter6D + DexNet stack")
    ap.add_argument("--gc6d-root", type=str, default="/mnt/ssd/ziyaochen/GraspClutter6D")
    args = ap.parse_args()

    root = _repo_root()
    data_root = Path(args.data_root)
    index_path = data_root / "index" / "index_train.jsonl"
    py = _python_with_torch()
    print(f"pipeline_validate: using Python: {py}")

    checks: List[Tuple[str, bool, str]] = []

    if args.run_build:
        build_cmd = [
            py,
            str(root / "scripts" / "build_gc6d_lift3d_dataset.py"),
            "--output-root",
            str(data_root),
            "--max-samples",
            str(args.max_samples),
            "--resume",
        ]
        code, out = _run_capture(build_cmd, root)
        checks.append(("dataset_build_exit0", code == 0, f"exit={code}"))
        if code != 0:
            print(out[-4000:])

    n_ep = _count_index_episodes(index_path)
    checks.append(("episodes_gt_0", n_ep > 0, f"n={n_ep} index={index_path}"))

    sanity_cmd = [py, str(root / "scripts" / "sanity_check_episodes.py"), "--data-root", str(data_root)]
    code, san_out = _run_capture(sanity_cmd, root)
    checks.append(("sanity_exit0", code == 0, f"exit={code}"))
    m = re.search(r"ok=(\d+)", san_out)
    n_ok = int(m.group(1)) if m else 0
    checks.append(("sanity_ok_gt_0", n_ok > 0, f"ok={n_ok}"))

    train_cmd = [
        py,
        str(root / "scripts" / "train_lift3d_gc6d.py"),
        "--config",
        str(root / args.config) if not Path(args.config).is_absolute() else args.config,
        "--data-root",
        str(data_root),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--print-json-summary",
    ]
    code, train_out = _run_capture(train_cmd, root)
    checks.append(("train_exit0", code == 0, f"exit={code}"))
    train_summary: Dict[str, Any] = {}
    loss_ok = True
    if code == 0:
        try:
            train_summary = _parse_json_line("TRAIN_SUMMARY_JSON:", train_out)
            losses: List[float] = train_summary.get("epoch_mean_total_loss") or []
            if not args.skip_loss_decrease_check:
                if len(losses) >= 2:
                    loss_ok = losses[-1] < losses[0]
                elif len(losses) == 0:
                    loss_ok = False
        except Exception as e:
            train_summary = {"error": str(e)}
            loss_ok = False
    else:
        train_summary = {}
        loss_ok = False
    check_loss_name = "loss_last_lt_first" if not args.skip_loss_decrease_check else "loss_check_skipped"
    checks.append((check_loss_name, loss_ok, json.dumps(train_summary.get("epoch_mean_total_loss", []))))

    eval_cmd = [
        py,
        str(root / "scripts" / "eval_lift3d_gc6d.py"),
        "--data-root",
        str(data_root),
        "--full-test-inference",
        "--dump-official-dir",
        str(data_root / "pred_17d" / "official_dump"),
        "--assert-full-coverage",
        "--run-official-eval-all",
        "--gc6d-root",
        args.gc6d_root,
        "--camera",
        "realsense-d415",
        "--split",
        "test",
        "--top-k",
        "50",
        "--json-summary",
    ]
    if args.run_gc6d_evaluator:
        eval_cmd.extend(["--run-gc6d-evaluator"])

    code, eval_out = _run_capture(eval_cmd, root)
    checks.append(("eval_exit0", code == 0, f"exit={code}"))

    eval_summary: Dict[str, Any] = {}
    geom_ok = True
    gc6d_ok = True
    coverage_ok = False
    ap_ok = False
    if code == 0:
        try:
            eval_summary = _parse_json_line("EVAL_SUMMARY_JSON:", eval_out)
            coverage_ok = bool(eval_summary.get("full_coverage", False))
            ge = eval_summary.get("gc6d_eval") or {}
            ap_ok = (
                float(ge.get("ap", 0.0)) > 0.0
                and float(ge.get("ap04", 0.0)) > 0.0
                and float(ge.get("ap08", 0.0)) > 0.0
            )
        except Exception as e:
            eval_summary = {"error": str(e)}
            geom_ok = False
            coverage_ok = False
            ap_ok = False
        g = eval_summary.get("gc6d_eval") or {}
        if args.run_gc6d_evaluator:
            gc6d_ok = bool(g.get("ok")) and not g.get("skipped")
        else:
            gc6d_ok = True
    checks.append(("inference_full_coverage", coverage_ok, json.dumps({"n_pred_files": eval_summary.get("n_pred_files"), "n_test_images": eval_summary.get("n_test_images")})))
    checks.append(("official_ap_nonzero", ap_ok, json.dumps((eval_summary.get("gc6d_eval") or {}))))
    checks.append(("geometry_thresholds", geom_ok, json.dumps({k: eval_summary.get(k) for k in ("center_err_mean", "rot_trace_mean", "warn_geometry")})))
    checks.append(("gc6d_evaluator", gc6d_ok, str(eval_summary.get("gc6d_eval", {}))))

    # AP / collision proxy: optional reporting
    ap_note = ""
    g = eval_summary.get("gc6d_eval") or {}
    if isinstance(g, dict) and g.get("mean_collision_free_rate") is not None:
        ap_note = f" eval_grasp_proxy_rate={g.get('mean_collision_free_rate')}"

    all_pass = all(c[1] for c in checks)
    print("\n=== PIPELINE VALIDATION REPORT ===")
    for name, ok, detail in checks:
        print(f"  [{('OK' if ok else 'FAIL')}] {name}: {detail}")
    if all_pass:
        print("\nPIPELINE VALIDATION PASSED" + ap_note)
        sys.exit(0)
    print("\nPIPELINE VALIDATION FAILED" + ap_note)
    sys.exit(1)


if __name__ == "__main__":
    main()
