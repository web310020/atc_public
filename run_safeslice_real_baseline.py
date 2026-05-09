"""
SafeSlice (CPO-style) baseline runner @ K=3 + K=5.

单全局 lambda 在 aggregate (mean) per-slice violation 上 dual ascent —
和 ATC 的 per-slice mu_k 互斥. K=3 50 seeds + K=5 30 seeds, 各 120K steps.

用法: python run_safeslice_real_baseline.py --workers 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from multiprocessing import Pool, set_start_method
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_k3_experiment import run_one  # noqa: E402


def _worker_fn(job: dict) -> dict:
    return run_one(
        mode="safeslice",
        template=job["template"],
        seed=job["seed"],
        total_steps=job["steps"],
        use_kalman=False,
        out_dir=Path(job["out_dir"]),
        use_lagrangian=False,        # 关 per-slice (ATC 机制)
        safeslice_mode=True,         # 开 single-lambda global Lagrangian
        normalize_per_slice_rewards=False,
        lr_dual=job.get("lr_dual", 1e-3),
        alpha_floor=job.get("alpha_floor", 1.0),
    )


def _run_one_K(K_label: str, template: str, seeds: list, steps: int,
               workers: int, parent_out: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = parent_out / f"safeslice_real_{K_label}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Real SafeSlice baseline — {K_label.upper()}")
    print("=" * 70)
    print(f"  Template:    {template}")
    print(f"  Seeds:       {len(seeds)} (range {seeds[0]}-{seeds[-1]})")
    print(f"  Steps/job:   {steps:,}")
    print(f"  Workers:     {workers}")
    print(f"  Output:      {out_dir}")
    print("=" * 70)

    config = {
        "experiment": f"safeslice_real_{K_label}",
        "purpose": "SafeSlice (CPO-style) reproduction with single global lambda",
        "template": template,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "steps_per_training": steps,
        "workers": workers,
        "started_at": datetime.now().isoformat(),
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    job_args = [{"template": template, "seed": s, "steps": steps,
                 "out_dir": str(out_dir)} for s in seeds]

    n_workers = min(workers, len(job_args))
    t0 = time.time()
    if n_workers > 1:
        try:
            set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        with Pool(processes=n_workers) as pool:
            results = pool.map(_worker_fn, job_args)
    else:
        results = [_worker_fn(j) for j in job_args]
    wall = time.time() - t0
    print(f"  {K_label} train+eval done: {wall/60:.1f} min")

    # Aggregate
    n_stable = sum(1 for r in results if r.get("status") == "ok" and r.get("stable", False))
    n_error = sum(1 for r in results if r.get("status") != "ok")
    summary = {
        "timestamp": timestamp,
        "K_label": K_label,
        "template": template,
        "n_seeds": len(seeds),
        "n_stable": n_stable,
        "n_error": n_error,
        "wall_clock_seconds": wall,
        "wall_clock_minutes": wall / 60.0,
    }
    (out_dir / "safeslice_real_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  {K_label}: {n_stable}/{len(seeds)} stable, {n_error} errored")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Real SafeSlice baseline at K=3 + K=5")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=120_000)
    parser.add_argument("--k3-seeds", type=int, default=50, help="K=3 seeds (default 50)")
    parser.add_argument("--k5-seeds", type=int, default=30, help="K=5 seeds (default 30)")
    parser.add_argument("--k3-only", action="store_true")
    parser.add_argument("--k5-only", action="store_true")
    args = parser.parse_args()

    parent_out = Path("experiments")
    parent_out.mkdir(exist_ok=True)

    t_total = time.time()
    if not args.k5_only:
        _run_one_K("k3", "A", list(range(args.k3_seeds)),
                   args.steps, args.workers, parent_out)
    if not args.k3_only:
        _run_one_K("k5", "K5_A", list(range(args.k5_seeds)),
                   args.steps, args.workers, parent_out)
    total_wall = time.time() - t_total
    print(f"\n[ALL DONE] Total wall: {total_wall/60:.1f} min")


if __name__ == "__main__":
    set_start_method("spawn", force=True)
    main()
