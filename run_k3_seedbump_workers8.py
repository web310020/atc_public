#!/usr/bin/env python
"""
K=3 seed-bump 并行 runner.

在已有的 K=3 主 run 基础上补 seed (默认 30..49), 用来收紧 worst-slice
violation rate 和 aggregate U gain 的 confidence interval.

直接复用 run_k3_experiment.py 的 run_one(), 这样 ATC vs Vanilla 对比
跟主 pipeline 保持一致.

并行: multiprocessing.Pool(workers), 用 'spawn' 启动方式
(PyTorch 在 fork-safety 限制下必须用 spawn).

Usage:
    # 默认: seeds 30..49, workers=8, template A, 120K steps/seed/mode
    python run_k3_seedbump_workers8.py \
        --seed-start 30 --seed-end 49 \
        --workers 8 \
        --template A \
        --steps 120000 \
        --output experiments/<run_name>/

    # 只跑 ATC:
    python run_k3_seedbump_workers8.py --modes proposed --workers 8

After completion, aggregate with the existing run via
scripts/aggregate_seed_bump.py (or merge results.json manually).

Hardware notes:
- Each worker spawns its own PyTorch process (~2-4 GB GPU memory per ATC,
  ~1-2 GB per Vanilla). With workers=8 ensure >= 16 GB GPU available, or
  run on CPU (slower but no memory ceiling).
- For CPU-only: set CUDA_VISIBLE_DEVICES="" before invocation.
- For GPU split across workers: PyTorch will auto-share device by default; if
  conflict, set --workers 4 or 6.

Output structure (matches run_k3_experiment.py convention):
    experiments/k3_r4_seedbump_30to49/
        runs/proposed_seed30/...
        runs/proposed_seed31/...
        ...
        runs/vanilla_ppo_seed49/...
        seedbump_summary.json    ← aggregate of this batch
        run_config.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from multiprocessing import get_context, set_start_method
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the canonical single-seed runner
from run_k3_experiment import run_one  # noqa: E402


def _worker_fn(job: dict) -> dict:
    """Run a single (mode, seed) job. Called inside a Pool worker process."""
    mode = job["mode"]
    template = job["template"]
    seed = job["seed"]
    steps = job["steps"]
    use_kalman = job["use_kalman"]
    out_dir = Path(job["out_dir"])

    # Match run_k3_experiment.py main() defaults:
    # - ATC ('proposed') uses lagrangian + reward-norm
    # - Vanilla-PPO uses neither (clean baseline)
    # - Other modes (oracle/safeslice/etc.) follow vanilla settings
    if mode == "proposed":
        use_lagrangian = True
        normalize_per_slice_rewards = True
    else:
        use_lagrangian = False
        normalize_per_slice_rewards = False

    return run_one(
        mode=mode,
        template=template,
        seed=seed,
        total_steps=steps,
        use_kalman=use_kalman,
        out_dir=out_dir,
        use_lagrangian=use_lagrangian,
        normalize_per_slice_rewards=normalize_per_slice_rewards,
        lr_dual=job.get("lr_dual", 1e-3),
        alpha_floor=job.get("alpha_floor", 1.0),
        sla_budget_override=job.get("sla_budget_override", None),
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description="K=3 seed-bump parallel runner (workers=8 by default)")
    p.add_argument("--seed-start", type=int, default=30,
                   help="seed range start (inclusive); default 30 (extends existing 0..29)")
    p.add_argument("--seed-end", type=int, default=49,
                   help="seed range end (inclusive); default 49 (yields 20 new seeds)")
    p.add_argument("--workers", type=int, default=8,
                   help="parallel worker count (default 8). Each worker = 1 process.")
    p.add_argument("--template", choices=["A", "B", "C"], default="A",
                   help="K=3 env template; A is the canonical heterogeneous mix")
    p.add_argument("--steps", type=int, default=120_000,
                   help="PPO training steps PER seed PER mode (default 120K, matching paper)")
    p.add_argument("--use-kalman", action="store_true",
                   help="use Kalman belief engine (default: 3-state discrete, matches K=1 ATC)")
    p.add_argument("--modes", nargs="+",
                   default=["proposed", "vanilla_ppo"],
                   choices=["proposed", "vanilla_ppo", "oracle", "safeslice",
                            "static_slicing", "guardrail_only", "lstm_predictive"],
                   help="modes to train (default: ATC + Vanilla-PPO for paired comparison)")
    p.add_argument("--output", default=None,
                   help="output dir (default: experiments/k3_seedbump_<timestamp>/)")
    p.add_argument("--lr-dual", type=float, default=1e-3,
                   help="per-slice Lagrangian lr (matches run_k3_experiment.py default)")
    p.add_argument("--alpha-floor", type=float, default=1.0,
                   help="Dirichlet alpha floor (matches run_k3_experiment.py default)")
    p.add_argument("--sla-budget", type=str, default=None,
                   help="comma-separated per-slice budget override, e.g. '0.03,0.05,0.10'")
    args = p.parse_args()

    if args.seed_end < args.seed_start:
        print(f"!!! --seed-end ({args.seed_end}) < --seed-start ({args.seed_start})")
        return 1
    n_seeds = args.seed_end - args.seed_start + 1
    if n_seeds < args.workers:
        print(f"!! Warning: {n_seeds} seeds < {args.workers} workers; "
              f"will under-utilize. Reducing workers to {n_seeds}.")
        args.workers = n_seeds

    sla_budget_override = None
    if args.sla_budget:
        try:
            sla_budget_override = [float(x) for x in args.sla_budget.split(",")]
            assert len(sla_budget_override) == 3
        except Exception as e:
            print(f"!!! Bad --sla-budget '{args.sla_budget}': {e}")
            return 1

    tz = timezone(timedelta(hours=9))
    ts = datetime.now(tz).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output or f"experiments/k3_seedbump_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    n_jobs = n_seeds * len(args.modes)
    print("=" * 60)
    print(f"K=3 SEED BUMP (parallel workers={args.workers})")
    print("=" * 60)
    print(f"Output dir       : {out_dir}")
    print(f"Seed range       : {args.seed_start}..{args.seed_end}  ({n_seeds} seeds)")
    print(f"Modes            : {args.modes}")
    print(f"Total jobs       : {n_jobs}  ({n_seeds} seeds × {len(args.modes)} modes)")
    print(f"Steps per job    : {args.steps:,}")
    print(f"Template         : {args.template}")
    print(f"Workers          : {args.workers}  (multiprocessing.Pool, spawn)")
    print("=" * 60)

    run_cfg = {
        "timestamp": ts,
        "args": vars(args),
        "n_jobs": n_jobs,
        "purpose": "K=3 seed-bump for tighter confidence intervals",
    }
    (out_dir / "run_config.json").write_text(
        json.dumps(run_cfg, indent=2), encoding="utf-8")

    jobs = []
    for mode in args.modes:
        for seed in range(args.seed_start, args.seed_end + 1):
            jobs.append({
                "mode": mode,
                "template": args.template,
                "seed": seed,
                "steps": args.steps,
                "use_kalman": args.use_kalman,
                "out_dir": str(out_dir),
                "lr_dual": args.lr_dual,
                "alpha_floor": args.alpha_floor,
                "sla_budget_override": sla_budget_override,
            })

    # Dispatch via multiprocessing.Pool with spawn start method
    # (PyTorch + CUDA require spawn or forkserver, NOT fork)
    t0 = time.time()
    ctx = get_context("spawn")
    with ctx.Pool(processes=args.workers) as pool:
        results = pool.map(_worker_fn, jobs)
    wall = time.time() - t0

    by_mode = {}
    for job, r in zip(jobs, results):
        by_mode.setdefault(job["mode"], []).append({
            "seed": job["seed"],
            "stable": r.get("stable") if isinstance(r, dict) else None,
            "summary": r,
        })

    summary = {
        "timestamp": ts,
        "seed_range_inclusive": [args.seed_start, args.seed_end],
        "n_seeds": n_seeds,
        "n_jobs": n_jobs,
        "workers": args.workers,
        "wall_clock_seconds": round(wall, 1),
        "wall_clock_minutes": round(wall / 60.0, 1),
        "by_mode": {m: {"n": len(v), "n_stable": sum(1 for x in v if x["stable"])}
                    for m, v in by_mode.items()},
    }
    (out_dir / "seedbump_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"DONE. Wall clock: {wall:.1f}s ({wall / 60.0:.1f} min)")
    for mode, info in summary["by_mode"].items():
        print(f"  {mode}: {info['n_stable']}/{info['n']} stable seeds")
    print(f"\nResults: {out_dir}/")
    print(f"Aggregate: {out_dir}/seedbump_summary.json")
    print("=" * 60)
    print("\nNext step: merge with existing experiments/k3_r2_01_templateA_main/")
    print("  to compute joint statistics over 50 seeds total.")
    return 0


if __name__ == "__main__":
    # PyTorch + multiprocessing requires spawn start method
    set_start_method("spawn", force=True)
    sys.exit(main())
