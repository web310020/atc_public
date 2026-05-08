#!/usr/bin/env python
"""
K=3 baseline pool 扩展: 在 K=3 / 50 seed 下训练 FullInfo-PPO (D=0 reference)
和 SafeSlice, 把 K=3 baseline pool 补全到跟 K=1 一致.

Modes:
  - oracle    = FullInfo-PPO (D=0 telemetry; tactic 周期 10 ms;
                Dirichlet PPO + beta=200)
  - safeslice = SafeSlice constraint-aware baseline (共享 Dirichlet trunk)

直接复用 run_k3_experiment.py 的 run_one(), 这样 ATC vs baseline 对比
跟主 pipeline 完全一致 (同一个 Dirichlet head, baseline 上把 per-slice
Lagrangian 关掉, env 和 template 也一样).

Usage:
    # 默认: 2 mode x 50 seed = 100 jobs, 8 worker, 每 job 120K step
    python run_k3_baseline_pool_overnight.py --workers 8 \
        --output experiments/<run_name>/

    # Smoke test (5 seed, 30K step, wall ~5-10 min)
    python run_k3_baseline_pool_overnight.py --seed-end 4 --steps 30000 \
        --workers 4 --output experiments/<smoke>/

    # Run only the oracle (FullInfo-PPO)
    python run_k3_baseline_pool_overnight.py --modes oracle

Hardware: 1 CPU process per worker, PyTorch CPU mode (no GPU). 100 jobs
across 8 workers ~ 1-2 h wall, ~10-15 GB RAM.
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

# Reuse canonical single-seed runner
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
    # - ATC ('proposed') uses lagrangian + reward-norm (per-slice safety)
    # - All baselines (vanilla_ppo, oracle, safeslice, etc.) use neither (clean baseline)
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
        description="K=3 baseline pool expansion: FullInfo-PPO (oracle) + SafeSlice "
                    "@ 50 seeds, parallel workers (default 8)")
    p.add_argument("--seed-start", type=int, default=0,
                   help="seed range start (inclusive); default 0")
    p.add_argument("--seed-end", type=int, default=49,
                   help="seed range end (inclusive); default 49 (50 seeds total)")
    p.add_argument("--workers", type=int, default=8,
                   help="parallel worker count (default 8). Each worker = 1 process.")
    p.add_argument("--template", choices=["A", "B", "C"], default="A",
                   help="K=3 env template; A is the canonical heterogeneous mix "
                        "(URLLC/eMBB/mMTC) matching ATC seed runs")
    p.add_argument("--steps", type=int, default=120_000,
                   help="PPO training steps per seed per mode (default 120K)")
    p.add_argument("--use-kalman", action="store_true",
                   help="use Kalman belief engine (default: 3-state discrete, "
                        "matches K=1 ATC + K=3 ATC canonical runs)")
    p.add_argument("--modes", nargs="+",
                   default=["oracle", "safeslice"],
                   choices=["oracle", "safeslice", "vanilla_ppo", "proposed",
                            "static_slicing", "guardrail_only", "lstm_predictive"],
                   help="baseline modes to train (default: oracle + safeslice)")
    p.add_argument("--output", default=None,
                   help="output dir (default: experiments/k3_baseline_pool_<modes>_<timestamp>/)")
    p.add_argument("--lr-dual", type=float, default=1e-3,
                   help="per-slice Lagrangian lr (matches run_k3_experiment.py default; "
                        "unused for non-proposed modes)")
    p.add_argument("--alpha-floor", type=float, default=1.0,
                   help="Dirichlet alpha floor (matches run_k3_experiment.py default)")
    p.add_argument("--sla-budget", type=str, default=None,
                   help="comma-separated per-slice budget override, e.g. '0.03,0.05,0.10'; "
                        "unused for non-proposed modes")
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
    modes_tag = "_".join(sorted(args.modes))
    out_dir = Path(args.output or f"experiments/k3_baseline_pool_{modes_tag}_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    n_jobs = n_seeds * len(args.modes)
    print("=" * 70)
    print(f"K=3 BASELINE POOL EXPANSION (parallel workers={args.workers})")
    print("=" * 70)
    print(f"Output dir       : {out_dir}")
    print(f"Seed range       : {args.seed_start}..{args.seed_end}  ({n_seeds} seeds)")
    print(f"Modes            : {args.modes}")
    print(f"Total jobs       : {n_jobs}  ({n_seeds} seeds × {len(args.modes)} modes)")
    print(f"Steps per job    : {args.steps:,}")
    print(f"Template         : {args.template}  (heterogeneous URLLC/eMBB/mMTC)")
    print(f"Workers          : {args.workers}  (multiprocessing.Pool, spawn)")
    print(f"Use Kalman       : {args.use_kalman}")
    print("=" * 70)
    print("Estimated wall (CPU, ~5-10 min/job):")
    print(f"  optimistic: {n_jobs * 5 / args.workers:.0f} min ≈ {n_jobs * 5 / args.workers / 60:.1f} h")
    print(f"  pessimistic: {n_jobs * 10 / args.workers:.0f} min ≈ {n_jobs * 10 / args.workers / 60:.1f} h")
    print("=" * 70)

    run_cfg = {
        "timestamp": ts,
        "args": vars(args),
        "n_jobs": n_jobs,
        "purpose": ("K=3 baseline pool expansion: adds FullInfo-PPO "
                    "(zero-delay reference) and SafeSlice (safety lower bound) "
                    "to match the K=1 baseline pool."),
        "filed_at": datetime.now(timezone.utc).isoformat(),
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
        "wall_clock_hours": round(wall / 3600.0, 2),
        "by_mode": {m: {"n": len(v), "n_stable": sum(1 for x in v if x["stable"])}
                    for m, v in by_mode.items()},
        "next_step": (
            f"Aggregate this run with the existing K=3 ATC/Vanilla runs via "
            f"scripts/aggregate_belief_experiments.py to produce the K=3 "
            f"main-results table (50-seed Welch t-test, per-slice violation "
            f"rates, aggregate U with CI)."
        ),
    }
    (out_dir / "baseline_pool_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"DONE. Wall clock: {wall:.1f}s ({wall / 60.0:.1f} min, {wall / 3600.0:.2f} h)")
    for mode, info in summary["by_mode"].items():
        print(f"  {mode}: {info['n_stable']}/{info['n']} stable seeds")
    print(f"\nResults: {out_dir}/")
    print(f"Aggregate: {out_dir}/baseline_pool_summary.json")
    print("=" * 70)
    print("\nNext step:")
    print("  Run scripts/aggregate_belief_experiments.py to merge this run")
    print("  with the existing K=3 ATC/Vanilla runs and produce the K=3")
    print("  main-results table.")
    print()
    return 0


if __name__ == "__main__":
    # PyTorch + multiprocessing requires spawn start method
    set_start_method("spawn", force=True)
    sys.exit(main())
