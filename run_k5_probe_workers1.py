#!/usr/bin/env python
"""
K=5 probe 并行 runner.

验证 ATC 机制能不能从 K=3 进一步扩到 K=5 多 slice 设置.

复用 run_k3_experiment.py 的 run_one() (代码本身 K-agnostic,
k3_env.py 里 K = len(cfg["names"])). 在 TEMPLATES 里加了 K5_A
template (URLLC + V2X + eMBB + mMTC + IoT_burst).

并行: multiprocessing.Pool(workers), spawn 启动.

推荐 worker 数: 2-3. K=5 的 action space 比 K=3 略大 (+10-20% 内存),
--workers 1 容易 OOM, 不行就先降到 2-3.

Usage:
    conda activate belief_telemetry_env

    # Default: K=5 probe with 10 seeds (0..9), workers=2, template K5_A,
    # 120K steps per seed/mode (matches K=3 main run cadence)
    python run_k5_probe_workers1.py \\
        --seed-start 0 --seed-end 9 \\
        --workers 2 \\
        --template K5_A \\
        --steps 120000 \\
        --output experiments/k5_probe_v1/

    # Larger probe (more credibility, more compute):
    python run_k5_probe_workers1.py --seed-end 19 --workers 3

    # Sanity-only quick test (less steps, fewer seeds):
    python run_k5_probe_workers1.py --seed-end 3 --steps 30000 --workers 2 \\
        --output experiments/k5_sanity/

Output structure:
    experiments/k5_probe_v1/
        runs/proposed_seed0/...
        runs/proposed_seed1/...
        ...
        runs/vanilla_ppo_seed9/...
        k5_probe_summary.json    <- aggregate
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

from run_k3_experiment import run_one  # noqa: E402


def _worker_fn(job: dict) -> dict:
    mode = job["mode"]
    template = job["template"]
    seed = job["seed"]
    steps = job["steps"]
    use_kalman = job["use_kalman"]
    out_dir = Path(job["out_dir"])

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
        description="K=5 probe parallel runner")
    p.add_argument("--seed-start", type=int, default=0,
                   help="seed range start (inclusive); default 0")
    p.add_argument("--seed-end", type=int, default=9,
                   help="seed range end (inclusive); default 9 (yields 10 seeds)")
    p.add_argument("--workers", type=int, default=2,
                   help="parallel worker count (default 2; 2-3 recommended on a single CPU)")
    p.add_argument("--template", default="K5_A",
                   help="K=5 template name (default K5_A: URLLC+V2X+eMBB+mMTC+IoT_burst)")
    p.add_argument("--steps", type=int, default=120_000,
                   help="PPO steps PER seed PER mode (default 120K, matches K=3 main run)")
    p.add_argument("--use-kalman", action="store_true",
                   help="use Kalman belief engine (default: 3-state discrete)")
    p.add_argument("--modes", nargs="+",
                   default=["proposed", "vanilla_ppo"],
                   choices=["proposed", "vanilla_ppo", "oracle", "safeslice",
                            "static_slicing", "guardrail_only", "lstm_predictive"],
                   help="modes to train (default: ATC + Vanilla-PPO for paired comparison)")
    p.add_argument("--output", default=None,
                   help="output dir (default: experiments/k5_probe_<timestamp>/)")
    p.add_argument("--lr-dual", type=float, default=1e-3,
                   help="per-slice Lagrangian lr (matches K=3 default)")
    p.add_argument("--alpha-floor", type=float, default=1.0,
                   help="Dirichlet alpha floor (matches K=3 default)")
    p.add_argument("--sla-budget", type=str, default=None,
                   help="comma-separated 5-slice budget override, e.g. '0.005,0.01,0.05,0.10,0.15'")
    args = p.parse_args()

    if args.seed_end < args.seed_start:
        print(f"!!! --seed-end ({args.seed_end}) < --seed-start ({args.seed_start})")
        return 1
    n_seeds = args.seed_end - args.seed_start + 1
    if n_seeds < args.workers:
        print(f"!! Warning: {n_seeds} seeds < {args.workers} workers; "
              f"reducing workers to {n_seeds}.")
        args.workers = n_seeds

    sla_budget_override = None
    if args.sla_budget:
        try:
            sla_budget_override = [float(x) for x in args.sla_budget.split(",")]
            assert len(sla_budget_override) == 5
        except Exception as e:
            print(f"!!! Bad --sla-budget '{args.sla_budget}' (expected 5 comma-sep floats): {e}")
            return 1

    tz = timezone(timedelta(hours=9))
    ts = datetime.now(tz).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output or f"experiments/k5_probe_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    n_jobs = n_seeds * len(args.modes)
    print("=" * 60)
    print(f"K=5 PROBE (parallel workers={args.workers})")
    print("=" * 60)
    print(f"Output dir       : {out_dir}")
    print(f"Seed range       : {args.seed_start}..{args.seed_end}  ({n_seeds} seeds)")
    print(f"Modes            : {args.modes}")
    print(f"Total jobs       : {n_jobs}  ({n_seeds} seeds x {len(args.modes)} modes)")
    print(f"Steps per job    : {args.steps:,}")
    print(f"Template         : {args.template}  (5 slices: URLLC/V2X/eMBB/mMTC/IoT_burst)")
    print(f"Workers          : {args.workers}  (multiprocessing.Pool, spawn)")
    print("=" * 60)

    run_cfg = {
        "timestamp": ts,
        "args": vars(args),
        "n_jobs": n_jobs,
        "purpose": "K=5 probe to validate scaling beyond K=3",
        "K": 5,
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
        "K": 5,
        "template": args.template,
        "seed_range_inclusive": [args.seed_start, args.seed_end],
        "n_seeds": n_seeds,
        "n_jobs": n_jobs,
        "workers": args.workers,
        "wall_clock_seconds": round(wall, 1),
        "wall_clock_minutes": round(wall / 60.0, 1),
        "by_mode": {m: {"n": len(v), "n_stable": sum(1 for x in v if x["stable"])}
                    for m, v in by_mode.items()},
    }
    (out_dir / "k5_probe_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"DONE. Wall clock: {wall:.1f}s ({wall / 60.0:.1f} min)")
    for mode, info in summary["by_mode"].items():
        print(f"  {mode}: {info['n_stable']}/{info['n']} stable seeds")
    print(f"\nResults: {out_dir}/")
    print(f"Aggregate: {out_dir}/k5_probe_summary.json")
    print("=" * 60)
    print("\nNext step: paper integration as 'Preliminary K=5 finding'")
    print("  -> 1-paragraph addition to Sec V.G or Conclusion;")
    print("  -> upgrades 'Future work K>3' -> 'demonstrated K=5 transfer'.")
    return 0


if __name__ == "__main__":
    set_start_method("spawn", force=True)
    sys.exit(main())
