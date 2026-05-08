"""
K=5 + 每 slice 自己的 violation threshold tau_k.

把 global tau 换成 mixed-criticality 设置:
    URLLC: 0.3 (最紧)
    V2X:   0.4
    eMBB:  0.5
    mMTC:  0.6
    IoT_burst: 0.5
每个 slice 的 violation rate 按它自己的 tau_k 算.

Usage:
    python core/run_k5_per_slice_tau.py --output <dir> --workers 2
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from multiprocessing import Pool, set_start_method
from pathlib import Path

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO

from core.telemetry_env import E2_Node_Simulator, SLICE_PROFILES


# K=5 slice configuration with per-slice τ_k
K5_SLICE_NAMES = ["URLLC", "V2X", "eMBB", "mMTC", "IoT_burst"]
K5_PROFILES = {
    "URLLC":     SLICE_PROFILES["URLLC"],
    "V2X":       SLICE_PROFILES["V2X"],
    "eMBB":      SLICE_PROFILES["eMBB"],
    "mMTC":      {"sigma": 0.04, "base_util": 0.25, "target_prb": 0.55, "label": "mMTC"},
    "IoT_burst": {"sigma": 0.10, "base_util": 0.20, "target_prb": 0.50, "label": "IoT_burst"},
}
# Per-slice τ_k: URLLC tightest (0.3), eMBB/IoT_burst medium (0.5), mMTC loosest (0.6)
PER_SLICE_TAU_DEFAULT = [0.3, 0.4, 0.5, 0.6, 0.5]


def make_env_per_slice_tau(per_slice_tau, seed=None):
    """Make K=5 ATC env with per-slice τ_k array."""
    env = E2_Node_Simulator(
        mode="proposed", K=5, tau=0.5, beta=200, use_kalman=False,
        slice_profiles=K5_PROFILES,
    )
    # Override tau_k with per-slice values
    env.tau_k = np.array(per_slice_tau, dtype=np.float64)
    if seed is not None:
        try:
            env.reset(seed=seed)
        except TypeError:
            np.random.seed(seed)
            env.reset()
    return env


def train_atc_per_slice_tau(per_slice_tau, output_dir, total_steps=120000, train_seed=42):
    """Train K=5 ATC with per-slice τ_k."""
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Training K=5 ATC with per-slice τ_k = {per_slice_tau}")
    print(f"  ({total_steps} steps, seed={train_seed})...")
    env = make_env_per_slice_tau(per_slice_tau)
    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4, batch_size=64, n_steps=2048, n_epochs=10,
        gamma=0.99, ent_coef=0.0,
        policy_kwargs={"net_arch": [512, 512, 256]},
        verbose=0, device='cpu', seed=train_seed,
    )
    t0 = time.time()
    model.learn(total_timesteps=total_steps, progress_bar=False)
    train_time = time.time() - t0
    print(f"  Training done: {train_time/60:.1f} min")

    model_path = models_dir / "ppo_atc_k5_per_slice_tau_final"
    model.save(str(model_path))
    return str(model_path), train_time


def eval_seed_worker(args):
    seed, model_path, per_slice_tau = args
    np.random.seed(seed)

    model = PPO.load(model_path, device='cpu')
    env = make_env_per_slice_tau(per_slice_tau, seed=seed)
    try:
        obs, _ = env.reset(seed=seed)
    except TypeError:
        obs, _ = env.reset()

    util_per_step = []
    viol_per_slice_per_step = [[] for _ in range(5)]
    cost_per_step = []

    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        util_per_step.append(info.get("true_util", 0.0))
        per_slice_viol = info.get("per_slice_violations", [0]*5)
        for k in range(5):
            viol_per_slice_per_step[k].append(per_slice_viol[k] if k < len(per_slice_viol) else 0)
        cost_per_step.append(info.get("sig_cost", 0))
        if done:
            obs, _ = env.reset()

    per_slice_viol_pct = [
        100.0 * float(np.mean(viol_per_slice_per_step[k])) for k in range(5)
    ]

    return {
        "seed": seed,
        "U_mean": float(np.mean(util_per_step)),
        "per_slice_viol_pct": per_slice_viol_pct,
        "worst_slice_viol_pct": max(per_slice_viol_pct),
        "all_5_below_2.5pct": all(v < 2.5 for v in per_slice_viol_pct),
        "cost_total": float(np.sum(cost_per_step)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--total-steps", type=int, default=120000)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--per-slice-tau", type=str, default=None,
                        help="Comma-separated per-slice τ_k (default: 0.3,0.4,0.5,0.6,0.5)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.per_slice_tau:
        per_slice_tau = [float(x) for x in args.per_slice_tau.split(",")]
    else:
        per_slice_tau = PER_SLICE_TAU_DEFAULT

    if len(per_slice_tau) != 5:
        print(f"ERROR: per-slice-tau must have 5 values, got {len(per_slice_tau)}")
        sys.exit(1)

    metadata = {
        "experiment": "k5_per_slice_tau",
        "K": 5,
        "slice_names": K5_SLICE_NAMES,
        "per_slice_tau": per_slice_tau,
        "per_slice_tau_rationale": "URLLC tighter (ultra-reliable); mMTC looser (low-priority)",
        "n_eval_seeds": args.seeds,
        "n_eval_steps_per_seed": 1000,
        "beta": 200,
        "training_total_steps": args.total_steps,
        "training_seed": args.train_seed,
        "purpose": "K=5 mixed-criticality eval with per-slice tau_k",
        "started_at": datetime.now().isoformat(),
    }
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    # Train
    print(f"\n[k5_per_slice_tau] Step 1: Train K=5 ATC with per-slice τ")
    model_path, train_time = train_atc_per_slice_tau(
        per_slice_tau, output_dir,
        total_steps=args.total_steps, train_seed=args.train_seed,
    )

    # Eval
    print(f"\n[k5_per_slice_tau] Step 2: Eval {args.seeds} seeds (workers={args.workers})")
    worker_args = [(seed, model_path, per_slice_tau) for seed in range(args.seeds)]
    n_workers = min(args.workers, args.seeds)
    t0 = time.time()
    if n_workers > 1:
        try:
            set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        with Pool(processes=n_workers) as pool:
            results = pool.map(eval_seed_worker, worker_args)
    else:
        results = [eval_seed_worker(wa) for wa in worker_args]
    eval_time = time.time() - t0
    print(f"  Eval done: {eval_time/60:.1f} min")

    # Aggregate
    U_means = [r["U_mean"] for r in results]
    worst_slice_pcts = [r["worst_slice_viol_pct"] for r in results]
    per_slice_arrays = [
        [r["per_slice_viol_pct"][k] for r in results] for k in range(5)
    ]

    summary = {
        "n_seeds": args.seeds,
        "K": 5,
        "per_slice_tau": per_slice_tau,
        "U_mean": float(np.mean(U_means)),
        "U_std": float(np.std(U_means)),
        "worst_slice_viol_mean": float(np.mean(worst_slice_pcts)),
        "worst_slice_viol_std": float(np.std(worst_slice_pcts)),
        "all_5_below_2.5pct_seeds": int(sum(1 for r in results if r["all_5_below_2.5pct"])),
        "all_5_below_2.5pct_fraction": float(sum(1 for r in results if r["all_5_below_2.5pct"]) / args.seeds),
        "per_slice_viol_pct_mean": [float(np.mean(per_slice_arrays[k])) for k in range(5)],
        "per_slice_viol_pct_std": [float(np.std(per_slice_arrays[k])) for k in range(5)],
        "training_time_sec": train_time,
        "eval_time_sec": eval_time,
    }
    with open(output_dir / "summary_stats.json", 'w') as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / "raw_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Markdown
    md = [
        "# K=5 with per-slice tau_k results",
        "",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Per-slice tau_k: " + ", ".join(f"{n}={t}" for n, t in zip(K5_SLICE_NAMES, per_slice_tau)),
        f"Protocol: K=5, {args.seeds} seeds x 1000 steps, ATC mode",
        "",
        "## Aggregate",
        "",
        f"- U mean +/- std: {summary['U_mean']:.3f} +/- {summary['U_std']:.3f}",
        f"- Worst-slice Viol% mean +/- std: {summary['worst_slice_viol_mean']:.2f} +/- {summary['worst_slice_viol_std']:.2f}",
        f"- All 5 slices below 2.5%: {summary['all_5_below_2.5pct_seeds']}/{args.seeds} seeds ({100*summary['all_5_below_2.5pct_fraction']:.0f}%)",
        "",
        "## Per-slice violation (% mean +/- std, against per-slice tau_k)",
        "",
        "| Slice | tau_k | Viol% mean | Viol% std | Below 2.5%? |",
        "|---|---|---|---|---|",
    ]
    for i, name in enumerate(K5_SLICE_NAMES):
        viol_mean = summary['per_slice_viol_pct_mean'][i]
        viol_std = summary['per_slice_viol_pct_std'][i]
        below_25 = "yes" if viol_mean < 2.5 else "no"
        md.append(
            f"| {name} | {per_slice_tau[i]} | {viol_mean:.2f}% | "
            f"{viol_std:.2f}% | {below_25} |"
        )
    md.extend([
        "",
        "## Timing",
        f"- Training: {train_time/60:.1f} min",
        f"- Eval: {eval_time/60:.1f} min",
        "",
    ])
    with open(output_dir / "summary_stats.md", 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))

    print(f"\n[k5_per_slice_tau] DONE - see {output_dir}/summary_stats.md")
    print(f"  U={summary['U_mean']:.3f}, all_5_below_2.5pct={summary['all_5_below_2.5pct_seeds']}/{args.seeds}")


if __name__ == "__main__":
    main()
