"""
K=5 30-seed 扩展实验.

K=5 (URLLC/V2X/eMBB/mMTC/IoT_burst) 配置, 训练 ATC 120K steps,
然后在 30 个独立 seed x 1000 steps 上 eval, 用来收紧 per-slice
指标的 confidence interval.

Usage:
    python core/run_k5_30seed.py --output <dir> --workers 4
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

# Make repo root importable (script may be invoked from various CWDs)
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO

from core.telemetry_env import E2_Node_Simulator, SLICE_PROFILES


# K=5 slice configuration (matches existing k5_probe_v1)
K5_SLICE_NAMES = ["URLLC", "V2X", "eMBB", "mMTC", "IoT_burst"]
K5_PROFILES = {
    "URLLC":     SLICE_PROFILES["URLLC"],
    "V2X":       SLICE_PROFILES["V2X"],
    "eMBB":      SLICE_PROFILES["eMBB"],
    "mMTC":      {"sigma": 0.04, "base_util": 0.25, "target_prb": 0.55, "label": "mMTC"},
    "IoT_burst": {"sigma": 0.10, "base_util": 0.20, "target_prb": 0.50, "label": "IoT_burst"},
}


def make_env(seed=None):
    """Make K=5 ATC env with the canonical slice mix."""
    env = E2_Node_Simulator(
        mode="proposed", K=5, tau=0.5, beta=200, use_kalman=False,
        slice_profiles=K5_PROFILES,
    )
    if seed is not None:
        try:
            env.reset(seed=seed)
        except TypeError:
            np.random.seed(seed)
            env.reset()
    return env


def eval_seed_worker(args):
    """Eval one seed; return per-slice violation + U stats."""
    seed, model_path, K = args
    np.random.seed(seed)

    model = PPO.load(model_path, device='cpu')
    env = make_env(seed=seed)
    obs, _ = env.reset() if not hasattr(env, "_seeded_reset_done") else (env._last_obs, {})

    util_per_step = []
    viol_per_slice_per_step = [[] for _ in range(K)]
    cost_per_step = []

    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        util_per_step.append(info.get("true_util", 0.0))
        per_slice_viol = info.get("per_slice_violations", [0]*K)
        for k in range(K):
            viol_per_slice_per_step[k].append(per_slice_viol[k] if k < len(per_slice_viol) else 0)
        cost_per_step.append(info.get("sig_cost", 0))
        if done:
            obs, _ = env.reset()

    per_slice_viol_pct = [
        100.0 * float(np.mean(viol_per_slice_per_step[k])) for k in range(K)
    ]

    return {
        "seed": seed,
        "U_mean": float(np.mean(util_per_step)),
        "U_std_within": float(np.std(util_per_step)),
        "per_slice_viol_pct": per_slice_viol_pct,
        "worst_slice_viol_pct": max(per_slice_viol_pct),
        "cost_total": float(np.sum(cost_per_step)),
    }


def train_k5_atc(output_dir, total_steps=120000, train_seed=42):
    """Train K=5 ATC PPO model; save to output_dir/models/."""
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Training K=5 ATC (PPO, {total_steps} steps, seed={train_seed})...")
    env = make_env()
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

    model_path = models_dir / "ppo_atc_k5_final"
    model.save(str(model_path))
    print(f"  Saved: {model_path}.zip")
    return str(model_path), train_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=30,
                        help="Number of eval seeds (default 30 closes T2.3)")
    parser.add_argument("--total-steps", type=int, default=120000,
                        help="PPO training steps (default 120K)")
    parser.add_argument("--train-seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "experiment": "k5_30seed",
        "K": 5,
        "slice_names": K5_SLICE_NAMES,
        "n_eval_seeds": args.seeds,
        "n_eval_steps_per_seed": 1000,
        "tau": 0.5,
        "beta": 200,
        "training_total_steps": args.total_steps,
        "training_seed": args.train_seed,
        "purpose": "K=5 30-seed evaluation",
        "started_at": datetime.now().isoformat(),
    }
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    # Step 1: train K=5 ATC
    print(f"\n[k5_30seed] Step 1: Train K=5 ATC")
    model_path, train_time = train_k5_atc(
        output_dir, total_steps=args.total_steps, train_seed=args.train_seed
    )

    # Step 2: eval 30 seeds
    print(f"\n[k5_30seed] Step 2: Eval {args.seeds} seeds (workers={args.workers})")
    worker_args = [(seed, model_path, 5) for seed in range(args.seeds)]

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

    # Step 3: aggregate
    U_means = [r["U_mean"] for r in results]
    worst_slice_pcts = [r["worst_slice_viol_pct"] for r in results]
    per_slice_arrays = [
        [r["per_slice_viol_pct"][k] for r in results] for k in range(5)
    ]

    summary = {
        "n_seeds": args.seeds,
        "K": 5,
        "U_mean": float(np.mean(U_means)),
        "U_std": float(np.std(U_means)),
        "worst_slice_viol_mean": float(np.mean(worst_slice_pcts)),
        "worst_slice_viol_std": float(np.std(worst_slice_pcts)),
        "all_5_below_2.5pct_seeds": int(sum(1 for r in results if r["worst_slice_viol_pct"] < 2.5)),
        "all_5_below_2.5pct_fraction": float(sum(1 for r in results if r["worst_slice_viol_pct"] < 2.5) / args.seeds),
        "per_slice_viol_pct_mean": [float(np.mean(per_slice_arrays[k])) for k in range(5)],
        "per_slice_viol_pct_std": [float(np.std(per_slice_arrays[k])) for k in range(5)],
        "training_time_sec": train_time,
        "eval_time_sec": eval_time,
    }
    with open(output_dir / "summary_stats.json", 'w') as f:
        json.dump(summary, f, indent=2)

    with open(output_dir / "raw_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Markdown summary
    md = [
        "# K=5 30-seed extension results",
        "",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"30 seeds x 1000 steps, ATC mode, K=5",
        f"Protocol: K=5 ({', '.join(K5_SLICE_NAMES)}), {args.seeds} seeds x 1000 steps, ATC mode",
        "",
        "## Aggregate",
        "",
        f"- **U mean ± std**: {summary['U_mean']:.3f} ± {summary['U_std']:.3f}",
        f"- **Worst-slice Viol% mean ± std**: {summary['worst_slice_viol_mean']:.2f} ± {summary['worst_slice_viol_std']:.2f}",
        f"- **All 5 slices below 2.5%**: {summary['all_5_below_2.5pct_seeds']}/{args.seeds} seeds ({100*summary['all_5_below_2.5pct_fraction']:.0f}%)",
        "",
        "## Per-slice violation (% mean ± std)",
        "",
        "| Slice | Viol% mean | Viol% std |",
        "|---|---|---|",
    ]
    for i, name in enumerate(K5_SLICE_NAMES):
        md.append(
            f"| {name} | {summary['per_slice_viol_pct_mean'][i]:.2f}% | "
            f"{summary['per_slice_viol_pct_std'][i]:.2f}% |"
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

    print(f"\n[k5_30seed] DONE - see {output_dir}/summary_stats.md")


if __name__ == "__main__":
    main()
