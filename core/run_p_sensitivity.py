"""
P-kernel 校准敏感性扫描.

K=1 下用标准 sigma kernel 训练 ATC, 然后在 5 个 sigma_eval scale
(0.7x, 0.85x, 1.0x, 1.15x, 1.3x 训练 sigma) 上 eval, 用来量化对
transition kernel mis-specification 的鲁棒性 (比如部署到一个 traffic
统计跟训练时不一样的网络上时的退化).

Usage:
    python core/run_p_sensitivity.py --output <dir> --workers 4
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

from core.telemetry_env import E2_Node_Simulator


# Scales applied to sigma_k at EVAL time (training uses scale=1.0)
P_EVAL_SCALES = [0.7, 0.85, 1.0, 1.15, 1.3]
N_SEEDS_PER_SCALE = 50
N_EVAL_STEPS = 1000


def make_env_with_sigma_scale(scale=1.0, seed=None, K=1):
    """Make ATC env, then override sigma_k by scale factor."""
    env = E2_Node_Simulator(mode="proposed", K=K, tau=0.5, beta=200, use_kalman=False)
    # Scale the per-slice noise (P kernel proxy)
    env.sigma_k = env.sigma_k * scale
    if seed is not None:
        try:
            env.reset(seed=seed)
        except TypeError:
            np.random.seed(seed)
            env.reset()
    return env


def eval_seed_worker(args):
    """Eval one seed at given sigma scale; return U + Viol stats."""
    seed, model_path, sigma_scale = args
    np.random.seed(seed)

    model = PPO.load(model_path, device='cpu')
    env = make_env_with_sigma_scale(scale=sigma_scale, seed=seed, K=1)
    obs, _ = env.reset()

    util_per_step = []
    viol_per_step = []
    cost_per_step = []

    for _ in range(N_EVAL_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        util_per_step.append(info.get("true_util", 0.0))
        viol_per_step.append(info.get("is_violation", 0))
        cost_per_step.append(info.get("sig_cost", 0))
        if done:
            obs, _ = env.reset()

    return {
        "seed": seed,
        "sigma_scale": sigma_scale,
        "U_mean": float(np.mean(util_per_step)),
        "viol_pct": 100.0 * float(np.mean(viol_per_step)),
        "cost_total": float(np.sum(cost_per_step)),
    }


def train_atc_baseline(output_dir, total_steps=120000, seed=42):
    """Train K=1 ATC at scale=1.0 (the standard kernel)."""
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Training baseline ATC (K=1, scale=1.0, {total_steps} steps)...")
    env = make_env_with_sigma_scale(scale=1.0, K=1)
    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4, batch_size=64, n_steps=2048, n_epochs=10,
        gamma=0.99, ent_coef=0.0,
        policy_kwargs={"net_arch": [512, 512, 256]},
        verbose=0, device='cpu', seed=seed,
    )
    t0 = time.time()
    model.learn(total_timesteps=total_steps, progress_bar=False)
    train_time = time.time() - t0
    print(f"  Training done: {train_time/60:.1f} min")

    model_path = models_dir / "ppo_atc_p_baseline_final"
    model.save(str(model_path))
    print(f"  Saved: {model_path}.zip")
    return str(model_path), train_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--scales", nargs="*", type=float, default=P_EVAL_SCALES)
    parser.add_argument("--seeds", type=int, default=N_SEEDS_PER_SCALE)
    parser.add_argument("--total-steps", type=int, default=120000)
    parser.add_argument("--train-seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "experiment": "p_sensitivity",
        "K": 1,
        "p_eval_scales": args.scales,
        "n_seeds_per_scale": args.seeds,
        "n_eval_steps_per_seed": N_EVAL_STEPS,
        "tau": 0.5,
        "beta": 200,
        "training_total_steps": args.total_steps,
        "training_seed": args.train_seed,
        "purpose": "Quantify ATC robustness to P-kernel mis-specification",
        "started_at": datetime.now().isoformat(),
    }
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    # Step 1: train baseline ATC at scale 1.0
    print(f"\n[p_sensitivity] Step 1: Train baseline ATC")
    model_path, train_time = train_atc_baseline(
        output_dir, total_steps=args.total_steps, seed=args.train_seed
    )

    # Step 2: eval at each scale
    print(f"\n[p_sensitivity] Step 2: Eval at {len(args.scales)} scales × {args.seeds} seeds")
    print(f"  Scales: {args.scales}")
    print(f"  Workers: {args.workers}")

    worker_args = [
        (seed, model_path, scale)
        for scale in args.scales
        for seed in range(args.seeds)
    ]

    n_workers = min(args.workers, len(worker_args))
    t0 = time.time()
    if n_workers > 1:
        try:
            set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        with Pool(processes=n_workers) as pool:
            all_results = pool.map(eval_seed_worker, worker_args)
    else:
        all_results = [eval_seed_worker(wa) for wa in worker_args]
    eval_time = time.time() - t0
    print(f"  Eval done: {eval_time/60:.1f} min")

    # Step 3: aggregate per scale
    by_scale = {}
    for r in all_results:
        s = r["sigma_scale"]
        by_scale.setdefault(s, []).append(r)

    degradation_curve = []
    for scale in args.scales:
        scale_results = by_scale.get(scale, [])
        U_arr = [r["U_mean"] for r in scale_results]
        V_arr = [r["viol_pct"] for r in scale_results]
        degradation_curve.append({
            "sigma_scale": scale,
            "U_mean": float(np.mean(U_arr)),
            "U_std": float(np.std(U_arr)),
            "viol_pct_mean": float(np.mean(V_arr)),
            "viol_pct_std": float(np.std(V_arr)),
            "n_seeds": len(scale_results),
        })

    summary = {
        "scales_tested": args.scales,
        "n_seeds_per_scale": args.seeds,
        "degradation_curve": degradation_curve,
        "U_at_scale_1.0": next((r["U_mean"] for r in degradation_curve if r["sigma_scale"] == 1.0), None),
        "U_degradation_at_scale_1.3_pct": (
            (next((r["U_mean"] for r in degradation_curve if r["sigma_scale"] == 1.0), 0)
             - next((r["U_mean"] for r in degradation_curve if r["sigma_scale"] == 1.3), 0))
            / max(next((r["U_mean"] for r in degradation_curve if r["sigma_scale"] == 1.0), 1e-9), 1e-9)
            * 100
        ),
        "training_time_sec": train_time,
        "eval_time_sec": eval_time,
    }
    with open(output_dir / "summary_stats.json", 'w') as f:
        json.dump(summary, f, indent=2)

    with open(output_dir / "raw_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    csv_lines = ["sigma_scale,U_mean,U_std,viol_pct_mean,viol_pct_std,n_seeds"]
    for r in degradation_curve:
        csv_lines.append(
            f"{r['sigma_scale']},{r['U_mean']:.4f},{r['U_std']:.4f},"
            f"{r['viol_pct_mean']:.3f},{r['viol_pct_std']:.3f},{r['n_seeds']}"
        )
    with open(output_dir / "degradation_curve.csv", 'w') as f:
        f.write('\n'.join(csv_lines))

    # Markdown summary
    md = [
        "# P calibration sensitivity results",
        "",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Protocol: train ATC at sigma_train; eval at sigma_eval = scale * sigma_train",
        f"Scales tested: {args.scales}",
        f"Per-scale: {args.seeds} seeds x {N_EVAL_STEPS} steps",
        "",
        "## Degradation curve",
        "",
        "| sigma_scale | U mean +/- std | Viol% mean +/- std | n_seeds |",
        "|---|---|---|---|",
    ]
    for r in degradation_curve:
        md.append(
            f"| {r['sigma_scale']:.2f} | "
            f"{r['U_mean']:.3f} +/- {r['U_std']:.3f} | "
            f"{r['viol_pct_mean']:.2f} +/- {r['viol_pct_std']:.2f} | "
            f"{r['n_seeds']} |"
        )
    md.extend([
        "",
        "## Key finding",
        "",
        f"- U at scale 1.0 (matched P): {summary['U_at_scale_1.0']:.3f}",
        f"- U degradation at scale 1.3 (+30% noise): {summary['U_degradation_at_scale_1.3_pct']:.1f}%",
        "",
        f"## Timing",
        f"- Training: {train_time/60:.1f} min",
        f"- Eval ({len(args.scales)} scales × {args.seeds} seeds): {eval_time/60:.1f} min",
        "",
    ])
    with open(output_dir / "summary_stats.md", 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))

    print(f"\n[p_sensitivity] DONE - see {output_dir}/summary_stats.md")


if __name__ == "__main__":
    main()
