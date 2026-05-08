"""
beta 敏感性扫描: 在 K=1 设置下用 5 个不同的 SLA penalty
(beta in {10, 50, 100, 150, 200}) 训练 Vanilla-PPO, 用来刻画
Conservatism Trap 的崩溃阈值 (即 D=200 ms feedback delay 下
Vanilla-PPO 跌到 U <= 0.15 时最小的 beta).

Usage:
    python core/run_beta_sensitivity.py --output <dir> --workers 4
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


BETA_VALUES = [10, 50, 100, 150, 200]  # paper uses β=200
N_SEEDS_PER_BETA = 50
N_EVAL_STEPS = 1000
COLLAPSE_THRESHOLD_U = 0.15


def make_vanilla_env(beta=200, seed=None, K=1):
    """Make Vanilla-PPO env (no belief, no trust fusion) at given β."""
    env = E2_Node_Simulator(
        mode="vanilla_ppo", K=K, tau=0.5, beta=beta, use_kalman=False,
    )
    if seed is not None:
        try:
            env.reset(seed=seed)
        except TypeError:
            np.random.seed(seed)
            env.reset()
    return env


def train_vanilla_at_beta(beta, output_dir, total_steps=120000, seed=42):
    """Train Vanilla-PPO at given β value."""
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Training Vanilla-PPO at β={beta} ({total_steps} steps)...")
    env = make_vanilla_env(beta=beta, K=1)
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
    print(f"  β={beta} training done: {train_time/60:.1f} min")

    model_path = models_dir / f"ppo_vanilla_beta_{beta}_final"
    model.save(str(model_path))
    return str(model_path), train_time


def _train_worker(args):
    """Pool worker for parallel β training."""
    beta, output_dir, total_steps, train_seed = args
    try:
        model_path, train_time = train_vanilla_at_beta(
            beta, output_dir, total_steps=total_steps, seed=train_seed
        )
        return {"beta": beta, "model_path": model_path,
                "train_time_sec": train_time, "status": "OK"}
    except Exception as e:
        import traceback
        return {"beta": beta, "status": "FAILED",
                "error": str(e), "traceback": traceback.format_exc()}


def eval_seed_worker(args):
    """Eval one seed for one β value."""
    seed, model_path, beta = args
    np.random.seed(seed)

    model = PPO.load(model_path, device='cpu')
    env = make_vanilla_env(beta=beta, seed=seed, K=1)
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
        "beta": beta,
        "U_mean": float(np.mean(util_per_step)),
        "viol_pct": 100.0 * float(np.mean(viol_per_step)),
        "cost_total": float(np.sum(cost_per_step)),
    }


def find_collapse_threshold(by_beta_summary, threshold=COLLAPSE_THRESHOLD_U):
    """Find smallest β where Vanilla-PPO U drops below threshold."""
    sorted_betas = sorted(by_beta_summary.keys())
    for b in sorted_betas:
        u_mean = by_beta_summary[b]["U_mean"]
        if u_mean <= threshold:
            return b
    return None  # never collapses (rare)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--betas", nargs="*", type=float, default=BETA_VALUES)
    parser.add_argument("--seeds", type=int, default=N_SEEDS_PER_BETA)
    parser.add_argument("--total-steps", type=int, default=120000)
    parser.add_argument("--train-seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "experiment": "beta_sensitivity",
        "K": 1,
        "betas_tested": args.betas,
        "n_seeds_per_beta": args.seeds,
        "n_eval_steps_per_seed": N_EVAL_STEPS,
        "tau": 0.5,
        "training_total_steps": args.total_steps,
        "training_seed": args.train_seed,
        "collapse_threshold_U": COLLAPSE_THRESHOLD_U,
        "purpose": "Vanilla-PPO Conservatism Trap collapse threshold vs beta",
        "started_at": datetime.now().isoformat(),
    }
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    # Step 1: train Vanilla-PPO at each β (parallel)
    print(f"\n[beta_sensitivity] Step 1: Train {len(args.betas)} Vanilla-PPO models")
    train_args = [(b, output_dir, args.total_steps, args.train_seed) for b in args.betas]

    n_train_workers = min(args.workers, len(args.betas))
    t_train_total = time.time()
    if n_train_workers > 1:
        try:
            set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        with Pool(processes=n_train_workers) as pool:
            train_results = pool.map(_train_worker, train_args)
    else:
        train_results = [_train_worker(ta) for ta in train_args]
    train_total_time = time.time() - t_train_total
    print(f"  All training done: {train_total_time/60:.1f} min")

    beta_to_model = {}
    for r in train_results:
        if r["status"] == "OK":
            beta_to_model[r["beta"]] = r["model_path"]
        else:
            print(f"  WARNING: β={r['beta']} training FAILED: {r.get('error', '?')}")

    # Step 2: eval each β at multiple seeds
    print(f"\n[beta_sensitivity] Step 2: Eval at {len(beta_to_model)} βs × {args.seeds} seeds")
    eval_args = [
        (seed, beta_to_model[b], b)
        for b in beta_to_model
        for seed in range(args.seeds)
    ]

    n_eval_workers = min(args.workers, len(eval_args))
    t_eval = time.time()
    if n_eval_workers > 1:
        try:
            set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        with Pool(processes=n_eval_workers) as pool:
            all_results = pool.map(eval_seed_worker, eval_args)
    else:
        all_results = [eval_seed_worker(ea) for ea in eval_args]
    eval_time = time.time() - t_eval
    print(f"  Eval done: {eval_time/60:.1f} min")

    # Step 3: aggregate per β
    by_beta = {}
    for r in all_results:
        b = r["beta"]
        by_beta.setdefault(b, []).append(r)

    by_beta_summary = {}
    for b in sorted(by_beta.keys()):
        results = by_beta[b]
        U_arr = [r["U_mean"] for r in results]
        V_arr = [r["viol_pct"] for r in results]
        by_beta_summary[b] = {
            "beta": b,
            "U_mean": float(np.mean(U_arr)),
            "U_std": float(np.std(U_arr)),
            "viol_pct_mean": float(np.mean(V_arr)),
            "viol_pct_std": float(np.std(V_arr)),
            "n_seeds": len(results),
            "collapsed": float(np.mean(U_arr)) <= COLLAPSE_THRESHOLD_U,
        }

    collapse_beta = find_collapse_threshold(by_beta_summary)

    summary = {
        "betas_tested": args.betas,
        "n_seeds_per_beta": args.seeds,
        "by_beta": by_beta_summary,
        "collapse_threshold_beta": collapse_beta,
        "collapse_threshold_U": COLLAPSE_THRESHOLD_U,
        "U_at_beta_200_paper_default": by_beta_summary.get(200, {}).get("U_mean"),
        "training_total_time_sec": train_total_time,
        "eval_time_sec": eval_time,
    }
    with open(output_dir / "summary_stats.json", 'w') as f:
        json.dump(summary, f, indent=2)

    with open(output_dir / "raw_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    # CSV for plotting
    csv_lines = ["beta,U_mean,U_std,viol_pct_mean,viol_pct_std,collapsed"]
    for b in sorted(by_beta_summary.keys()):
        r = by_beta_summary[b]
        csv_lines.append(
            f"{r['beta']},{r['U_mean']:.4f},{r['U_std']:.4f},"
            f"{r['viol_pct_mean']:.3f},{r['viol_pct_std']:.3f},{r['collapsed']}"
        )
    with open(output_dir / "collapse_curve.csv", 'w') as f:
        f.write('\n'.join(csv_lines))

    # Markdown
    md = [
        "# beta sensitivity sweep results",
        "",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Protocol: train Vanilla-PPO at each beta; eval at K=1, D=200 ms, tau=0.5",
        f"Collapse threshold: U <= {COLLAPSE_THRESHOLD_U}",
        "",
        "## Collapse curve",
        "",
        "| beta | U mean +/- std | Viol% mean +/- std | Collapsed? |",
        "|---|---|---|---|",
    ]
    for b in sorted(by_beta_summary.keys()):
        r = by_beta_summary[b]
        collapse_str = "YES" if r["collapsed"] else "no"
        md.append(
            f"| {b} | {r['U_mean']:.3f} +/- {r['U_std']:.3f} | "
            f"{r['viol_pct_mean']:.2f} +/- {r['viol_pct_std']:.2f} | "
            f"{collapse_str} |"
        )
    md.extend([
        "",
        "## Key finding",
        "",
        f"- U at beta=200: {summary['U_at_beta_200_paper_default']:.3f}",
        f"- Collapse threshold (smallest beta where U <= {COLLAPSE_THRESHOLD_U}): "
        f"beta={collapse_beta if collapse_beta else 'NONE (no collapse in tested range)'}",
        "",
        "## Timing",
        f"- Training ({len(args.betas)} models, parallel): {train_total_time/60:.1f} min",
        f"- Eval ({len(args.betas)} betas x {args.seeds} seeds): {eval_time/60:.1f} min",
        "",
    ])
    with open(output_dir / "summary_stats.md", 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))

    print(f"\n[beta_sensitivity] DONE - see {output_dir}/summary_stats.md")


if __name__ == "__main__":
    main()
