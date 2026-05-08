"""
Eval 驱动: ATC 各 ablation + HRL baseline.

每个训练好的 mode 跑 50-seed eval, 算 U / violation rate / signaling /
latency 的 mean +/- std, 写一个 summary markdown.

前置: 需要先跑过 train_l3_ablations.py, model 存在 experiments/<exp_dir>/.

Usage:
    python run_l3_ablations_eval.py
    python run_l3_ablations_eval.py --exp-dir experiments/<run_name>
    python run_l3_ablations_eval.py --n-runs 50
"""

import os
import sys
import json
import glob
import time
import argparse
import numpy as np
from datetime import datetime
from multiprocessing import Pool, set_start_method
from scipy.stats import pearsonr

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from stable_baselines3 import PPO

from core.telemetry_env import E2_Node_Simulator
from core.telemetry_env_ablations_v3 import (
    ATC_NoAdaptiveTelemetry, ATC_NoSCU, ATC_BangBang
)
from core.habib_hrl_baseline import E2_Habib_HRL


N_EVAL_STEPS = 1000


MODE_LABELS = {
    "no_adaptive_telem": r"w/o Adaptive Telemetry",
    "no_scu":            r"w/o L4 SCU",
    "bang_bang":         r"L6 Bang-Bang",
    "kalman_belief":     r"Kalman Belief (continuous)",
    "habib_hrl":         r"Habib HRL \cite{hrl_intent_oran_mass_2023}",
}


MODE_ENV_CLASSES = {
    "no_adaptive_telem": (ATC_NoAdaptiveTelemetry, {"mode": "proposed"}),
    "no_scu":            (ATC_NoSCU, {"mode": "proposed"}),
    "bang_bang":         (ATC_BangBang, {"mode": "proposed"}),
    "kalman_belief":     (E2_Node_Simulator, {"mode": "proposed", "use_kalman": True}),
    "habib_hrl":         (E2_Habib_HRL, {}),
}


def evaluate_mode(mode_name, model, env_class, env_kwargs, K, n_runs):
    """Run n_runs independent evaluations, return per-run stats."""
    runs = []
    for run_i in range(n_runs):
        env = env_class(**{**env_kwargs, "K": K})
        if hasattr(env, 'set_a1_policy'):
            env.set_a1_policy(0.5)
        obs, _ = env.reset()

        stats = {"util": [], "viol": [], "viol_depth": [], "cost": [],
                 "reward": [], "latency": [], "belief_mean": []}
        for _ in range(N_EVAL_STEPS):
            t0 = time.time()
            action, _ = model.predict(obs, deterministic=True)
            latency = (time.time() - t0) * 1000  # ms
            obs, reward, _, _, info = env.step(action)
            u = info["true_util"]
            stats["util"].append(u)
            stats["belief_mean"].append(info.get("belief_mean", 0))
            stats["viol_depth"].append(max(0, u - 0.5))
            stats["viol"].append(info["is_violation"])
            stats["cost"].append(info["sig_cost"])
            stats["reward"].append(reward)
            stats["latency"].append(latency)

        runs.append({
            "U": np.mean(stats["util"]),
            "viol_pct": 100 * np.mean(stats["viol"]),
            "viol_depth": np.mean(stats["viol_depth"]),
            "cost_per_episode": np.sum(stats["cost"]),
            "latency_ms": np.mean(stats["latency"]),
            "belief_mean": np.mean(stats["belief_mean"]),
            "reward_total": np.sum(stats["reward"]),
        })
    return runs


def aggregate_runs(runs):
    """Compute mean ± std across runs for each metric."""
    keys = runs[0].keys()
    return {k: (np.mean([r[k] for r in runs]),
                np.std([r[k] for r in runs])) for k in keys}


def write_md_table(results, out_path):
    """Generate a markdown summary of the ablation results."""
    lines = [
        "# Ablation and HRL-baseline evaluation results",
        "",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Protocol: 50 independent runs x 1000 steps per run, K=1, tau=0.5",
        "",
        "## Results",
        "",
        "| Variant | U up | psi down | Viol% down | Signaling (ev/ep) down | Latency (ms) down |",
        "|---------|------|----------|------------|------------------------|-------------------|",
    ]
    for mode_name, label in MODE_LABELS.items():
        if mode_name not in results:
            continue
        r = results[mode_name]
        lines.append(
            f"| {label} | "
            f"{r['U'][0]:.3f} +/- {r['U'][1]:.3f} | "
            f"{r['viol_depth'][0]:.4f} | "
            f"{r['viol_pct'][0]:.1f} +/- {r['viol_pct'][1]:.1f}\\% | "
            f"{r['cost_per_episode'][0]:.0f} +/- {r['cost_per_episode'][1]:.0f} | "
            f"{r['latency_ms'][0]:.2f} |"
        )

    lines.extend([
        "",
        "## Notes",
        "",
        "- 50-seed reports.",
        "- habib_hrl is the HRL baseline.",
        "- Ablation modes: no_adaptive_telem, no_scu, bang_bang, kalman_belief.",
    ])

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Markdown summary: {out_path}")


def _eval_mode_worker(args_tuple):
    """Top-level worker for multiprocessing.Pool (must be picklable)."""
    mode_name, exp_dir, n_runs, K = args_tuple
    if mode_name not in MODE_ENV_CLASSES:
        return mode_name, None, f"unknown mode"

    model_path = os.path.join(exp_dir, mode_name, "models", f"ppo_{mode_name}_final.zip")
    if not os.path.exists(model_path):
        return mode_name, None, f"model not found at {model_path}"

    try:
        model = PPO.load(model_path.replace(".zip", ""), device='cpu')
        env_class, env_kwargs = MODE_ENV_CLASSES[mode_name]
        t0 = time.time()
        runs = evaluate_mode(mode_name, model, env_class, env_kwargs, K, n_runs)
        agg = aggregate_runs(runs)
        elapsed = time.time() - t0
        return mode_name, agg, f"OK ({elapsed:.1f}s, {elapsed/60:.1f} min)"
    except Exception as e:
        import traceback
        return mode_name, None, f"ERROR: {e}\n{traceback.format_exc()}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", type=str, default=None,
                        help="Experiment dir (default: latest experiments/l3_ablations_*)")
    parser.add_argument("--n-runs", type=int, default=50,
                        help="Independent evaluation runs per mode (default 50)")
    parser.add_argument("--K", type=int, default=1)
    parser.add_argument("--modes", type=str, default="all",
                        help="Comma-separated subset or 'all'")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel eval processes (default 1; "
                             "3 recommended for 5-mode run)")
    args = parser.parse_args()

    # Locate experiment dir
    if args.exp_dir is None:
        candidates = sorted(glob.glob("experiments/l3_ablations_*"))
        if not candidates:
            print("ERROR: no experiments/l3_ablations_* found. Run train_l3_ablations.py first.")
            sys.exit(1)
        args.exp_dir = candidates[-1]

    # Filter modes
    if args.modes == "all":
        modes_to_eval = list(MODE_LABELS.keys())
    else:
        modes_to_eval = [m.strip() for m in args.modes.split(",")]

    n_workers = min(args.workers, len(modes_to_eval))

    print(f"\n{'='*65}")
    print(f"  L3 ABLATIONS EVAL: {args.n_runs} runs/mode, K={args.K}")
    print(f"  Modes: {modes_to_eval}")
    print(f"  Workers: {n_workers} (parallel)")
    print(f"  Experiment dir: {args.exp_dir}")
    print(f"{'='*65}")

    # Build worker args
    worker_args = [(m, args.exp_dir, args.n_runs, args.K) for m in modes_to_eval]

    results = {}
    t_total = time.time()
    if n_workers > 1:
        print(f"  >>> Using multiprocessing Pool with {n_workers} workers <<<")
        try:
            set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        with Pool(processes=n_workers) as pool:
            outputs = pool.map(_eval_mode_worker, worker_args)
    else:
        print(f"  >>> Serial eval (1 worker) <<<")
        outputs = [_eval_mode_worker(wa) for wa in worker_args]

    # Print + collect
    for mode_name, agg, status in outputs:
        print(f"\n--- {mode_name} ({MODE_LABELS.get(mode_name, '?')}) ---")
        print(f"  Status: {status}")
        if agg is not None:
            results[mode_name] = agg
            print(f"  U = {agg['U'][0]:.3f} ± {agg['U'][1]:.3f}")
            print(f"  Viol% = {agg['viol_pct'][0]:.1f} ± {agg['viol_pct'][1]:.1f}")
            print(f"  Signaling = {agg['cost_per_episode'][0]:.0f} ± {agg['cost_per_episode'][1]:.0f}")

    elapsed_total = time.time() - t_total
    print(f"\n{'='*65}")
    print(f"  Eval done: {elapsed_total/60:.1f} min")
    print(f"{'='*65}")

    # Write summary
    out_md = os.path.join(args.exp_dir, "summary_stats_l3_ablations.md")
    write_md_table(results, out_md)

    # Also save raw JSON for reproducibility
    raw_json_path = os.path.join(args.exp_dir, "summary_stats_l3_ablations.json")
    json_safe = {m: {k: list(v) for k, v in agg.items()} for m, agg in results.items()}
    with open(raw_json_path, 'w') as f:
        json.dump(json_safe, f, indent=2)
    print(f"  Raw JSON: {raw_json_path}")


if __name__ == "__main__":
    main()
