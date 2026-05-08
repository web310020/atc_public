"""
Vanilla-PPO baseline 验证.

只做 eval, 不训练. 用一个已经训练好的 K=1 Vanilla-PPO model
(通过 --model-path 指定, 或自动从 experiments/ 下找), 跑标准
eval protocol 看是否能复现期望的 utilization.

Usage:
    python core/run_vanilla_verify.py --output <dir> --workers 2
"""

import argparse
import glob
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


N_EVAL_STEPS = 1000
N_SEEDS = 50
VANILLA_U_REFERENCE = 0.12


def find_vanilla_model():
    """Locate a previously trained Vanilla-PPO checkpoint."""
    candidates = []
    candidates.extend(sorted(glob.glob(
        "experiments/*main*/vanilla_ppo/models/*.zip")))
    candidates.extend(sorted(glob.glob(
        "experiments/**/vanilla_ppo/models/*.zip", recursive=True)))
    if not candidates:
        return None
    return candidates[0].replace(".zip", "")


def eval_seed_worker(args):
    seed, model_path = args
    np.random.seed(seed)

    model = PPO.load(model_path, device='cpu')
    env = E2_Node_Simulator(
        mode="vanilla_ppo", K=1, tau=0.5, beta=200, use_kalman=False
    )
    try:
        obs, _ = env.reset(seed=seed)
    except TypeError:
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
        "U_mean": float(np.mean(util_per_step)),
        "viol_pct": 100.0 * float(np.mean(viol_per_step)),
        "cost_total": float(np.sum(cost_per_step)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    parser.add_argument("--model-path", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.model_path or find_vanilla_model()
    if model_path is None or not os.path.exists(model_path + ".zip"):
        msg = ("No Vanilla-PPO model found under experiments/. "
               "Pass --model-path or train one first.")
        print(f"[vanilla_verify] ERROR: {msg}")
        with open(output_dir / "summary_stats.json", 'w') as f:
            json.dump({"status": "SKIPPED", "reason": msg}, f, indent=2)
        with open(output_dir / "summary_stats.md", 'w') as f:
            f.write(f"# Vanilla verify SKIPPED\n\n{msg}\n")
        return

    print(f"\n[vanilla_verify] Using model: {model_path}")
    print(f"[vanilla_verify] Running {args.seeds}-seed eval (workers={args.workers})")

    metadata = {
        "experiment": "vanilla_verify",
        "model_path": model_path,
        "K": 1,
        "n_seeds": args.seeds,
        "n_eval_steps": N_EVAL_STEPS,
        "tau": 0.5,
        "beta": 200,
        "vanilla_U_reference": VANILLA_U_REFERENCE,
        "purpose": "Verify the existing Vanilla-PPO checkpoint reproduces its reference U",
        "started_at": datetime.now().isoformat(),
    }
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    worker_args = [(seed, model_path) for seed in range(args.seeds)]
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
    print(f"  Eval done: {eval_time:.1f}s")

    U_arr = [r["U_mean"] for r in results]
    V_arr = [r["viol_pct"] for r in results]

    summary = {
        "n_seeds": args.seeds,
        "U_mean": float(np.mean(U_arr)),
        "U_std": float(np.std(U_arr)),
        "viol_pct_mean": float(np.mean(V_arr)),
        "viol_pct_std": float(np.std(V_arr)),
        "vanilla_U_reference": VANILLA_U_REFERENCE,
        "discrepancy_vs_reference": float(np.mean(U_arr)) - VANILLA_U_REFERENCE,
        "matches_reference_within_0_02": abs(float(np.mean(U_arr)) - VANILLA_U_REFERENCE) < 0.02,
        "eval_time_sec": eval_time,
    }
    with open(output_dir / "summary_stats.json", 'w') as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / "raw_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    md = [
        "# Vanilla-PPO baseline verification",
        "",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Model: `{model_path}.zip`",
        f"Protocol: K=1, beta=200, {args.seeds} seeds x {N_EVAL_STEPS} steps",
        "",
        "## Results",
        "",
        f"- U mean +/- std: {summary['U_mean']:.4f} +/- {summary['U_std']:.4f}",
        f"- Viol% mean +/- std: {summary['viol_pct_mean']:.2f} +/- {summary['viol_pct_std']:.2f}",
        f"- Reference U: {VANILLA_U_REFERENCE}",
        f"- Discrepancy: {summary['discrepancy_vs_reference']:+.4f}",
        f"- Matches reference within +/-0.02: {'YES' if summary['matches_reference_within_0_02'] else 'NO'}",
        "",
        "## Timing",
        f"- Eval: {eval_time:.1f}s ({eval_time/60:.1f} min)",
        "",
    ]
    with open(output_dir / "summary_stats.md", 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))

    print(f"\n[vanilla_verify] DONE - see {output_dir}/summary_stats.md")
    print(f"  U={summary['U_mean']:.4f} +/- {summary['U_std']:.4f}, "
          f"reference {VANILLA_U_REFERENCE}, "
          f"matches: {summary['matches_reference_within_0_02']}")


if __name__ == "__main__":
    main()
