"""
训练驱动: ATC 各 ablation + HRL baseline.

ABLATIONS (per-component 贡献分析):
  1. ATC_NoAdaptiveTelemetry  (固定 signaling, 没有 entropy gating)
  2. ATC_NoSCU                (跳过 safety calibration unit)
  3. ATC_BangBang             (binary reflex 代替 proportional)
  4. ATC_Kalman               (use_kalman=True, 连续 belief 版本)

BASELINE:
  5. Habib_HRL                (2 层 meta + primitive HRL)

输出: experiments/<exp_dir>/<mode>/ 下的 PPO checkpoint, 每个 mode
跑 120K PPO step. 单核 CPU 上每个 mode 大约 25-35 min.

Usage:
    python train_l3_ablations.py
    python train_l3_ablations.py --total-steps 60000   # smoke test
    python train_l3_ablations.py --modes habib_hrl,no_scu
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from datetime import datetime
from multiprocessing import Pool, set_start_method

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from core.telemetry_env import E2_Node_Simulator
from core.telemetry_env_ablations_v3 import (
    ATC_NoAdaptiveTelemetry, ATC_NoSCU, ATC_BangBang
)
from core.habib_hrl_baseline import E2_Habib_HRL


MODES = {
    # Ablations: each isolates one component of "proposed"
    "no_adaptive_telem": {
        "env_class": ATC_NoAdaptiveTelemetry,
        "env_kwargs": {"mode": "proposed"},
        "label": "w/o adaptive telemetry",
        "policy": "MlpPolicy",
    },
    "no_scu": {
        "env_class": ATC_NoSCU,
        "env_kwargs": {"mode": "proposed"},
        "label": "w/o L4 SCU",
        "policy": "MlpPolicy",
    },
    "bang_bang": {
        "env_class": ATC_BangBang,
        "env_kwargs": {"mode": "proposed"},
        "label": "L6 bang-bang (no proportional)",
        "policy": "MlpPolicy",
    },
    "kalman_belief": {
        "env_class": E2_Node_Simulator,
        "env_kwargs": {"mode": "proposed", "use_kalman": True},
        "label": "Kalman-filter belief (continuous)",
        "policy": "MlpPolicy",
    },
    # Habib MASS 2023 HRL baseline
    "habib_hrl": {
        "env_class": E2_Habib_HRL,
        "env_kwargs": {},
        "label": "Habib MASS 2023 HRL",
        "policy": "MlpPolicy",
    },
}


def train_mode(mode_name, mode_config, exp_dir, total_steps=120000, K=1, seed=42):
    """Train one mode, save model to exp_dir/<mode_name>/models/."""
    out_dir = os.path.join(exp_dir, mode_name)
    os.makedirs(os.path.join(out_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)

    env_class = mode_config["env_class"]
    env_kwargs = {**mode_config["env_kwargs"], "K": K}
    env = env_class(**env_kwargs)
    env.set_a1_policy(0.5) if hasattr(env, 'set_a1_policy') else None

    print(f"\n{'='*65}")
    print(f"  Training: {mode_name} ({mode_config['label']})")
    print(f"  Env: {env_class.__name__}, K={K}, total_steps={total_steps}")
    print(f"{'='*65}")

    # PPO hyperparameters
    model = PPO(
        mode_config["policy"], env,
        learning_rate=3e-4,
        batch_size=64,
        n_steps=2048,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.0,
        policy_kwargs={"net_arch": [512, 512, 256]},
        verbose=0,
        device='cpu',
        seed=seed,
    )

    # Save metadata
    metadata = {
        "standard": "O-RAN-R005-2026",
        "K": K,
        "use_kalman": mode_config["env_kwargs"].get("use_kalman", False),
        "intent_type": "Latency-Aware-Slicing",
        "kpm_sources": ["DRB.PRB.UsedDl", "DRB.PdcpPduDelay"],
        "mode": mode_name,
        "label": mode_config["label"],
        "total_steps": total_steps,
        "obs_dim": env.observation_space.shape[0],
        "act_dim": env.action_space.shape[0],
    }
    with open(os.path.join(out_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=4)

    # Train
    t0 = time.time()
    model.learn(total_timesteps=total_steps, progress_bar=False)
    train_time = time.time() - t0
    print(f"  Training done: {train_time:.1f}s ({train_time/60:.1f} min)")

    # Save model
    model_path = os.path.join(out_dir, "models", f"ppo_{mode_name}_final")
    model.save(model_path)
    print(f"  Saved: {model_path}.zip")

    return model_path, train_time


def _train_mode_worker(args_tuple):
    """Top-level worker for multiprocessing.Pool (must be picklable)."""
    mode, mode_config, exp_dir, total_steps, K, seed = args_tuple
    try:
        model_path, train_time = train_mode(
            mode, mode_config, exp_dir,
            total_steps=total_steps, K=K, seed=seed
        )
        return {"mode": mode, "model_path": model_path,
                "train_time_sec": train_time, "status": "OK"}
    except Exception as e:
        import traceback
        return {"mode": mode, "status": "FAILED",
                "error": str(e), "traceback": traceback.format_exc()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=120000,
                        help="PPO total timesteps per mode (default 120K)")
    parser.add_argument("--K", type=int, default=1, help="Number of slices")
    parser.add_argument("--seed", type=int, default=42, help="Training seed")
    parser.add_argument("--modes", type=str, default="all",
                        help="Comma-separated mode names or 'all'")
    parser.add_argument("--exp-suffix", type=str, default="",
                        help="Optional suffix for experiment dir name")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel training processes (default 1; "
                             "3 recommended for 5-mode run; cap at min(modes,workers))")
    args = parser.parse_args()

    # Filter modes
    if args.modes == "all":
        modes_to_train = list(MODES.keys())
    else:
        modes_to_train = [m.strip() for m in args.modes.split(",")]
        unknown = [m for m in modes_to_train if m not in MODES]
        if unknown:
            print(f"ERROR: unknown modes: {unknown}")
            print(f"Available: {list(MODES.keys())}")
            sys.exit(1)

    # Cap workers
    n_workers = min(args.workers, len(modes_to_train))

    # Setup experiment dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.exp_suffix}" if args.exp_suffix else ""
    exp_dir = os.path.join("experiments", f"l3_ablations_{timestamp}{suffix}")
    os.makedirs(exp_dir, exist_ok=True)

    # Save experiment-level config
    exp_config = {
        "K": args.K,
        "total_steps": args.total_steps,
        "seed": args.seed,
        "timestamp": timestamp,
        "modes": modes_to_train,
        "n_workers": n_workers,
        "purpose": "Per-component ablations and the HRL baseline",
    }
    with open(os.path.join(exp_dir, "experiment_config.json"), 'w') as f:
        json.dump(exp_config, f, indent=4)

    print(f"\n{'='*65}")
    print(f"  ABLATIONS TRAINING")
    print(f"  Modes: {modes_to_train}")
    print(f"  Workers: {n_workers} (parallel)")
    print(f"  total_steps: {args.total_steps} per mode")
    print(f"  Experiment dir: {exp_dir}")
    print(f"{'='*65}")

    # Build worker args list
    worker_args = [
        (mode, MODES[mode], exp_dir, args.total_steps, args.K, args.seed)
        for mode in modes_to_train
    ]

    # Train (parallel or serial)
    t_total = time.time()
    if n_workers > 1:
        print(f"  >>> Using multiprocessing Pool with {n_workers} workers <<<")
        # Windows requires spawn; safe across platforms
        try:
            set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # already set
        with Pool(processes=n_workers) as pool:
            summary = pool.map(_train_mode_worker, worker_args)
    else:
        print(f"  >>> Serial training (1 worker) <<<")
        summary = [_train_mode_worker(wa) for wa in worker_args]

    elapsed_total = time.time() - t_total
    print(f"\n{'='*65}")
    print(f"  All training done: {elapsed_total/60:.1f} min "
          f"(workers={n_workers}, modes={len(modes_to_train)})")
    print(f"  Experiment dir: {exp_dir}")
    print(f"{'='*65}")

    # Save summary
    with open(os.path.join(exp_dir, "training_summary.json"), 'w') as f:
        json.dump(summary, f, indent=4)
    for s in summary:
        status_str = s.get('status', '?')
        time_str = f" ({s.get('train_time_sec', 0):.0f}s)" if status_str == 'OK' else ''
        print(f"  {s['mode']:<25} {status_str}{time_str}")
        if status_str == 'FAILED':
            print(f"      error: {s.get('error', '?')}")


if __name__ == "__main__":
    main()
