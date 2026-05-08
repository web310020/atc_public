# batch_train.py
import os
import time
import pytz
import json
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from core.telemetry_env import E2_Node_Simulator
from sb3_contrib import RecurrentPPO


os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""

# --- 配置 ---
K = 1                  # slice 数 (K=1 为代表性单 slice 设置)
USE_KALMAN = False     # 用 discrete 3-state belief engine
TOTAL_STEPS = 120000   # 每个 mode 的训练 steps

# --- 初始化 experiment 目录 ---
tz = pytz.timezone('Asia/Seoul')
timestamp = datetime.now(tz).strftime("%Y%m%d_%H%M%S%f")[:-3]
base_dir = os.path.join("experiments", timestamp)
os.makedirs(base_dir, exist_ok=True)


class A1_Policy_Callback(BaseCallback):
    """
    Simulates A1 interface dynamic policy injection.
    Changes SLA target mid-training to improve generalization.
    """
    def __init__(self, verbose=0):
        super(A1_Policy_Callback, self).__init__(verbose)

    def _on_step(self) -> bool:
        if self.num_timesteps == self.locals["total_timesteps"] // 2:
            new_sla = 0.50
            new_tau = 0.45
            self.training_env.env_method("set_a1_policy", new_sla)
            self.training_env.env_method("set_tau", new_tau)
            if self.verbose > 0:
                print(f"\n[A1] Intent change: SLA={new_sla}, tau={new_tau}")
        return True


def run_experiment(mode, total_steps=TOTAL_STEPS, timing_log=None):
    t_start = time.time()
    print(f"\n{'=' * 50}")
    print(f">>> Training: {mode.upper()} (K={K}, KF={'ON' if USE_KALMAN else 'OFF'})")
    print(f"{'=' * 50}")

    mode_dir = os.path.join(base_dir, mode)
    os.makedirs(os.path.join(mode_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(mode_dir, "logs"), exist_ok=True)

    # Initialize environment with K slices
    env = E2_Node_Simulator(mode=mode, K=K, use_kalman=USE_KALMAN)

    # Policy network size scales with observation/action dimensions
    obs_dim = env.observation_space.shape[0]  # 2 + 3*K
    act_dim = env.action_space.shape[0]       # K

    if mode == "safeslice":
        policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
        model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs,
                     verbose=0, tensorboard_log=os.path.join(mode_dir, "logs"), device='cpu')
    elif mode == "lstm_predictive":
        model = RecurrentPPO("MlpLstmPolicy", env,
                             verbose=0, tensorboard_log=os.path.join(mode_dir, "logs"), device='cpu')
    else:
        # Default: 3-layer MLP scaled for K-dimensional I/O
        policy_kwargs = dict(net_arch=dict(pi=[512, 512, 256], vf=[512, 512, 256]))
        model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs,
                     verbose=0, tensorboard_log=os.path.join(mode_dir, "logs"), device='cpu')

    callback = A1_Policy_Callback(verbose=1)
    model.learn(total_timesteps=total_steps, callback=callback)
    model.save(os.path.join(mode_dir, "models", f"ppo_{mode}_final"))

    # Quick diagnostic: run 200 steps and log key parameters
    diag_env = E2_Node_Simulator(mode=mode, K=K, use_kalman=USE_KALMAN)
    diag_env.set_a1_policy(0.5)
    obs, _ = diag_env.reset()
    diag_data = {"gamma": [], "entropy": [], "period": [], "util": [], "sig_cost": []}
    for _ in range(200):
        if mode == "static_slicing":
            act = np.full(K, 1.0 / K) if K > 1 else np.array([0.5])
        else:
            act, _ = model.predict(obs, deterministic=True)
        obs, _, _, _, info = diag_env.step(act)
        diag_data["gamma"].append(info["gamma"])
        diag_data["entropy"].append(info["entropy"])
        diag_data["period"].append(info["period"])
        diag_data["util"].append(info["true_util"])
        diag_data["sig_cost"].append(info["sig_cost"])

    diag_md = [
        f"# Diagnostic: {mode}",
        f"| Metric | Mean | Min | Max |",
        f"|--------|------|-----|-----|",
        f"| gamma | {np.mean(diag_data['gamma']):.3f} | {np.min(diag_data['gamma']):.3f} | {np.max(diag_data['gamma']):.3f} |",
        f"| entropy | {np.mean(diag_data['entropy']):.3f} | {np.min(diag_data['entropy']):.3f} | {np.max(diag_data['entropy']):.3f} |",
        f"| period | {np.mean(diag_data['period']):.1f} | {np.min(diag_data['period']):.1f} | {np.max(diag_data['period']):.1f} |",
        f"| util | {np.mean(diag_data['util']):.3f} | {np.min(diag_data['util']):.3f} | {np.max(diag_data['util']):.3f} |",
        f"| total_sig | {np.sum(diag_data['sig_cost']):.0f} | — | — |",
    ]
    with open(os.path.join(mode_dir, "diagnostic.md"), "w") as f:
        f.write("\n".join(diag_md))

    # Metadata
    metadata = {
        "standard": "O-RAN-R005-2026",
        "K": K,
        "use_kalman": USE_KALMAN,
        "intent_type": "Latency-Aware-Slicing",
        "kpm_sources": ["DRB.PRB.UsedDl", "DRB.PdcpPduDelay"],
        "mode": mode,
        "total_steps": total_steps,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
    }
    with open(os.path.join(mode_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    elapsed = time.time() - t_start
    print(f">>> {mode.upper()} done in {elapsed:.1f}s")
    if timing_log is not None:
        timing_log.append({"mode": mode, "seconds": round(elapsed, 1)})


if __name__ == "__main__":
    import time as time_mod
    total_start = time_mod.time()

    target_modes = [
        "proposed", "vanilla_ppo", "static_slicing",
        "safeslice", "lstm_predictive", "oracle", "guardrail_only"
    ]

    config = {"K": K, "use_kalman": USE_KALMAN, "total_steps": TOTAL_STEPS,
              "timestamp": timestamp, "modes": target_modes}
    with open(os.path.join(base_dir, "experiment_config.json"), "w") as f:
        json.dump(config, f, indent=4)

    timing_log = []
    for mode in target_modes:
        if mode == "static_slicing":
            mode_dir = os.path.join(base_dir, mode)
            os.makedirs(os.path.join(mode_dir, "models"), exist_ok=True)
            with open(os.path.join(mode_dir, "models", f"ppo_{mode}_final.zip"), "w") as f:
                f.write("dummy")
            timing_log.append({"mode": "static_slicing", "seconds": 0.0})
            continue

        run_experiment(mode=mode, timing_log=timing_log)

    total_elapsed = time_mod.time() - total_start

    md_lines = [
        f"# Training Timing Report",
        f"",
        f"**Experiment:** `{base_dir}`",
        f"**Config:** K={K}, KF={'ON' if USE_KALMAN else 'OFF'}, steps={TOTAL_STEPS}",
        f"**Total time:** {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)",
        f"",
        f"| Mode | Time (s) | Time (min) |",
        f"|------|----------|------------|",
    ]
    for entry in timing_log:
        md_lines.append(f"| {entry['mode']} | {entry['seconds']:.1f} | {entry['seconds']/60:.1f} |")
    md_lines.append(f"| **Total** | **{total_elapsed:.1f}** | **{total_elapsed/60:.1f}** |")

    timing_path = os.path.join(base_dir, "training_timing.md")
    with open(timing_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\n{'=' * 60}")
    print(f"All training complete in {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"Results in: {base_dir}")
    print(f"Timing report: {timing_path}")
    print(f"{'=' * 60}")
