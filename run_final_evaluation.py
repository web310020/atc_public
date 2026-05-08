"""
最终 50 次独立 eval (跑完 batch_train 之后用).

7 个 mode 各跑 50 次 eval, 输出 summary_stats_final.md
(mean +/- std, 直接对接 paper 主表).

Usage:
    python run_final_evaluation.py
"""

import os
import sys
import glob
import json
import time
import numpy as np
from datetime import datetime
from scipy.stats import pearsonr

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from core.telemetry_env import E2_Node_Simulator

N_RUNS = 100
N_EVAL_STEPS = 1000


def run_final_evaluation():
    # Find latest experiment
    list_of_experiments = glob.glob(os.path.join("experiments", "*"))
    exp_dir = max(list_of_experiments, key=os.path.getmtime)

    config_path = os.path.join(exp_dir, "experiment_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        K = config.get("K", 1)
        use_kalman = config.get("use_kalman", False)
    else:
        K = 1
        use_kalman = False

    print(f"\n{'='*65}")
    print(f"  FINAL EVALUATION: {N_RUNS} runs × 7 modes")
    print(f"  Experiment: {exp_dir}")
    print(f"  Config: K={K}, KF={'ON' if use_kalman else 'OFF'}")
    print(f"{'='*65}")

    modes_mapping = {
        "proposed":        r"\textbf{ATC (Proposed)}",
        "safeslice":       r"SafeSlice \cite{SafeSlice_2025}",
        "vanilla_ppo":     r"Vanilla-PPO \cite{ppo_2017}",
        "lstm_predictive": r"LSTM-Pred \cite{deepcog_2020}",
        "static_slicing":  r"Static-Slicing \cite{threegpp_static}",
        "oracle":          r"Oracle (ZD-Utility)",
        "guardrail_only":  r"Guardrail-only"
    }

    all_results = {}
    t_total = time.time()

    for m, label in modes_mapping.items():
        model_path = os.path.join(exp_dir, m, "models", f"ppo_{m}_final")
        if m != "static_slicing" and not os.path.exists(model_path + ".zip"):
            print(f"  Skipping {m} (model not found)")
            continue

        if m == "static_slicing":
            model = None
        elif m == "lstm_predictive":
            model = RecurrentPPO.load(model_path, device='cpu')
        else:
            model = PPO.load(model_path, device='cpu')

        print(f"\n--- {m}: {N_RUNS} runs ---")
        runs = []
        for run_i in range(N_RUNS):
            env = E2_Node_Simulator(mode=m, K=K, use_kalman=use_kalman)
            env.set_a1_policy(0.5)
            obs, _ = env.reset()

            stats = {"util": [], "belief_mean": [], "viol": [], "viol_depth": [],
                     "cost": [], "reward": [], "latency": []}

            for _ in range(N_EVAL_STEPS):
                t0 = time.time()
                if m == "static_slicing":
                    action = np.full(K, 1.0 / K) if K > 1 else np.array([0.5])
                    latency = 0.01
                else:
                    action, _ = model.predict(obs, deterministic=True)
                    latency = (time.time() - t0) * 1000
                obs, reward, _, _, info = env.step(action)
                u = info["true_util"]
                stats["util"].append(u)
                stats["belief_mean"].append(info["belief_mean"])
                stats["viol_depth"].append(max(0, u - 0.5))
                stats["viol"].append(info["is_violation"])
                stats["cost"].append(info["sig_cost"])
                stats["reward"].append(reward)
                stats["latency"].append(latency)

            try:
                r_all, _ = pearsonr(stats["util"], stats["belief_mean"])
            except:
                r_all = 0.0

            u_arr = np.array(stats["util"])
            b_arr = np.array(stats["belief_mean"])
            c_arr = np.array(stats["cost"])
            silent = c_arr == 0
            try:
                r_sil, _ = pearsonr(u_arr[silent], b_arr[silent]) if np.sum(silent) > 10 else (0, 0)
            except:
                r_sil = 0.0

            runs.append({
                "U": np.mean(stats["util"]),
                "delta": np.mean(stats["viol_depth"]),
                "viol_pct": np.mean(stats["viol"]) * 100,
                "r_all": r_all,
                "r_sil": r_sil,
                "sig_cost": np.sum(stats["cost"]),
                "latency": np.mean(stats["latency"]),
                "psi": abs(np.mean(stats["reward"])),
            })

            if (run_i + 1) % 10 == 0:
                avg_u = np.mean([r["U"] for r in runs])
                avg_v = np.mean([r["viol_pct"] for r in runs])
                print(f"  {run_i+1}/{N_RUNS}: avg U={avg_u:.3f}, avg Viol={avg_v:.1f}%")

        all_results[m] = runs

    total_time = time.time() - t_total
    save_final_md(all_results, modes_mapping, exp_dir, K, use_kalman, total_time)


def save_final_md(all_results, modes_mapping, exp_dir, K, use_kalman, total_time):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    lines = [
        "# Final Evaluation Results (50 runs)",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Experiment:** `{exp_dir}`",
        f"**Config:** K={K}, KF={'ON' if use_kalman else 'OFF'}, L6=Proportional(m=0.15)",
        f"**Protocol:** {N_RUNS} independent runs × {N_EVAL_STEPS} steps per run",
        f"**Total time:** {total_time:.0f}s ({total_time/60:.1f} min)",
        "",
        "## Main Results (for Paper Table III)",
        "",
        "| Method | U ↑ | δ ↓ | Viol.% ↓ | r_all | r_sil | Cost | Lat.(ms) | Ψ† |",
        "|--------|-----|-----|----------|-------|-------|------|----------|----|",
    ]

    display_order = ["proposed", "safeslice", "vanilla_ppo", "lstm_predictive",
                     "static_slicing", "oracle", "guardrail_only"]

    summary = {}
    for m in display_order:
        if m not in all_results:
            continue
        runs = all_results[m]
        agg = {}
        for key in runs[0].keys():
            vals = [r[key] for r in runs]
            agg[key] = {"mean": np.mean(vals), "std": np.std(vals)}
        summary[m] = agg
        label = modes_mapping.get(m, m)
        lines.append(
            f"| {label} | "
            f"{agg['U']['mean']:.2f} ± {agg['U']['std']:.2f} | "
            f"{agg['delta']['mean']:.4f} ± {agg['delta']['std']:.4f} | "
            f"{agg['viol_pct']['mean']:.1f} ± {agg['viol_pct']['std']:.1f}% | "
            f"{agg['r_all']['mean']:.3f} ± {agg['r_all']['std']:.3f} | "
            f"{agg['r_sil']['mean']:.3f} ± {agg['r_sil']['std']:.3f} | "
            f"{agg['sig_cost']['mean']:.0f} | "
            f"{agg['latency']['mean']:.2f} | "
            f"{agg['psi']['mean']:.2f} ± {agg['psi']['std']:.2f} |"
        )

    # Key metrics for paper
    if "proposed" in summary:
        atc = summary["proposed"]
        lines.extend([
            "",
            "## Key Paper Numbers",
            "",
            f"- **ATC U:** {atc['U']['mean']:.3f} ± {atc['U']['std']:.3f}",
            f"- **ATC δ:** {atc['delta']['mean']:.4f} ± {atc['delta']['std']:.4f}",
            f"- **ATC Viol.%:** {atc['viol_pct']['mean']:.1f} ± {atc['viol_pct']['std']:.1f}%",
            f"- **ATC Signaling:** {atc['sig_cost']['mean']:.0f} ± {atc['sig_cost']['std']:.0f}",
            f"- **ATC r_all / r_sil:** {atc['r_all']['mean']:.3f} / {atc['r_sil']['mean']:.3f}",
        ])

        if "vanilla_ppo" in summary:
            vp = summary["vanilla_ppo"]
            gain = atc['U']['mean'] / vp['U']['mean'] if vp['U']['mean'] > 0 else 0
            lines.append(f"- **Gain vs Vanilla-PPO:** {gain:.1f}×")

        if "oracle" in summary:
            orc = summary["oracle"]
            sig_ratio = atc['sig_cost']['mean'] / orc['sig_cost']['mean'] * 100 if orc['sig_cost']['mean'] > 0 else 0
            lines.append(f"- **Signaling vs Oracle:** {sig_ratio:.1f}% of Oracle budget")
            lines.append(f"- **Signaling reduction:** {100-sig_ratio:.1f}%")

    lines.extend(["", "---",
                   f"Generated: {ts}",
                   f"{N_RUNS} runs, L6 = proportional (m=0.15), 3-state discrete belief"])

    filename = f"summary_stats_final_{ts}.md"
    exp_path = os.path.join(exp_dir, filename)
    with open(exp_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    exp_latest = os.path.join(exp_dir, "summary_stats_final.md")
    with open(exp_latest, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n>>> Results saved to: {exp_path}")
    print(f">>> Total time: {total_time:.0f}s ({total_time/60:.1f} min)")


if __name__ == "__main__":
    run_final_evaluation()
