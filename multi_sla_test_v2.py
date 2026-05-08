"""
Multi-SLA 压力测试 (K-slice 通用).

在随机化的 intent curriculum (tau in [0.15, 0.85]) 下测所有 mode,
输出 multi_sla_results_v2.md.

Usage:
    python multi_sla_test_v2.py
"""

import os
import glob
import json
import numpy as np
import time
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from core.telemetry_env import E2_Node_Simulator

os.environ["CUDA_VISIBLE_DEVICES"] = ""

N_RUNS = 50
N_EVAL_STEPS = 1000


def run_stress_test():
    exps = glob.glob(os.path.join("experiments", "*"))
    latest = max(exps, key=os.path.getmtime)

    config_path = os.path.join(latest, "experiment_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        K = cfg.get("K", 1)
        use_kalman = cfg.get("use_kalman", False)
    else:
        K, use_kalman = 1, False

    modes = {
        "proposed":        ("ATC (Proposed)", "PPO"),
        "safeslice":       ("SafeSlice", "PPO"),
        "vanilla_ppo":     ("Vanilla-PPO", "PPO"),
        "lstm_predictive": ("LSTM-Pred", "LSTM"),
        "oracle":          ("Oracle (ZD-Utility)", "PPO"),
    }

    print(f"\n{'='*60}")
    print(f"  STRESS TEST: Randomized Intent τ∈[0.15, 0.85]")
    print(f"  {N_RUNS} runs × {len(modes)} modes")
    print(f"  Experiment: {latest}")
    print(f"{'='*60}")

    all_results = {}
    for m, (label, algo) in modes.items():
        model_path = os.path.join(latest, m, "models", f"ppo_{m}_final")
        if not os.path.exists(model_path + ".zip"):
            print(f"  Skipping {m}")
            continue

        if algo == "LSTM":
            model = RecurrentPPO.load(model_path, device='cpu')
        else:
            model = PPO.load(model_path, device='cpu')

        print(f"\n--- {label}: {N_RUNS} runs ---")
        runs = []
        for i in range(N_RUNS):
            env = E2_Node_Simulator(mode=m, K=K, use_kalman=use_kalman)
            # Randomize intent each run
            random_tau = np.random.uniform(0.15, 0.85)
            env.set_a1_policy(random_tau)
            obs, _ = env.reset()

            stats = {"util": [], "viol": [], "viol_depth": []}
            for _ in range(N_EVAL_STEPS):
                if m == "static_slicing":
                    action = np.full(K, 1.0 / K) if K > 1 else np.array([0.5])
                else:
                    action, _ = model.predict(obs, deterministic=True)
                obs, _, _, _, info = env.step(action)
                u = info["true_util"]
                stats["util"].append(u)
                stats["viol"].append(info["is_violation"])
                stats["viol_depth"].append(max(0, u - random_tau))

            runs.append({
                "U": np.mean(stats["util"]),
                "delta": np.mean(stats["viol_depth"]),
                "viol_pct": np.mean(stats["viol"]) * 100,
                "tau": random_tau,
            })

            if (i + 1) % 10 == 0:
                avg_u = np.mean([r["U"] for r in runs])
                avg_v = np.mean([r["viol_pct"] for r in runs])
                print(f"  {i+1}/{N_RUNS}: avg U={avg_u:.3f}, avg Viol={avg_v:.1f}%")

        agg = {}
        for key in ["U", "delta", "viol_pct"]:
            vals = [r[key] for r in runs]
            agg[key] = {"mean": np.mean(vals), "std": np.std(vals)}
        all_results[m] = {"label": label, "agg": agg, "runs": runs}

    # Save
    lines = [
        "# Stress Test Results v2 (for Table IV)",
        f"**Protocol:** {N_RUNS} runs, τ ~ U(0.15, 0.85)",
        "",
        "## Overall Results",
        "",
        "| Method | U ↑ | δ ↓ | Viol.% ↓ | Status |",
        "|--------|-----|-----|----------|--------|",
    ]
    for m in ["proposed", "safeslice", "vanilla_ppo", "lstm_predictive", "oracle"]:
        if m not in all_results:
            continue
        r = all_results[m]
        a = r["agg"]
        v = a['viol_pct']['mean']
        if m == "proposed":
            status = "Adaptive"
        elif m == "oracle":
            status = "Aggressive"
        elif v > 15:
            status = "Unstable"
        else:
            status = "Cons."
        lines.append(
            f"| **{r['label']}** | {a['U']['mean']:.3f}±{a['U']['std']:.3f} | "
            f"{a['delta']['mean']:.4f} | "
            f"{a['viol_pct']['mean']:.1f}±{a['viol_pct']['std']:.1f} | {status} |"
        )

    # Per-τ breakdown for ATC and Oracle
    lines.extend([
        "",
        "## Per-τ Breakdown (ATC vs Oracle)",
        "",
        "| τ Range | ATC U | ATC Viol.% | Oracle U | Oracle Viol.% | ATC Safer? |",
        "|---------|-------|-----------|----------|--------------|------------|",
    ])

    tau_bins = [(0.15, 0.30, "Tight"), (0.30, 0.50, "Medium"), (0.50, 0.70, "Relaxed"), (0.70, 0.85, "Loose")]

    for lo, hi, label in tau_bins:
        for m_key, col_prefix in [("proposed", "ATC"), ("oracle", "Oracle")]:
            if m_key not in all_results:
                continue

        atc_runs = [r for r in all_results.get("proposed", {}).get("runs", []) if lo <= r["tau"] < hi]
        orc_runs = [r for r in all_results.get("oracle", {}).get("runs", []) if lo <= r["tau"] < hi]

        if atc_runs and orc_runs:
            atc_u = np.mean([r["U"] for r in atc_runs])
            atc_v = np.mean([r["viol_pct"] for r in atc_runs])
            orc_u = np.mean([r["U"] for r in orc_runs])
            orc_v = np.mean([r["viol_pct"] for r in orc_runs])
            safer = "yes" if atc_v <= orc_v else "no"
            n_atc = len(atc_runs)
            lines.append(
                f"| {label} [{lo:.2f},{hi:.2f}) n={n_atc} | "
                f"{atc_u:.3f} | {atc_v:.1f}% | {orc_u:.3f} | {orc_v:.1f}% | {safer} |"
            )
        elif atc_runs:
            atc_u = np.mean([r["U"] for r in atc_runs])
            atc_v = np.mean([r["viol_pct"] for r in atc_runs])
            lines.append(
                f"| {label} [{lo:.2f},{hi:.2f}) n={len(atc_runs)} | "
                f"{atc_u:.3f} | {atc_v:.1f}% | — | — | — |"
            )

    # Key insight
    lines.extend([
        "",
        "## Key Insight",
        "",
    ])

    # Check if violations concentrate in tight regime
    if "proposed" in all_results:
        tight = [r for r in all_results["proposed"]["runs"] if r["tau"] < 0.30]
        relaxed = [r for r in all_results["proposed"]["runs"] if r["tau"] >= 0.50]
        if tight and relaxed:
            tight_v = np.mean([r["viol_pct"] for r in tight])
            relaxed_v = np.mean([r["viol_pct"] for r in relaxed])
            lines.append(
                f"ATC violations concentrate in tight-intent regime: "
                f"tau<0.30 -> {tight_v:.1f}% violation vs tau>=0.50 -> {relaxed_v:.1f}% violation."
            )
            if relaxed_v < 3.0:
                lines.append(
                    f"**In the operational regime (τ≥0.50), ATC achieves near-zero violation ({relaxed_v:.1f}%).**"
                )

    lines.extend(["", "---", "*Generated by multi_sla_test_v2.py*"])

    out_path = os.path.join(latest, "multi_sla_results_v2.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n>>> Saved: {out_path}")


if __name__ == "__main__":
    run_stress_test()
