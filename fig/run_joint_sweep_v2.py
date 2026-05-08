# Joint Latency x Beta sweep, K-slice 通用.
# 生成 fig_joint_landscape_1x4.pdf 以及 joint_sensitivity_v2.csv
# (后者会被 plot_empirical_trap.py 用到).
#
# Usage: python fig/run_joint_sweep_v2.py

import os, glob, json, numpy as np, pandas as pd
import matplotlib
from sb3_contrib import RecurrentPPO

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from core.telemetry_env import E2_Node_Simulator
from stable_baselines3 import PPO
import matplotlib.patches as patches

os.environ["CUDA_VISIBLE_DEVICES"] = ""

plt.rcParams.update({
    "text.usetex": True, "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    # Source font sizes scaled ~3.36x for a 1x5 panel layout at 1.0\linewidth (~0.298x per panel).
    "font.size": 28, "axes.labelsize": 28, "xtick.labelsize": 26, "ytick.labelsize": 26,
    "savefig.dpi": 300,
    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amsfonts} \usepackage{amssymb}"
})

def save_dual_format(fig, out_dir, base_name):
    """Save figure to BOTH PDF (vector, paper) and PNG (raster).
    Handles text.usetex=True silent-fail on PNG by retrying with usetex=False."""
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, base_name + ".pdf")
    png_path = os.path.join(out_dir, base_name + ".png")
    fig.savefig(pdf_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    print(f"  PDF saved: {pdf_path}")
    try:
        fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
        print(f"  PNG saved: {png_path}")
    except Exception as e:
        print(f"  PNG with usetex failed ({type(e).__name__}: {e}); retry without usetex")
        with plt.rc_context({"text.usetex": False}):
            fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
        print(f"  PNG saved (no usetex): {png_path}")


PAPER_FIG_DIR = "paper_draft/figures" if os.path.isdir("paper_draft/figures") \
    else os.path.join("..", "paper_draft", "figures")

def run_joint_sweep(exp_dir=None):
    if exp_dir is None:
        exps = glob.glob(os.path.join("experiments", "*"))
        # K=1 main-run filter
        _K1_EXCLUDE = ("k3_", "k5_", "k_exploration", "sanity", "decision", "seedbump", "probe")
        exps_k1 = [e for e in exps if not any(kw in os.path.basename(e) for kw in _K1_EXCLUDE)]
        if not exps_k1:
            raise FileNotFoundError("No K=1 main experiment dir found under experiments/.")
        latest_exp = max(exps_k1, key=os.path.getmtime)
        print(f">>> Using K=1 main run: {latest_exp}")
    else:
        latest_exp = exp_dir

    config_path = os.path.join(latest_exp, "experiment_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        K = cfg.get("K", 1)
        use_kalman = cfg.get("use_kalman", False)
    else:
        K, use_kalman = 1, False

    latency_range = [10, 50, 100, 200, 400, 600, 800, 1000]
    beta_range = [0.001, 0.01, 0.1, 1, 10, 100, 200]
    modes = ["proposed", "safeslice", "vanilla_ppo", "lstm_predictive"]

    print(f"\n>>> Joint Sweep (5-seed avg): {latest_exp} (K={K})\n")

    models = {}
    for m in modes:
        path = os.path.join(latest_exp, m, "models", f"ppo_{m}_final")
        if os.path.exists(path + ".zip"):
            if m == "lstm_predictive":
                models[m] = RecurrentPPO.load(path, device='cpu')
            else:
                models[m] = PPO.load(path, device='cpu')

    results = []
    eval_seeds = [42, 123, 456, 789, 1010]

    for d in latency_range:
        for b in beta_range:
            temp_res = {"Latency": d, "Beta": b}

            for m in modes:
                if m not in models:
                    temp_res[m] = 0.0
                    continue

                seed_means = []
                for seed in eval_seeds:
                    env = E2_Node_Simulator(mode=m, K=K, use_kalman=use_kalman,
                                           base_period=d, beta=b)
                    env.set_a1_policy(0.5)
                    obs, _ = env.reset(seed=seed)
                    utils = []
                    for _ in range(500):
                        action, _ = models[m].predict(obs, deterministic=True)
                        obs, _, _, _, info = env.step(action)
                        utils.append(info.get("true_util", 0))
                    seed_means.append(np.mean(utils))
                temp_res[m] = np.mean(seed_means)

            baselines_utils = [temp_res[m] for m in modes if m != "proposed"]
            best_baseline = max(baselines_utils) if baselines_utils else 0.001
            temp_res["Gain_vs_Best"] = (temp_res["proposed"] - best_baseline) / (best_baseline + 1e-6) * 100

            all_vals = {m: temp_res[m] for m in modes}
            winner = max(all_vals, key=all_vals.get)

            print("-" * 80)
            print(f"| Latency:{d:4}ms | Beta:{b:6.3f} |")
            print(f"| Proposed: {temp_res['proposed']:.3f} (*) | SafeSlice: {temp_res['safeslice']:.3f} | "
                  f"Vanilla: {temp_res['vanilla_ppo']:.3f} | LSTM-Pred: {temp_res['lstm_predictive']:.3f} |")
            print(f"| [WINNER]: {winner:<15} | [GAIN vs BEST]: {temp_res['Gain_vs_Best']:>6.1f}% |")

            results.append(temp_res)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(latest_exp, "joint_sensitivity_v2.csv"), index=False)
    plot_faceted_heatmaps(df, latest_exp)


def plot_faceted_heatmaps(df, save_dir):
    modes_plot = ["Proposed", "LSTM-Pred", "Vanilla-PPO", "SafeSlice"]
    fig, axes = plt.subplots(1, 5, figsize=(24, 5.5),
                              gridspec_kw={'width_ratios': [1, 1, 1, 1, 0.06]})
    hm_axes = axes[:4]
    cbar_ax = axes[4]

    # Display name -> DataFrame column key mapping
    display_to_key = {
        "Proposed": "proposed",
        "LSTM-Pred": "lstm_predictive",
        "Vanilla-PPO": "vanilla_ppo",
        "SafeSlice": "safeslice",
    }
    for i, m in enumerate(modes_plot):
        mName = m
        m_key = display_to_key.get(m, m.lower())
        if mName == "Proposed": mName = "ATC (Ours)"

        pivot = df.pivot(index="Beta", columns="Latency", values=m_key).sort_index(ascending=True)

        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", ax=hm_axes[i],
                    vmin=0, vmax=0.45,
                    cbar=(i == 3),
                    cbar_ax=cbar_ax if i == 3 else None,
                    cbar_kws={'label': r'Avg. Saturation $\mathbb{E}[U]$'} if i == 3 else {})

        hm_axes[i].set_title(rf"\textbf{{{mName}}}", fontsize=18, pad=10)
        hm_axes[i].set_xlabel("Latency $D$ (ms)", fontsize=14)

        if i == 0:
            hm_axes[i].set_ylabel(r"Penalty $\beta$", fontsize=16)
        else:
            hm_axes[i].set_ylabel("")

        hm_axes[i].invert_yaxis()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(wspace=0.10)

    #save_path = os.path.join(save_dir, "fig_joint_landscape_1x4.pdf")
    #plt.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
    #print(f"\n>>> Saved: {save_path}")
    save_dual_format(plt, PAPER_FIG_DIR, "fig_joint_landscape_1x4")
    plt.close()


if __name__ == "__main__":
    import sys
    exp = sys.argv[1] if len(sys.argv) > 1 else None
    run_joint_sweep(exp_dir=exp)
