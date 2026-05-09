"""
plot_perf_distributions.py — NEW Fig 3 (lean-6 design)

Single-col 2-panel CDF: Saturation (left) + Violation Depth (right).
Replaces the 1x4 figure* horizontal subfig layout in current Fig 4
(drops fidelity + scores subfigs; merges saturation_cdf + violation_depth into 2-panel).

Data: same K=1 main-run rollouts used by auto_plot_v2.py.
We re-use auto_plot_v2's data collection but plot in 2-panel single-col.

Usage: python -m fig.plot_perf_distributions
Output: paper_draft/figures/fig_perf_distributions.pdf + .png
"""
import os, glob, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from core.telemetry_env import E2_Node_Simulator

os.environ["CUDA_VISIBLE_DEVICES"] = ""

plt.rcParams.update({
    "text.usetex": True, "font.family": "serif",
    "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7,
    "lines.linewidth": 1.4, "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": ":",
    "savefig.dpi": 600,
    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amsfonts} \usepackage{amssymb}"
})


def save_dual_format(fig, out_dir, base_name):
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


METHOD_STYLE = {
    # Baseline: MatchedRef-PPO (D=0) is the zero-delay full-information reference
    "oracle":          {"label": "MatchedRef-PPO ($D{=}0$)", "color": "black",   "ls": "--", "lw": 1.5, "alpha":0.7},
    "proposed":        {"label": r"\textbf{ATC (Ours)}",  "color": "#d62728", "ls": "-",  "lw": 2.0, "alpha":0.7},
    "lstm_predictive": {"label": "LSTM-Pred",             "color": "#1f77b4", "ls": "-.", "lw": 1.3, "alpha":0.7},
    "vanilla_ppo":     {"label": "Vanilla-PPO",           "color": "#2ca02c", "ls": ":",  "lw": 1.3, "alpha":0.7},
    "safeslice":       {"label": "SafeSlice",             "color": "#ff7f0e", "ls": (0, (3, 1, 1, 1)), "lw": 1.3, "alpha":0.7},
}

ROLLOUT_LEN = 10000


def find_K1_main():
    _K1_EXCLUDE = ("k3_", "k5_", "k_exploration", "sanity", "decision", "seedbump", "probe", "beta_tuned", "bestref", "multiseed", "path_gamma", "l3_", "ablations", "habib")
    candidates = glob.glob("experiments/*")
    k1_dirs = [e for e in candidates if not any(kw in os.path.basename(e) for kw in _K1_EXCLUDE)]
    if not k1_dirs:
        raise FileNotFoundError("No K=1 experiment dir found under experiments/")
    return max(k1_dirs, key=os.path.getmtime)


def collect_rollouts(exp_dir):
    """Run rollouts for each method to collect U trajectory + violation depths."""
    cfg_path = os.path.join(exp_dir, "experiment_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        K = cfg.get("K", 1)
        use_kalman = cfg.get("use_kalman", False)
    else:
        K, use_kalman = 1, False

    tau = 0.5
    data = {}
    for mode in METHOD_STYLE.keys():
        path = os.path.join(exp_dir, mode, "models", f"ppo_{mode}_final")
        if not os.path.exists(path + ".zip"):
            print(f"  ! model not found: {path}.zip — skipping {mode}")
            continue
        if mode == "lstm_predictive":
            model = RecurrentPPO.load(path, device="cpu")
        else:
            model = PPO.load(path, device="cpu")
        env = E2_Node_Simulator(mode=mode, K=K, use_kalman=use_kalman)
        env.set_a1_policy(0.5)
        obs, _ = env.reset(seed=42)
        u_list, viol_depth = [], []
        for _ in range(ROLLOUT_LEN):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, _, _, info = env.step(action)
            u = info.get("true_util", 0.0)
            u_list.append(u)
            if u > tau:
                viol_depth.append(u - tau)
            else:
                viol_depth.append(0.0)
        data[mode] = {"U": np.asarray(u_list), "viol_depth": np.asarray(viol_depth)}
        print(f"  ok: {mode} (U_mean={data[mode]['U'].mean():.3f})")
    return data, tau


def plot_2panel(data, tau, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(3.49, 2.2))
    ax_l, ax_r = axes

    # Left: Saturation CDF
    for mode, d in data.items():
        u = np.sort(d["U"])
        cdf = np.arange(1, len(u) + 1) / len(u)
        cfg = METHOD_STYLE[mode]
        ax_l.plot(u, cdf, label=cfg["label"], color=cfg["color"], ls=cfg["ls"], lw=cfg["lw"], alpha=cfg.get("alpha", 0.8))
    ax_l.axvline(tau, color="#d62728", linestyle=":", lw=0.9, alpha=0.5)
    ax_l.text(tau + 0.02, 0.05, r"SLA $\tau$", color="#d62728", fontsize=6.5)
    ax_l.set_xlabel(r"Saturation $U_t$")
    ax_l.set_ylabel(r"$F(U_t)$")
    ax_l.set_xlim(0, 0.7)
    ax_l.set_ylim(0, 1.02)

    # Right: Violation Depth CDF (only positive depths)
    for mode, d in data.items():
        depths = d["viol_depth"]
        nonzero = depths[depths > 0]
        if len(nonzero) == 0:
            # All zero -> draw a step at 0
            ax_r.plot([0, 0.1], [1, 1], label=METHOD_STYLE[mode]["label"],
                       color=METHOD_STYLE[mode]["color"], ls=METHOD_STYLE[mode]["ls"],
                       lw=METHOD_STYLE[mode]["lw"])
            continue
        s = np.sort(nonzero)
        cdf = np.arange(1, len(s) + 1) / len(s)
        cfg = METHOD_STYLE[mode]
        ax_r.plot(s, cdf, label=cfg["label"], color=cfg["color"], ls=cfg["ls"], lw=cfg["lw"], alpha=cfg.get("alpha", 0.8))
    # psi = violation depth (paper notation; avoids overload with delta_t = tactical-slot length)
    ax_r.set_xlabel(r"Violation depth $\psi$")
    ax_r.set_ylabel(r"$F(\psi)$")
    ax_r.set_xlim(0, 0.12)
    ax_r.set_ylim(0, 1.02)

    # Single legend for both panels
    handles, labels = ax_l.get_legend_handles_labels()

    # --- 2. 定义手动排序映射 (0 为第一位，4 为最后一位) ---
    order_map = {
        r"\textbf{ATC (Ours)}": 0,
        "LSTM-Pred": 1,
        "Vanilla-PPO": 2,
        "SafeSlice": 3,
        r"MatchedRef-PPO ($D{=}0$)": 4,  # 设置为最大值，排在最后
    }
    # --- 3. 执行排序 ---
    new_handles_labels = sorted(zip(handles, labels), key=lambda x: order_map.get(x[1], 99))
    ordered_handles, ordered_labels = zip(*new_handles_labels)

    fig.legend(ordered_handles, ordered_labels, loc="upper center", bbox_to_anchor=(0.54, 1.04),
                ncol=3, frameon=True, framealpha=0.8, edgecolor="gray",
                fontsize=6.5,
                #bbox_to_anchor = (-0.01, 1.014),
                #fontsize = 7, framealpha = 1.0, ncol = 2,
                labelspacing = 0.2, handletextpad = 0.2, columnspacing = 0.4,
                handlelength = 1.5, borderpad = 0.2,)
    leg = fig.legends[0] if fig.legends else None
    if leg is not None:
        leg.get_frame().set_linewidth(0.4)

    fig.tight_layout(pad=0.3, rect=[0, 0, 1, 0.92])
    save_dual_format(fig, out_dir, "fig_perf_distributions")
    plt.close(fig)


def main():
    exp_dir = find_K1_main()
    print(f">>> Using K=1 main run: {exp_dir}")
    data, tau = collect_rollouts(exp_dir)
    plot_2panel(data, tau, PAPER_FIG_DIR)


if __name__ == "__main__":
    main()
