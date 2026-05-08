"""
Empirical Conservatism Trap 图: U vs telemetry latency D, 4 个 method
对比, 显示 baseline 在 D 增大时崩溃, ATC 保持平稳.

数据: joint_sensitivity_v2.csv (Latency x Beta x methods x U), 在 Beta
维度上平均, 单独看 D 的影响.

Usage:
    python -m fig.plot_empirical_trap

Output: paper_draft/figures/fig_empirical_trap.pdf + .png
"""
import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# IEEE conf single-column figure (\linewidth ~3.49")
# Source figsize 3.49 x 2.4 -> rendered at native; fonts at 8pt floor
plt.rcParams.update({
    "text.usetex": True, "font.family": "serif",
    "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7,
    "lines.linewidth": 1.5,
    "savefig.dpi": 600,
    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amsfonts} \usepackage{amssymb}"
})


def save_dual_format(fig, out_dir, base_name):
    """Save PDF + PNG (with usetex fallback for PNG)."""
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


def main():
    # K=1 main run filter (excludes K=3/K=5 dirs from glob)
    _K1_EXCLUDE = ("k3_", "k5_", "k_exploration", "sanity", "decision", "seedbump", "probe")
    candidates = glob.glob("experiments/*")
    k1_dirs = [e for e in candidates if not any(kw in os.path.basename(e) for kw in _K1_EXCLUDE)]
    if not k1_dirs:
        raise FileNotFoundError("No K=1 experiment dir found under experiments/")
    exp_dir = max(k1_dirs, key=os.path.getmtime)
    csv_path = os.path.join(exp_dir, "joint_sensitivity_v2.csv")
    print(f">>> Reading {csv_path}")
    df = pd.read_csv(csv_path)
    print(f">>> Schema: {list(df.columns)}; rows={len(df)}")

    # Average over Beta to isolate the Latency (D) dimension
    methods = {
        "proposed":         {"label": r"\textbf{ATC (Ours)}", "color": "#d62728", "ls": "-",  "lw": 2.0, "marker": "o"},
        "lstm_predictive":  {"label": "LSTM-Pred",            "color": "#1f77b4", "ls": "--", "lw": 1.4, "marker": "s"},
        "vanilla_ppo":      {"label": "Vanilla-PPO",          "color": "#2ca02c", "ls": "-.", "lw": 1.4, "marker": "^"},
        "safeslice":        {"label": "SafeSlice",            "color": "#ff7f0e", "ls": ":",  "lw": 1.4, "marker": "D"},
    }

    grouped = df.groupby("Latency").agg({m: ["mean", "std"] for m in methods.keys()})

    fig, ax = plt.subplots(figsize=(3.49, 2.45))

    # Shaded "Conservatism Trap" region: D >= 100ms where baselines collapse below 0.15
    ax.axhspan(0.0, 0.15, color="#d62728", alpha=0.06, zorder=0)
    ax.text(9, 0.142, r"\textit{Trap region: $U{<}0.15$}", color="#d62728", fontsize=8, va="bottom", ha="left")

    # FullInfo-PPO (D=0) zero-delay reference ceiling at U=0.36.
    ax.axhline(0.36, color="black", linestyle=":", lw=1.2, alpha=0.85, zorder=2, label=r"FullInfo-PPO ($D{=}0$ ref.)")

    # Operational latency marker (D=200ms nominal).
    ax.axvline(200, color="black", linestyle="--", lw=0.7, alpha=0.5, zorder=1)
    ax.text(210, 0.3, r"$D{=}200$\,ms", fontsize=7, va="bottom", ha="left", alpha=0.7)

    # Plot 4 method curves
    latencies = grouped.index.values
    for mode, cfg in methods.items():
        means = grouped[(mode, "mean")].values
        stds  = grouped[(mode, "std")].values
        ax.plot(latencies, means, label=cfg["label"], color=cfg["color"],
                linestyle=cfg["ls"], lw=cfg["lw"], marker=cfg["marker"],
                markersize=4, zorder=3)
        ax.fill_between(latencies, np.maximum(0, means - stds), means + stds, color=cfg["color"], alpha=0.10, zorder=2)

    ax.set_xscale("log")
    ax.set_xlabel(r"Telemetry Delay $D$ (ms)")
    ax.set_ylabel(r"Spectral Saturation $U$")
    ax.set_xlim(8, 1100)
    ax.set_ylim(0, 0.52)
    ax.set_xticks([10, 50, 100, 200, 400, 1000])
    ax.set_xticklabels(["10", "50", "100", "200", "400", "1000"])

    # Manual legend order: ATC + FullInfo on top (oracle-parity pair), baselines below
    handles, labels = ax.get_legend_handles_labels()
    # 重新定义顺序：ATC 第一，三个基准方法紧随其后，参考线最后
    order_map = {
        r"\textbf{ATC (Ours)}": 0,
        "LSTM-Pred": 1,
        "Vanilla-PPO": 2,
        "SafeSlice": 3,
        r"FullInfo-PPO ($D{=}0$ ref.)": 4,  # 设置为最大值，排在最后
    }

    new_handles_labels = sorted(zip(handles, labels), key=lambda x: order_map.get(x[1], 99))
    ordered_handles, ordered_labels = zip(*new_handles_labels)
    ax.legend(ordered_handles, ordered_labels,
              loc="upper left", frameon=True, edgecolor="gray",
              bbox_to_anchor=(-0.01, 1.014),
              fontsize=7, framealpha=1.0, ncol=2,
              labelspacing=0.2, handletextpad=0.2, columnspacing=0.4,
              handlelength=1.5, borderpad=0.2,)
    leg = ax.get_legend()
    if leg is not None:
        leg.get_frame().set_linewidth(0.4)

    ax.grid(True, alpha=0.3, axis='y', ls=':')

    fig.tight_layout(pad=0.3)
    save_dual_format(fig, PAPER_FIG_DIR, "fig_empirical_trap")
    plt.close(fig)


if __name__ == "__main__":
    main()
