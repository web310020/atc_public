"""
K=3 + K=5 per-slice violation 合并图 (单列 2 row).

上排: K=3 (URLLC / eMBB / mMTC, 50 seeds)
下排: K=5 (URLLC / V2X / eMBB / mMTC / IoT_burst, 10 seeds, safety
      threshold 2.5%)

数据从 experiments/ 下自动 glob (run_k3_seedbump_workers1.py 和
run_k5_probe_workers1.py 跑出来的 dir).

Usage:
    python -m fig.plot_k_combined

Output: paper_draft/figures/fig_k_combined.pdf + .png
"""
import os, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True, "font.family": "serif",
    "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
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


def collect_per_slice(run_dir: Path, mode: str, seed_range):
    arr = []
    for s in seed_range:
        p = run_dir / "runs" / f"{mode}_seed{s}" / f"{mode}_seed{s}_summary.json"
        if not p.exists():
            print(f"  ! missing: {p}")
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        es = d.get("eval_summary") or d
        arr.append(np.asarray(es["per_slice_viol_rate"], dtype=np.float64))
    return np.asarray(arr)


def draw_panel(ax, slice_names, atc_rates, van_rates, panel_title,
                show_threshold=False, safety_threshold_pct=2.5,
                show_legend=False, value_fontsize=5.5):
    """Draw one panel of per-slice bars (ATC vs Vanilla)."""
    n_slices = len(slice_names)
    x = np.arange(n_slices)
    width = 0.36

    atc_mean_pct = atc_rates.mean(axis=0) * 100
    atc_std_pct  = atc_rates.std(axis=0, ddof=1) * 100
    van_mean_pct = van_rates.mean(axis=0) * 100
    van_std_pct  = van_rates.std(axis=0, ddof=1) * 100

    bars_atc = ax.bar(x - width/2, atc_mean_pct, width, yerr=atc_std_pct, capsize=2,
                       color="#d62728", edgecolor="black", lw=0.4,
                       label=r"\textbf{ATC (Ours)}", error_kw={"lw": 0.5, "ecolor": "#3a0a0b"})
    bars_van = ax.bar(x + width/2, van_mean_pct, width, yerr=van_std_pct, capsize=2,
                       color="#1f77b4", edgecolor="black", lw=0.4,
                       label="Vanilla-PPO", error_kw={"lw": 0.5, "ecolor": "#0a2a55"})

    if show_threshold:
        ax.axhline(safety_threshold_pct, color="#2ca02c", linestyle="--", lw=2, alpha=0.7, zorder=1, label=rf"Safety Threshold (${safety_threshold_pct:.1f}\%$)")
        # IEEE 6pt annotation floor.
        #ax.text(n_slices - 0.5, safety_threshold_pct + 0.4, rf"safety ${safety_threshold_pct:.1f}\%$", fontsize=6.0, color="#2ca02c", ha="right", va="bottom")

    # Annotate bar values
    for rect, val in zip(bars_atc, atc_mean_pct):
        ax.text(rect.get_x() + rect.get_width()/2, val + 0.15,
                f"{val:.1f}", ha="center", va="bottom", fontsize=value_fontsize)
    for rect, val in zip(bars_van, van_mean_pct):
        ax.text(rect.get_x() + rect.get_width()/2, val + 0.15,
                f"{val:.1f}", ha="center", va="bottom", fontsize=value_fontsize)

    ax.set_xticks(x)
    ax.set_xticklabels(slice_names, fontsize=7.5)
    ax.set_ylabel(r"Viol.\ rate (\%)")
    ymax = max(van_mean_pct.max() + van_std_pct.max(), atc_mean_pct.max() + atc_std_pct.max()) * 1.2
    ax.set_ylim(0, max(ymax, 5))
    ax.set_title(panel_title, fontsize=8.5, pad=2)

    if show_legend:
        # --- 核心修改：手动控制图例顺序 ---
        handles, labels = ax.get_legend_handles_labels()

        # 定义你想要的顺序映射
        order_map = {
            r"\textbf{ATC (Ours)}": 0,
            "Vanilla-PPO": 1,
            rf"Safety Thr. (${safety_threshold_pct:.1f}\%$)": 2
        }

        # 按映射排序，没在映射里的放最后
        new_handles_labels = sorted(zip(handles, labels),
                                    key=lambda x: order_map.get(x[1], 99))
        ordered_handles, ordered_labels = zip(*new_handles_labels)

        ax.legend(ordered_handles, ordered_labels,
                    loc="upper left", frameon=True, framealpha=0.95, edgecolor="gray",
                      bbox_to_anchor=(-0.007, 1.022),
                      fontsize=7, ncol=1,
                      labelspacing=0.2, handletextpad=0.2, columnspacing=0.4,
                      handlelength=1.8, borderpad=0.2,)
        leg = ax.get_legend()
        if leg is not None:
            leg.get_frame().set_linewidth(0.4)


def main():
    REPO = Path(".")
    EXP_DIR = REPO / "experiments"

    # ---- K=3 (50 seeds) ----
    print(">>> K=3 per-slice (50 seeds: 0-29 + 30-49)")
    k3_old = EXP_DIR / "k3_r3_3_30seed_power"
    k3_new = EXP_DIR / "k3_r4_seedbump_30to49"
    atc_k3 = np.vstack([
        collect_per_slice(k3_old, "proposed", range(0, 30)),
        collect_per_slice(k3_new, "proposed", range(30, 50)),
    ])
    van_k3 = np.vstack([
        collect_per_slice(k3_old, "vanilla_ppo", range(0, 30)),
        collect_per_slice(k3_new, "vanilla_ppo", range(30, 50)),
    ])
    print(f"  K=3 ATC shape: {atc_k3.shape}, Vanilla shape: {van_k3.shape}")

    # ---- K=5 (10 seeds) ----
    print(">>> K=5 per-slice (10 seeds)")
    k5 = EXP_DIR / "k5_probe_v1"
    atc_k5 = collect_per_slice(k5, "proposed", range(0, 10))
    van_k5 = collect_per_slice(k5, "vanilla_ppo", range(0, 10))
    print(f"  K=5 ATC shape: {atc_k5.shape}, Vanilla shape: {van_k5.shape}")

    # ---- Combined 2-row vstack figure (compact 2.6" tall to fit 9-page budget) ----
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(3.49, 2.7),
                                          gridspec_kw={"height_ratios": [1, 1.05]})

    # 增加这一行：手动插入一个带标签的虚线代理，用于在 Legend 中显示
    ax_top.plot([], [], color="#2ca02c", linestyle="--", lw=0.8, label=r"Safety Thr. ($2.5\%$)")

    # Top: K=3 (with shared legend in upper-right corner of top panel; compact)
    draw_panel(
        ax_top,
        slice_names=["URLLC", "eMBB", "mMTC"],
        atc_rates=atc_k3, van_rates=van_k3,
        panel_title=r"$K{=}3$ (50 seeds)",
        show_threshold=False,
        show_legend=False,
        value_fontsize=6.0,
    )

    # Bottom: K=5 (with safety threshold)
    draw_panel(
        ax_bot,
        slice_names=["URLLC", "V2X", "eMBB", "mMTC", r"IoT$_{\text{burst}}$"],
        atc_rates=atc_k5, van_rates=van_k5,
        panel_title=r"$K{=}5$ (10 seeds)",
        show_threshold=True, safety_threshold_pct=2.5,
        show_legend=True,
        value_fontsize=6.0,
    )

    fig.tight_layout(pad=0.2, h_pad=0.5)
    save_dual_format(fig, PAPER_FIG_DIR, "fig_k_combined")
    plt.close(fig)


if __name__ == "__main__":
    main()
