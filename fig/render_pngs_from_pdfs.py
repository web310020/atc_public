"""
PDF → PNG 重栅格 (按 paper 实际渲染宽度算 dpi target).

避开 matplotlib default + LaTeX scaling 双重缩放导致 PNG 字号偏大.
用法: python -m fig.render_pngs_from_pdfs
"""
import os
import fitz  # PyMuPDF

# IEEE 2-col page (IEEEtran):
#   textwidth = 7.16 in (figure* 双栏)
#   linewidth = 3.49 in (figure 单栏)
LINEWIDTH_IN = 3.49
TEXTWIDTH_IN = 7.16

# Per-figure 目标 rendered 宽度 (匹配 \includegraphics)
TARGETS = {
    # 主图 (1 system arch + 5 data figs)
    "fig_atc_architecture":       (LINEWIDTH_IN,            r"\linewidth single col, Fig 1 system arch"),
    "fig_empirical_trap":         (LINEWIDTH_IN,            r"\linewidth single col, Fig 2 empirical trap"),
    "fig_perf_distributions":     (LINEWIDTH_IN,            r"\linewidth single col, Fig 3 perf distributions"),
    "fig_k_combined":             (LINEWIDTH_IN,            r"\linewidth single col, Fig 4 K=3+K=5 combined"),
    "fig_pareto_front":           (LINEWIDTH_IN,            r"\linewidth single col, Fig 5 Pareto front"),

    # 旧版 K=3 / K=5 per-slice 拆开版 (back-compat)
    "fig_k3_perslice":            (LINEWIDTH_IN,            r"\linewidth K=3 per-slice"),
    "fig_k5_perslice":            (LINEWIDTH_IN,            r"\linewidth K=5 per-slice"),

    # 历史 entries (PDF 可能仍在 disk, 但 canonical_tex 已不引)
    "fig_theory_trap":            (LINEWIDTH_IN,            r"\linewidth single col"),
    "fig_fidelity":               (0.244 * TEXTWIDTH_IN,    r"0.244\textwidth subfig"),
    "fig_saturation_cdf":         (0.244 * TEXTWIDTH_IN,    r"0.244\textwidth subfig"),
    "fig_violation_depth":        (0.244 * TEXTWIDTH_IN,    r"0.244\textwidth subfig"),
    "fig_scores":                 (0.244 * TEXTWIDTH_IN,    r"0.244\textwidth subfig"),
    "fig_joint_landscape_1x4":    (TEXTWIDTH_IN,            r"1.0\linewidth in figure* (full 2-col)"),
    "fig_sensitivity_analysis":   (0.9 * LINEWIDTH_IN,      r"0.9\linewidth single col"),
}

DPI = 300

PAPER_FIG_DIR = "paper_draft/figures" if os.path.isdir("paper_draft/figures") \
    else os.path.join("..", "paper_draft", "figures")


def render_one(base_name: str, target_width_in: float, src_label: str) -> bool:
    pdf_path = os.path.join(PAPER_FIG_DIR, base_name + ".pdf")
    png_path = os.path.join(PAPER_FIG_DIR, base_name + ".png")

    if not os.path.exists(pdf_path):
        print(f"  [SKIP] {base_name}: PDF not found ({pdf_path})")
        return False

    doc = fitz.open(pdf_path)
    page = doc[0]
    src_w_in = page.rect.width / 72.0   # PDF pts -> inches
    src_h_in = page.rect.height / 72.0

    # Scale factor so output is target_width_in wide at DPI
    scale = (target_width_in * DPI) / page.rect.width
    matrix = fitz.Matrix(scale, scale)

    pix = page.get_pixmap(matrix=matrix, alpha=False)
    out_w_px = pix.width
    out_h_px = pix.height
    out_h_in = out_h_px / DPI
    pix.save(png_path)
    doc.close()

    print(
        f"  [OK]   {base_name:32s}  "
        f"PDF {src_w_in:.2f}x{src_h_in:.2f}in  ->  "
        f"PNG {out_w_px}x{out_h_px}px ({target_width_in:.2f}x{out_h_in:.2f}in @ {DPI}dpi)  "
        f"[{src_label}]"
    )
    return True


def main():
    print(f">>> Rendering PNGs from PDFs in {PAPER_FIG_DIR}")
    print(f">>> Target: paper-rendered size (per \\includegraphics in canonical .tex)")
    print(f">>> DPI: {DPI}")
    print()

    n_ok, n_skip = 0, 0
    for base_name, (target_w, src_label) in TARGETS.items():
        if render_one(base_name, target_w, src_label):
            n_ok += 1
        else:
            n_skip += 1

    print()
    print(f">>> Done: {n_ok} OK, {n_skip} SKIP")


if __name__ == "__main__":
    main()
