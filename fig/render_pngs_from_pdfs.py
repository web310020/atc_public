"""
按 paper 里的实际渲染宽度 (\\includegraphics 之后那个尺寸) 把每个 PDF
图重新栅格成 PNG, 保证 PNG 里的字号和比例跟正稿里看到的一致.

绘图脚本里 source font size 故意设得偏大, 因为 LaTeX 里会再 scale 一次
(\\includegraphics width=...). matplotlib 直接导 PNG (figsize x dpi)
跳过了这层 scale, 所以独立看 PNG 字会显得很大. 这个脚本修正这一点.

Usage:
    python -m fig.render_pngs_from_pdfs
"""
import os
import fitz  # PyMuPDF

# IEEE conf two-column page (sig-alternate / IEEEtran):
#   textwidth   = 7.16 in (full 2-col width, used by figure*)
#   linewidth   = 3.49 in (single column, used by figure)
# Constants below are conservative; pixel deltas <5% don't change visual.
LINEWIDTH_IN = 3.49
TEXTWIDTH_IN = 7.16

# Per-figure target rendered width matched against the \includegraphics
# directives in the LaTeX source. Tuple is (target_width_in_inches, label).
TARGETS = {
    # paper 里实际用到的图: 1 张架构示意 + 3 张数据图
    "fig_atc_architecture":     (LINEWIDTH_IN, r"\linewidth (system architecture, hand-drawn)"),
    "fig_empirical_trap":       (LINEWIDTH_IN, r"\linewidth (empirical Conservatism Trap)"),
    "fig_perf_distributions":   (LINEWIDTH_IN, r"\linewidth (perf distributions 2-panel)"),
    "fig_k_combined":           (LINEWIDTH_IN, r"\linewidth (K=3+K=5 per-slice 2-row)"),

    # run_joint_sweep_v2 输出的 1x5 panel 大图 (用作 plot_empirical_trap 的数据源)
    "fig_joint_landscape_1x4":  (TEXTWIDTH_IN, r"1.0\linewidth (full 2-col, figure*)"),
}

DPI = 300  # output PNG DPI; 300 = print-quality, gives crisp on-screen view

PAPER_FIG_DIR = "paper_draft/figures" if os.path.isdir("paper_draft/figures") \
    else os.path.join("..", "paper_draft", "figures")


def render_one(base_name: str, target_width_in: float, src_label: str) -> bool:
    """Rasterize PDF at target width inches x DPI; overwrite same-name PNG."""
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
    print(f">>> Target: paper-rendered size (per \\includegraphics in the LaTeX source)")
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
