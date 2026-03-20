"""
generate_new_figures.py
=======================
Generates 4 new thesis figures (N1–N4) for the thesis:
  "Uncertainty Quantification for Graph Neural Network Surrogates
   of Agent-Based Transport Models"

Run from the figures/ directory:
    python generate_new_figures.py

Outputs (PDF + PNG for each):
    fig11_thesis_workflow.pdf/.png
    fig12_trial_progression.pdf/.png
    fig13_mc_dropout_inference.pdf/.png
    fig14_conformal_workflow.pdf/.png
"""

import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Pastel / soft palette  (matches generate_all_thesis_figures.py)
# ---------------------------------------------------------------------------
BG = "#FAFBFC"
PANEL = "#F0F4F8"
P_BLUE = "#5B8DB8"
P_BLUE_LT = "#A8C8E8"
P_BLUE_DK = "#2E6494"
P_CORAL = "#E07A5F"
P_CORAL_LT = "#F2B5A0"
P_GREEN = "#6BAB8C"
P_GREEN_LT = "#B8D4C0"
P_AMBER = "#E8A84C"
P_SLATE = "#5C6B7A"
P_DGRAY = "#3A4A5A"
P_MGRAY = "#7A8A9A"
P_LGRAY = "#D0D8E0"
P_XLGRAY = "#E8EDF2"
WHITE = "#FFFFFF"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,
        "figure.dpi": 150,
        "axes.facecolor": BG,
        "figure.facecolor": BG,
    }
)


def save(fig, name):
    pdf = os.path.join(OUT, name + ".pdf")
    png = os.path.join(OUT, name + ".png")
    fig.savefig(pdf, dpi=300, bbox_inches="tight", facecolor=BG)
    fig.savefig(png, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  saved {name}.pdf")


# ===========================================================================
# N1 — fig11_thesis_workflow: Overall thesis pipeline  (v2 — redesigned)
# ===========================================================================
def fig11_thesis_workflow():
    """Redesigned thesis pipeline — professional academic style (v4).

    Design principles:
    - Taller canvas (5.0 in) to give more breathing room to boxes and circles.
    - Larger step-number circles (radius 0.26) above each stage.
    - White background for clean PDF printing.
    - No title baked into the figure.
    - T8 box rendered slightly larger with coral border for cohesive highlight.
    - Clean diagonal fork/merge arrows in matching accent colours.
    """
    fig, ax = plt.subplots(figsize=(13.5, 6.5))
    fig.patch.set_facecolor(BG)
    ax.set_xlim(0, 14)
    ax.set_ylim(-0.1, 5.8)
    ax.axis("off")

    # ── Layout constants ────────────────────────────────────────────────────
    Y = 2.70  # main pipeline y-centre (raised for extra height)
    BW, BH = 1.55, 1.35  # standard box (slightly taller)
    T8W, T8H = 1.75, 1.55  # T8 box — slightly larger
    UW, UH = 1.85, 1.10  # UQ branch boxes
    PW, PH = 1.55, 1.15  # policy/output box

    Xs = [0.95, 2.75, 4.55, 6.35, 8.20]  # stage x-centres
    Xuq = 10.30  # UQ branch x-centre
    Ymc = 3.90  # MC Dropout y-centre (raised)
    Ycf = 1.30  # Conformal y-centre
    Xpol = 12.80  # policy box x-centre

    # ── Helpers ─────────────────────────────────────────────────────────────
    def rbox(cx, cy, w, h, txt, fc, tc=P_DGRAY, fs=10.5, lw=0.8, ec=None):
        ec = ec or P_LGRAY
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (cx - w / 2, cy - h / 2),
                w,
                h,
                boxstyle="round,pad=0.09",
                linewidth=lw,
                edgecolor=ec,
                facecolor=fc,
                zorder=3,
            )
        )
        ax.text(
            cx,
            cy,
            txt,
            ha="center",
            va="center",
            fontsize=fs,
            color=tc,
            fontweight="bold",
            multialignment="center",
            zorder=4,
            linespacing=1.45,
        )

    def arr(x0, y0, x1, y1, col=P_MGRAY):
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", color=col, lw=1.0, mutation_scale=12),
            zorder=2,
        )

    # ── Step-number circles above each main stage ────────────────────────────
    sfc = [P_MGRAY, P_MGRAY, P_BLUE_LT, P_BLUE, P_CORAL]
    stc = [WHITE, WHITE, P_DGRAY, WHITE, WHITE]
    btop = [Y + BH / 2] * 4 + [Y + T8H / 2]
    for i, (xi, fc_, tc_, bt) in enumerate(zip(Xs, sfc, stc, btop)):
        yc = bt + 0.52  # larger gap above box for the bigger circle
        ax.add_patch(
            plt.Circle((xi, yc), 0.30, facecolor=fc_, edgecolor="none", zorder=5)
        )
        ax.text(
            xi,
            yc,
            str(i + 1),
            ha="center",
            va="center",
            fontsize=10.5,
            color=tc_,
            fontweight="bold",
            zorder=6,
        )

    # ── Main pipeline boxes (stages 1–5) ─────────────────────────────────────
    rbox(Xs[0], Y, BW, BH, "MATSim\nScenarios", P_XLGRAY, P_SLATE)
    rbox(Xs[1], Y, BW, BH, "1,000 of 10,000\n(10% subset)", P_XLGRAY, P_SLATE)
    rbox(
        Xs[2],
        Y,
        BW,
        BH,
        "Line-graph\nconstruction\n31,635 nodes",
        P_BLUE_LT,
        P_DGRAY,
    )
    rbox(Xs[3], Y, BW, BH, "GNN training\n8 trials (T1\u2013T8)", P_BLUE, WHITE)

    # T8: larger box, coral border
    rbox(
        Xs[4],
        Y,
        T8W,
        T8H,
        "Best model\nTrial 8\nR\u00b2\u2009=\u20090.5957\nMAE\u2009=\u20093.96\u2009veh/h",
        P_CORAL,
        WHITE,
        lw=2.0,
        ec=P_CORAL,
    )

    # ── Main-flow arrows (horizontal) ────────────────────────────────────────
    for i in range(4):
        x0 = Xs[i] + BW / 2
        x1 = (Xs[i + 1] - BW / 2) if i < 3 else (Xs[4] - T8W / 2)
        arr(x0 + 0.04, Y, x1 - 0.04, Y)

    # ── UQ branch boxes ───────────────────────────────────────────────────────
    rbox(
        Xuq,
        Ymc,
        UW,
        UH,
        "MC Dropout\n\u03c1\u2009=\u20090.4820\u2002(S\u2009=\u200930)",
        P_AMBER,
        P_DGRAY,
        fs=10.0,
    )

    rbox(
        Xuq,
        Ycf,
        UW,
        UH,
        "Conformal prediction\n90%\u2009/\u200995%\u2002coverage",
        P_GREEN_LT,
        P_DGRAY,
        fs=10.0,
    )

    # ── Policy / output box ───────────────────────────────────────────────────
    rbox(
        Xpol,
        Y,
        PW,
        PH,
        "Uncertainty-guided\npolicy decisions",
        P_BLUE_DK,
        WHITE,
        fs=10.0,
    )

    # ── Fork arrows: T8 → MC Dropout, T8 → Conformal ────────────────────────
    t8rx = Xs[4] + T8W / 2
    uqlx = Xuq - UW / 2
    arr(t8rx + 0.05, Y + 0.30, uqlx - 0.05, Ymc, col=P_AMBER)
    arr(t8rx + 0.05, Y - 0.30, uqlx - 0.05, Ycf, col=P_GREEN)

    # ── Merge arrows: MC Dropout → Policy, Conformal → Policy ───────────────
    uqrx = Xuq + UW / 2
    pollx = Xpol - PW / 2
    arr(uqrx + 0.05, Ymc, pollx - 0.05, Y + 0.30, col=P_BLUE_DK)
    arr(uqrx + 0.05, Ycf, pollx - 0.05, Y - 0.30, col=P_BLUE_DK)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.04)
    save(fig, "fig11_thesis_workflow")


# ===========================================================================
# N2 — fig12_trial_progression: Trial T1→T8 table-style chart
# ===========================================================================
def fig12_trial_progression():
    trials = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"]
    r2 = [0.7860, 0.5117, 0.2246, 0.2426, 0.5553, 0.5223, 0.5471, 0.5957]
    mae = [2.97, 4.33, 5.99, 6.08, 4.24, 4.32, 4.06, 3.96]

    # What changed per trial (short description)
    changes = [
        "Baseline GNN\n(Linear head)",
        "GATConv\nhead (×2)",
        "Reduced\ncapacity",
        "Augmented\nfeatures",
        "MC Dropout\nadded (p=0.3)",
        "Dropout\np=0.5",
        "Dropout\np=0.1",
        "Dropout p=0.3\n+ larger hidden",
    ]

    x = np.arange(len(trials))
    # Color gradient: low R² = TUM_LGRAY, high = TUM_BLUE, best = TUM_ORANGE
    bar_colors = []
    for i, v in enumerate(r2):
        if i == 7:
            bar_colors.append(P_CORAL)
        elif v >= 0.50:
            bar_colors.append(P_BLUE)
        elif v >= 0.35:
            bar_colors.append(P_BLUE_LT)
        else:
            bar_colors.append(P_XLGRAY)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.patch.set_facecolor(BG)

    # Top: R² bars
    bars1 = ax1.bar(
        x, r2, color=bar_colors, edgecolor=P_LGRAY, linewidth=0.5, width=0.65
    )
    ax1.bar_label(bars1, fmt="%.4f", fontsize=8, padding=2)
    ax1.set_ylabel("R²")
    ax1.set_ylim(0, 0.95)
    ax1.yaxis.grid(True, color=P_LGRAY)
    ax1.set_axisbelow(True)
    ax1.set_title("R² per Trial (higher = better)", fontsize=10, color=P_DGRAY)
    ax1.axhline(0.5957, color=P_CORAL, lw=0.8, ls="--", alpha=0.6)
    ax1.text(7.4, 0.61, "T8 best", fontsize=7.5, color=P_CORAL, style="italic")

    # Bottom: MAE bars (inverted sense — lower better)
    bars2 = ax2.bar(
        x, mae, color=bar_colors, edgecolor=P_LGRAY, linewidth=0.5, width=0.65
    )
    ax2.bar_label(bars2, fmt="%.2f", fontsize=8, padding=2)
    ax2.set_ylabel("MAE (veh/h)")
    ax2.set_ylim(0, 7.5)
    ax2.yaxis.grid(True, color=P_LGRAY)
    ax2.set_axisbelow(True)
    ax2.set_title("MAE per Trial (lower = better)", fontsize=10, color=P_DGRAY)
    ax2.axhline(3.96, color=P_CORAL, lw=0.8, ls="--", alpha=0.6)
    ax2.text(7.4, 4.1, "T8 best", fontsize=7.5, color=P_CORAL, style="italic")

    # Change annotations below x-axis
    ax2.set_xticks(x)
    ax2.set_xticklabels(trials, fontsize=9)

    # Change description text below each bar
    for i, (ch, xi) in enumerate(zip(changes, x)):
        ax2.text(
            xi,
            -1.5,
            ch,
            ha="center",
            va="top",
            fontsize=6.5,
            color=P_SLATE,
            multialignment="center",
            transform=ax2.get_xaxis_transform(),
        )

    # T1 architectural note — Linear head vs GATConv (T2–T8)
    ax1.annotate(
        "Linear\nhead",
        xy=(0, r2[0]),
        xytext=(0.55, r2[0] + 0.09),
        fontsize=6.5,
        color=P_MGRAY,
        style="italic",
        ha="left",
        arrowprops=dict(arrowstyle="-", color=P_LGRAY, lw=0.7),
    )
    ax2.annotate(
        "Linear\nhead",
        xy=(0, mae[0]),
        xytext=(0.55, mae[0] + 0.35),
        fontsize=6.5,
        color=P_MGRAY,
        style="italic",
        ha="left",
        arrowprops=dict(arrowstyle="-", color=P_LGRAY, lw=0.7),
    )
    # Footnote below chart
    fig.text(
        0.5,
        0.01,
        "† T1 uses a Linear output head; T2–T8 use GATConv. T1 is not directly comparable to T2–T8.",
        ha="center",
        fontsize=7.5,
        color=P_MGRAY,
        style="italic",
    )

    # Legend patches
    p_best = mpatches.Patch(color=P_CORAL, label="T8 (best)")
    p_good = mpatches.Patch(color=P_BLUE, label="R² ≥ 0.50")
    p_med = mpatches.Patch(color=P_BLUE_LT, label="0.35 ≤ R² < 0.50")
    p_low = mpatches.Patch(color=P_XLGRAY, label="R² < 0.35")
    fig.legend(
        handles=[p_best, p_good, p_med, p_low],
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
        fontsize=8.5,
    )

    fig.suptitle(
        "Trial Progression T1→T8: Architecture Changes and Performance",
        fontsize=11,
        fontweight="bold",
        color=P_DGRAY,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    save(fig, "fig12_trial_progression")


# ===========================================================================
# N3 — fig13_mc_dropout_inference: MC Dropout forward-pass diagram
# ===========================================================================
def fig13_mc_dropout_inference():
    fig, ax = plt.subplots(figsize=(12, 4.0))
    fig.patch.set_facecolor(BG)
    ax.set_xlim(0, 12)
    ax.set_ylim(0.8, 4.4)
    ax.axis("off")

    def box(cx, cy, w, h, label, fc, tc=P_DGRAY, fs=8.0):
        r = mpatches.FancyBboxPatch(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            boxstyle="round,pad=0.12",
            linewidth=0.8,
            edgecolor=P_LGRAY,
            facecolor=fc,
            zorder=3,
        )
        ax.add_patch(r)
        ax.text(
            cx,
            cy,
            label,
            ha="center",
            va="center",
            fontsize=fs,
            color=tc,
            fontweight="bold",
            multialignment="center",
            zorder=4,
        )

    def arrow(x0, y0, x1, y1, label="", color=P_SLATE, ls="-"):
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="-|>", color=color, lw=1.2, mutation_scale=13, linestyle=ls
            ),
            zorder=2,
        )
        if label:
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            ax.text(
                mx,
                my + 0.18,
                label,
                ha="center",
                fontsize=7.5,
                color=color,
                style="italic",
            )

    # Input graph
    box(
        1.3,
        2.45,
        1.8,
        1.6,
        "Input\nGraph\n(road network\nscenario)",
        P_XLGRAY,
        P_SLATE,
    )

    # GNN with dropout
    box(
        4.0,
        2.45,
        2.2,
        2.0,
        "GNN (T8)\nPointNetTransfGAT\nDropout ON\nduring inference",
        P_BLUE,
        WHITE,
    )

    # 30 forward passes fan-out
    arrow(5.1, 2.45, 5.8, 2.45)

    # Stacked pass boxes
    pass_y = [3.7, 3.0, 2.45, 1.9, 1.2]
    pass_lbl = ["ŷ₁, σ₁", "ŷ₂, σ₂", "  ···", "ŷ₂₉, σ₂₉", "ŷ₃₀, σ₃₀"]
    for py, pl in zip(pass_y, pass_lbl):
        box(7.0, py, 1.5, 0.45, pl, P_BLUE_LT, P_DGRAY, fs=7.5)
        ax.annotate(
            "",
            xy=(5.8, py),
            xytext=(5.1, 2.45),
            arrowprops=dict(
                arrowstyle="-|>", color=P_BLUE_LT, lw=0.9, mutation_scale=10
            ),
            zorder=2,
        )

    ax.text(
        6.05,
        2.45,
        "S=30\npasses",
        ha="center",
        va="center",
        fontsize=7.5,
        color=P_CORAL,
        fontweight="bold",
    )

    # Aggregation box
    arrow(7.75, 2.45, 8.7, 2.45, label="aggregate")
    box(9.8, 2.45, 2.0, 1.6, "Per-node\nOutput\nŷ = mean\nσ = std", P_CORAL, WHITE)

    # Dropout bracket label
    ax.annotate(
        "",
        xy=(5.0, 4.1),
        xytext=(3.0, 4.1),
        arrowprops=dict(arrowstyle="<->", color=P_CORAL, lw=1.1, mutation_scale=11),
        zorder=2,
    )
    ax.text(
        4.0,
        4.28,
        "Dropout active at inference time",
        ha="center",
        fontsize=7.5,
        color=P_CORAL,
        style="italic",
    )

    ax.set_title(
        "MC Dropout Inference: S=30 Stochastic Forward Passes → Per-Node Mean ŷ and σ  (T8)",
        fontsize=10.5,
        fontweight="bold",
        color=P_DGRAY,
        pad=8,
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.04)
    save(fig, "fig13_mc_dropout_inference")


# ===========================================================================
# N4 — fig14_conformal_workflow: Conformal prediction diagram
# ===========================================================================
def fig14_conformal_workflow():
    fig, ax = plt.subplots(figsize=(14, 4.2))
    fig.patch.set_facecolor(BG)
    ax.set_xlim(0, 14)
    ax.set_ylim(0.4, 4.5)
    ax.axis("off")

    def box(cx, cy, w, h, label, fc, tc=P_DGRAY, fs=8.0):
        r = mpatches.FancyBboxPatch(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            boxstyle="round,pad=0.1",
            linewidth=0.8,
            edgecolor=P_LGRAY,
            facecolor=fc,
            zorder=3,
        )
        ax.add_patch(r)
        ax.text(
            cx,
            cy,
            label,
            ha="center",
            va="center",
            fontsize=fs,
            color=tc,
            fontweight="bold",
            multialignment="center",
            zorder=4,
        )

    def arrow(x0, y0, x1, y1, label="", color=P_SLATE):
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2, mutation_scale=13),
            zorder=2,
        )
        if label:
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            ax.text(
                mx,
                my + 0.18,
                label,
                ha="center",
                fontsize=7.5,
                color=color,
                style="italic",
            )

    # Step 1: 100 test graphs
    box(
        1.3,
        2.8,
        2.0,
        1.6,
        "100 Test\nGraphs\n(T8 inference\nwith MC dropout)",
        P_XLGRAY,
        P_SLATE,
    )

    # Step 2: split
    arrow(2.3, 2.8, 3.0, 2.8)
    box(3.9, 3.8, 1.8, 1.0, "Calibration\n50 graphs", P_BLUE_LT, P_DGRAY)
    box(3.9, 1.8, 1.8, 1.0, "Evaluation\n50 graphs", P_BLUE, WHITE)

    ax.annotate(
        "",
        xy=(3.0, 3.8),
        xytext=(3.0, 2.8),
        arrowprops=dict(arrowstyle="-|>", color=P_SLATE, lw=1.1, mutation_scale=11),
    )
    ax.annotate(
        "",
        xy=(3.0, 1.8),
        xytext=(3.0, 2.8),
        arrowprops=dict(arrowstyle="-|>", color=P_SLATE, lw=1.1, mutation_scale=11),
    )

    # Step 3: residuals on calibration set
    arrow(4.8, 3.8, 5.8, 3.8)
    box(6.9, 3.8, 2.0, 1.0, "Compute\nResiduals\n|ŷᵢ − yᵢ|", P_BLUE_LT, P_DGRAY)

    # Step 4: quantile
    arrow(7.9, 3.8, 9.0, 3.8)
    box(
        10.1,
        3.8,
        2.0,
        1.0,
        "q = (1−α)(1+1/n)\nquantile of\nresiduals",
        P_CORAL,
        WHITE,
    )

    # Step 5: apply to evaluation set
    arrow(4.8, 1.8, 5.8, 1.8)
    box(6.9, 1.8, 2.0, 1.0, "Prediction\nIntervals\nŷ ± q", P_BLUE, WHITE)

    # Step 6: coverage check
    arrow(7.9, 1.8, 9.0, 1.8)
    box(10.1, 1.8, 2.0, 1.0, "Coverage\nCheck\nPICP ≥ 1−α?", P_AMBER, P_DGRAY)

    # Step 7: results
    arrow(11.1, 3.8, 12.1, 3.8)
    box(12.9, 3.8, 1.6, 1.0, "90%: q=9.92\nPICP=90.02%", P_GREEN, P_DGRAY, fs=7.5)

    arrow(11.1, 1.8, 12.1, 1.8)
    box(12.9, 1.8, 1.6, 1.0, "95%: q=14.68\nPICP=95.01%", P_GREEN, P_DGRAY, fs=7.5)

    # Guarantee note
    ax.text(
        12.9,
        0.72,
        "Marginal coverage\nguarantee holds",
        ha="center",
        fontsize=7.5,
        color=P_GREEN,
        style="italic",
        fontweight="bold",
    )

    # Vertical connector between cal → quantile and eval → intervals
    ax.plot(
        [10.1, 10.1], [2.4, 3.2], color=P_CORAL, lw=1.1, ls="--", alpha=0.6, zorder=2
    )

    ax.set_title(
        "Conformal Prediction Workflow  (T8 — 50 calibration + 50 evaluation graphs)",
        fontsize=10.5,
        fontweight="bold",
        color=P_DGRAY,
        pad=8,
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.04)
    save(fig, "fig14_conformal_workflow")


# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    print("Generating new thesis figures (N1–N4)...")
    fig11_thesis_workflow()
    fig12_trial_progression()
    fig13_mc_dropout_inference()
    fig14_conformal_workflow()
    print("Done. All new figures written to:", OUT)
