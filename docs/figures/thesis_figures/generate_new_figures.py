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
import sys

# Add figures directory to path for thesis_style import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from thesis_style import *

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))


def save(fig, name):
    pdf = os.path.join(OUT, name + ".pdf")
    png = os.path.join(OUT, name + ".png")
    fig.savefig(pdf, dpi=300, bbox_inches="tight", facecolor=BG)
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor=BG)
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

    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.04)
    save(fig, "fig11_thesis_workflow")


# ===========================================================================
# N2 — fig12_trial_progression: Trial T1→T8 table-style chart
# ===========================================================================
def fig12_trial_progression():
    """Fig 3.5 — Trial Progression T1-T8 (2D bars) with hyperparameter table."""

    trials = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"]
    r2 = [0.7860, 0.5117, 0.2246, 0.2426, 0.5553, 0.5223, 0.5471, 0.5957]
    mae = [2.97, 4.33, 5.99, 6.08, 4.24, 4.32, 4.06, 3.96]

    # Color coding: T8 best = coral, R2>=0.50 = blue, <0.35 = light gray
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

    x = np.arange(len(trials))
    bw = 0.55  # bar width

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    fig.patch.set_facecolor(BG)

    # ================================================================
    # Panel (a) — R² per Trial
    # ================================================================
    bars1 = ax1.bar(x, r2, width=bw, color=bar_colors, edgecolor="white", linewidth=0.6)
    # Value labels on top
    for i, (bar, v) in enumerate(zip(bars1, r2)):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.015,
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=P_DGRAY,
        )

    # T8 best reference line
    ax1.axhline(y=0.5957, color=P_CORAL, lw=0.9, ls="--", alpha=0.5, zorder=0)
    ax1.text(
        -0.4,
        0.5957 + 0.012,
        "T8 best",
        fontsize=7.5,
        color=P_CORAL,
        ha="left",
        va="bottom",
        style="italic",
    )

    ax1.set_ylabel(r"R$^2$", fontsize=11)
    ax1.set_ylim(0, 0.92)
    ax1.set_title(
        "(a)  R\u00b2 per Trial (higher = better)", fontsize=11, color=P_DGRAY, pad=8
    )
    ax1.grid(axis="y", alpha=0.3, color=P_LGRAY, ls="--")
    ax1.set_axisbelow(True)

    # ================================================================
    # Panel (b) — MAE per Trial
    # ================================================================
    bars2 = ax2.bar(
        x, mae, width=bw, color=bar_colors, edgecolor="white", linewidth=0.6
    )
    for i, (bar, v) in enumerate(zip(bars2, mae)):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.08,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=P_DGRAY,
        )

    # T8 best reference line
    ax2.axhline(y=3.96, color=P_CORAL, lw=0.9, ls="--", alpha=0.5, zorder=0)
    ax2.text(
        -0.4,
        3.96 + 0.06,
        "T8 best",
        fontsize=7.5,
        color=P_CORAL,
        ha="left",
        va="bottom",
        style="italic",
    )

    ax2.set_ylabel("MAE (veh/h)", fontsize=11)
    ax2.set_ylim(0, 7.2)
    ax2.set_title(
        "(b)  MAE per Trial (lower = better)", fontsize=11, color=P_DGRAY, pad=8
    )
    ax2.grid(axis="y", alpha=0.3, color=P_LGRAY, ls="--")
    ax2.set_axisbelow(True)

    # Shared x-axis: trial names
    ax2.set_xticks(x)
    ax2.set_xticklabels(trials, fontsize=10, fontweight="bold")

    # ================================================================
    # Hyperparameter table below the bars (using matplotlib table)
    # ================================================================
    table_data = [
        # Row labels: Final Layer, Dropout, Batch, LR, Weighted, Split
        [
            "Linear\u2020",
            "GATConv",
            "GATConv",
            "GATConv",
            "GATConv",
            "GATConv",
            "GATConv",
            "GATConv",
        ],
        ["0.0", "0.3", "0.0", "0.0", "0.3", "0.3", "0.3", "0.2"],
        ["32", "16", "16", "16", "8", "8", "8", "8"],
        [
            "1e\u22123",
            "5e\u22124",
            "5e\u22124",
            "5e\u22124",
            "5e\u22124",
            "3e\u22124",
            "6e\u22124",
            "5e\u22124",
        ],
        ["No", "No", "wMSE", "wMSE", "No", "No", "No", "No"],
        [
            "80/15/5",
            "80/15/5",
            "80/15/5",
            "80/15/5",
            "80/15/5",
            "80/15/5",
            "80/10/10",
            "80/10/10",
        ],
    ]
    row_labels = ["Final Layer", "Dropout", "Batch Size", "LR", "Weighted", "Split"]

    tbl = ax2.table(
        cellText=table_data,
        rowLabels=row_labels,
        colLabels=trials,
        loc="bottom",
        cellLoc="center",
        bbox=[0.0, -0.92, 1.0, 0.55],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)

    # Style the table
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor(P_LGRAY)
        cell.set_linewidth(0.5)
        if row == 0:
            # Header row (trial names)
            cell.set_facecolor("#E8EDF2")
            cell.set_text_props(fontweight="bold", fontsize=8.5, color=P_DGRAY)
        elif col == -1:
            # Row labels
            cell.set_facecolor("#F0F3F7")
            cell.set_text_props(fontweight="bold", fontsize=7.5, color=P_DGRAY)
        else:
            cell.set_facecolor(BG)
            cell.set_text_props(fontsize=7.5, color=P_SLATE)
        # Highlight T8 column (col index 7)
        if col == 7 and row >= 1:
            cell.set_facecolor("#FFF0EC")

    # Legend — place inside panel (a) upper-left (above low T3/T4 bars)
    p_best = mpatches.Patch(color=P_CORAL, label="T8 (best)")
    p_good = mpatches.Patch(color=P_BLUE, label=r"R$^2$ $\geq$ 0.50")
    p_low = mpatches.Patch(color=P_XLGRAY, label=r"R$^2$ < 0.35")
    ax1.legend(
        handles=[p_best, p_good, p_low],
        loc="upper center",
        ncol=3,
        frameon=True,
        fontsize=8.5,
        framealpha=0.9,
        edgecolor=P_LGRAY,
    )

    # Footnote
    fig.text(
        0.02,
        0.005,
        "\u2020 T1: use_dropout=False \u2192 effective dropout = 0.0",
        fontsize=7.5,
        color=P_MGRAY,
        style="italic",
        va="bottom",
    )

    fig.suptitle(
        "Trial Progression T1\u2192T8: Architecture Changes and Performance",
        fontsize=13,
        fontweight="bold",
        color=P_DGRAY,
        y=0.99,
    )
    fig.subplots_adjust(left=0.10, right=0.97, top=0.93, bottom=0.32, hspace=0.28)
    save(fig, "fig12_trial_progression")


# ===========================================================================
# N3 — fig13_mc_dropout_inference: MC Dropout forward-pass diagram
# ===========================================================================
def fig13_mc_dropout_inference():
    fig, ax = plt.subplots(figsize=(12, 4.5))
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
    pass_lbl = [
        "\u0177\u2081, \u03c3\u2081",
        "\u0177\u2082, \u03c3\u2082",
        "  \u00b7\u00b7\u00b7",
        "\u0177\u2082\u2089, \u03c3\u2082\u2089",
        "\u0177\u2083\u2080, \u03c3\u2083\u2080",
    ]
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
    box(
        9.8,
        2.45,
        2.0,
        1.6,
        "Per-node\nOutput\n\u0177 = mean\n\u03c3 = std",
        P_CORAL,
        WHITE,
    )

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
        "MC Dropout Inference: S=30 Stochastic Forward Passes \u2192 Per-Node Mean \u0177 and \u03c3  (T8)",
        fontsize=13,
        fontweight="bold",
        color=P_DGRAY,
        pad=8,
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.85, bottom=0.04)
    save(fig, "fig13_mc_dropout_inference")


# ===========================================================================
# N4 — fig14_conformal_workflow: Conformal prediction diagram
# ===========================================================================
def fig14_conformal_workflow():
    """Three-column layout: Calibration | Quantile Bridge | Evaluation."""
    fig, ax = plt.subplots(figsize=(13, 10))
    fig.patch.set_facecolor(BG)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Column centres
    LX, CX, RX = 2.5, 6.5, 10.5
    BW = 2.8  # standard box width
    BH = 0.9  # standard box height

    # ── Helpers ──────────────────────────────────────────────────────
    def box(cx, cy, w, h, label, fc, tc=P_DGRAY, fs=9.5):
        r = mpatches.FancyBboxPatch(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            boxstyle="round,pad=0.12",
            linewidth=1.0,
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

    def arr(x0, y0, x1, y1, color=P_SLATE, ls="-"):
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=1.3,
                mutation_scale=14,
                linestyle=ls,
            ),
            zorder=2,
        )

    # ── Title ────────────────────────────────────────────────────────
    ax.set_title(
        "Split Conformal Prediction Workflow (T8)",
        fontsize=13,
        fontweight="bold",
        color=P_DGRAY,
        pad=12,
    )

    # ── Column headers (subtle) ──────────────────────────────────────
    ax.text(
        LX,
        9.65,
        "CALIBRATION",
        ha="center",
        fontsize=9,
        fontweight="bold",
        color=P_BLUE,
        alpha=0.6,
    )
    ax.text(
        CX,
        9.65,
        "QUANTILE BRIDGE",
        ha="center",
        fontsize=9,
        fontweight="bold",
        color=P_CORAL,
        alpha=0.6,
    )
    ax.text(
        RX,
        9.65,
        "EVALUATION",
        ha="center",
        fontsize=9,
        fontweight="bold",
        color=P_GREEN,
        alpha=0.6,
    )

    # ── Subtle column separators (thin dotted vertical lines) ────────
    for xsep in [4.5, 8.5]:
        ax.plot(
            [xsep, xsep], [0.3, 9.5], color=P_LGRAY, lw=0.5, ls=":", alpha=0.4, zorder=0
        )

    # ── Row 0: Source (centre column) ────────────────────────────────
    box(CX, 9.0, 3.2, BH, "100 Test Graphs\n(T8, MC Dropout, S=30)", P_XLGRAY, P_SLATE)

    # Split arrows + label
    ax.text(
        CX,
        8.28,
        "50 / 50 random split",
        ha="center",
        fontsize=8.5,
        color=P_SLATE,
        style="italic",
    )
    arr(CX - 0.6, 8.50, LX + 0.6, 7.78)  # → cal set
    arr(CX + 0.6, 8.50, RX - 0.6, 7.78)  # → eval set

    # ── Row 1: Cal Set (left) + Eval Set (right) ────────────────────
    box(
        LX,
        7.3,
        BW,
        BH,
        "Calibration Set\n50 graphs \u00b7 1.58M nodes",
        P_BLUE_LT,
        P_DGRAY,
    )
    box(
        RX,
        7.3,
        BW,
        BH,
        "Evaluation Set\n50 graphs \u00b7 1.58M nodes",
        P_GREEN_LT,
        P_DGRAY,
    )

    # ── Row 2: Compute Residuals (left only) ─────────────────────────
    arr(LX, 7.3 - BH / 2, LX, 5.8 + BH / 2)
    box(
        LX,
        5.8,
        BW,
        BH,
        "Compute Residuals\n|\u0177\u1d62 \u2212 y\u1d62| for each node",
        P_BLUE_LT,
        P_DGRAY,
    )

    # ── Eval Set → Apply Intervals (long down arrow, right column) ───
    arr(RX, 7.3 - BH / 2, RX, 4.1 + BH / 2)

    # ── Row 3: Quantile q (centre) + Apply Intervals (right) ────────
    # Diagonal arrow: Residuals → Quantile (left-col → centre-col)
    QH = 1.1  # quantile box slightly taller (3 lines)
    arr(LX + BW / 2 - 0.1, 5.55, CX - BW / 2 + 0.1, 4.1 + QH / 2)

    box(
        CX,
        4.1,
        BW,
        QH,
        "Conformal Quantile\nq\u2089\u2080 = 9.92 veh/h\nq\u2089\u2085 = 14.68 veh/h",
        P_CORAL,
        WHITE,
    )

    # Horizontal dashed arrow: Quantile → Apply Intervals (centre → right)
    arr(CX + BW / 2, 4.1, RX - BW / 2, 4.1, color=P_CORAL, ls="--")
    ax.text(
        (CX + RX) / 2,
        4.32,
        "q",
        ha="center",
        fontsize=9.5,
        color=P_CORAL,
        style="italic",
        fontweight="bold",
    )

    box(RX, 4.1, BW, BH, "Apply Intervals\n\u0177 \u00b1 q", P_GREEN_LT, P_DGRAY, fs=10)

    # ── Row 4: Coverage Check (right only) ───────────────────────────
    arr(RX, 4.1 - BH / 2, RX, 2.5 + BH / 2)
    box(
        RX,
        2.5,
        BW,
        BH,
        "Verify Coverage\nPICP \u2265 1\u2212\u03b1 ?",
        P_AMBER,
        P_DGRAY,
    )

    # ── Row 5: Results (right column) ────────────────────────────────
    arr(RX, 2.5 - BH / 2, RX, 1.0 + BH / 2)
    box(RX, 1.0, BW, BH, "90%: PICP = 90.02%\n95%: PICP = 95.01%", P_GREEN, WHITE)

    # ── Guarantee note (centred at bottom) ───────────────────────────
    ax.text(
        CX,
        0.18,
        "Marginal coverage guarantee holds for both levels",
        ha="center",
        fontsize=10,
        color=P_GREEN,
        style="italic",
        fontweight="bold",
    )

    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02)
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
