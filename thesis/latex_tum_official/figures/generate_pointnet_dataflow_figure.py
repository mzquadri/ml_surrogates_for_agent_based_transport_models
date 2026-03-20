"""
generate_pointnet_dataflow_figure.py
=====================================
Generates Fig 3.4: Data flow diagram from road network to GNN surrogate
prediction with uncertainty quantification.

All data values verified against thesis results (T8 model, test set):
  - R² = 0.5957, MAE = 3.96 veh/h, RMSE = 7.12 veh/h
  - MC Dropout ρ = 0.4820 (Spearman, uncertainty vs |error|)
  - Architecture: PointNet → TransformerConv → GATConv, hidden=512

Run from the figures/ directory:
    python generate_pointnet_dataflow_figure.py

Outputs:
    pointnet_data_flow.pdf
    pointnet_data_flow.png
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
    print(f"  saved {name}.pdf  +  {name}.png")


def fig_pointnet_dataflow():
    fig, ax = plt.subplots(figsize=(14, 6.5))
    fig.patch.set_facecolor(BG)
    ax.set_xlim(0, 14)
    ax.set_ylim(0.3, 6.2)
    ax.axis("off")

    # -----------------------------------------------------------------------
    # Layout constants
    # -----------------------------------------------------------------------
    cx_list = [1.5, 4.2, 7.0, 9.8, 12.5]  # box centre x
    bw, bh = 2.2, 2.3  # box width / height (taller)
    cy = 3.7  # box centre y (raised)
    BADGE_Y = 5.35  # step-number circle y (raised)

    # -----------------------------------------------------------------------
    # Stage definitions — (title, detail lines, face-color, text-color,
    #                       badge-fill, badge-text-color)
    # -----------------------------------------------------------------------
    stages = [
        (
            "Road Network",
            "31,635 nodes / scenario\n5 node features\n1,000 MATSim runs",
            P_XLGRAY,
            P_SLATE,
            P_MGRAY,
            WHITE,
        ),
        (
            "Feature Matrix",
            "VOL · CAP · CAP_RED\nSPD · LEN\nN × 5 per graph",
            P_GREEN_LT,
            P_DGRAY,
            P_GREEN,
            WHITE,
        ),
        (
            "GNN Surrogate (T8)",
            "PointNet encoder\n+ TransformerConv\n+ GATConv  |  h = 512",
            P_BLUE_LT,
            P_DGRAY,
            P_BLUE,
            WHITE,
        ),
        (
            "Traffic Prediction",
            "Δv per segment (veh/h)\nN × 1 node output\nR² = 0.5957",
            PANEL,
            P_DGRAY,
            P_BLUE_DK,
            WHITE,
        ),
        (
            "MC Uncertainty",
            "S = 30 forward passes\nσ = std(ŷ),  ρ = 0.4820\n90 % / 95 % intervals",
            P_CORAL_LT,
            P_DGRAY,
            P_CORAL,
            WHITE,
        ),
    ]

    for i, (title, detail, fc, tc, badge_fc, badge_tc) in enumerate(stages):
        cx = cx_list[i]
        bx = cx - bw / 2
        by = cy - bh / 2

        # --- main box -------------------------------------------------------
        rect = mpatches.FancyBboxPatch(
            (bx, by),
            bw,
            bh,
            boxstyle="round,pad=0.12",
            linewidth=0.7,
            edgecolor=P_LGRAY,
            facecolor=fc,
            zorder=2,
        )
        ax.add_patch(rect)

        # --- title (upper portion of box) -----------------------------------
        ax.text(
            cx,
            cy + 0.55,
            title,
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color=tc,
            zorder=3,
        )

        # --- thin rule between title and detail -----------------------------
        rule_y = cy + 0.15
        ax.plot(
            [bx + 0.18, bx + bw - 0.18],
            [rule_y, rule_y],
            color=P_LGRAY,
            lw=0.6,
            zorder=3,
        )

        # --- detail lines (lower portion of box) ----------------------------
        ax.text(
            cx,
            cy - 0.45,
            detail,
            ha="center",
            va="center",
            fontsize=8.2,
            color=tc,
            multialignment="center",
            linespacing=1.50,
            zorder=3,
        )

        # --- step-number badge above box ------------------------------------
        badge = plt.Circle((cx, BADGE_Y), 0.23, color=badge_fc, zorder=5)
        ax.add_patch(badge)
        ax.text(
            cx,
            BADGE_Y,
            str(i + 1),
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=badge_tc,
            zorder=6,
        )

        # --- stage label below box ------------------------------------------
        stage_names = [
            "Input data",
            "Graph features",
            "GNN model",
            "Prediction",
            "Uncertainty",
        ]
        ax.text(
            cx,
            by - 0.20,
            stage_names[i],
            ha="center",
            va="top",
            fontsize=8,
            color=P_MGRAY,
            style="italic",
        )

    # -----------------------------------------------------------------------
    # Connector arrows between boxes
    # -----------------------------------------------------------------------
    gap = 0.10
    for i in range(4):
        x0 = cx_list[i] + bw / 2 + gap
        x1 = cx_list[i + 1] - bw / 2 - gap
        ax.annotate(
            "",
            xy=(x1, cy),
            xytext=(x0, cy),
            arrowprops=dict(
                arrowstyle="-|>",
                color=P_LGRAY,
                lw=1.1,
                mutation_scale=13,
            ),
            zorder=4,
        )

    # -----------------------------------------------------------------------
    # Footer — thin rule + two clean metric lines (no heavy dark box)
    # -----------------------------------------------------------------------
    footer_y = 1.95
    ax.axhline(footer_y, xmin=0.03, xmax=0.97, color=P_LGRAY, lw=0.6, zorder=1)

    ax.text(
        7.0,
        footer_y - 0.16,
        "T8 best model  ·  Test R² = 0.5957  ·  MAE = 3.96 veh/h  ·  "
        "RMSE = 7.12 veh/h  ·  MC Dropout ρ = 0.4820",
        ha="center",
        va="top",
        fontsize=8.5,
        color=P_SLATE,
        multialignment="center",
    )
    ax.text(
        7.0,
        footer_y - 0.52,
        "80 / 10 / 10 train–val–test split  ·  LR = 1×10⁻³  ·  "
        "100 epochs  ·  Dropout p = 0.15  ·  1,000 of 10,000 MATSim scenarios",
        ha="center",
        va="top",
        fontsize=8.0,
        color=P_MGRAY,
        multialignment="center",
    )

    # -----------------------------------------------------------------------
    # Title
    # -----------------------------------------------------------------------
    ax.set_title(
        "Data Flow: Road Network  →  GNN Surrogate  →  "
        "Traffic Prediction with Uncertainty Quantification",
        fontsize=12,
        fontweight="bold",
        color=P_DGRAY,
        pad=12,
    )

    fig.tight_layout(pad=0.6)
    save(fig, "pointnet_data_flow")


if __name__ == "__main__":
    print("Generating pointnet data flow figure...")
    fig_pointnet_dataflow()
    print("Done.")
