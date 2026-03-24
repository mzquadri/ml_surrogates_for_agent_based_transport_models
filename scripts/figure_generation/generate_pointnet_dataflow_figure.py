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
import sys

# Ensure thesis_style is importable from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from thesis_style import *

import matplotlib.patches as mpatches
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))


def save(fig, name):
    save_fig(fig, name, out_dir=OUT, bg=BG, skip_tight=True)


def fig_pointnet_dataflow():
    fig, ax = plt.subplots(figsize=(13, 5.5))
    fig.patch.set_facecolor(BG)
    ax.set_xlim(0, 13)
    ax.set_ylim(0.2, 5.6)
    ax.axis("off")

    # -----------------------------------------------------------------------
    # Layout — tighter spacing, larger boxes
    # -----------------------------------------------------------------------
    cx_list = [1.3, 3.8, 6.5, 9.2, 11.7]
    bw, bh = 2.1, 2.1
    cy = 3.45

    # Light fills + dark text (consistent with Fig 3.3 style)
    stages = [
        (
            "Road Network",
            "31,635 nodes / scenario\n5 node features\n1,000 MATSim runs",
            P_XLGRAY,
            P_SLATE,
        ),
        (
            "Feature Matrix",
            "VOL \u00b7 CAP \u00b7 CAP_RED\nSPD \u00b7 LEN\nN \u00d7 5 per graph",
            "#C4DECE",
            P_DGRAY,
        ),
        (
            "GNN Surrogate (T8)",
            "PointNet encoder\n+ TransformerConv\n+ GATConv  |  h = 512",
            "#D2E4F0",
            P_DGRAY,
        ),
        (
            "Traffic Prediction",
            "\u0394v per segment (veh/h)\nN \u00d7 1 node output\nR\u00b2 = 0.5957",
            "#B8D4E8",
            P_DGRAY,
        ),
        (
            "MC Uncertainty",
            "S = 30 forward passes\n\u03c3 = std(\u0177),  \u03c1 = 0.4820\n90 % / 95 % intervals",
            "#F0D5C8",
            P_DGRAY,
        ),
    ]

    stage_names = [
        "Input data",
        "Graph features",
        "GNN model",
        "Prediction",
        "Uncertainty",
    ]

    for i, (title, detail, fc, tc) in enumerate(stages):
        cx = cx_list[i]
        bx = cx - bw / 2

        # Main box
        rect = mpatches.FancyBboxPatch(
            (bx, cy - bh / 2),
            bw,
            bh,
            boxstyle="round,pad=0.10",
            linewidth=0.9,
            edgecolor=P_LGRAY,
            facecolor=fc,
            zorder=2,
        )
        ax.add_patch(rect)

        # Title (upper)
        ax.text(
            cx,
            cy + 0.50,
            title,
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color=tc,
            zorder=3,
        )

        # Thin rule
        rule_y = cy + 0.12
        ax.plot(
            [bx + 0.15, bx + bw - 0.15],
            [rule_y, rule_y],
            color=P_LGRAY,
            lw=0.6,
            zorder=3,
        )

        # Detail lines (lower)
        ax.text(
            cx,
            cy - 0.42,
            detail,
            ha="center",
            va="center",
            fontsize=9,
            color=tc,
            multialignment="center",
            linespacing=1.45,
            zorder=3,
        )

        # Stage label below box
        ax.text(
            cx,
            cy - bh / 2 - 0.18,
            stage_names[i],
            ha="center",
            va="top",
            fontsize=8.5,
            color=P_MGRAY,
            style="italic",
        )

    # -----------------------------------------------------------------------
    # Connector arrows
    # -----------------------------------------------------------------------
    for i in range(4):
        x0 = cx_list[i] + bw / 2 + 0.05
        x1 = cx_list[i + 1] - bw / 2 - 0.05
        ax.annotate(
            "",
            xy=(x1, cy),
            xytext=(x0, cy),
            arrowprops=dict(arrowstyle="-|>", color=P_SLATE, lw=1.2, mutation_scale=13),
            zorder=4,
        )

    # -----------------------------------------------------------------------
    # Footer — metric summary
    # -----------------------------------------------------------------------
    footer_y = 1.75
    ax.axhline(footer_y, xmin=0.03, xmax=0.97, color=P_LGRAY, lw=0.6)

    ax.text(
        6.5,
        footer_y - 0.14,
        "T8 best model  \u00b7  Test R\u00b2 = 0.5957  \u00b7  MAE = 3.96 veh/h  \u00b7  "
        "RMSE = 7.12 veh/h  \u00b7  MC Dropout \u03c1 = 0.4820",
        ha="center",
        va="top",
        fontsize=8.5,
        color=P_SLATE,
    )

    ax.text(
        6.5,
        footer_y - 0.48,
        "80 / 10 / 10 train\u2013val\u2013test split  \u00b7  LR = 1\u00d710\u207b\u00b3  \u00b7  "
        "100 epochs  \u00b7  Dropout p = 0.15  \u00b7  1,000 of 10,000 MATSim scenarios",
        ha="center",
        va="top",
        fontsize=8,
        color=P_MGRAY,
    )

    # Title
    ax.set_title(
        "Data Flow: Road Network  \u2192  GNN Surrogate  \u2192  "
        "Traffic Prediction with Uncertainty Quantification",
        fontsize=13,
        fontweight="bold",
        color=P_DGRAY,
        pad=12,
    )

    fig.tight_layout(pad=1.2)
    save(fig, "pointnet_data_flow")


if __name__ == "__main__":
    print("Generating pointnet data flow figure...")
    fig_pointnet_dataflow()
    print("Done.")
