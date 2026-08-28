"""Regenerate fig35: 3-tier uncertainty-guided decision framework."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from plot_style import set_style, save_both, TUM_BLUE, TUM_ORANGE, TUM_GRAY, TUM_DARK, PASS_GREEN, FAIL_RED

set_style()
OUT = "../../../document/figures/new"

def box(ax, x, y, w, h, text, fc=TUM_BLUE, txt_color="white", fontsize=10, fontweight="bold", ec="black", lw=1.0):
    p = FancyBboxPatch((x - w/2, y - h/2), w, h,
                       boxstyle="round,pad=0.05,rounding_size=0.12",
                       facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, color=txt_color)

def arrow(ax, x1, y1, x2, y2, color=TUM_DARK, lw=1.6):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=lw, color=color,
                                shrinkA=4, shrinkB=4, mutation_scale=18))

# ============================================================
# Figure: 3-tier decision framework
# ============================================================
fig, ax = plt.subplots(figsize=(13.5, 7.2))
ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis("off")

# Title
ax.text(7, 7.55, "Uncertainty-guided decision framework for GNN traffic surrogates",
        ha="center", fontsize=13, fontweight="bold", color=TUM_DARK)
ax.text(7, 7.15, "Route each prediction by MC Dropout $\\sigma$ percentile",
        ha="center", fontsize=10, style="italic", color=TUM_DARK)

# Top: input
box(ax, 1.6, 5.9, 2.4, 1.1, "GNN prediction\n$(\\hat{y}_i, \\sigma_i)$\nper node",
    fc=TUM_BLUE, fontsize=10)

# Decision / ranker in middle
box(ax, 5.3, 5.9, 2.8, 1.1, "Rank by $\\sigma_i$\n(percentile across test set)",
    fc=TUM_GRAY, fontsize=10)
arrow(ax, 2.8, 5.9, 3.9, 5.9)

# Three tiers on the right
# ACCEPT tier (bottom 50% sigma)
box(ax, 10.7, 6.5, 3.8, 0.9, "ACCEPT",
    fc=PASS_GREEN, fontsize=12.5, fontweight="bold")
ax.text(10.7, 5.9,
        "Bottom 50\\% $\\sigma$  \\ (most confident)\n"
        "Expected MAE $\\approx 2.32$ veh/h  (\\textminus 41.2\\%)\n"
        "\\textbf{Use prediction directly}",
        ha="center", va="center", fontsize=9, color=TUM_DARK)
arrow(ax, 6.7, 6.1, 8.8, 6.5, color=PASS_GREEN, lw=1.8)

# FLAG tier (50-90% sigma)
box(ax, 10.7, 4.2, 3.8, 0.9, "FLAG for review",
    fc=TUM_ORANGE, fontsize=12.5, fontweight="bold")
ax.text(10.7, 3.6,
        "50\\%--90\\% $\\sigma$\n"
        "Sensitivity analysis or\ncross-check with alternate model",
        ha="center", va="center", fontsize=9, color=TUM_DARK)
arrow(ax, 6.7, 5.8, 8.8, 4.2, color=TUM_ORANGE, lw=1.8)

# REJECT tier (top 10% sigma)
box(ax, 10.7, 1.9, 3.8, 0.9, "REJECT / abstain",
    fc=FAIL_RED, fontsize=12.5, fontweight="bold")
ax.text(10.7, 1.3,
        "Top 10\\% $\\sigma$  \\ (least confident)\n"
        "Trigger full MATSim re-simulation\nor discard prediction",
        ha="center", va="center", fontsize=9, color=TUM_DARK)
arrow(ax, 6.7, 5.6, 8.8, 1.9, color=FAIL_RED, lw=1.8)

# Threshold scale on left side
ax.text(1.6, 3.9, "$\\sigma$ percentile", ha="center", fontsize=10, fontweight="bold", color=TUM_DARK)
# Vertical gradient bar
from matplotlib.patches import Polygon
grad_x = 1.6; grad_ytop = 3.5; grad_ybot = 0.6; grad_w = 0.5
# Segmented bar: 3 regions
ax.add_patch(Rectangle((grad_x - grad_w/2, grad_ybot), grad_w, (grad_ytop - grad_ybot) * 0.50,
                       facecolor=PASS_GREEN, edgecolor="black", linewidth=0.8))
ax.add_patch(Rectangle((grad_x - grad_w/2, grad_ybot + (grad_ytop - grad_ybot) * 0.50),
                       grad_w, (grad_ytop - grad_ybot) * 0.40,
                       facecolor=TUM_ORANGE, edgecolor="black", linewidth=0.8))
ax.add_patch(Rectangle((grad_x - grad_w/2, grad_ybot + (grad_ytop - grad_ybot) * 0.90),
                       grad_w, (grad_ytop - grad_ybot) * 0.10,
                       facecolor=FAIL_RED, edgecolor="black", linewidth=0.8))

# Labels on scale
ax.text(grad_x + 0.55, grad_ybot, "0\\%", fontsize=9, va="center", color=TUM_DARK)
ax.text(grad_x + 0.55, grad_ybot + (grad_ytop - grad_ybot) * 0.50, "50\\%", fontsize=9, va="center", color=TUM_DARK)
ax.text(grad_x + 0.55, grad_ybot + (grad_ytop - grad_ybot) * 0.90, "90\\%", fontsize=9, va="center", color=TUM_DARK)
ax.text(grad_x + 0.55, grad_ytop, "100\\%", fontsize=9, va="center", color=TUM_DARK)

# Footer with source
ax.text(7, 0.15,
        "Thresholds learned from selective-prediction curve (Section 5.7): " +
        "50\\% retention saves \\textminus 41.2\\% MAE,\\ \\ 10\\% reject flags top-$\\sigma$ segments " +
        "(AUROC $= 0.7548$ for top-10\\% errors)",
        ha="center", fontsize=8.5, color=TUM_DARK, style="italic")

save_both(fig, OUT, "fig35_policy_decision_framework")
print("fig35 regenerated")
