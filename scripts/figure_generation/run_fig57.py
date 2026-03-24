"""
Regenerate Fig 5.7: fig7_calibration
Scaling factor k95 for ±k*sigma to achieve 95% coverage.
Ideal Gaussian (1.96) vs T8 MC Dropout (11.34).
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from thesis_style import *

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# ── Verified values ─────────────────────────────────────────────────────
# Source: docs/verified/phase3_results/verify_all_metrics_summary.json
# k95 = 11.34, Gaussian k95 = 1.96
labels = ["Ideal\nGaussian", "T8\nMC Dropout"]
k95 = [1.96, 11.34]
colors = [P_BLUE, P_CORAL]

# ── Figure ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 5))
fig.patch.set_facecolor(BG)

x = np.arange(2)
bars = ax.bar(x, k95, 0.45, color=colors, edgecolor=P_LGRAY, linewidth=0.6, zorder=3)

# Value labels on top of bars
for i, (bar, val) in enumerate(zip(bars, k95)):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.25,
        f"{val:.2f}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color=P_DGRAY,
    )

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("$k_{95}$  (scaling factor for 95% coverage)", color=P_DGRAY)
ax.set_ylim(0, 14.5)
ax.grid(True, axis="y", alpha=0.3, zorder=0)
ax.set_axisbelow(True)
ax.tick_params(colors=P_SLATE)

# Reference dashed line at k=1.96
ax.axhline(1.96, color=P_BLUE, lw=1.0, ls="--", alpha=0.6, zorder=2)
ax.text(
    -0.45,
    3.2,
    "Calibrated Gaussian:\n$k_{95}$ = 1.96",
    fontsize=8.5,
    color=P_BLUE,
    style="italic",
    ha="left",
)

# Ratio annotation — double arrow to the right of bars
ratio = 11.34 / 1.96
arrow_x = 1.38  # right of the T8 bar
ax.annotate(
    "",
    xy=(arrow_x, 11.34),
    xytext=(arrow_x, 1.96),
    arrowprops=dict(arrowstyle="<->", color=P_SLATE, lw=1.2),
    annotation_clip=False,
)
ax.text(
    arrow_x + 0.08,
    6.5,
    f"{ratio:.1f}$\\times$",
    fontsize=11,
    fontweight="bold",
    color=P_SLATE,
    ha="left",
    va="center",
)

# Expand x-axis to make room for arrow on right
ax.set_xlim(-0.5, 2.0)

# Title
ax.set_title(
    "MC Dropout Calibration: Scaling Factor $k_{95}$\n"
    "for $\\pm k\\sigma$ to Achieve 95% Coverage (T8)",
    fontweight="bold",
    color=P_DGRAY,
    pad=12,
)

fig.tight_layout(pad=1.5)

# ── Save ────────────────────────────────────────────────────────────────
out_dir = os.path.join(REPO, "thesis", "latex_tum_official", "figures")
out_pdf = os.path.join(out_dir, "fig7_calibration.pdf")
out_png = os.path.join(out_dir, "fig7_calibration.png")
fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor=BG)
fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved: fig7_calibration.pdf + .png")
print(f"\nVerified: k95_gaussian=1.96, k95_t8=11.34, ratio={ratio:.1f}x")
