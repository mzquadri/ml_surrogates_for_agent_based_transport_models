"""
Regenerate Fig 5.6: t8_conformal_conditional
Conditional conformal coverage by MC Dropout uncertainty decile.
Global vs Adaptive conformal — values from verified JSON.
"""

import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from thesis_style import *

# ── Load verified JSON ──────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
JSON_PATH = os.path.join(
    REPO_ROOT,
    "docs",
    "verified",
    "phase3_results",
    "conformal_conditional_coverage_t8.json",
)
with open(JSON_PATH, "r") as f:
    data = json.load(f)

deciles = data["sigma_deciles"]

# ── Extract arrays ──────────────────────────────────────────────────────
d_nums = [d["decile"] for d in deciles]
glob90 = [d["global_coverage_90"] * 100 for d in deciles]
glob95 = [d["global_coverage_95"] * 100 for d in deciles]
adapt90 = [d["adaptive_coverage_90"] * 100 for d in deciles]
adapt95 = [d["adaptive_coverage_95"] * 100 for d in deciles]

x = np.arange(len(d_nums))
width = 0.35

# ── Figure ──────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8), sharey=True)
fig.patch.set_facecolor(BG)

# ── Shared y-axis range for fair comparison ─────────────────────────────
Y_MIN, Y_MAX = 55, 102

# ── PANEL (a): Global Conformal ─────────────────────────────────────────
panel_label(ax1, "(a)", x=-0.06, y=1.04)

bars_90a = ax1.bar(
    x - width / 2,
    glob90,
    width,
    color=P_BLUE,
    edgecolor=P_LGRAY,
    linewidth=0.5,
    label="90% coverage",
    zorder=3,
)
bars_95a = ax1.bar(
    x + width / 2,
    glob95,
    width,
    color=P_BLUE_DK,
    edgecolor=P_LGRAY,
    linewidth=0.5,
    label="95% coverage",
    zorder=3,
)

# Nominal target lines
ax1.axhline(90, color=P_CORAL, linestyle="--", linewidth=1.3, alpha=0.8, zorder=2)
ax1.axhline(95, color=P_CORAL_LT, linestyle="--", linewidth=1.3, alpha=0.8, zorder=2)

# Target labels on right edge
ax1.text(
    9.6,
    90,
    "90% target",
    fontsize=8,
    color=P_CORAL,
    va="bottom",
    ha="right",
    fontweight="semibold",
)
ax1.text(
    9.6,
    95,
    "95% target",
    fontsize=8,
    color=P_CORAL_LT,
    va="bottom",
    ha="right",
    fontweight="semibold",
)

ax1.set_xticks(x)
ax1.set_xticklabels([f"D{d}" for d in d_nums], fontsize=9)
ax1.set_xlabel(
    "Uncertainty decile (D1 = lowest $\\sigma$, D10 = highest $\\sigma$)", color=P_DGRAY
)
ax1.set_ylabel("Coverage (%)", color=P_DGRAY)
ax1.set_title("Global Conformal", color=P_DGRAY, fontweight="bold")
ax1.set_ylim(Y_MIN, Y_MAX)
ax1.grid(True, axis="y", alpha=0.3, zorder=0)
ax1.tick_params(colors=P_SLATE)
ax1.legend(fontsize=9, loc="upper right", framealpha=0.9)

# Value labels on D1 and D10
for idx, label_idx in [(0, "D1"), (9, "D10")]:
    # 90% bar
    ax1.text(
        x[idx] - width / 2,
        glob90[idx] + 0.6,
        f"{glob90[idx]:.1f}%",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=P_BLUE,
        fontweight="semibold",
    )
    # 95% bar
    ax1.text(
        x[idx] + width / 2,
        glob95[idx] + 0.6,
        f"{glob95[idx]:.1f}%",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=P_BLUE_DK,
        fontweight="semibold",
    )

# Range annotation box
glob90_range = max(glob90) - min(glob90)
ax1.text(
    0.03,
    0.03,
    f"90% range: {glob90_range:.1f} pp\n(D1: {glob90[0]:.1f}% $\\to$ D10: {glob90[-1]:.1f}%)",
    transform=ax1.transAxes,
    fontsize=8.5,
    color=P_DGRAY,
    bbox=dict(boxstyle="round,pad=0.4", facecolor=WHITE, edgecolor=P_LGRAY, alpha=0.95),
    va="bottom",
    ha="left",
)

# ── PANEL (b): Adaptive Conformal ───────────────────────────────────────
panel_label(ax2, "(b)", x=-0.04, y=1.04)

bars_90b = ax2.bar(
    x - width / 2,
    adapt90,
    width,
    color=P_BLUE,
    edgecolor=P_LGRAY,
    linewidth=0.5,
    label="90% coverage",
    zorder=3,
)
bars_95b = ax2.bar(
    x + width / 2,
    adapt95,
    width,
    color=P_BLUE_DK,
    edgecolor=P_LGRAY,
    linewidth=0.5,
    label="95% coverage",
    zorder=3,
)

# Nominal target lines
ax2.axhline(90, color=P_CORAL, linestyle="--", linewidth=1.3, alpha=0.8, zorder=2)
ax2.axhline(95, color=P_CORAL_LT, linestyle="--", linewidth=1.3, alpha=0.8, zorder=2)

# Target labels
ax2.text(
    9.6,
    90,
    "90% target",
    fontsize=8,
    color=P_CORAL,
    va="bottom",
    ha="right",
    fontweight="semibold",
)
ax2.text(
    9.6,
    95,
    "95% target",
    fontsize=8,
    color=P_CORAL_LT,
    va="bottom",
    ha="right",
    fontweight="semibold",
)

ax2.set_xticks(x)
ax2.set_xticklabels([f"D{d}" for d in d_nums], fontsize=9)
ax2.set_xlabel(
    "Uncertainty decile (D1 = lowest $\\sigma$, D10 = highest $\\sigma$)", color=P_DGRAY
)
ax2.set_title(
    "Adaptive Conformal ($\\sigma$-normalized)", color=P_DGRAY, fontweight="bold"
)
ax2.set_ylim(Y_MIN, Y_MAX)
ax2.grid(True, axis="y", alpha=0.3, zorder=0)
ax2.tick_params(colors=P_SLATE)
ax2.legend(fontsize=9, loc="upper left", framealpha=0.9)

# Value labels on D1 and D10
for idx, label_idx in [(0, "D1"), (9, "D10")]:
    # 90% bar
    ax2.text(
        x[idx] - width / 2,
        adapt90[idx] + 0.6,
        f"{adapt90[idx]:.1f}%",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=P_BLUE,
        fontweight="semibold",
    )
    # 95% bar
    ax2.text(
        x[idx] + width / 2,
        adapt95[idx] + 0.6,
        f"{adapt95[idx]:.1f}%",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=P_BLUE_DK,
        fontweight="semibold",
    )

# Also label D2 (the dip) for 90%
ax2.text(
    x[1] - width / 2,
    adapt90[1] - 2.5,
    f"{adapt90[1]:.1f}%",
    ha="center",
    va="top",
    fontsize=7.5,
    color=P_BLUE,
    fontweight="semibold",
)

# Range annotation box
adapt90_range = max(adapt90) - min(adapt90)
adapt90_min_d = d_nums[adapt90.index(min(adapt90))]
ax2.text(
    0.97,
    0.03,
    f"90% range: {adapt90_range:.1f} pp\n(D{adapt90_min_d}: {min(adapt90):.1f}% $\\to$ D10: {adapt90[-1]:.1f}%)",
    transform=ax2.transAxes,
    fontsize=8.5,
    color=P_DGRAY,
    bbox=dict(boxstyle="round,pad=0.4", facecolor=WHITE, edgecolor=P_LGRAY, alpha=0.95),
    va="bottom",
    ha="right",
)

# ── Suptitle ────────────────────────────────────────────────────────────
fig.suptitle(
    "Conditional Conformal Coverage by Uncertainty Decile (T8, 20/80 cal/eval split)",
    fontsize=13,
    color=P_DGRAY,
    fontweight="bold",
    y=0.99,
)

# ── Footnote ────────────────────────────────────────────────────────────
fig.text(
    0.5,
    -0.02,
    "Results based on 1,000 of 10,000 available MATSim scenarios (10% subset).",
    ha="center",
    fontsize=8,
    color=P_MGRAY,
    style="italic",
)

fig.tight_layout(pad=1.8, rect=[0, 0.02, 1, 0.94])

# ── Save ────────────────────────────────────────────────────────────────
out_pdf = os.path.join(SCRIPT_DIR, "t8_conformal_conditional.pdf")
out_png = os.path.join(SCRIPT_DIR, "t8_conformal_conditional.png")
fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor=BG)
fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"Saved: t8_conformal_conditional.pdf + .png")

# ── Print verified values for confirmation ──────────────────────────────
print("\n=== Verified values used ===")
print(
    f"Global 90%:   D1={glob90[0]:.1f}%, D10={glob90[-1]:.1f}%, range={max(glob90) - min(glob90):.1f}pp"
)
print(f"Global 95%:   D1={glob95[0]:.1f}%, D10={glob95[-1]:.1f}%")
print(
    f"Adaptive 90%: D1={adapt90[0]:.1f}%, D2={adapt90[1]:.1f}%(min), D10={adapt90[-1]:.1f}%, range={max(adapt90) - min(adapt90):.1f}pp"
)
print(f"Adaptive 95%: D1={adapt95[0]:.1f}%, D10={adapt95[-1]:.1f}%")
