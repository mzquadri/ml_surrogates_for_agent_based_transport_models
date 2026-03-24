"""
Regenerate Fig 5.11: t8_reliability_diagram
Two panels: (a) Reliability diagram — observed vs nominal coverage
            (b) Calibration gap bar chart
Values from verified JSON: reliability_diagram_t8.json
"""

import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from thesis_style import *

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
JSON_PATH = os.path.join(
    REPO, "docs", "verified", "phase3_results", "reliability_diagram_t8.json"
)

FOOTNOTE = (
    "Results based on 1,000 of 10,000 available MATSim scenarios "
    "(10% subset), Trial 8 test set."
)

# ── Load verified JSON ──────────────────────────────────────────────────
with open(JSON_PATH, "r") as f:
    data = json.load(f)

nominal = np.array(data["nominal_levels"])
observed = np.array(data["observed_coverage"])
ece = data["expected_calibration_error_ECE"]
n_nodes = data["n_nodes"]
n_graphs = data["n_graphs"]

# Gaps in percentage points
gaps_pp = (observed - nominal) * 100

print("=== Verified values ===")
print(f"ECE = {ece:.3f}")
print(f"Nodes: {n_nodes:,}   Graphs: {n_graphs}")
for i, nom in enumerate(nominal):
    print(f"  {int(nom * 100):2d}%: observed={observed[i]:.4f}  gap={gaps_pp[i]:.1f}pp")

# ── Cross-checks ────────────────────────────────────────────────────────
assert abs(ece - 0.265) < 0.001, f"FAIL: ECE={ece}"
assert abs(observed[-1] - 0.5555) < 0.001, f"FAIL: obs@95%={observed[-1]}"
print("\n[OK] Cross-checks passed.")

# ── Figure ──────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
fig.patch.set_facecolor(BG)

# ── Panel (a): Reliability Diagram ──────────────────────────────────────
ax = ax1
panel_label(ax, "(a)", x=-0.08, y=1.04)

# Perfect calibration diagonal
ax.plot(
    [0, 1], [0, 1], "--", color=P_LGRAY, lw=1.5, label="Perfect calibration", zorder=2
)

# MC Dropout observed coverage
ax.plot(
    nominal,
    observed,
    "-o",
    color=P_CORAL,
    lw=2.2,
    ms=6,
    label="T8 MC Dropout",
    zorder=4,
)

# Shade the gap
ax.fill_between(nominal, observed, nominal, alpha=0.15, color=P_CORAL, zorder=1)

# ECE annotation — inside the shaded region
ax.text(
    0.55,
    0.18,
    f"ECE = {ece:.3f}",
    fontsize=11,
    fontweight="bold",
    color=P_CORAL,
    bbox=dict(
        boxstyle="round,pad=0.3", facecolor=WHITE, edgecolor=P_CORAL, alpha=0.9, lw=1.2
    ),
    ha="center",
    zorder=5,
)

ax.set_xlabel("Nominal coverage level", color=P_DGRAY)
ax.set_ylabel("Observed coverage", color=P_DGRAY)
ax.set_title("Reliability Diagram", fontsize=12, color=P_DGRAY, pad=8)
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.0)
ax.tick_params(colors=P_SLATE, labelsize=9)
ax.legend(fontsize=9, framealpha=0.9, loc="upper left")
ax.grid(True, alpha=0.3, zorder=0)

# ── Panel (b): Calibration Gap ──────────────────────────────────────────
ax = ax2
panel_label(ax, "(b)", x=-0.08, y=1.04)

x_pos = np.arange(len(nominal))
x_labels = [f"{int(n * 100)}%" for n in nominal]

bars = ax.bar(x_pos, gaps_pp, 0.65, color=P_CORAL, alpha=0.85, zorder=3)

# Value labels on each bar
for i, bar in enumerate(bars):
    h = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        h - 1.2,
        f"{gaps_pp[i]:.1f}pp",
        ha="center",
        va="top",
        fontsize=7.5,
        fontweight="semibold",
        color=P_DGRAY,
    )

ax.axhline(0, color=P_SLATE, lw=0.8, zorder=2)

ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels, rotation=45, ha="right")
ax.set_xlabel("Nominal coverage level", color=P_DGRAY)
ax.set_ylabel("Coverage gap (percentage points)", color=P_DGRAY)
ax.set_title("Calibration Gap", fontsize=12, color=P_DGRAY, pad=8)
ax.tick_params(colors=P_SLATE, labelsize=9)
ax.grid(True, axis="y", alpha=0.3, zorder=0)

# Give some breathing room below the bars
ax.set_ylim(min(gaps_pp) * 1.15, 2)

# Suptitle
fig.suptitle(
    f"T8 MC Dropout Calibration: Reliability Diagram (S=30, {n_graphs} graphs, {n_nodes:,} nodes)",
    fontsize=13,
    fontweight="bold",
    color=P_DGRAY,
    y=0.99,
)

fig.text(0.5, -0.02, FOOTNOTE, ha="center", fontsize=8, color=P_MGRAY, style="italic")

fig.tight_layout(pad=1.8, rect=[0, 0.02, 1, 0.94])

# ── Save ────────────────────────────────────────────────────────────────
THESIS_FIG = os.path.join(REPO, "thesis", "latex_tum_official", "figures")
for ext in ("pdf", "png"):
    out = os.path.join(THESIS_FIG, f"t8_reliability_diagram.{ext}")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("\nSaved: t8_reliability_diagram.pdf + .png")
