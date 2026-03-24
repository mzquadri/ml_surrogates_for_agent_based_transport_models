"""
Regenerate Fig 5.13: t8_pit_histogram
PIT histogram for T8 MC Dropout (S=30).
Values from verified JSON: pit_t8.json
"""

import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from thesis_style import *

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
JSON_PATH = os.path.join(REPO, "docs", "verified", "phase3_results", "pit_t8.json")

FOOTNOTE = (
    "Results based on 1,000 of 10,000 available MATSim scenarios "
    "(10% subset), Trial 8 test set."
)

# ── Load verified JSON ──────────────────────────────────────────────────
with open(JSON_PATH, "r") as f:
    data = json.load(f)

hist_density = np.array(data["histogram_density"])
bin_edges = np.array(data["bin_edges"])
n_bins = data["n_bins"]
pit_mean = data["pit_mean"]
pit_std = data["pit_std"]
pit_median = data["pit_median"]
ks_stat = data["ks_test_subsample"]["ks_stat"]
n_nodes = data["n_nodes"]
first_bin = hist_density[0]

print("=== Verified values ===")
print(f"PIT mean = {pit_mean}")
print(f"PIT std  = {pit_std}")
print(f"PIT median = {pit_median}")
print(f"KS stat  = {ks_stat}")
print(f"First bin density = {first_bin:.4f}")
print(f"Last bin density  = {hist_density[-1]:.4f}")
print(f"N nodes  = {n_nodes:,}")

# ── Cross-checks ────────────────────────────────────────────────────────
assert abs(pit_mean - 0.4331) < 0.001, f"FAIL: mean={pit_mean}"
assert abs(pit_std - 0.3991) < 0.001, f"FAIL: std={pit_std}"
assert abs(ks_stat - 0.2445) < 0.001, f"FAIL: KS={ks_stat}"
assert abs(first_bin - 0.2839) < 0.001, f"FAIL: first_bin={first_bin}"
print("\n[OK] Cross-checks passed.")

# ── Figure ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))
fig.patch.set_facecolor(BG)

# Bar width from bin edges
bin_widths = np.diff(bin_edges)
bin_centers = bin_edges[:-1] + bin_widths / 2

# Plot histogram bars
ax.bar(
    bin_centers,
    hist_density,
    width=bin_widths * 0.95,
    color=P_BLUE,
    alpha=0.85,
    zorder=3,
    label="Observed PIT",
)

# Uniform reference line
uniform_level = 1.0 / n_bins
ax.axhline(
    uniform_level,
    color=P_CORAL,
    lw=2.0,
    ls="--",
    label=f"Uniform (1/{n_bins} = {uniform_level:.3f})",
    zorder=4,
)

# Stats annotation box — top right
stats_text = f"KS = {ks_stat:.4f}\nMean = {pit_mean:.3f}\nStd = {pit_std:.3f}"
ax.text(
    0.97,
    0.95,
    stats_text,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="top",
    horizontalalignment="right",
    bbox=dict(
        boxstyle="round,pad=0.4", facecolor=P_XLGRAY, edgecolor=P_LGRAY, alpha=0.95
    ),
    fontfamily="monospace",
    color=P_DGRAY,
    zorder=5,
)

# First bin annotation — highlight the spike
ax.annotate(
    f"{first_bin * 100:.1f}% in first bin\n(expected 5%)",
    xy=(bin_centers[0], first_bin),
    xytext=(0.18, first_bin - 0.02),
    fontsize=8.5,
    color=P_CORAL,
    fontweight="semibold",
    arrowprops=dict(arrowstyle="->", color=P_CORAL, lw=1.0),
    ha="left",
    va="top",
)

ax.set_xlabel("PIT value", color=P_DGRAY)
ax.set_ylabel("Relative frequency", color=P_DGRAY)
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, max(hist_density) * 1.08)
ax.tick_params(colors=P_SLATE, labelsize=10)
ax.legend(fontsize=9, framealpha=0.9, loc="upper center")
ax.grid(True, axis="y", alpha=0.3, zorder=0)

fig.suptitle(
    f"PIT Histogram: T8 MC Dropout (S=30, {n_nodes:,} nodes)",
    fontsize=13,
    fontweight="bold",
    color=P_DGRAY,
    y=0.99,
)

fig.text(0.5, -0.02, FOOTNOTE, ha="center", fontsize=8, color=P_MGRAY, style="italic")

fig.tight_layout(pad=1.5, rect=[0, 0.02, 1, 0.94])

# ── Save ────────────────────────────────────────────────────────────────
THESIS_FIG = os.path.join(REPO, "thesis", "latex_tum_official", "figures")
for ext in ("pdf", "png"):
    out = os.path.join(THESIS_FIG, f"t8_pit_histogram.{ext}")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("\nSaved: t8_pit_histogram.pdf + .png")
