"""
Regenerate Fig 5.18: t7_vs_t8_uq_comparison (v3 — fixed)
2x2 panel comparing T7 and T8 UQ quality metrics.
Fixes: uniform bar widths, amber reference lines, no label overlap.
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from thesis_style import *

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

FOOTNOTE = (
    "Results based on 1,000 of 10,000 available MATSim scenarios "
    "(10% subset), 100 test graphs each, S=30."
)

# ── Verified values ─────────────────────────────────────────────────────
T7 = {
    "rho": 0.4460,
    "mae": 4.07,
    "auroc_10": 0.7416,
    "auroc_20": 0.7151,
    "sel_50_mae": 2.51,
    "sel_90_mae": 3.32,
    "k90": 10.45,
    "k95": 16.15,
    "raw_cov_95": 0.484,
}
T8 = {
    "rho": 0.4820,
    "mae": 3.95,
    "auroc_10": 0.7585,
    "auroc_20": 0.7401,
    "sel_50_mae": 2.32,
    "sel_90_mae": 3.23,
    "k90": 7.56,
    "k95": 11.34,
    "raw_cov_95": 0.556,
}
z95 = 1.96

# ── Cross-checks ────────────────────────────────────────────────────────
print("-- Cross-checks --")
checks = [
    ("T7 rho", T7["rho"], 0.4460, 0.001),
    ("T7 k90", T7["k90"], 10.45, 0.01),
    ("T7 k95", T7["k95"], 16.15, 0.01),
    ("T7 MAE", T7["mae"], 4.07, 0.01),
    ("T7 AUROC top-10%", T7["auroc_10"], 0.7416, 0.001),
    ("T7 AUROC top-20%", T7["auroc_20"], 0.7151, 0.001),
    ("T7 sel 50% MAE", T7["sel_50_mae"], 2.51, 0.01),
    ("T7 sel 90% MAE", T7["sel_90_mae"], 3.32, 0.01),
    ("T7 raw 95% cov", T7["raw_cov_95"], 0.484, 0.002),
    ("T8 rho", T8["rho"], 0.4820, 0.001),
    ("T8 k90", T8["k90"], 7.56, 0.01),
    ("T8 k95", T8["k95"], 11.34, 0.01),
    ("T8 MAE", T8["mae"], 3.95, 0.01),
    ("T8 AUROC top-10%", T8["auroc_10"], 0.7585, 0.001),
    ("T8 AUROC top-20%", T8["auroc_20"], 0.7401, 0.001),
    ("T8 sel 50% MAE", T8["sel_50_mae"], 2.32, 0.01),
    ("T8 sel 90% MAE", T8["sel_90_mae"], 3.23, 0.01),
    ("T8 raw 95% cov", T8["raw_cov_95"], 0.556, 0.002),
]
passed = 0
for name, val, expected, tol in checks:
    ok = abs(val - expected) < tol
    passed += ok
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] {name}: {val} (expected {expected})")
print(f"\nCross-checks: {passed}/{len(checks)} PASSED")
assert passed == len(checks), "SOME CROSS-CHECKS FAILED"

# ── Figure ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.patch.set_facecolor(BG)

COL_T7 = P_PURPLE
COL_T8 = P_BLUE
COL_REF = P_AMBER  # amber for reference lines — distinct from bar colors


def plot_grouped_bars(
    ax,
    labels,
    t7_vals,
    t8_vals,
    fmt=".3f",
    ylabel="",
    title="",
    panel="(a)",
    legend_loc="upper right",
    ylim=None,
    ylim_bottom=0,
):
    """Plot a consistently-styled grouped bar panel."""
    n_groups = len(labels)
    # Bar width scales with number of groups for visual consistency
    bar_w = 0.28 if n_groups >= 3 else 0.24
    x = np.arange(n_groups)

    ax.bar(
        x - bar_w / 2, t7_vals, bar_w, color=COL_T7, alpha=0.85, label="T7", zorder=3
    )
    ax.bar(
        x + bar_w / 2, t8_vals, bar_w, color=COL_T8, alpha=0.85, label="T8", zorder=3
    )

    # Value labels
    y_range = (ylim if ylim else max(max(t7_vals), max(t8_vals)) * 1.15) - ylim_bottom
    offset = y_range * 0.015
    for i, (v7, v8) in enumerate(zip(t7_vals, t8_vals)):
        ax.text(
            x[i] - bar_w / 2,
            v7 + offset,
            f"{v7:{fmt}}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=P_DGRAY,
        )
        ax.text(
            x[i] + bar_w / 2,
            v8 + offset,
            f"{v8:{fmt}}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=P_DGRAY,
        )

    # Consistent x-axis padding so bars look uniform across panels
    pad = 0.6
    ax.set_xlim(-pad, n_groups - 1 + pad)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel, color=P_DGRAY)
    ax.set_title(title, color=P_DGRAY, fontweight="bold", fontsize=12)
    if ylim:
        ax.set_ylim(ylim_bottom, ylim)
    else:
        ax.set_ylim(ylim_bottom, max(max(t7_vals), max(t8_vals)) * 1.15)
    ax.legend(fontsize=8.5, framealpha=0.9, loc=legend_loc)
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    ax.tick_params(colors=P_SLATE, labelsize=10)
    panel_label(ax, panel)

    return bar_w


# ── Panel (a): Error Detection AUROC ── (2 groups) ─────────────────────
plot_grouped_bars(
    axes[0, 0],
    labels=["AUROC\ntop-10%", "AUROC\ntop-20%"],
    t7_vals=[T7["auroc_10"], T7["auroc_20"]],
    t8_vals=[T8["auroc_10"], T8["auroc_20"]],
    fmt=".3f",
    ylabel="AUROC",
    title="Error Detection",
    panel="(a)",
    legend_loc="upper left",
    ylim=0.84,
    ylim_bottom=0.45,
)
# Random baseline
axes[0, 0].axhline(
    0.5, color=P_MGRAY, linestyle=":", linewidth=0.8, alpha=0.5, label="Random"
)

# ── Panel (b): Correlation & Raw Coverage ── (2 groups, 0-1) ──────────
ax_b = axes[0, 1]
n_b = 2
bar_w_b = 0.24
x_b = np.arange(n_b)
panel_label(ax_b, "(b)")

ax_b.bar(
    x_b - bar_w_b / 2,
    [T7["rho"], T7["raw_cov_95"]],
    bar_w_b,
    color=COL_T7,
    alpha=0.85,
    label="T7",
    zorder=3,
)
ax_b.bar(
    x_b + bar_w_b / 2,
    [T8["rho"], T8["raw_cov_95"]],
    bar_w_b,
    color=COL_T8,
    alpha=0.85,
    label="T8",
    zorder=3,
)

# Custom formatted labels
for i, (v7, v8) in enumerate(
    zip([T7["rho"], T7["raw_cov_95"]], [T8["rho"], T8["raw_cov_95"]])
):
    l7 = f"{v7:.4f}" if i == 0 else f"{v7:.1%}"
    l8 = f"{v8:.4f}" if i == 0 else f"{v8:.1%}"
    ax_b.text(
        x_b[i] - bar_w_b / 2,
        v7 + 0.018,
        l7,
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=P_DGRAY,
    )
    ax_b.text(
        x_b[i] + bar_w_b / 2,
        v8 + 0.018,
        l8,
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=P_DGRAY,
    )

# Nominal 95% reference — amber, clearly visible
ax_b.axhline(0.95, color=COL_REF, linestyle="--", linewidth=1.5, alpha=0.9, zorder=4)
ax_b.text(
    1.10,
    0.96,
    "Nominal 95%",
    fontsize=8,
    color=COL_REF,
    va="bottom",
    ha="center",
    fontweight="bold",
)

ax_b.set_xlim(-0.6, 1.6)
ax_b.set_xticks(x_b)
ax_b.set_xticklabels(["Spearman $\\rho$", "Raw 95%\ncoverage"])
ax_b.set_ylabel("Value", color=P_DGRAY)
ax_b.set_title(
    "Correlation & Raw Coverage", color=P_DGRAY, fontweight="bold", fontsize=12
)
ax_b.set_ylim(0, 1.10)
ax_b.legend(fontsize=8.5, framealpha=0.9, loc="upper left")
ax_b.grid(True, axis="y", alpha=0.3, zorder=0)
ax_b.tick_params(colors=P_SLATE, labelsize=10)

# ── Panel (c): Selective Prediction MAE ── (3 groups) ──────────────────
plot_grouped_bars(
    axes[1, 0],
    labels=["50% retain", "90% retain", "Full (100%)"],
    t7_vals=[T7["sel_50_mae"], T7["sel_90_mae"], T7["mae"]],
    t8_vals=[T8["sel_50_mae"], T8["sel_90_mae"], T8["mae"]],
    fmt=".2f",
    ylabel="MAE (veh/h)",
    title="Selective Prediction",
    panel="(c)",
    legend_loc="upper left",
)

# ── Panel (d): Calibration k-multiplier ── (2 groups) ──────────────────
ax_d = axes[1, 1]
bar_w_d = 0.24
x_d = np.arange(2)
panel_label(ax_d, "(d)")

ax_d.bar(
    x_d - bar_w_d / 2,
    [T7["k90"], T7["k95"]],
    bar_w_d,
    color=COL_T7,
    alpha=0.85,
    label="T7",
    zorder=3,
)
ax_d.bar(
    x_d + bar_w_d / 2,
    [T8["k90"], T8["k95"]],
    bar_w_d,
    color=COL_T8,
    alpha=0.85,
    label="T8",
    zorder=3,
)

# Value labels
for i, (v7, v8) in enumerate(zip([T7["k90"], T7["k95"]], [T8["k90"], T8["k95"]])):
    ax_d.text(
        x_d[i] - bar_w_d / 2,
        v7 + 0.35,
        f"{v7:.2f}",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=P_DGRAY,
    )
    ax_d.text(
        x_d[i] + bar_w_d / 2,
        v8 + 0.35,
        f"{v8:.2f}",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=P_DGRAY,
    )

# Single z95 reference line — amber, clean, no overlap issue
ax_d.axhline(z95, color=COL_REF, linestyle="--", linewidth=1.5, alpha=0.9, zorder=4)
ax_d.text(
    1.42,
    z95 + 0.4,
    f"$z_{{95}}$={z95}",
    fontsize=8,
    color=COL_REF,
    va="bottom",
    ha="center",
    fontweight="bold",
)

ax_d.set_xlim(-0.6, 1.6)
ax_d.set_xticks(x_d)
ax_d.set_xticklabels(["$k_{90}$", "$k_{95}$"])
ax_d.set_ylabel("k-multiplier", color=P_DGRAY)
ax_d.set_title("Calibration Gap", color=P_DGRAY, fontweight="bold", fontsize=12)
ax_d.set_ylim(0, T7["k95"] * 1.2)
ax_d.legend(fontsize=8.5, framealpha=0.9, loc="upper left")
ax_d.grid(True, axis="y", alpha=0.3, zorder=0)
ax_d.tick_params(colors=P_SLATE, labelsize=10)

# ── Suptitle and footnote ──────────────────────────────────────────────
fig.suptitle(
    "T7 vs T8 Uncertainty Quality Comparison",
    fontsize=13,
    fontweight="bold",
    color=P_DGRAY,
    y=0.99,
)
fig.text(0.5, -0.01, FOOTNOTE, ha="center", fontsize=8, color=P_MGRAY, style="italic")

fig.tight_layout(pad=1.5, rect=[0, 0.02, 1, 0.94])

# ── Save ────────────────────────────────────────────────────────────────
THESIS_FIG = os.path.join(REPO, "thesis", "latex_tum_official", "figures")
save_fig(fig, "t7_vs_t8_uq_comparison", out_dir=THESIS_FIG)
plt.close(fig)
print("\nSaved: t7_vs_t8_uq_comparison.pdf + .png")
