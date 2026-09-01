"""
Regenerate Fig 5.12: t8_temperature_scaling
Four panels: (a) Before temp scaling (b) After temp scaling
             (c) Before vs After overlay (d) Coverage gap reduction
Values from verified JSON: temperature_scaling_t8.json
"""

import os, sys, json
import numpy as np

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "thesis",
        "latex_tum_official",
        "figures",
    ),
)
from thesis_style import *

REPO = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(
    REPO, "docs", "verified", "phase3_results", "temperature_scaling_t8.json"
)

FOOTNOTE = (
    "Results based on 1,000 of 10,000 available MATSim scenarios "
    "(10% subset), Trial 8 test set."
)

# ── Load verified JSON ──────────────────────────────────────────────────
with open(JSON_PATH, "r") as f:
    data = json.load(f)

nominal = np.array(data["nominal_levels"])
T_opt = data["optimal_temperature_T"]
ev = data["evaluation_set"]
cov_before = np.array(ev["coverage_before"])
cov_after = np.array(ev["coverage_after"])
ece_before = ev["ece_before"]
ece_after = ev["ece_after"]
ece_improve = ev["ece_improvement_pct"]

gaps_before = (cov_before - nominal) * 100
gaps_after = (cov_after - nominal) * 100

print("=== Verified values ===")
print(f"T = {T_opt:.4f}")
print(f"ECE before = {ece_before:.3f}")
print(f"ECE after  = {ece_after:.3f}")
print(f"Improvement = {ece_improve:.1f}%")
for i, nom in enumerate(nominal):
    print(
        f"  {int(nom * 100):2d}%: before={cov_before[i]:.4f}  after={cov_after[i]:.4f}  "
        f"gap_before={gaps_before[i]:.1f}pp  gap_after={gaps_after[i]:.1f}pp"
    )

# ── Cross-checks ────────────────────────────────────────────────────────
assert abs(T_opt - 2.70) < 0.02, f"FAIL: T={T_opt}"
assert abs(ece_before - 0.269) < 0.002, f"FAIL: ECE_before={ece_before}"
assert abs(ece_after - 0.048) < 0.002, f"FAIL: ECE_after={ece_after}"
print("\n[OK] Cross-checks passed.")

# ── Figure: 2x2 ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.patch.set_facecolor(BG)


# Helper: draw a reliability diagram
def draw_reliability(ax, nom, obs, color, label, ece_val, shade_color):
    ax.plot(
        [0, 1],
        [0, 1],
        "--",
        color=P_LGRAY,
        lw=1.5,
        label="Perfect calibration",
        zorder=2,
    )
    ax.plot(nom, obs, "-o", color=color, lw=2.2, ms=6, label=label, zorder=4)
    ax.fill_between(nom, obs, nom, alpha=0.15, color=shade_color, zorder=1)
    ax.text(
        0.55,
        0.15,
        f"ECE = {ece_val:.3f}",
        fontsize=11,
        fontweight="bold",
        color=color,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor=WHITE,
            edgecolor=color,
            alpha=0.9,
            lw=1.2,
        ),
        ha="center",
        zorder=5,
    )
    ax.set_xlabel("Nominal coverage level", color=P_DGRAY)
    ax.set_ylabel("Observed coverage", color=P_DGRAY)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(colors=P_SLATE, labelsize=9)
    ax.legend(fontsize=8.5, framealpha=0.9, loc="upper left")
    ax.grid(True, alpha=0.3, zorder=0)


# ── Panel (a): Before ───────────────────────────────────────────────────
ax = axes[0, 0]
panel_label(ax, "(a)", x=-0.10, y=1.06)
ax.set_title("Before Temperature Scaling", fontsize=12, color=P_DGRAY, pad=8)
draw_reliability(
    ax, nominal, cov_before, P_CORAL, "Before (T=1.0)", ece_before, P_CORAL
)

# ── Panel (b): After ────────────────────────────────────────────────────
ax = axes[0, 1]
panel_label(ax, "(b)", x=-0.10, y=1.06)
ax.set_title(
    f"After Temperature Scaling (T={T_opt:.2f})", fontsize=12, color=P_DGRAY, pad=8
)
draw_reliability(
    ax, nominal, cov_after, P_GREEN, f"After (T={T_opt:.2f})", ece_after, P_GREEN
)

# ── Panel (c): Overlay ──────────────────────────────────────────────────
ax = axes[1, 0]
panel_label(ax, "(c)", x=-0.10, y=1.06)
ax.set_title("Before vs After Comparison", fontsize=12, color=P_DGRAY, pad=8)

ax.plot([0, 1], [0, 1], "--", color=P_LGRAY, lw=1.5, label="Perfect", zorder=2)
ax.plot(
    nominal,
    cov_before,
    "-o",
    color=P_CORAL,
    lw=2.0,
    ms=5,
    label=f"Before (ECE={ece_before:.3f})",
    zorder=3,
    alpha=0.7,
)
ax.plot(
    nominal,
    cov_after,
    "-s",
    color=P_GREEN,
    lw=2.2,
    ms=6,
    label=f"After (ECE={ece_after:.3f})",
    zorder=4,
)
ax.set_xlabel("Nominal coverage level", color=P_DGRAY)
ax.set_ylabel("Observed coverage", color=P_DGRAY)
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.0)
ax.tick_params(colors=P_SLATE, labelsize=9)
ax.legend(fontsize=8.5, framealpha=0.9, loc="upper left")
ax.grid(True, alpha=0.3, zorder=0)

# ── Panel (d): Coverage Gap Reduction ───────────────────────────────────
ax = axes[1, 1]
panel_label(ax, "(d)", x=-0.10, y=1.06)
ax.set_title("Coverage Gap Reduction", fontsize=12, color=P_DGRAY, pad=8)

x_pos = np.arange(len(nominal))
x_labels = [f"{int(n * 100)}%" for n in nominal]
bar_w = 0.35

b1 = ax.bar(
    x_pos - bar_w / 2,
    gaps_before,
    bar_w,
    label="Before",
    color=P_CORAL,
    alpha=0.80,
    zorder=3,
)
b2 = ax.bar(
    x_pos + bar_w / 2,
    gaps_after,
    bar_w,
    label="After",
    color=P_GREEN,
    alpha=0.80,
    zorder=3,
)

ax.axhline(0, color=P_SLATE, lw=0.8, zorder=2)

ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels, rotation=45, ha="right")
ax.set_xlabel("Nominal coverage level", color=P_DGRAY)
ax.set_ylabel("Coverage gap (pp)", color=P_DGRAY)
ax.tick_params(colors=P_SLATE, labelsize=9)
ax.legend(fontsize=9, framealpha=0.9, loc="lower left")
ax.grid(True, axis="y", alpha=0.3, zorder=0)
ax.set_ylim(min(gaps_before) * 1.15, max(gaps_after) * 1.5 + 3)

# ── Suptitle ────────────────────────────────────────────────────────────
fig.suptitle(
    f"T8 Temperature Scaling Calibration (T={T_opt:.2f}, 20/80 graph split, S=30)",
    fontsize=14,
    fontweight="bold",
    color=P_DGRAY,
    y=0.99,
)

fig.text(0.5, -0.01, FOOTNOTE, ha="center", fontsize=8, color=P_MGRAY, style="italic")

fig.tight_layout(pad=2.0, rect=[0, 0.02, 1, 0.95])

# ── Save ────────────────────────────────────────────────────────────────
THESIS_FIG = os.path.join(REPO, "thesis", "latex_tum_official", "figures")
for ext in ("pdf", "png"):
    out = os.path.join(THESIS_FIG, f"t8_temperature_scaling.{ext}")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("\nSaved: t8_temperature_scaling.pdf + .png")
