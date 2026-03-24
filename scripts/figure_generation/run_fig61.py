"""
Regenerate Fig 6.1: t8_s_convergence
Two-panel S-convergence figure justifying S=30 choice.
  (a) Spearman rho vs S (aggregate + per-graph mean with std band)
  (b) Mean sigma vs S
Data from docs/verified/phase3_results/s_convergence_results.json.
"""

import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from thesis_style import *

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
JSON_PATH = os.path.join(
    REPO, "docs", "verified", "phase3_results", "s_convergence_results.json"
)

FOOTNOTE = "Trial 8, 10 test graphs (316,350 nodes), weight-remapped model."

# ── Load data ───────────────────────────────────────────────────────────
with open(JSON_PATH) as f:
    data = json.load(f)

S_vals = [r["S"] for r in data["aggregate_convergence"]]
agg_rho = [r["spearman_rho"] for r in data["aggregate_convergence"]]
mean_sigma = [r["mean_sigma"] for r in data["aggregate_convergence"]]
mae_vals = [r["mae"] for r in data["aggregate_convergence"]]

pg_mean = np.array([data["per_graph_mean_rho"][str(s)]["mean"] for s in S_vals])
pg_std = np.array([data["per_graph_mean_rho"][str(s)]["std"] for s in S_vals])

s30_idx = S_vals.index(30)

# ── Cross-checks ────────────────────────────────────────────────────────
print("-- Cross-checks vs JSON --")
checks = [
    ("S=30 agg rho", agg_rho[s30_idx], 0.4584, 0.001),
    ("S=30 pg mean rho", pg_mean[s30_idx], 0.4449, 0.001),
    ("S=30 pg std rho", pg_std[s30_idx], 0.0268, 0.001),
    ("S=30 mean sigma", mean_sigma[s30_idx], 1.986, 0.01),
    ("S=30 MAE", mae_vals[s30_idx], 4.019, 0.01),
    ("S=5 agg rho", agg_rho[0], 0.4101, 0.001),
    ("S=50 agg rho", agg_rho[-1], 0.4632, 0.001),
    ("S=50 mean sigma", mean_sigma[-1], 2.015, 0.01),
    ("n_S_values", len(S_vals), 10, 0),
]
passed = 0
for name, val, expected, tol in checks:
    ok = abs(val - expected) <= tol
    passed += ok
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] {name}: {val:.4f} (expected {expected})")
print(f"\nCross-checks: {passed}/{len(checks)} PASSED")
assert passed == len(checks), "SOME CROSS-CHECKS FAILED"

# ── Figure ──────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
fig.patch.set_facecolor(BG)

# ── Panel (a): Spearman rho vs S ───────────────────────────────────────
panel_label(ax1, "(a)")

# Aggregate rho line
ax1.plot(
    S_vals,
    agg_rho,
    "o-",
    color=P_BLUE,
    linewidth=2,
    markersize=6,
    label="Aggregate $\\rho$",
    zorder=3,
)

# Per-graph mean rho with std band
ax1.plot(
    S_vals,
    pg_mean,
    "s--",
    color=P_AMBER,
    linewidth=1.5,
    markersize=5,
    label="Mean per-graph $\\rho$",
    zorder=3,
)
ax1.fill_between(
    S_vals, pg_mean - pg_std, pg_mean + pg_std, alpha=0.2, color=P_AMBER, zorder=2
)

# S=50 reference line (asymptote)
ax1.axhline(
    agg_rho[-1],
    color=P_LGRAY,
    linestyle="--",
    linewidth=1.2,
    alpha=0.8,
    zorder=1,
    label=f"$S$=50: $\\rho$={agg_rho[-1]:.4f}",
)

# S=30 marker
ax1.axvline(30, color=P_CORAL, linestyle=":", linewidth=1.2, alpha=0.7, zorder=2)
ax1.scatter(
    [30],
    [agg_rho[s30_idx]],
    s=120,
    zorder=5,
    facecolors="none",
    edgecolors=P_CORAL,
    linewidths=2,
)
ax1.annotate(
    f"$S$=30\n$\\rho$={agg_rho[s30_idx]:.3f}",
    xy=(30, agg_rho[s30_idx]),
    xytext=(37, agg_rho[s30_idx] - 0.015),
    fontsize=9,
    color=P_CORAL,
    arrowprops=dict(arrowstyle="->", color=P_CORAL, lw=1.2),
)

ax1.set_xlabel("Number of MC samples ($S$)", color=P_DGRAY)
ax1.set_ylabel("Spearman $\\rho$ (uncertainty vs $|e|$)", color=P_DGRAY)
ax1.set_title(
    "Uncertainty ranking quality", color=P_DGRAY, fontweight="bold", fontsize=12
)
ax1.legend(fontsize=9, loc="lower right", framealpha=0.9)
ax1.set_xlim(0, 55)
ax1.grid(True, alpha=0.3, zorder=0)
ax1.tick_params(colors=P_SLATE, labelsize=10)

# ── Panel (b): Mean sigma vs S ─────────────────────────────────────────
panel_label(ax2, "(b)")

ax2.plot(S_vals, mean_sigma, "o-", color=P_GREEN, linewidth=2, markersize=6, zorder=3)

# S=50 reference line (asymptote)
ax2.axhline(
    mean_sigma[-1],
    color=P_LGRAY,
    linestyle="--",
    linewidth=1.2,
    alpha=0.8,
    zorder=1,
    label=f"$S$=50: $\\bar{{\\sigma}}$={mean_sigma[-1]:.2f}",
)

# S=30 marker
ax2.axvline(30, color=P_CORAL, linestyle=":", linewidth=1.2, alpha=0.7, zorder=2)
ax2.scatter(
    [30],
    [mean_sigma[s30_idx]],
    s=120,
    zorder=5,
    facecolors="none",
    edgecolors=P_CORAL,
    linewidths=2,
)
ax2.annotate(
    f"$S$=30\n$\\bar{{\\sigma}}$={mean_sigma[s30_idx]:.2f}",
    xy=(30, mean_sigma[s30_idx]),
    xytext=(37, mean_sigma[s30_idx] - 0.07),
    fontsize=9,
    color=P_CORAL,
    arrowprops=dict(arrowstyle="->", color=P_CORAL, lw=1.2),
)

ax2.set_xlabel("Number of MC samples ($S$)", color=P_DGRAY)
ax2.set_ylabel("Mean uncertainty $\\bar{\\sigma}$ (veh/h)", color=P_DGRAY)
ax2.set_title(
    "Uncertainty magnitude stabilisation", color=P_DGRAY, fontweight="bold", fontsize=12
)
ax2.legend(fontsize=9, loc="lower right", framealpha=0.9)
ax2.set_xlim(0, 55)
ax2.grid(True, alpha=0.3, zorder=0)
ax2.tick_params(colors=P_SLATE, labelsize=10)

# ── Suptitle and footnote ──────────────────────────────────────────────
fig.suptitle(
    "$S$-Convergence of MC Dropout Uncertainty Quality (T8)",
    fontsize=13,
    fontweight="bold",
    color=P_DGRAY,
    y=0.99,
)
fig.text(0.5, -0.01, FOOTNOTE, ha="center", fontsize=8, color=P_MGRAY, style="italic")

fig.tight_layout(pad=1.5, rect=[0, 0.02, 1, 0.94])

# ── Save ────────────────────────────────────────────────────────────────
THESIS_FIG = os.path.join(REPO, "thesis", "latex_tum_official", "figures")
save_fig(fig, "t8_s_convergence", out_dir=THESIS_FIG)
plt.close(fig)
print("\nSaved: t8_s_convergence.pdf + .png")
