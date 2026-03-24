"""
Phase 4: Generate S-convergence figure for thesis.
Two-panel figure:
  Left: Spearman rho vs S (aggregate + per-graph mean with shaded std band)
  Right: Mean sigma vs S (with S=30 marker)
"""

import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys, os

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "thesis",
        "latex_tum_official",
        "figures",
    ),
)
from thesis_style import *

RESULTS_PATH = r"C:\Users\zamin\OneDrive\Desktop\Nazim_thesis\ml_surrogates_for_agent_based_transport_models\docs\verified\phase3_results\s_convergence_results.json"

# Output paths
FIGURES_DIR = r"C:\Users\zamin\OneDrive\Desktop\Nazim_thesis\ml_surrogates_for_agent_based_transport_models\thesis\latex_tum_official\figures"
VERIFIED_DIR = r"C:\Users\zamin\OneDrive\Desktop\Nazim_thesis\ml_surrogates_for_agent_based_transport_models\docs\verified\figures"

with open(RESULTS_PATH) as f:
    data = json.load(f)

S_vals = [r["S"] for r in data["aggregate_convergence"]]
agg_rho = [r["spearman_rho"] for r in data["aggregate_convergence"]]
mean_sigma = [r["mean_sigma"] for r in data["aggregate_convergence"]]
mae_vals = [r["mae"] for r in data["aggregate_convergence"]]

# Per-graph mean rho and std
pg_mean = [data["per_graph_mean_rho"][str(s)]["mean"] for s in S_vals]
pg_std = [data["per_graph_mean_rho"][str(s)]["std"] for s in S_vals]
pg_mean = np.array(pg_mean)
pg_std = np.array(pg_std)

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

# --- Left panel: Spearman rho vs S ---
ax1.plot(
    S_vals,
    agg_rho,
    "o-",
    color=P_BLUE,
    linewidth=2,
    markersize=6,
    label="Aggregate $\\rho$",
)
ax1.plot(
    S_vals,
    pg_mean,
    "s--",
    color=P_AMBER,
    linewidth=1.5,
    markersize=5,
    label="Mean per-graph $\\rho$",
)
ax1.fill_between(S_vals, pg_mean - pg_std, pg_mean + pg_std, alpha=0.2, color=P_AMBER)

# Mark S=30
s30_idx = S_vals.index(30)
ax1.axvline(x=30, color="gray", linestyle=":", alpha=0.5)
ax1.scatter(
    [30],
    [agg_rho[s30_idx]],
    s=120,
    zorder=5,
    facecolors="none",
    edgecolors="red",
    linewidths=2,
)
ax1.annotate(
    f"$S=30$\n$\\rho={agg_rho[s30_idx]:.3f}$",
    xy=(30, agg_rho[s30_idx]),
    xytext=(36, agg_rho[s30_idx] - 0.02),
    fontsize=9,
    arrowprops=dict(arrowstyle="->", color="red", lw=1.2),
    color="red",
)

ax1.set_xlabel("Number of MC samples ($S$)", fontsize=13)
ax1.set_ylabel("Spearman $\\rho$ (uncertainty vs $|e|$)", fontsize=13)
ax1.set_title("(a) Uncertainty ranking quality", fontsize=13)
ax1.legend(fontsize=10, loc="lower right")
ax1.set_xlim(0, 55)
ax1.grid(True, alpha=0.3)

# --- Right panel: Mean sigma vs S ---
ax2.plot(S_vals, mean_sigma, "o-", color=P_GREEN, linewidth=2, markersize=6)

# Mark S=30
ax2.axvline(x=30, color="gray", linestyle=":", alpha=0.5)
ax2.scatter(
    [30],
    [mean_sigma[s30_idx]],
    s=120,
    zorder=5,
    facecolors="none",
    edgecolors="red",
    linewidths=2,
)
ax2.annotate(
    f"$S=30$\n$\\bar{{\\sigma}}={mean_sigma[s30_idx]:.2f}$",
    xy=(30, mean_sigma[s30_idx]),
    xytext=(36, mean_sigma[s30_idx] - 0.08),
    fontsize=9,
    arrowprops=dict(arrowstyle="->", color="red", lw=1.2),
    color="red",
)

ax2.set_xlabel("Number of MC samples ($S$)", fontsize=13)
ax2.set_ylabel("Mean uncertainty $\\bar{\\sigma}$ (veh/h)", fontsize=13)
ax2.set_title("(b) Uncertainty magnitude stabilisation", fontsize=13)
ax2.set_xlim(0, 55)
ax2.grid(True, alpha=0.3)

plt.tight_layout(pad=2.0)

# Save
for d, ext in [(FIGURES_DIR, "pdf"), (VERIFIED_DIR, "pdf"), (VERIFIED_DIR, "png")]:
    path = f"{d}/t8_s_convergence.{ext}"
    fig.savefig(path, dpi=300 if ext == "png" else None, bbox_inches="tight")
    print(f"Saved: {path}")

plt.close()
print("Done.")
