"""EDA 9: UQ Metrics Dashboard - Clean Academic Style"""

import matplotlib

matplotlib.use("Agg")
import json
import numpy as np
import matplotlib.pyplot as plt

UQ_PATH = r"C:\Users\zamin\Downloads\Nazim\Thesis\Nazim\ml_surrogates_thesis_final\code\data\TR-C_Benchmarks\point_net_transf_gat_8th_trial_lower_dropout\uq_results\uq_comparison_model8.json"
with open(UQ_PATH, "r") as f:
    uq = json.load(f)

plt.rcParams.update(
    {
        "font.size": 10,
        "font.family": "serif",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle(
    "Uncertainty Quantification Results (Trial 8)",
    fontsize=16,
    fontweight="bold",
    y=0.98,
)

# Extract coverage & width
levels = [50, 80, 90, 95]
mc_int = uq.get("mc_dropout", {}).get("prediction_intervals", {})
cf_int = uq.get("conformal", {}).get("prediction_intervals", {})

mc_cov = [mc_int.get(str(l), {}).get("picp", 0) * 100 for l in levels]
cf_cov = [cf_int.get(str(l), {}).get("picp", 0) * 100 for l in levels]
mc_wid = [mc_int.get(str(l), {}).get("mpiw", 0) for l in levels]
cf_wid = [cf_int.get(str(l), {}).get("mpiw", 0) for l in levels]

# Fix any values that might already be percentages
mc_cov = [v if v <= 100 else v for v in mc_cov]
cf_cov = [v if v <= 100 else v for v in cf_cov]

# Panel 1: Coverage comparison
ax = axes[0, 0]
xp = np.arange(len(levels))
w = 0.35
ax.bar(xp - w / 2, mc_cov, w, label="MC Dropout", color="#4878A8", alpha=0.8)
ax.bar(xp + w / 2, cf_cov, w, label="Conformal", color="#5DA573", alpha=0.8)
for i, l in enumerate(levels):
    ax.plot([i - 0.5, i + 0.5], [l, l], "k--", alpha=0.3, linewidth=1)
ax.set_xticks(xp)
ax.set_xticklabels([f"{l}%" for l in levels])
ax.set_ylabel("Actual Coverage (%)")
ax.set_title("Coverage Comparison", fontweight="bold")
ax.legend(fontsize=9)

# Panel 2: Interval width
ax = axes[0, 1]
ax.bar(xp - w / 2, mc_wid, w, label="MC Dropout", color="#4878A8", alpha=0.8)
ax.bar(xp + w / 2, cf_wid, w, label="Conformal", color="#5DA573", alpha=0.8)
for bars, vals in [(xp - w / 2, mc_wid), (xp + w / 2, cf_wid)]:
    for b, v in zip(bars, vals):
        if v > 0:
            ax.text(b, v + 0.3, f"{v:.1f}", ha="center", fontsize=7)
ax.set_xticks(xp)
ax.set_xticklabels([f"{l}%" for l in levels])
ax.set_ylabel("Mean Interval Width (veh/hr)")
ax.set_title("Interval Width Comparison", fontweight="bold")
ax.legend(fontsize=9)

# Panel 3: Spearman rho comparison
ax = axes[0, 2]
methods = ["MC\nDropout", "Ensemble\nVariance", "Combined", "Multi-\nModel"]
rhos = [0.4820, 0.4370, 0.4909, 0.4333]
bars = ax.bar(
    range(4),
    rhos,
    color=["#4878A8", "#D66B6B", "#5DA573", "#D4A843"],
    alpha=0.8,
    width=0.5,
)
for bar, v in zip(bars, rhos):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        v + 0.003,
        f"{v:.4f}",
        ha="center",
        fontsize=9,
        fontweight="bold",
    )
ax.set_xticks(range(4))
ax.set_xticklabels(methods, fontsize=9)
ax.set_ylabel("Spearman $\\rho$")
ax.set_title("UQ Quality Ranking", fontweight="bold")
ax.set_ylim(0.4, 0.52)

# Panel 4: Experiment A
ax = axes[1, 0]
ea_m = ["MC (S=30)", "Ens. Var.", "Combined"]
ea_r = [0.4908, 0.4370, 0.4909]
bars = ax.bar(
    range(3), ea_r, color=["#4878A8", "#D66B6B", "#5DA573"], alpha=0.8, width=0.5
)
for bar, v in zip(bars, ea_r):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        v + 0.003,
        f"{v:.4f}",
        ha="center",
        fontsize=9,
    )
ax.set_xticks(range(3))
ax.set_xticklabels(ea_m, fontsize=9)
ax.set_ylabel("Spearman $\\rho$")
ax.set_title("Exp. A: Same Model Ensemble", fontweight="bold")
ax.set_ylim(0.42, 0.52)

# Panel 5: Experiment B
ax = axes[1, 1]
eb_m = ["T2", "T5", "T6", "T7", "T8", "Ensemble"]
eb_r2 = [0.5117, 0.4200, 0.5050, 0.5600, 0.5957, 0.5656]
colors = ["#4878A8"] * 5 + ["#5DA573"]
bars = ax.bar(range(6), eb_r2, color=colors, alpha=0.8, width=0.6)
for bar, v in zip(bars, eb_r2):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        v + 0.005,
        f"{v:.3f}",
        ha="center",
        fontsize=8,
    )
ax.set_xticks(range(6))
ax.set_xticklabels(eb_m)
ax.set_ylabel("R$^2$")
ax.set_title("Exp. B: Multi-Model Ensemble", fontweight="bold")

# Panel 6: Key metrics summary
ax = axes[1, 2]
ax.axis("off")
summary = (
    "Key UQ Metrics (Trial 8)\n"
    "=" * 35 + "\n\n"
    "MC Dropout: $\\rho$=0.4820, k95=11.34\n"
    "ECE: 0.265 -> 0.048 (T=2.70)\n\n"
    "Conformal (90%): 90.02% at +/-9.92\n"
    "Conformal (95%): 95.01% at +/-14.68\n\n"
    "Selective Pred.: 41.2% MAE reduction\n"
    "  at 50% retention (3.95 -> 2.32)\n\n"
    "AUROC: 0.76 (top-10% errors)\n"
    "CRPS/MAE: 0.857"
)
ax.text(
    0.05,
    0.95,
    summary,
    transform=ax.transAxes,
    fontsize=10,
    va="top",
    family="monospace",
    bbox=dict(boxstyle="round", facecolor="#f0f0f0", edgecolor="gray"),
)

fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(
    r"C:\Users\zamin\Downloads\Nazim\eda_9_uq_dashboard.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="white",
)
plt.close()
print("Saved: eda_9_uq_dashboard.png")
