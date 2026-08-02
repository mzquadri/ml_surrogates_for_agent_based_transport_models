"""Generate F1, F2, F3: trial comparison + UQ ranking."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from plot_style import set_style, save_both, TUM_BLUE, TUM_ORANGE, TUM_LIGHTBLUE, TUM_GRAY, PALETTE

set_style()
OUT = "../../../document/figures/new"

# ============================================================
# F1: Trial bars R2 / MAE / RMSE for T1-T8 (3 sub-panels)
# ============================================================
trials = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"]
r2 = [0.7860, 0.5117, 0.2246, 0.2426, 0.5553, 0.5223, 0.5471, 0.5957]
mae = [2.972, 4.328, 5.990, 6.080, 4.242, 4.324, 4.060, 3.957]
rmse = [5.396, 8.151, 10.270, 10.151, 7.778, 8.061, 7.534, 7.118]

# T1 has different architecture (Linear head, no dropout) and different split
colors = [TUM_GRAY if t == "T1" else (TUM_ORANGE if t == "T8" else TUM_BLUE) for t in trials]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
for ax, vals, ylabel, title in [
    (axes[0], r2, "$R^2$", "Coefficient of determination"),
    (axes[1], mae, "MAE (veh/h)", "Mean absolute error"),
    (axes[2], rmse, "RMSE (veh/h)", "Root mean square error"),
]:
    bars = ax.bar(trials, vals, color=colors, edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + (max(vals)*0.02), f"{v:.3f}" if ylabel == "$R^2$" else f"{v:.2f}",
                ha="center", va="bottom", fontsize=8.5)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, max(vals) * 1.15)

# Legend
import matplotlib.patches as mpatches
leg_handles = [
    mpatches.Patch(color=TUM_GRAY, label="T1 (Linear head, 80/15/5)"),
    mpatches.Patch(color=TUM_BLUE, label="T2–T7 (GATConv head)"),
    mpatches.Patch(color=TUM_ORANGE, label="T8 (primary model)"),
]
fig.legend(handles=leg_handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03), frameon=False)
fig.suptitle("Trial progression across point-prediction metrics (T1–T8)", y=1.02)
fig.tight_layout()
save_both(fig, OUT, "fig01_trial_bars_r2_mae_rmse")
print("F1 saved")

# ============================================================
# F2: Single-panel R2 bar with architectural annotations
# ============================================================
fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(trials, r2, color=colors, edgecolor="black", linewidth=0.5)
ax.axhline(y=0.55, color=TUM_GRAY, linestyle=":", linewidth=1, alpha=0.6)
ax.text(7.4, 0.56, "$R^2 = 0.55$ (T9 gate)", fontsize=8, color=TUM_GRAY, va="bottom", ha="right")

annotations = {
    "T1": "Linear head,\ndropout=0",
    "T2": "GATConv head,\ndropout=0.3",
    "T3": "+ weighted loss,\ndropout=0",
    "T4": "+ weighted loss,\ndropout=0",
    "T5": "batch=8,\ndropout=0.3",
    "T6": "lr=3e-4,\ndropout=0.3",
    "T7": "80/10/10 split,\nlr=6e-4",
    "T8": "dropout=0.2\n(primary)",
}
for i, t in enumerate(trials):
    ax.text(i, r2[i] + 0.02, f"{r2[i]:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.text(i, -0.08, annotations[t], ha="center", va="top", fontsize=8, color="#555555")

ax.set_ylabel("Test $R^2$")
ax.set_title("Training trial progression with architectural changes")
ax.set_ylim(-0.15, 0.9)
ax.set_xlim(-0.5, 7.5)
ax.spines["bottom"].set_position(("data", 0))
fig.tight_layout()
save_both(fig, OUT, "fig02_trial_progression_annotated")
print("F2 saved")

# ============================================================
# F3: UQ method ranking (horizontal Spearman rho bar chart)
# ============================================================
methods = [
    "T8 MC Dropout (primary)",
    "MC avg, 5 seeds (Exp A)",
    "Combined MC + seed var",
    "T9 heteroscedastic $\\sigma_{tot}$",
    "T7 MC Dropout",
    "Seed ensemble variance",
    "Multi-model ensemble (Exp B)",
    "Deep Ensemble (5 models)",
]
rhos = [0.4820, 0.4908, 0.4909, 0.4797, 0.4437, 0.4370, 0.4333, 0.3997]
# Sort descending
order = np.argsort(rhos)[::-1]
methods_s = [methods[i] for i in order]
rhos_s = [rhos[i] for i in order]

# Highlight primary T8 MC Dropout
bar_colors = [TUM_ORANGE if "T8 MC Dropout (primary)" in m else TUM_BLUE for m in methods_s]

fig, ax = plt.subplots(figsize=(9.5, 5))
y_pos = np.arange(len(methods_s))
bars = ax.barh(y_pos, rhos_s, color=bar_colors, edgecolor="black", linewidth=0.5)
for b, v in zip(bars, rhos_s):
    ax.text(v + 0.005, b.get_y() + b.get_height()/2, f"{v:.4f}", va="center", fontsize=9)

ax.set_yticks(y_pos)
ax.set_yticklabels(methods_s)
ax.invert_yaxis()
ax.set_xlabel(r"Spearman $\rho$ between $\sigma$ and $|y - \hat{y}|$")
ax.set_title("UQ method ranking by uncertainty-error correlation (T8 baseline)")
ax.set_xlim(0, 0.55)
ax.axvline(x=0, color="black", linewidth=0.5)
fig.tight_layout()
save_both(fig, OUT, "fig03_uq_ranking_spearman")
print("F3 saved")
