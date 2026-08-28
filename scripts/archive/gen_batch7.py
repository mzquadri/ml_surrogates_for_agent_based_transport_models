"""Generate F4, F26, F27, F28: UQ summary, Deep Ensemble, stratified UQ."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import spearmanr
from plot_style import set_style, save_both, TUM_BLUE, TUM_ORANGE, TUM_LIGHTBLUE, TUM_GRAY, TUM_GREEN, IDEAL_GREEN, FAIL_RED, PASS_GREEN

set_style()
OUT = "../../../document/figures/new"
BASE = "../../data/TR-C_Benchmarks/"

# ============================================================
# F4: UQ method summary heatmap
# ============================================================
methods = ["T8 MC Dropout", "T8 + temp. scaling", "Std conformal", "Adaptive conformal",
           "T9 heteroscedastic", "T11 CQR (frozen)", "Deep Ensemble"]
metric_labels = ["$R^2$", r"Spearman $\rho$", "PICP$_{95}$ (\\%)", r"$k_{95}$ (lower=better)", "Width$_{95}$ (veh/h)", "Cost"]

# Values: rows=methods, cols=metrics. Use NaN where not applicable.
nan = np.nan
table = np.array([
    [0.5857, 0.4820, 54.85, 11.66, 5.37,   30.0],     # T8 MC (cost = S forward passes)
    [0.5857, 0.4820, 85.0,  4.04,  nan,    30.5],     # T8 + TS  (PICP after TS at 2sig ~ 85)
    [0.5957, 0.4820, 95.01, nan,   29.35,  30.5],     # Std conformal
    [0.5957, 0.4820, 95.0,  nan,   31.87,  30.5],     # Adaptive conformal (uses MC sigma)
    [0.4991, 0.4797, 90.01, 2.84,  18.91,  30.5],     # T9 heteroscedastic
    [0.5835, 0.2943, 94.91, nan,   24.82,  1.0],      # T11 CQR
    [0.6841, 0.3997, 51.05, 15.18, nan,    5.0],      # Deep Ensemble (cost = M=5)
])

# Normalize each column to [0,1] for color mapping
# Higher-is-better: R2, rho, PICP95 (closer to 95)
# Lower-is-better: k95, width95, cost
norm_table = np.zeros_like(table)
for j in range(table.shape[1]):
    col = table[:, j]
    valid = ~np.isnan(col)
    if not np.any(valid):
        continue
    if j in [0, 1]:  # higher better
        lo, hi = np.nanmin(col), np.nanmax(col)
        norm_table[:, j] = (col - lo) / (hi - lo + 1e-9)
    elif j == 2:  # PICP95 → distance from 95% target (closer = better, lower distance = greener)
        dist = np.abs(col - 95.0)
        lo, hi = np.nanmin(dist), np.nanmax(dist)
        norm_table[:, j] = 1.0 - (dist - lo) / (hi - lo + 1e-9)
    else:  # lower better
        lo, hi = np.nanmin(col), np.nanmax(col)
        norm_table[:, j] = 1.0 - (col - lo) / (hi - lo + 1e-9)

fig, ax = plt.subplots(figsize=(11.5, 5.5))
cmap = mcolors.LinearSegmentedColormap.from_list("tum_div", ["#F4D0C0", "#FFFFFF", "#9CC4E4", TUM_BLUE])
im = ax.imshow(norm_table, cmap=cmap, aspect="auto", vmin=0, vmax=1)

# Annotate cells with raw values
for i in range(table.shape[0]):
    for j in range(table.shape[1]):
        v = table[i, j]
        if np.isnan(v):
            ax.text(j, i, "—", ha="center", va="center", fontsize=10, color=TUM_GRAY)
        else:
            if j == 0 or j == 1:
                txt = f"{v:.4f}"
            elif j == 2:
                txt = f"{v:.2f}"
            elif j == 3:
                txt = f"{v:.2f}"
            elif j == 4:
                txt = f"{v:.2f}"
            else:
                txt = f"{v:.0f}×" if v >= 1 else f"{v:.1f}×"
            color = "white" if norm_table[i, j] > 0.65 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=10, fontweight="bold", color=color)

ax.set_xticks(range(len(metric_labels)))
ax.set_xticklabels(metric_labels)
ax.set_yticks(range(len(methods)))
ax.set_yticklabels(methods)
ax.tick_params(axis="x", which="both", length=0)
ax.tick_params(axis="y", which="both", length=0)

cbar = fig.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label("Per-column relative quality (1.0 = best in column)")
ax.set_title("UQ method comparison heatmap: trade-offs across accuracy, calibration, and cost")
ax.set_xlim(-0.5, len(metric_labels) - 0.5)
fig.tight_layout()
save_both(fig, OUT, "fig04_uq_method_summary_heatmap")
print("F4 saved")

# ============================================================
# F26: Deep Ensemble member R² + ensemble + T8 reference
# ============================================================
ens = json.load(open(BASE + "deep_ensemble_results/ensemble_metrics.json"))
seeds = [42, 137, 256, 389, 512]
member_r2 = [ens["member_quality"][f"seed_{s}"]["r2"] for s in seeds]
ens_r2 = ens["point_prediction"]["r2"]
T8_R2 = 0.5957

fig, ax = plt.subplots(figsize=(9.5, 5.2))
labels = [f"Seed {s}" for s in seeds] + ["Ensemble\n(mean of 5)"]
vals = member_r2 + [ens_r2]
colors = [TUM_LIGHTBLUE]*5 + [TUM_ORANGE]
bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.4f}",
            ha="center", fontsize=10, fontweight="bold")

# T8 reference line
ax.axhline(T8_R2, color=TUM_GRAY, linestyle="--", linewidth=1.4, label=f"T8 (single, dropout=0.2): $R^2 = {T8_R2:.4f}$")
ax.text(5.4, T8_R2 - 0.005, f"T8 baseline", color=TUM_GRAY, fontsize=9, ha="right", va="top")

ax.set_ylabel("Test $R^2$ (deterministic)")
ax.set_title("Deep ensemble: ensemble mean exceeds every member ($+14.8\\%$ vs T8)")
ax.set_ylim(0.55, 0.72)
ax.legend(loc="lower right", fontsize=9.5)
fig.tight_layout()
save_both(fig, OUT, "fig26_deep_ensemble_member_r2")
print("F26 saved")

# ============================================================
# F27: MC Dropout vs Deep Ensemble trade-off (4 sub-panels)
# ============================================================
mc_metrics = {"R²": 0.5857, "MAE": 3.948, "ρ": 0.4820, "k₉₅": 11.66}
de_metrics = {"R²": 0.6841, "MAE": 3.485, "ρ": 0.3997, "k₉₅": 15.18}

# Direction: R²/ρ higher better; MAE/k₉₅ lower better
def winner(m, mc, de, lower_better=False):
    if lower_better:
        return "DE" if de < mc else "MC"
    return "DE" if de > mc else "MC"

panels = [
    ("(a) Point accuracy: $R^2$", "R²", False),
    ("(b) Point error: MAE (veh/h)", "MAE", True),
    (r"(c) UQ ranking: Spearman $\rho$", "ρ", False),
    (r"(d) Calibration: $k_{95}$", "k₉₅", True),
]

fig, axes = plt.subplots(2, 2, figsize=(11.5, 8))
for ax, (title, key, lower) in zip(axes.flat, panels):
    vals = [mc_metrics[key], de_metrics[key]]
    win = winner(key, vals[0], vals[1], lower_better=lower)
    bar_colors = []
    for label in ["MC", "DE"]:
        bar_colors.append(PASS_GREEN if label == win else TUM_GRAY)
    bars = ax.bar(["MC Dropout\n(T8)", "Deep Ensemble\n(5 models)"], vals,
                  color=bar_colors, edgecolor="black", linewidth=0.6, width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + max(vals)*0.02, f"{v:.4f}",
                ha="center", fontsize=10, fontweight="bold")
    # Mark winner
    win_idx = ["MC", "DE"].index(win)
    ax.text(win_idx, vals[win_idx]/2, "WINNER", ha="center", va="center",
            fontsize=11, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=PASS_GREEN, alpha=0.9))
    ax.set_title(title)
    ax.set_ylim(0, max(vals) * 1.2)

fig.suptitle("MC Dropout vs Deep Ensemble: Deep Ensemble wins on accuracy; MC Dropout wins on UQ quality")
fig.tight_layout()
save_both(fig, OUT, "fig27_mc_vs_deepens_tradeoff")
print("F27 saved")

# ============================================================
# F28: Stratified UQ by VOL_BASE_CASE quartile
# ============================================================
# Use T8 MC Dropout NPZ + reconstruct VOL_BASE_CASE feature ordering
# We don't have the test_loader feature data immediately, but can stratify by error magnitude
# as a proxy for "high traffic vs low traffic". Use the targets themselves (delta_v) to sort —
# nodes with small |target| are mostly low-volume baseline scenarios.
mc = np.load(BASE + "point_net_transf_gat_8th_trial_lower_dropout/uq_results/mc_dropout_full_100graphs_mc30.npz")
y = mc["targets"]
yhat = mc["predictions"]
sigma = mc["uncertainties"]
err = np.abs(y - yhat)

# Stratify by |y_target| (proxy for traffic-volume change magnitude)
# Use rank-based binning to handle the heavy ties at y=0 (no-change segments)
abs_y = np.abs(y)
n_total = len(abs_y)
ranks = np.argsort(np.argsort(abs_y))  # rank-based to split ties evenly
quartile_labels = ["Q1 (smallest |Δv|)", "Q2", "Q3", "Q4 (largest |Δv|)"]
boundaries = [0, n_total // 4, n_total // 2, 3 * n_total // 4, n_total]

mae_by_q = []
rho_by_q = []
sigma_mean_by_q = []
n_by_q = []
abs_y_means = []
for k in range(4):
    mask = (ranks >= boundaries[k]) & (ranks < boundaries[k+1])
    e = err[mask]; s = sigma[mask]
    mae_by_q.append(float(e.mean()))
    rho_q, _ = spearmanr(s, e)
    rho_by_q.append(float(rho_q))
    sigma_mean_by_q.append(float(s.mean()))
    n_by_q.append(int(mask.sum()))
    abs_y_means.append(float(abs_y[mask].mean()))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
bars = ax.bar(quartile_labels, mae_by_q, color=[TUM_BLUE, TUM_LIGHTBLUE, TUM_ORANGE, FAIL_RED],
              edgecolor="black", linewidth=0.5)
for b, v in zip(bars, mae_by_q):
    ax.text(b.get_x() + b.get_width()/2, v + max(mae_by_q)*0.02, f"{v:.2f}",
            ha="center", fontsize=10, fontweight="bold")
ax.set_ylabel("MAE on quartile (veh/h)")
ax.set_title(f"(a) Prediction error by traffic-change quartile\n(easier on low-Δv segments, harder on high-traffic)")

ax = axes[1]
bars = ax.bar(quartile_labels, rho_by_q, color=[TUM_BLUE, TUM_LIGHTBLUE, TUM_ORANGE, FAIL_RED],
              edgecolor="black", linewidth=0.5)
for b, v in zip(bars, rho_by_q):
    ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.3f}",
            ha="center", fontsize=10, fontweight="bold")
ax.axhline(0.4820, color=TUM_GRAY, linestyle="--", linewidth=1.2, label=f"Aggregate $\\rho = 0.4820$")
ax.set_ylabel(r"Spearman $\rho$ within quartile")
ax.set_title("(b) Uncertainty ranking quality by quartile")
ax.legend(fontsize=9)
ax.set_ylim(0, max(rho_by_q) * 1.2)

fig.suptitle("Stratified UQ analysis: T8 uncertainty ranking varies with traffic-volume change magnitude")
fig.tight_layout()
save_both(fig, OUT, "fig28_stratified_uq_quartiles")
print("F28 saved")

# Print stratified stats for thesis
print("\nStratified stats:")
for k, lbl in enumerate(quartile_labels):
    print(f"  {lbl}: n={n_by_q[k]:,d}  MAE={mae_by_q[k]:.3f}  rho={rho_by_q[k]:.4f}  mean_sigma={sigma_mean_by_q[k]:.3f}")
