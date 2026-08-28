"""Generate F5–F8: S-convergence, per-graph rho, selective prediction, AUROC."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import spearmanr
from plot_style import set_style, save_both, TUM_BLUE, TUM_ORANGE, TUM_LIGHTBLUE, TUM_GRAY, TUM_GREEN, PASS_GREEN

set_style()
OUT = "../../../document/figures/new"
BASE = "../../data/TR-C_Benchmarks/"

# ============================================================
# F5: S-convergence of MC Dropout quality (2-panel)
# ============================================================
sc = json.load(open(BASE + "point_net_transf_gat_8th_trial_lower_dropout/uq_results/phase3_results/s_convergence_with_rho.json"))
graphs = sc["convergence_graphs"]
s_vals = [r["S"] for r in graphs[0]["s_results"]]
rhos_per_graph = np.array([[g["s_results"][i]["spearman_rho"] for g in graphs] for i in range(len(s_vals))])
sigmas_per_graph = np.array([[g["s_results"][i]["mean_sigma"] for g in graphs] for i in range(len(s_vals))])
rho_mean = rhos_per_graph.mean(axis=1)
rho_std = rhos_per_graph.std(axis=1)
sigma_mean = sigmas_per_graph.mean(axis=1)
sigma_std = sigmas_per_graph.std(axis=1)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
ax.fill_between(s_vals, rho_mean - rho_std, rho_mean + rho_std, alpha=0.2, color=TUM_BLUE, label="$\\pm1$ std across 10 graphs")
ax.plot(s_vals, rho_mean, "-o", color=TUM_BLUE, linewidth=1.8, markersize=6, label="Mean per-graph $\\rho$")
# Circle S=30
idx30 = s_vals.index(30)
ax.plot(30, rho_mean[idx30], "o", markerfacecolor="none", markeredgecolor=TUM_ORANGE, markersize=16, markeredgewidth=2, label="$S=30$ (operating point)")
ax.set_xlabel("Number of MC samples $S$")
ax.set_ylabel(r"Spearman $\rho$ between $\sigma$ and $|$error$|$")
ax.set_title("(a) Uncertainty ranking convergence")
ax.legend(loc="lower right", fontsize=9)
ax.set_xticks(s_vals)
ax.set_ylim(0.40, 0.50)

ax = axes[1]
ax.fill_between(s_vals, sigma_mean - sigma_std, sigma_mean + sigma_std, alpha=0.2, color=TUM_ORANGE)
ax.plot(s_vals, sigma_mean, "-s", color=TUM_ORANGE, linewidth=1.8, markersize=6, label="Mean $\\bar\\sigma$")
ax.plot(30, sigma_mean[idx30], "s", markerfacecolor="none", markeredgecolor=TUM_BLUE, markersize=16, markeredgewidth=2, label="$S=30$")
ax.set_xlabel("Number of MC samples $S$")
ax.set_ylabel(r"Mean uncertainty $\bar{\sigma}$ (veh/h)")
ax.set_title("(b) Mean uncertainty convergence")
ax.legend(loc="lower right", fontsize=9)
ax.set_xticks(s_vals)

fig.suptitle("$S$-convergence of MC Dropout (T8, 10 test graphs, 316,350 nodes)")
fig.tight_layout()
save_both(fig, OUT, "fig05_s_convergence")
print("F5 saved")

# ============================================================
# F6: Per-graph rho histogram with bootstrap CI
# ============================================================
boot = json.load(open(BASE + "point_net_transf_gat_8th_trial_lower_dropout/uq_results/phase3_results/bootstrap_ci_results.json"))
per_graph_rho = np.array(boot["per_graph_rho"])
mean_rho = boot["mean_rho"]
ci_lo, ci_hi = boot["ci_lo"], boot["ci_hi"]

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(per_graph_rho, bins=25, color=TUM_BLUE, edgecolor="black", linewidth=0.5, alpha=0.85)
ax.axvline(mean_rho, color=TUM_ORANGE, linewidth=2, label=f"Mean per-graph $\\rho = {mean_rho:.4f}$")
ax.axvspan(ci_lo, ci_hi, alpha=0.25, color=TUM_ORANGE, label=f"Bootstrap 95\\% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
ax.axvline(0.4820, color=TUM_GRAY, linestyle="--", linewidth=1.2, label=f"Aggregate $\\rho = 0.4820$")
ax.set_xlabel("Per-graph Spearman $\\rho$")
ax.set_ylabel("Number of test graphs")
ax.set_title(f"Distribution of per-graph $\\rho$ across 100 test graphs\n(min={per_graph_rho.min():.3f}, max={per_graph_rho.max():.3f}, std={per_graph_rho.std():.3f})")
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
save_both(fig, OUT, "fig06_per_graph_rho_distribution")
print("F6 saved")

# ============================================================
# F7: Selective prediction curve
# ============================================================
sp = json.load(open(BASE + "point_net_transf_gat_8th_trial_lower_dropout/uq_results/phase3_results/selective_prediction_s30.json"))
retentions = [r["retention"] for r in sp["retention_results"]]
maes = [r["mae"] for r in sp["retention_results"]]
reductions = [r["mae_reduction_pct"] for r in sp["retention_results"]]
# Sort by retention ascending
order = np.argsort(retentions)
retentions = np.array(retentions)[order]
maes = np.array(maes)[order]
reductions = np.array(reductions)[order]
full_mae = sp["full_mae"]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(retentions * 100, maes, "-o", color=TUM_BLUE, linewidth=2, markersize=7, label="Selective MAE")
ax.fill_between(retentions * 100, maes, full_mae, alpha=0.2, color=TUM_ORANGE, label="Reduction vs baseline")
ax.axhline(full_mae, color=TUM_GRAY, linestyle=":", linewidth=1.2, label=f"Full-set MAE = {full_mae:.2f} veh/h")

# Annotate 90% and 50% retention
for r_pct in [90, 50]:
    idx = np.argmin(np.abs(retentions * 100 - r_pct))
    ax.annotate(f"{r_pct}\\% retained\nMAE={maes[idx]:.2f}\n({reductions[idx]:+.1f}\\%)",
                xy=(r_pct, maes[idx]), xytext=(r_pct + 5, maes[idx] - 0.7),
                fontsize=9, color=TUM_ORANGE,
                arrowprops=dict(arrowstyle="->", color=TUM_ORANGE, lw=1))

ax.set_xlabel("Retention fraction (\\%)")
ax.set_ylabel("MAE on retained nodes (veh/h)")
ax.set_title("Selective prediction: MAE vs uncertainty-based retention (T8)")
ax.legend(loc="upper left", fontsize=9)
ax.set_xlim(0, 105)
ax.set_ylim(0.5, full_mae * 1.1)
fig.tight_layout()
save_both(fig, OUT, "fig07_selective_prediction_curve")
print("F7 saved")

# ============================================================
# F8: Error detection ROC curves (T8 + T7)
# ============================================================
mc_t8 = np.load(BASE + "point_net_transf_gat_8th_trial_lower_dropout/uq_results/mc_dropout_full_100graphs_mc30.npz")
mc_t7 = np.load(BASE + "point_net_transf_gat_7th_trial_80_10_10_split/uq_results/mc_dropout_full_100graphs_mc30.npz")

def roc_data(mc, pct_thr):
    err = np.abs(mc["targets"] - mc["predictions"])
    sigma = mc["uncertainties"]
    thr = np.quantile(err, pct_thr)
    labels = (err >= thr).astype(int)
    idx = np.random.RandomState(42).choice(len(err), 300000, replace=False)
    fpr, tpr, _ = roc_curve(labels[idx], sigma[idx])
    auc = roc_auc_score(labels[idx], sigma[idx])
    return fpr, tpr, auc

fpr_t8_10, tpr_t8_10, auc_t8_10 = roc_data(mc_t8, 0.90)
fpr_t8_20, tpr_t8_20, auc_t8_20 = roc_data(mc_t8, 0.80)
fpr_t7_10, tpr_t7_10, auc_t7_10 = roc_data(mc_t7, 0.90)
fpr_t7_20, tpr_t7_20, auc_t7_20 = roc_data(mc_t7, 0.80)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
ax = axes[0]
ax.plot(fpr_t8_10, tpr_t8_10, color=TUM_BLUE, linewidth=2, label=f"T8  AUROC = {auc_t8_10:.4f}")
ax.plot(fpr_t7_10, tpr_t7_10, color=TUM_ORANGE, linewidth=2, label=f"T7  AUROC = {auc_t7_10:.4f}")
ax.plot([0, 1], [0, 1], "--", color=TUM_GRAY, linewidth=1, label="Random (AUROC = 0.5)")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title(r"(a) Detecting top-10\% errors ($|err| \geq $ 90th percentile)")
ax.legend(loc="lower right", fontsize=9)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

ax = axes[1]
ax.plot(fpr_t8_20, tpr_t8_20, color=TUM_BLUE, linewidth=2, label=f"T8  AUROC = {auc_t8_20:.4f}")
ax.plot(fpr_t7_20, tpr_t7_20, color=TUM_ORANGE, linewidth=2, label=f"T7  AUROC = {auc_t7_20:.4f}")
ax.plot([0, 1], [0, 1], "--", color=TUM_GRAY, linewidth=1, label="Random")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title(r"(b) Detecting top-20\% errors ($|err| \geq $ 80th percentile)")
ax.legend(loc="lower right", fontsize=9)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

fig.suptitle("Uncertainty as a large-error classifier (T8 vs T7, 100-graph test set)")
fig.tight_layout()
save_both(fig, OUT, "fig08_error_detection_auroc")
print("F8 saved")
