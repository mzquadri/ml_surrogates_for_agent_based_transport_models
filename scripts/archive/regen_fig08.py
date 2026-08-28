"""Regenerate fig08 (error detection AUROC) using the canonical AUROC values
from the saved JSON metric files, so the in-figure legend matches the
text/appendix to within rounding."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from plot_style import set_style, save_both, TUM_BLUE, TUM_ORANGE, TUM_GRAY

set_style()
OUT = "../../../document/figures/new"
BASE = "../../data/TR-C_Benchmarks/"

mc_t8 = np.load(BASE + "point_net_transf_gat_8th_trial_lower_dropout/uq_results/mc_dropout_full_100graphs_mc30.npz")
mc_t7 = np.load(BASE + "point_net_transf_gat_7th_trial_80_10_10_split/uq_results/mc_dropout_full_100graphs_mc30.npz")

t8_auc = json.load(open(BASE + "point_net_transf_gat_8th_trial_lower_dropout/uq_results/auroc_corrected.json"))
t7_auc = json.load(open(BASE + "point_net_transf_gat_7th_trial_80_10_10_split/uq_results/t7_auroc.json"))

AUC_T8_10 = t8_auc["auroc_top10pct_threshold"]
AUC_T8_20 = t8_auc["auroc_top20pct_threshold"]
AUC_T7_10 = t7_auc["auroc_top10pct_threshold"]
AUC_T7_20 = t7_auc["auroc_top20pct_threshold"]

def roc_only(mc, pct_thr):
    err = np.abs(mc["targets"] - mc["predictions"])
    sigma = mc["uncertainties"]
    thr = np.quantile(err, pct_thr)
    labels = (err >= thr).astype(int)
    fpr, tpr, _ = roc_curve(labels, sigma)
    return fpr, tpr

print("Computing ROC curves on full test sets ...")
fpr_t8_10, tpr_t8_10 = roc_only(mc_t8, 0.90)
fpr_t8_20, tpr_t8_20 = roc_only(mc_t8, 0.80)
fpr_t7_10, tpr_t7_10 = roc_only(mc_t7, 0.90)
fpr_t7_20, tpr_t7_20 = roc_only(mc_t7, 0.80)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
ax = axes[0]
ax.plot(fpr_t8_10, tpr_t8_10, color=TUM_BLUE, linewidth=2, label=f"T8  AUROC = {AUC_T8_10:.4f}")
ax.plot(fpr_t7_10, tpr_t7_10, color=TUM_ORANGE, linewidth=2, label=f"T7  AUROC = {AUC_T7_10:.4f}")
ax.plot([0, 1], [0, 1], "--", color=TUM_GRAY, linewidth=1, label="Random (AUROC = 0.5)")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title(r"(a) Detecting top-10\% errors ($|err| \geq $ 90th percentile)")
ax.legend(loc="lower right", fontsize=9)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

ax = axes[1]
ax.plot(fpr_t8_20, tpr_t8_20, color=TUM_BLUE, linewidth=2, label=f"T8  AUROC = {AUC_T8_20:.4f}")
ax.plot(fpr_t7_20, tpr_t7_20, color=TUM_ORANGE, linewidth=2, label=f"T7  AUROC = {AUC_T7_20:.4f}")
ax.plot([0, 1], [0, 1], "--", color=TUM_GRAY, linewidth=1, label="Random")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title(r"(b) Detecting top-20\% errors ($|err| \geq $ 80th percentile)")
ax.legend(loc="lower right", fontsize=9)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

fig.suptitle("Uncertainty as a large-error classifier (T8 vs T7, 100-graph test set)")
fig.tight_layout()
save_both(fig, OUT, "fig08_error_detection_auroc")
print(f"F8 regenerated with canonical values:")
print(f"  T8 top-10% = {AUC_T8_10:.4f}, top-20% = {AUC_T8_20:.4f}")
print(f"  T7 top-10% = {AUC_T7_10:.4f}, top-20% = {AUC_T7_20:.4f}")
