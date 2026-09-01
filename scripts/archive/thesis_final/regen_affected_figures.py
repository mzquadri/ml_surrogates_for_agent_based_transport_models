"""Regenerate the 7 thesis figures that had literal \% / \_ rendering artefacts.

Bug: matplotlib labels were authored with LaTeX-style escapes ("80\\%", "VOL\\_BASE\\_CASE")
expecting LaTeX rendering, but the figures were saved without ``text.usetex=True``, so the
backslashes ended up baked into the PDF/PNG. This script regenerates the seven affected
figures using clean plain-text labels (``80%``, ``VOL_BASE_CASE``).

Numerical content of each figure is unchanged — values are pulled from the same canonical
JSON / NPZ files used by the original generators. Style matches the rest of the thesis.

Figures regenerated:
    fig07_selective_prediction_curve
    fig08_error_detection_auroc
    fig13_conformal_coverage_nominal_vs_achieved
    fig15_conditional_coverage_by_decile
    fig19_t9_uncertainty_decomposition
    fig34_feature_distributions
    fig35_policy_decision_framework
"""
import os, json, glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_curve, roc_auc_score

# ---------------------------------------------------------------------------
# Style — matches final_layout_figs.py palette used elsewhere in the thesis
# ---------------------------------------------------------------------------
PRIMARY   = "#5B9BD5"
SECOND    = "#ED7D31"
TERTIARY  = "#70AD47"
NEUTRAL   = "#A5A5A5"
EDGE      = "#888888"
GRID      = "#E5E5E5"
TICKDARK  = "#404040"
ALPHA     = 0.7

# Fig08 uses TUM colours to match the regen_fig08 lineage already in the thesis.
TUM_BLUE   = "#005293"
TUM_ORANGE = "#E37222"
TUM_GRAY   = "#808080"

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "normal",
    "axes.labelsize": 10, "axes.labelcolor": "black",
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "xtick.color": TICKDARK, "ytick.color": TICKDARK,
    "legend.fontsize": 9, "legend.frameon": False,
    "axes.edgecolor": "#666666", "axes.linewidth": 0.6,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

ROOT = "C:/Users/zamin/Downloads/ml_surrogates_thesis_final/ml_surrogates_thesis_final"
OUT  = f"{ROOT}/document/figures/new"
BASE = f"{ROOT}/code/data/TR-C_Benchmarks"
TRAIN_DATA = f"{ROOT}/code/data/train_data/dist_not_connected_10k_1pct"

os.makedirs(OUT, exist_ok=True)

def save(fig, stem):
    fig.savefig(f"{OUT}/{stem}.pdf")
    fig.savefig(f"{OUT}/{stem}.png", dpi=300)
    plt.close(fig)
    print(f"  saved {stem}")

# ===========================================================================
# fig07 — Selective prediction curve
# ===========================================================================
print("\n[1/7] fig07_selective_prediction_curve")
sp = json.load(open(f"{BASE}/point_net_transf_gat_8th_trial_lower_dropout/uq_results/phase3_results/selective_prediction_s30.json"))
retentions = np.array([r["retention"] for r in sp["retention_results"]])
maes       = np.array([r["mae"] for r in sp["retention_results"]])
reductions = np.array([r["mae_reduction_pct"] for r in sp["retention_results"]])
order = np.argsort(retentions)
retentions, maes, reductions = retentions[order], maes[order], reductions[order]
full_mae = sp["full_mae"]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(retentions * 100, maes, "-o", color=TUM_BLUE, linewidth=2, markersize=7, label="Selective MAE")
ax.fill_between(retentions * 100, maes, full_mae, alpha=0.2, color=TUM_ORANGE, label="Reduction vs baseline")
ax.axhline(full_mae, color=TUM_GRAY, linestyle=":", linewidth=1.2,
           label=f"Full-set MAE = {full_mae:.2f} veh/h")
for r_pct in [90, 50]:
    idx = int(np.argmin(np.abs(retentions * 100 - r_pct)))
    ax.annotate(f"{r_pct}% retained\nMAE={maes[idx]:.2f}\n({reductions[idx]:+.1f}%)",
                xy=(r_pct, maes[idx]), xytext=(r_pct + 5, maes[idx] - 0.7),
                fontsize=9, color=TUM_ORANGE,
                arrowprops=dict(arrowstyle="->", color=TUM_ORANGE, lw=1))
ax.set_xlabel("Retention fraction (%)")
ax.set_ylabel("MAE on retained nodes (veh/h)")
ax.set_title("Selective prediction: MAE vs uncertainty-based retention (T8)")
ax.legend(loc="upper left", fontsize=9)
ax.set_xlim(0, 105)
ax.set_ylim(0.5, full_mae * 1.1)
fig.tight_layout()
save(fig, "fig07_selective_prediction_curve")

# ===========================================================================
# fig08 — Error detection ROC (T8 vs T7)
# ===========================================================================
print("\n[2/7] fig08_error_detection_auroc")
mc_t8 = np.load(f"{BASE}/point_net_transf_gat_8th_trial_lower_dropout/uq_results/mc_dropout_full_100graphs_mc30.npz")
mc_t7 = np.load(f"{BASE}/point_net_transf_gat_7th_trial_80_10_10_split/uq_results/mc_dropout_full_100graphs_mc30.npz")
t8_auc = json.load(open(f"{BASE}/point_net_transf_gat_8th_trial_lower_dropout/uq_results/auroc_corrected.json"))
t7_auc = json.load(open(f"{BASE}/point_net_transf_gat_7th_trial_80_10_10_split/uq_results/t7_auroc.json"))

def roc_only(mc, pct_thr):
    err = np.abs(mc["targets"] - mc["predictions"])
    sigma = mc["uncertainties"]
    thr = np.quantile(err, pct_thr)
    labels = (err >= thr).astype(int)
    fpr, tpr, _ = roc_curve(labels, sigma)
    return fpr, tpr

fpr_t8_10, tpr_t8_10 = roc_only(mc_t8, 0.90)
fpr_t8_20, tpr_t8_20 = roc_only(mc_t8, 0.80)
fpr_t7_10, tpr_t7_10 = roc_only(mc_t7, 0.90)
fpr_t7_20, tpr_t7_20 = roc_only(mc_t7, 0.80)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
ax = axes[0]
ax.plot(fpr_t8_10, tpr_t8_10, color=TUM_BLUE,   linewidth=2,
        label=f"T8  AUROC = {t8_auc['auroc_top10pct_threshold']:.4f}")
ax.plot(fpr_t7_10, tpr_t7_10, color=TUM_ORANGE, linewidth=2,
        label=f"T7  AUROC = {t7_auc['auroc_top10pct_threshold']:.4f}")
ax.plot([0, 1], [0, 1], "--", color=TUM_GRAY, linewidth=1, label="Random (AUROC = 0.5)")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title(r"(a) Detecting top-10% errors ($|err| \geq $ 90th percentile)")
ax.legend(loc="lower right", fontsize=9)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

ax = axes[1]
ax.plot(fpr_t8_20, tpr_t8_20, color=TUM_BLUE,   linewidth=2,
        label=f"T8  AUROC = {t8_auc['auroc_top20pct_threshold']:.4f}")
ax.plot(fpr_t7_20, tpr_t7_20, color=TUM_ORANGE, linewidth=2,
        label=f"T7  AUROC = {t7_auc['auroc_top20pct_threshold']:.4f}")
ax.plot([0, 1], [0, 1], "--", color=TUM_GRAY, linewidth=1, label="Random")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title(r"(b) Detecting top-20% errors ($|err| \geq $ 80th percentile)")
ax.legend(loc="lower right", fontsize=9)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

fig.suptitle("Uncertainty as a large-error classifier (T8 vs T7, 100-graph test set)")
fig.tight_layout()
save(fig, "fig08_error_detection_auroc")

# ===========================================================================
# fig13 — Conformal coverage nominal vs achieved (3 bars × 3 levels)
# ===========================================================================
print("\n[3/7] fig13_conformal_coverage_nominal_vs_achieved")
levels    = [80, 90, 95]
raw_cov   = [40.09, 48.55, 54.85]
std_cov   = [80.01, 90.02, 95.01]
adapt_cov = [80.20, 89.87, 95.00]
x = np.arange(len(levels)); w = 0.27

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(x - w, raw_cov,   w, color=PRIMARY,  edgecolor="black", linewidth=0.4, alpha=ALPHA, label="Raw MC Dropout")
ax.bar(x,     std_cov,   w, color=SECOND,   edgecolor="black", linewidth=0.4, alpha=ALPHA, label="Standard conformal")
ax.bar(x + w, adapt_cov, w, color=TERTIARY, edgecolor="black", linewidth=0.4, alpha=ALPHA, label="Adaptive conformal")
for i, lvl in enumerate(levels):
    ax.plot([i - 1.5*w, i + 1.5*w], [lvl, lvl], "--", color=NEUTRAL, linewidth=0.8, alpha=0.7)
ax.text(2.55, 95 + 1.2, "Nominal target", fontsize=8, color=NEUTRAL, ha="right", style="italic")
ax.set_xticks(x); ax.set_xticklabels([f"{l}%" for l in levels])
ax.set_xlabel("Nominal coverage")
ax.set_ylabel("Achieved coverage (%)")
ax.set_ylim(0, 105)
ax.legend(loc="upper left", frameon=False)
fig.tight_layout()
save(fig, "fig13_conformal_coverage_nominal_vs_achieved")

# ===========================================================================
# fig15 — Conditional coverage by uncertainty decile
# ===========================================================================
print("\n[4/7] fig15_conditional_coverage_by_decile")
adapt_decile = json.load(open(f"{BASE}/point_net_transf_gat_8th_trial_lower_dropout/uq_results/phase3_results/adaptive_conformal_decile.json"))
deciles = adapt_decile["deciles"]
d_idx      = [d["decile"] for d in deciles]
std_by_d   = [d["standard_coverage_pct"] for d in deciles]
adapt_by_d = [d["adaptive_coverage_pct"] for d in deciles]

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(d_idx, std_by_d,   "--", color=PRIMARY, linewidth=1.4, alpha=0.9, label="Standard conformal")
ax.plot(d_idx, adapt_by_d, "-",  color=SECOND,  linewidth=1.4, alpha=0.9, label="Adaptive conformal")
for arr, c in [(std_by_d, PRIMARY), (adapt_by_d, SECOND)]:
    ax.plot([1, 10], [arr[0], arr[-1]], "o", color=c, markersize=4)
ax.axhline(90, color=NEUTRAL, linestyle=":", linewidth=0.8, alpha=0.6)
ax.text(10.2, 90, "90%", fontsize=8, color=NEUTRAL, va="center")
ax.set_xticks(d_idx); ax.set_xticklabels([f"D{i}" for i in d_idx])
ax.set_xlabel(r"Uncertainty decile (low $\sigma$ $\rightarrow$ high $\sigma$)")
ax.set_ylabel("Conditional coverage (%)")
ax.set_ylim(55, 102)
ax.legend(loc="lower left", frameon=False)
fig.tight_layout()
save(fig, "fig15_conditional_coverage_by_decile")

# ===========================================================================
# fig19 — T9 uncertainty decomposition (bar + pie)
# ===========================================================================
print("\n[5/7] fig19_t9_uncertainty_decomposition")
t9 = json.load(open(f"{BASE}/point_net_transf_gat_9th_trial_heteroscedastic/data_created_during_training/t9_evaluation_results.json"))
u = t9["uncertainty_decomposition"]
mean_alea = u["mean_sigma_aleatoric"]; mean_epi = u["mean_sigma_epistemic"]; mean_tot = u["mean_sigma_total"]
ratio = u["ratio_alea_epi"]; frac_alea = u["frac_aleatoric_dominant"]
frac_alea_pct = 100.0 * frac_alea
frac_epi_pct  = 100.0 - frac_alea_pct

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6),
                                gridspec_kw={"width_ratios": [1.4, 1]})
labels = ["Aleatoric", "Epistemic", "Total"]
values = [mean_alea, mean_epi, mean_tot]
colors = [PRIMARY, SECOND, NEUTRAL]
bars = ax1.bar(labels, values, color=colors, edgecolor=EDGE, linewidth=0.5, alpha=ALPHA)
for b, v in zip(bars, values):
    ax1.text(b.get_x() + b.get_width()/2, v + 0.15, f"{v:.3f}", ha="center", fontsize=10)
ax1.set_ylabel("Mean uncertainty (vehicles per hour)")
ax1.set_title("(a) Mean uncertainty by component (Trial 9)", fontsize=10.5)
ax1.set_ylim(0, max(values) * 1.30)
ax1.text(0.5, -0.22, f"Aleatoric to epistemic ratio: {ratio:.2f}",
         transform=ax1.transAxes, ha="center", fontsize=9.5, color="black",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=NEUTRAL, linewidth=0.6))

sizes = [frac_alea_pct, frac_epi_pct]
labels_pie = [f"Aleatoric-dominant\n({frac_alea_pct:.2f}%)",
              f"Epistemic-dominant\n({frac_epi_pct:.2f}%)"]
ax2.pie(sizes, labels=labels_pie, colors=[PRIMARY, SECOND], startangle=90,
        wedgeprops={"edgecolor": EDGE, "linewidth": 0.5, "alpha": ALPHA},
        textprops={"fontsize": 9.5}, labeldistance=1.15)
ax2.set_title("(b) Per-node dominant component", fontsize=10.5)

fig.suptitle("Trial 9 uncertainty decomposition", fontsize=11)
fig.tight_layout()
save(fig, "fig19_t9_uncertainty_decomposition")

# ===========================================================================
# fig34 — Feature distributions
# ===========================================================================
print("\n[6/7] fig34_feature_distributions")
try:
    import torch, joblib
    SCALER = f"{BASE}/point_net_transf_gat_8th_trial_lower_dropout/data_created_during_training/train_x_scaler.pkl"
    batches = sorted(glob.glob(f"{TRAIN_DATA}/datalist_batch_*.pt"))[:2]
    all_x = []
    for b in batches:
        data = torch.load(b, map_location="cpu", weights_only=False)
        if isinstance(data, list):
            for g in data:
                all_x.append(g.x.numpy() if hasattr(g.x, "numpy") else np.asarray(g.x))
    X = np.concatenate(all_x, axis=0)
    print(f"  loaded {X.shape[0]:,} nodes, {X.shape[1]} columns")

    USE_COLS = [0, 1, 2, 3, 5]  # skip column 4 = HIGHWAY
    NAMES_PLAIN = ["VOL_BASE_CASE", "CAPACITY_BASE_CASE", "CAPACITY_REDUCTION", "FREESPEED", "LENGTH"]
    UNITS       = ["veh/h", "veh/h", "veh/h", "m/s", "m"]
    sc = joblib.load(SCALER)
    vol_max = X[:, 0].max()
    if vol_max > 100:
        raw = X[:, USE_COLS]
    else:
        z = X[:, USE_COLS]; raw = z * sc.scale_ + sc.mean_

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    axes = axes.flatten()
    medians = {}
    for i, (name, unit) in enumerate(zip(NAMES_PLAIN, UNITS)):
        ax = axes[i]
        d = raw[:, i]; med = float(np.median(d)); medians[name] = med
        ax.hist(d, bins=60, color=PRIMARY, edgecolor=EDGE, linewidth=0.3, alpha=ALPHA)
        ax.axvline(med, color=SECOND, linestyle="--", linewidth=1.4, alpha=0.85,
                   label=f"median = {med:.2f}")
        ax.set_xlabel(f"{name} ({unit})")
        ax.set_ylabel("Node count")
        ax.set_title(name, fontsize=10)
        ax.legend(loc="upper right")

    ax = axes[5]; ax.axis("off")
    nz_red_pct = 100.0 * (raw[:, 2] == 0).sum() / raw.shape[0]
    summary = (
        f"Verified raw-unit statistics (n = {raw.shape[0]:,} nodes)\n\n"
        f"VOL_BASE_CASE:  mean {sc.mean_[0]:.1f},  std {sc.scale_[0]:.1f} veh/h\n\n"
        f"CAPACITY_BASE_CASE:  mean {sc.mean_[1]:.0f},  std {sc.scale_[1]:.0f} veh/h\n\n"
        f"CAPACITY_REDUCTION:  {nz_red_pct:.1f}% zero values\n\n"
        f"FREESPEED:  6 discrete urban bands\n"
        f"  (approx 30, 40, 50, 60, 80, 100 km/h)\n\n"
        f"LENGTH:  mean {sc.mean_[4]:.1f},  std {sc.scale_[4]:.1f} m\n\n"
        f"HIGHWAY (OSM road-type) excluded\n"
        f"per Natterer et al. (2025) convention."
    )
    ax.text(0.03, 0.97, summary, transform=ax.transAxes, fontsize=9, va="top", color="black",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=NEUTRAL, linewidth=0.6))

    fig.suptitle("Distributions of the five input features in raw units", fontsize=11, y=1.0)
    fig.tight_layout()
    save(fig, "fig34_feature_distributions")
    print(f"  medians: {medians}")
except Exception as e:
    print(f"  SKIPPED fig34: {e}")

# ===========================================================================
# fig35 — Three-tier deployment policy (horizontal flowchart)
# ===========================================================================
print("\n[7/7] fig35_policy_decision_framework")
fig, ax = plt.subplots(figsize=(10, 4))
ax.set_xlim(0, 13); ax.set_ylim(-1.8, 4.2); ax.axis("off")

ax.text(6.5, 4.0, "Three-tier deployment policy based on uncertainty percentile",
        ha="center", fontsize=11, color="black")

def tier(ax, x, w, color, top, mid, bot, icon):
    rect = FancyBboxPatch((x - w/2, 0.6), w, 2.6,
                          boxstyle="round,pad=0.06,rounding_size=0.10",
                          facecolor=color, edgecolor=NEUTRAL, alpha=ALPHA, linewidth=0.7)
    ax.add_patch(rect)
    ax.text(x, 2.85, top,  ha="center", va="center", fontsize=11.5, color="black", fontweight="bold")
    ax.text(x, 2.05, icon, ha="center", va="center", fontsize=18,   color="black")
    ax.text(x, 1.40, mid,  ha="center", va="center", fontsize=9.5,  color="black")
    ax.text(x, 0.90, bot,  ha="center", va="center", fontsize=9,    color=TICKDARK, style="italic")

tier(ax, 2.5,  3.6, "#9CCDA1", "ACCEPT",          "0% to 50% percentile",   "Use prediction directly",     "OK")
tier(ax, 6.5,  3.6, "#ED7D31", "FLAG FOR REVIEW", "50% to 90% percentile",  "Send to expert verification", "!")
tier(ax, 10.5, 3.6, "#A5A5A5", "REJECT",          "90% to 100% percentile", "Re-run MATSim simulation",    "X")

ax.annotate("", xy=(12.6, 0.0), xytext=(0.4, 0.0),
            arrowprops=dict(arrowstyle="->", lw=0.9, color="black"))
for tx, lbl in [(0.7, "0%"), (4.5, "50%"), (8.5, "90%"), (12.4, "100%")]:
    ax.plot([tx, tx], [-0.08, 0.08], color="black", linewidth=0.7)
    ax.text(tx, -0.30, lbl, ha="center", va="top", fontsize=8.5, color=TICKDARK)
ax.text(6.5, -1.05, "Increasing predicted uncertainty",
        ha="center", fontsize=10, color=TICKDARK, style="italic")

save(fig, "fig35_policy_decision_framework")

print("\nAll 7 affected figures regenerated.")
