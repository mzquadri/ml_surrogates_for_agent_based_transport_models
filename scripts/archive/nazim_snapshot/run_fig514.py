"""Fig 5.14 — T8 Error Detection: ROC + Precision-Recall curves (redesign)."""

import sys, os, numpy as np, pandas as pd

# thesis_style
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)

# ── Data ──────────────────────────────────────────────────────────────────────
REPO = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(
    REPO,
    "data",
    "TR-C_Benchmarks",
    "point_net_transf_gat_8th_trial_lower_dropout",
    "trial8_uq_ablation_results.csv",
)

print("Loading CSV …")
df = pd.read_csv(CSV)
sigma = df["pred_mc_std"].values.astype(np.float64)
errors = df["abs_error_det"].values.astype(np.float64)

p90 = float(np.percentile(errors, 90))
p80 = float(np.percentile(errors, 80))

# Full-data scalar metrics
labels_10 = (errors >= p90).astype(int)
labels_20 = (errors >= p80).astype(int)
auroc_10 = float(roc_auc_score(labels_10, sigma))
auroc_20 = float(roc_auc_score(labels_20, sigma))
auprc_10 = float(average_precision_score(labels_10, sigma))
auprc_20 = float(average_precision_score(labels_20, sigma))

# Subsample for smooth curves (memory-safe)
MAX_CURVE = 300_000
rng = np.random.RandomState(42)
idx = rng.choice(len(sigma), MAX_CURVE, replace=False)
sig_c, err_c = sigma[idx], errors[idx]

lab10_c = (err_c >= p90).astype(int)
lab20_c = (err_c >= p80).astype(int)

fpr10, tpr10, _ = roc_curve(lab10_c, sig_c)
fpr20, tpr20, _ = roc_curve(lab20_c, sig_c)

prec10, rec10, _ = precision_recall_curve(lab10_c, sig_c)
prec20, rec20, _ = precision_recall_curve(lab20_c, sig_c)

print(f"  AUROC  top-10%={auroc_10:.4f}  top-20%={auroc_20:.4f}")
print(f"  AUPRC  top-10%={auprc_10:.4f}  top-20%={auprc_20:.4f}")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
fig.patch.set_facecolor(BG)

# Panel labels
panel_label(ax1, "(a)")
panel_label(ax2, "(b)")


def _style(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=P_MGRAY, labelsize=10)
    for sp in ax.spines.values():
        sp.set_edgecolor(P_LGRAY)
    ax.grid(True, color=P_LGRAY, lw=0.4, ls="--", alpha=0.3)


# ── (a) ROC Curve ─────────────────────────────────────────────────────────────
ax1.plot(
    [0, 1], [0, 1], "--", color=P_MGRAY, lw=1.0, label="Random (AUROC = 0.50)", zorder=1
)
ax1.plot(
    fpr10,
    tpr10,
    color=P_CORAL,
    lw=2.0,
    zorder=3,
    label=f"Top-10% errors   AUROC = {auroc_10:.4f}",
)
ax1.plot(
    fpr20,
    tpr20,
    color=P_BLUE,
    lw=2.0,
    zorder=3,
    label=f"Top-20% errors   AUROC = {auroc_20:.4f}",
)

ax1.set_xlabel("False Positive Rate", color=P_SLATE)
ax1.set_ylabel("True Positive Rate", color=P_SLATE)
ax1.set_title("ROC Curve", fontsize=13, color=P_DGRAY, pad=8)
ax1.set_xlim(-0.02, 1.02)
ax1.set_ylim(-0.02, 1.05)
_style(ax1)
ax1.legend(fontsize=9, loc="lower right", framealpha=0.9, edgecolor=P_LGRAY)

# ── (b) Precision-Recall Curve ────────────────────────────────────────────────
# Random baselines
ax2.axhline(0.10, color=P_CORAL, lw=0.8, ls=":", alpha=0.5)
ax2.axhline(0.20, color=P_BLUE, lw=0.8, ls=":", alpha=0.5)

ax2.plot(
    rec10,
    prec10,
    color=P_CORAL,
    lw=2.0,
    zorder=3,
    label=f"Top-10% errors   AUPRC = {auprc_10:.3f}",
)
ax2.plot(
    rec20,
    prec20,
    color=P_BLUE,
    lw=2.0,
    zorder=3,
    label=f"Top-20% errors   AUPRC = {auprc_20:.3f}",
)

# Small baseline annotations on right side
ax2.text(
    0.97,
    0.115,
    "random (0.10)",
    fontsize=8,
    color=P_CORAL,
    alpha=0.7,
    ha="right",
    va="bottom",
)
ax2.text(
    0.97,
    0.215,
    "random (0.20)",
    fontsize=8,
    color=P_BLUE,
    alpha=0.7,
    ha="right",
    va="bottom",
)

ax2.set_xlabel("Recall", color=P_SLATE)
ax2.set_ylabel("Precision", color=P_SLATE)
ax2.set_title("Precision\u2013Recall Curve", fontsize=13, color=P_DGRAY, pad=8)
ax2.set_xlim(-0.02, 1.02)
ax2.set_ylim(-0.02, 1.05)
_style(ax2)
ax2.legend(fontsize=9, loc="upper right", framealpha=0.9, edgecolor=P_LGRAY)

fig.subplots_adjust(left=0.08, right=0.97, top=0.90, bottom=0.13, wspace=0.28)

# ── Save ──────────────────────────────────────────────────────────────────────
out_dir = os.path.join(REPO, "thesis", "latex_tum_official", "figures")
for ext in ("pdf", "png"):
    out = os.path.join(out_dir, f"t8_error_detection_auroc.{ext}")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {out}")
plt.close(fig)
print("Done.")
