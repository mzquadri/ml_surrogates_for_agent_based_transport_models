"""TASK 2: master summary figure (3 panels)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

PRIMARY = "#4A90C4"   # blue   = point-prediction trial
ACCENT  = "#E07B39"   # orange = UQ-extension trial (T9, T10, T11)
SUCCESS = "#5A9E6F"   # green  = ensemble (Deep Ens.)
NEUTRAL = "#888888"
GRID    = "#EEEEEE"
ALPHA   = 0.55

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.edgecolor": "#333333", "axes.linewidth": 0.7,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

OUT = "../../../document/figures/new"

# ---------- Verified master-table data ----------
# (label, R², category)  — categories: 'point', 'uq_ext', 'ensemble'
models = [
    ("T1",         0.7860, "point"),
    ("T2",         0.5117, "point"),
    ("T3",         0.2246, "point"),
    ("T4",         0.2426, "point"),
    ("T5",         0.5553, "point"),
    ("T6",         0.5223, "point"),
    ("T7",         0.5471, "point"),
    ("T8",         0.5957, "point"),
    ("T9",         0.4991, "uq_ext"),
    ("T10",        0.4057, "uq_ext"),
    ("T11",        0.5835, "uq_ext"),
    ("Deep Ens.",  0.6841, "ensemble"),
]

# UQ scatter: (label, R², Spearman ρ, inference cost in forward-pass equivalents)
uq_methods = [
    ("T8 MC Dropout",   0.5857, 0.4820, 30),
    ("T8 MC avg, 5 seeds", 0.5865, 0.4908, 150),
    ("Multi-model ens.", 0.5656, 0.4333, 5),
    ("T9 hetero.",       0.4991, 0.4797, 30),
    ("T11 CQR frozen",   0.5835, 0.2943, 1),
    ("Deep Ensemble",    0.6841, 0.3997, 5),
]

# k95 for UQ methods (lower better)
k95_methods = [
    ("Ideal Gaussian", 1.96),
    ("T9 hetero.",     2.84),
    ("T8 + TS",        4.04),
    ("T8 MC Dropout",  11.66),
    ("Deep Ensemble",  15.18),
]

CAT_COLOR = {"point": PRIMARY, "uq_ext": ACCENT, "ensemble": SUCCESS}

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

# ---------- Panel 1: R² bar chart, sorted by R² ----------
ax = axes[0]
sorted_models = sorted(models, key=lambda m: m[1], reverse=True)
labels = [m[0] for m in sorted_models]
vals   = [m[1] for m in sorted_models]
cols   = [CAT_COLOR[m[2]] for m in sorted_models]
ax.barh(np.arange(len(labels)), vals, color=cols, alpha=ALPHA, edgecolor="black", linewidth=0.4)
ax.set_yticks(np.arange(len(labels)))
ax.set_yticklabels(labels)
ax.invert_yaxis()
ax.set_xlabel("Test $R^2$")
ax.set_title("(a) Point accuracy across all models")
ax.set_xlim(0, 0.85)
# legend
import matplotlib.patches as mpatches
handles = [
    mpatches.Patch(color=PRIMARY, alpha=ALPHA, label="Point prediction"),
    mpatches.Patch(color=ACCENT,  alpha=ALPHA, label="UQ extension"),
    mpatches.Patch(color=SUCCESS, alpha=ALPHA, label="Ensemble"),
]
ax.legend(handles=handles, loc="lower right", frameon=False)

# ---------- Panel 2: UQ scatter R² vs Spearman ρ, size = log cost ----------
ax = axes[1]
for label, r2, rho, cost in uq_methods:
    s = 30 + 60 * np.log10(cost + 1)        # area scales with log(cost)
    cat = "ensemble" if "Ensemble" in label else ("uq_ext" if any(t in label for t in ["T9", "T11"]) else "point")
    ax.scatter(r2, rho, s=s, color=CAT_COLOR[cat], alpha=ALPHA, edgecolor="black", linewidth=0.4)
# annotate slightly offset to avoid overlap
offsets = {
    "T8 MC Dropout":      (0.005,  0.005),
    "T8 MC avg, 5 seeds": (0.005, -0.012),
    "Multi-model ens.":   (-0.04,  0.012),
    "T9 hetero.":         (0.005, -0.012),
    "T11 CQR frozen":     (0.005,  0.005),
    "Deep Ensemble":      (-0.07, -0.018),
}
for label, r2, rho, cost in uq_methods:
    dx, dy = offsets[label]
    ax.text(r2 + dx, rho + dy, label, fontsize=8, color="#222")
ax.set_xlabel("Test $R^2$")
ax.set_ylabel(r"Spearman $\rho$  ($\sigma$ vs $|err|$)")
ax.set_title("(b) UQ quality vs accuracy  (size $\\propto$ inference cost)")
ax.set_xlim(0.40, 0.72)
ax.set_ylim(0.27, 0.53)

# ---------- Panel 3: k95 bar chart, lower=better ----------
ax = axes[2]
labels_k = [m[0] for m in k95_methods]
vals_k   = [m[1] for m in k95_methods]
ideal_color = SUCCESS
bar_cols = [
    ideal_color if lbl == "Ideal Gaussian"
    else (ACCENT if lbl in ["T9 hetero.", "T8 + TS"] else PRIMARY)
    for lbl in labels_k
]
# Deep ensemble distinct
bar_cols[-1] = SUCCESS  # but mark deep ens green - actually use neutral
bar_cols = [
    SUCCESS if lbl == "Ideal Gaussian"
    else ACCENT if lbl == "T9 hetero."
    else "#9CC4E4" if lbl == "T8 + TS"
    else PRIMARY if lbl == "T8 MC Dropout"
    else NEUTRAL  # deep ensemble - worst
    for lbl in labels_k
]
ax.bar(labels_k, vals_k, color=bar_cols, alpha=ALPHA, edgecolor="black", linewidth=0.4)
ax.axhline(1.96, color=SUCCESS, linestyle=":", linewidth=0.8, alpha=0.7)
ax.set_ylabel(r"Empirical $k_{95}$  (lower better)")
ax.set_title("(c) Calibration of UQ methods")
ax.set_ylim(0, 17)
ax.tick_params(axis="x", labelrotation=18)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig00_all_models_summary.pdf"))
fig.savefig(os.path.join(OUT, "fig00_all_models_summary.png"), dpi=300)
plt.close(fig)
print("Task 2 done — fig00_all_models_summary saved")
