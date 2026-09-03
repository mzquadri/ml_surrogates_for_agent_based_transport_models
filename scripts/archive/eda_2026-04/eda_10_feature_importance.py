"""EDA 10: Feature Importance / Predictability - Clean Academic Style"""

import matplotlib

matplotlib.use("Agg")
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

DATA_PATH = r"C:\Users\zamin\Downloads\Nazim\Thesis\Nazim\ml_surrogates_thesis_final\code\data\train_data\dist_not_connected_10k_1pct\datalist_batch_1.pt"
data_list = torch.load(DATA_PATH, weights_only=False, map_location="cpu")

feature_names = [
    "VOL_BASE",
    "CAP_BASE",
    "CAP_REDUC",
    "FREESPEED",
    "HWY_TYPE*",
    "LENGTH",
]
colors = ["#4878A8", "#D66B6B", "#5DA573", "#D4A843", "#9B7EB8", "#C47A4E"]

np.random.seed(42)
sample_idx = np.random.choice(50, 10, replace=False)
all_x = np.vstack([data_list[i].x.numpy() for i in sample_idx])
all_y = np.concatenate([data_list[i].y.numpy().flatten() for i in sample_idx])

pc_vals, sc_vals = [], []
for i in range(6):
    pc, _ = pearsonr(all_x[:, i], all_y)
    sc, _ = spearmanr(all_x[:, i], all_y)
    pc_vals.append(pc)
    sc_vals.append(sc)

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
    "Feature Importance: Which Features Predict Traffic Change?",
    fontsize=15,
    fontweight="bold",
    y=0.98,
)

# Panel 1: |Pearson| ranking
ax = axes[0, 0]
s_idx = np.argsort(np.abs(pc_vals))[::-1]
bars = ax.barh(
    range(6),
    [abs(pc_vals[i]) for i in s_idx],
    color=[colors[i] for i in s_idx],
    alpha=0.8,
    height=0.5,
)
for i, idx in enumerate(s_idx):
    sign = "+" if pc_vals[idx] > 0 else ""
    ax.text(
        abs(pc_vals[idx]) + 0.002,
        i,
        f"{sign}{pc_vals[idx]:.4f}",
        va="center",
        fontsize=9,
    )
ax.set_yticks(range(6))
ax.set_yticklabels([feature_names[i] for i in s_idx])
ax.set_xlabel("|Pearson r|")
ax.set_title("Pearson Correlation with Target", fontweight="bold")

# Panel 2: |Spearman| ranking
ax = axes[0, 1]
s_idx2 = np.argsort(np.abs(sc_vals))[::-1]
bars = ax.barh(
    range(6),
    [abs(sc_vals[i]) for i in s_idx2],
    color=[colors[i] for i in s_idx2],
    alpha=0.8,
    height=0.5,
)
for i, idx in enumerate(s_idx2):
    sign = "+" if sc_vals[idx] > 0 else ""
    ax.text(
        abs(sc_vals[idx]) + 0.002,
        i,
        f"{sign}{sc_vals[idx]:.4f}",
        va="center",
        fontsize=9,
    )
ax.set_yticks(range(6))
ax.set_yticklabels([feature_names[i] for i in s_idx2])
ax.set_xlabel("|Spearman $\\rho$|")
ax.set_title("Spearman Correlation with Target", fontweight="bold")

# Panel 3: Pearson vs Spearman scatter
ax = axes[0, 2]
for i in range(6):
    ax.scatter(
        abs(pc_vals[i]),
        abs(sc_vals[i]),
        c=colors[i],
        s=120,
        edgecolors="black",
        linewidth=0.5,
        zorder=5,
    )
    ax.annotate(
        feature_names[i],
        (abs(pc_vals[i]), abs(sc_vals[i])),
        textcoords="offset points",
        xytext=(6, 4),
        fontsize=8,
        color=colors[i],
    )
ax.plot([0, 0.2], [0, 0.2], "k--", alpha=0.3)
ax.set_xlabel("|Pearson r|")
ax.set_ylabel("|Spearman $\\rho$|")
ax.set_title("Linear vs Monotonic Correlation", fontweight="bold")

# Panels 4-6: Scatter plots for top 3 features by |Pearson|
sample_mask = np.random.choice(len(all_y), 30000, replace=False)
xs, ys = all_x[sample_mask], all_y[sample_mask]

top3 = s_idx[:3]
for plot_i, feat_idx in enumerate(top3):
    ax = axes[1, plot_i]
    ax.scatter(xs[:, feat_idx], ys, c=colors[feat_idx], s=0.3, alpha=0.08)

    # Binned means
    n_bins = 25
    edges = np.percentile(xs[:, feat_idx], np.linspace(0, 100, n_bins + 1))
    bx, by = [], []
    for b in range(n_bins):
        mask = (xs[:, feat_idx] >= edges[b]) & (xs[:, feat_idx] < edges[b + 1])
        if np.sum(mask) > 5:
            bx.append(np.mean(xs[mask, feat_idx]))
            by.append(np.mean(ys[mask]))
    ax.plot(
        bx,
        by,
        "o-",
        color="black",
        linewidth=1.5,
        markersize=4,
        label="Binned mean",
        zorder=10,
    )

    ax.set_title(
        f"{feature_names[feat_idx]} vs y  (r={pc_vals[feat_idx]:.4f})",
        fontweight="bold",
    )
    ax.set_xlabel(feature_names[feat_idx])
    ax.set_ylabel("Delta Volume (y)")
    ax.legend(fontsize=8)

fig.tight_layout(rect=[0, 0.02, 1, 0.95])
fig.text(
    0.5,
    0.005,
    "All correlations are weak (<0.13), showing that simple linear features cannot predict traffic change — GNN captures non-linear spatial propagation.",
    ha="center",
    fontsize=9,
    color="gray",
    style="italic",
)

plt.savefig(
    r"C:\Users\zamin\Downloads\Nazim\eda_10_feature_importance.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="white",
)
plt.close()
print("Saved: eda_10_feature_importance.png")
