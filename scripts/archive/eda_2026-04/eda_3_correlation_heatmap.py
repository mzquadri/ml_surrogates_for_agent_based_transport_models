"""EDA 3: Correlation Heatmap - Clean Academic Style"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

DATA_PATH = r"C:\Users\zamin\Downloads\Nazim\Thesis\Nazim\ml_surrogates_thesis_final\code\data\train_data\dist_not_connected_10k_1pct\datalist_batch_1.pt"
data_list = torch.load(DATA_PATH, weights_only=False, map_location="cpu")

names = [
    "VOL_BASE",
    "CAP_BASE",
    "CAP_REDUC",
    "FREESPEED",
    "HWY_TYPE*",
    "LENGTH",
    "TARGET (y)",
]

np.random.seed(42)
sample_idx = np.random.choice(50, 10, replace=False)
all_data = []
for idx in sample_idx:
    g = data_list[idx]
    combined = np.hstack([g.x.numpy(), g.y.numpy().flatten()[:, np.newaxis]])
    all_data.append(combined)
all_data = np.vstack(all_data)

pearson_corr = np.corrcoef(all_data.T)
spearman_corr, _ = spearmanr(all_data)

plt.rcParams.update({"font.size": 10, "font.family": "serif"})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle(
    "Feature & Target Correlation Matrix", fontsize=16, fontweight="bold", y=1.01
)

for ax, corr, title in [
    (ax1, pearson_corr, "Pearson (Linear)"),
    (ax2, spearman_corr, "Spearman (Rank)"),
]:
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")

    for i in range(7):
        for j in range(7):
            val = corr[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color=color,
                fontweight="bold" if abs(val) > 0.3 else "normal",
            )

    ax.set_xticks(range(7))
    ax.set_yticks(range(7))
    ax.set_xticklabels(names, fontsize=8, rotation=45, ha="right")
    ax.set_yticklabels(names, fontsize=8)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Correlation")

fig.tight_layout()
fig.text(
    0.5,
    -0.02,
    "10 scenarios x 31,635 nodes = 316,350 data points  |  *HWY_TYPE excluded from training",
    ha="center",
    fontsize=9,
    color="gray",
)

plt.savefig(
    r"C:\Users\zamin\Downloads\Nazim\eda_3_correlation_heatmap.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="white",
)
plt.close()
print("Saved: eda_3_correlation_heatmap.png")
