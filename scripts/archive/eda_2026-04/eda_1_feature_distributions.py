"""EDA 1: Feature Distributions across 50 Scenarios - Clean Academic Style"""

import torch
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = r"C:\Users\zamin\Downloads\Nazim\Thesis\Nazim\ml_surrogates_thesis_final\code\data\train_data\dist_not_connected_10k_1pct\datalist_batch_1.pt"
data_list = torch.load(DATA_PATH, weights_only=False, map_location="cpu")

feature_names = [
    "VOL_BASE_CASE",
    "CAPACITY_BASE",
    "CAPACITY_REDUCTION",
    "FREESPEED",
    "HIGHWAY_TYPE*",
    "LENGTH",
]
units = ["(veh/hr)", "(veh/hr)", "(fraction)", "(km/hr)", "(categorical)", "(meters)"]

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

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle(
    "Feature Distributions Across 50 Transport Scenarios",
    fontsize=16,
    fontweight="bold",
    y=0.98,
)

muted_colors = ["#4878A8", "#D66B6B", "#5DA573", "#D4A843", "#9B7EB8", "#C47A4E"]

for idx in range(6):
    ax = axes[idx // 3, idx % 3]

    all_vals = np.concatenate([g.x[:, idx].numpy() for g in data_list])

    ax.hist(
        all_vals,
        bins=80,
        color=muted_colors[idx],
        alpha=0.7,
        edgecolor="white",
        linewidth=0.3,
        density=True,
    )

    mean_v, std_v = np.mean(all_vals), np.std(all_vals)
    ax.axvline(
        mean_v,
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label=f"Mean={mean_v:.2f}",
    )

    ax.set_title(f"{feature_names[idx]} {units[idx]}", fontsize=11, fontweight="bold")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8, framealpha=0.8)

    if idx == 4:
        ax.text(
            0.5,
            0.85,
            "*Excluded from training",
            transform=ax.transAxes,
            fontsize=9,
            color="red",
            ha="center",
            style="italic",
        )

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.text(
    0.5,
    0.01,
    "50 scenarios x 31,635 nodes per scenario = 1,581,750 data points",
    ha="center",
    fontsize=9,
    color="gray",
)

plt.savefig(
    r"C:\Users\zamin\Downloads\Nazim\eda_1_feature_distributions.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="white",
)
plt.close()
print("Saved: eda_1_feature_distributions.png")
