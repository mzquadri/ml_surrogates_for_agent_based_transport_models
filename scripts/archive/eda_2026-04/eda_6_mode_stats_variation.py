"""EDA 6: Mode Stats Variation - Clean Academic Style"""

import matplotlib

matplotlib.use("Agg")
import torch
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = r"C:\Users\zamin\Downloads\Nazim\Thesis\Nazim\ml_surrogates_thesis_final\code\data\train_data\dist_not_connected_10k_1pct\datalist_batch_1.pt"
data_list = torch.load(DATA_PATH, weights_only=False, map_location="cpu")

mode_names = ["Car", "PT", "Bike", "Walk", "Freight", "Ride"]
metric_names = ["Trips", "Distance", "Duration"]

abs_stats = np.array([g.mode_stats_diff.numpy() for g in data_list])  # [50, 6, 3]
perc_stats = np.array([g.mode_stats_diff_perc.numpy() for g in data_list])  # [50, 6, 3]

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

fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle(
    "Transport Mode Statistics Across 50 Scenarios",
    fontsize=15,
    fontweight="bold",
    y=0.98,
)

mode_colors = ["#4878A8", "#D66B6B", "#5DA573", "#D4A843", "#9B7EB8", "#C47A4E"]

# Panel 1: Mean absolute change heatmap
ax = axes[0, 0]
mean_abs = np.mean(abs_stats, axis=0)
im = ax.imshow(mean_abs, cmap="RdBu_r", aspect="auto")
for i in range(6):
    for j in range(3):
        ax.text(
            j,
            i,
            f"{mean_abs[i, j]:.1f}",
            ha="center",
            va="center",
            fontsize=9,
            color="white"
            if abs(mean_abs[i, j]) > np.max(np.abs(mean_abs)) * 0.5
            else "black",
        )
ax.set_xticks(range(3))
ax.set_xticklabels(metric_names)
ax.set_yticks(range(6))
ax.set_yticklabels(mode_names)
ax.set_title("Mean Absolute Change", fontweight="bold")
plt.colorbar(im, ax=ax, shrink=0.8)

# Panel 2: Mean percentage change heatmap
ax = axes[0, 1]
mean_perc = np.mean(perc_stats, axis=0)
im = ax.imshow(mean_perc, cmap="RdBu_r", aspect="auto")
for i in range(6):
    for j in range(3):
        ax.text(
            j,
            i,
            f"{mean_perc[i, j]:.2f}%",
            ha="center",
            va="center",
            fontsize=9,
            color="white"
            if abs(mean_perc[i, j]) > np.max(np.abs(mean_perc)) * 0.5
            else "black",
        )
ax.set_xticks(range(3))
ax.set_xticklabels(metric_names)
ax.set_yticks(range(6))
ax.set_yticklabels(mode_names)
ax.set_title("Mean Percentage Change", fontweight="bold")
plt.colorbar(im, ax=ax, shrink=0.8)

# Panel 3: Trips box plot per mode
ax = axes[1, 0]
bp = ax.boxplot(
    [abs_stats[:, m, 0] for m in range(6)],
    vert=True,
    patch_artist=True,
    medianprops=dict(color="black", linewidth=1.5),
    flierprops=dict(marker=".", markersize=3),
)
for patch, c in zip(bp["boxes"], mode_colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.6)
ax.set_xticklabels(mode_names, fontsize=9)
ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax.set_title("Trip Count Changes by Mode", fontweight="bold")
ax.set_ylabel("Delta Trips")

# Panel 4: Duration box plot per mode
ax = axes[1, 1]
bp = ax.boxplot(
    [abs_stats[:, m, 2] for m in range(6)],
    vert=True,
    patch_artist=True,
    medianprops=dict(color="black", linewidth=1.5),
    flierprops=dict(marker=".", markersize=3),
)
for patch, c in zip(bp["boxes"], mode_colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.6)
ax.set_xticklabels(mode_names, fontsize=9)
ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax.set_title("Duration Changes by Mode", fontweight="bold")
ax.set_ylabel("Delta Duration")

fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(
    r"C:\Users\zamin\Downloads\Nazim\eda_6_mode_stats_variation.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="white",
)
plt.close()
print("Saved: eda_6_mode_stats_variation.png")
