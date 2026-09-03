"""EDA 2: Target Variable (y) Deep Dive - Clean Academic Style"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

DATA_PATH = r"C:\Users\zamin\Downloads\Nazim\Thesis\Nazim\ml_surrogates_thesis_final\code\data\train_data\dist_not_connected_10k_1pct\datalist_batch_1.pt"
data_list = torch.load(DATA_PATH, weights_only=False, map_location="cpu")

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
    "Target Variable (y): Traffic Volume Change Analysis",
    fontsize=16,
    fontweight="bold",
    y=0.98,
)

all_y = np.concatenate([g.y.numpy().flatten() for g in data_list])

# Panel 1: Overall distribution
ax = axes[0, 0]
ax.hist(
    all_y,
    bins=150,
    color="#4878A8",
    alpha=0.7,
    edgecolor="white",
    linewidth=0.2,
    density=True,
)
ax.axvline(0, color="red", linestyle="--", linewidth=1, alpha=0.6)
ax.axvline(
    np.mean(all_y),
    color="black",
    linestyle="--",
    linewidth=1,
    label=f"Mean={np.mean(all_y):.3f}",
)
ax.set_title("Overall y Distribution", fontweight="bold")
ax.set_xlabel("Delta Volume (veh/hr)")
ax.set_ylabel("Density")
ax.legend(fontsize=8)

# Panel 2: Positive/Negative/Zero stacked bar
ax = axes[0, 1]
pos_pct, neg_pct, zero_pct = [], [], []
for g in data_list:
    yv = g.y.numpy().flatten()
    n = len(yv)
    pos_pct.append(100 * np.sum(yv > 0.5) / n)
    neg_pct.append(100 * np.sum(yv < -0.5) / n)
    zero_pct.append(100 * np.sum(np.abs(yv) <= 0.5) / n)

x_sc = np.arange(50)
ax.bar(
    x_sc,
    pos_pct,
    color="#5DA573",
    alpha=0.8,
    label=f"Positive ({np.mean(pos_pct):.0f}%)",
    width=0.9,
)
ax.bar(
    x_sc,
    zero_pct,
    bottom=pos_pct,
    color="#D4A843",
    alpha=0.8,
    label=f"Near-zero ({np.mean(zero_pct):.0f}%)",
    width=0.9,
)
ax.bar(
    x_sc,
    neg_pct,
    bottom=np.array(pos_pct) + np.array(zero_pct),
    color="#D66B6B",
    alpha=0.8,
    label=f"Negative ({np.mean(neg_pct):.0f}%)",
    width=0.9,
)
ax.set_title("Node Impact Split per Scenario", fontweight="bold")
ax.set_xlabel("Scenario")
ax.set_ylabel("% of Nodes")
ax.legend(fontsize=7, loc="center right")
ax.set_ylim(0, 100)

# Panel 3: Per-scenario mean y (sorted)
ax = axes[0, 2]
means = [np.mean(g.y.numpy()) for g in data_list]
sorted_means = sorted(means)
colors = ["#5DA573" if m >= 0 else "#D66B6B" for m in sorted_means]
ax.barh(range(50), sorted_means, color=colors, height=0.8, alpha=0.8)
ax.axvline(0, color="black", linestyle="-", linewidth=0.5)
ax.set_title("Mean y per Scenario (Sorted)", fontweight="bold")
ax.set_xlabel("Mean Delta Volume (veh/hr)")
ax.set_ylabel("Scenario (rank)")
ax.set_yticks([0, 10, 20, 30, 40, 49])

# Panel 4: Box plot (first 20 scenarios)
ax = axes[1, 0]
bp = ax.boxplot(
    [g.y.numpy().flatten() for g in data_list[:20]],
    vert=True,
    patch_artist=True,
    showfliers=False,
    medianprops=dict(color="black", linewidth=1.5),
)
for patch in bp["boxes"]:
    patch.set_facecolor("#4878A8")
    patch.set_alpha(0.5)
ax.axhline(0, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
ax.set_title("y Box Plot (Scenarios 1-20)", fontweight="bold")
ax.set_xlabel("Scenario")
ax.set_ylabel("Delta Volume (veh/hr)")

# Panel 5: Spatial map - mean y
ax = axes[1, 1]
pos = data_list[0].pos.numpy()
lon, lat = pos[:, 2, 0], pos[:, 2, 1]
y_stack = np.stack([g.y.numpy().flatten() for g in data_list])
mean_y = np.mean(y_stack, axis=0)

norm = TwoSlopeNorm(
    vmin=np.percentile(mean_y, 2), vcenter=0, vmax=np.percentile(mean_y, 98)
)
sc = ax.scatter(lon, lat, c=mean_y, cmap="RdBu_r", s=0.08, alpha=0.5, norm=norm)
plt.colorbar(sc, ax=ax, label="Mean Delta Vol.", shrink=0.8)
ax.set_title("Spatial: Mean y (50 Scenarios)", fontweight="bold")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_aspect("equal")

# Panel 6: Spatial map - std y
ax = axes[1, 2]
std_y = np.std(y_stack, axis=0)
sc2 = ax.scatter(
    lon, lat, c=std_y, cmap="OrRd", s=0.08, alpha=0.5, vmax=np.percentile(std_y, 95)
)
plt.colorbar(sc2, ax=ax, label="Std Delta Vol.", shrink=0.8)
ax.set_title("Spatial: y Variability (Std)", fontweight="bold")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_aspect("equal")

fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(
    r"C:\Users\zamin\Downloads\Nazim\eda_2_target_deep_dive.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="white",
)
plt.close()
print("Saved: eda_2_target_deep_dive.png")
