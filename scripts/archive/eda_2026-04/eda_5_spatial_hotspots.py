"""EDA 5: Spatial Hotspot Maps - Clean Academic Style"""

import matplotlib

matplotlib.use("Agg")
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

DATA_PATH = r"C:\Users\zamin\Downloads\Nazim\Thesis\Nazim\ml_surrogates_thesis_final\code\data\train_data\dist_not_connected_10k_1pct\datalist_batch_1.pt"
data_list = torch.load(DATA_PATH, weights_only=False, map_location="cpu")

pos = data_list[0].pos.numpy()
lon, lat = pos[:, 2, 0], pos[:, 2, 1]

y_stack = np.stack([g.y.numpy().flatten() for g in data_list])
cap_stack = np.stack([g.x[:, 2].numpy() for g in data_list])

mean_y = np.mean(y_stack, axis=0)
std_y = np.std(y_stack, axis=0)
max_abs_y = np.max(np.abs(y_stack), axis=0)
n_affected = np.sum(cap_stack > 0, axis=0)

plt.rcParams.update({"font.size": 9, "font.family": "serif"})
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle(
    "Paris Road Network: Spatial Analysis (50 Scenarios)",
    fontsize=15,
    fontweight="bold",
    y=0.98,
)

# Panel 1: Capacity reduction locations
ax = axes[0, 0]
ax.scatter(lon, lat, c="lightgray", s=0.02, alpha=0.2)
mask = n_affected > 0
sc = ax.scatter(
    lon[mask],
    lat[mask],
    c=n_affected[mask],
    cmap="YlOrRd",
    s=0.5,
    alpha=0.8,
    vmin=1,
    vmax=50,
)
plt.colorbar(sc, ax=ax, label="# Scenarios Affected", shrink=0.8)
ax.set_title(
    f"Capacity Reduction Locations ({np.sum(mask):,} roads)", fontweight="bold"
)
ax.set_aspect("equal")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

# Panel 2: Mean traffic change
ax = axes[0, 1]
norm = TwoSlopeNorm(
    vmin=np.percentile(mean_y, 2), vcenter=0, vmax=np.percentile(mean_y, 98)
)
sc = ax.scatter(lon, lat, c=mean_y, cmap="RdBu_r", s=0.08, alpha=0.5, norm=norm)
plt.colorbar(sc, ax=ax, label="Mean Delta Vol. (veh/hr)", shrink=0.8)
ax.set_title("Mean Traffic Volume Change", fontweight="bold")
ax.set_aspect("equal")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

# Panel 3: High variability areas
ax = axes[1, 0]
ax.scatter(lon, lat, c="lightgray", s=0.02, alpha=0.2)
threshold = np.percentile(std_y, 90)
mask_hv = std_y > threshold
sc = ax.scatter(
    lon[mask_hv], lat[mask_hv], c=std_y[mask_hv], cmap="Oranges", s=0.8, alpha=0.8
)
plt.colorbar(sc, ax=ax, label="Std Delta Vol.", shrink=0.8)
ax.set_title(
    f"High-Variability Roads (Top 10%, std>{threshold:.1f})", fontweight="bold"
)
ax.set_aspect("equal")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

# Panel 4: Top 100 most impacted
ax = axes[1, 1]
ax.scatter(lon, lat, c="lightgray", s=0.02, alpha=0.2)
top100 = np.argsort(max_abs_y)[-100:]
sc = ax.scatter(
    lon[top100],
    lat[top100],
    c=mean_y[top100],
    cmap="RdBu_r",
    s=20,
    alpha=0.9,
    edgecolors="black",
    linewidth=0.3,
    norm=TwoSlopeNorm(vmin=-50, vcenter=0, vmax=50),
)
plt.colorbar(sc, ax=ax, label="Mean Delta Vol.", shrink=0.8)
ax.set_title("Top 100 Most-Impacted Roads", fontweight="bold")
ax.set_aspect("equal")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(
    r"C:\Users\zamin\Downloads\Nazim\eda_5_spatial_hotspots.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="white",
)
plt.close()
print("Saved: eda_5_spatial_hotspots.png")
