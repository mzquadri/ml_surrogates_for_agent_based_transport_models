"""EDA 4: Scenario Variation Analysis - Clean Academic Style"""

import matplotlib

matplotlib.use("Agg")
import torch
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = r"C:\Users\zamin\Downloads\Nazim\Thesis\Nazim\ml_surrogates_thesis_final\code\data\train_data\dist_not_connected_10k_1pct\datalist_batch_1.pt"
data_list = torch.load(DATA_PATH, weights_only=False, map_location="cpu")

feature_names = [
    "VOL_BASE",
    "CAP_BASE",
    "CAP_REDUC",
    "FREESPEED",
    "HWY_TYPE*",
    "LENGTH",
    "TARGET (y)",
]
colors = ["#4878A8", "#D66B6B", "#5DA573", "#D4A843", "#9B7EB8", "#C47A4E", "#3A7CA5"]

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

# Compute per-scenario means
scenario_means = {name: [] for name in feature_names}
for g in data_list:
    for i in range(6):
        scenario_means[feature_names[i]].append(np.mean(g.x[:, i].numpy()))
    scenario_means["TARGET (y)"].append(np.mean(g.y.numpy()))

# Coefficient of variation
cvs = []
for name in feature_names:
    m = scenario_means[name]
    cv = np.std(m) / (np.abs(np.mean(m)) + 1e-10) * 100
    cvs.append(cv)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Scenario Variation Analysis: What Changes Between Scenarios?",
    fontsize=15,
    fontweight="bold",
    y=0.98,
)

# Panel 1: CV ranking
ax = axes[0, 0]
sorted_idx = np.argsort(cvs)[::-1]
bars = ax.barh(
    range(7),
    [cvs[i] for i in sorted_idx],
    color=[colors[i] for i in sorted_idx],
    alpha=0.8,
    height=0.6,
)
ax.set_yticks(range(7))
ax.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=9)
ax.set_xlabel("Coefficient of Variation (%)")
ax.set_title("Which Features Vary Across Scenarios?", fontweight="bold")
for i, (bar, idx) in enumerate(zip(bars, sorted_idx)):
    ax.text(
        cvs[idx] + 0.5,
        i,
        f"{cvs[idx]:.1f}%",
        va="center",
        fontsize=8,
        fontweight="bold",
    )

# Panel 2: CAP_REDUC per scenario (the main varying feature)
ax = axes[0, 1]
cap_means = scenario_means["CAP_REDUC"]
ax.bar(range(50), cap_means, color="#5DA573", alpha=0.8, width=0.8)
ax.set_title("CAPACITY_REDUCTION Mean per Scenario", fontweight="bold")
ax.set_xlabel("Scenario Index")
ax.set_ylabel("Mean Capacity Reduction")

# Panel 3: TARGET y per scenario
ax = axes[1, 0]
y_means = scenario_means["TARGET (y)"]
cols = ["#5DA573" if m >= 0 else "#D66B6B" for m in y_means]
ax.bar(range(50), y_means, color=cols, alpha=0.8, width=0.8)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_title("Mean Target (y) per Scenario", fontweight="bold")
ax.set_xlabel("Scenario Index")
ax.set_ylabel("Mean Delta Volume")

# Panel 4: Box plot of scenario means for all features (normalized for comparison)
ax = axes[1, 1]
normalized_means = []
labels = []
for i, name in enumerate(feature_names):
    vals = np.array(scenario_means[name])
    if np.std(vals) > 0:
        normalized_means.append((vals - np.mean(vals)) / np.std(vals))
    else:
        normalized_means.append(vals - np.mean(vals))
    labels.append(name)

bp = ax.boxplot(
    normalized_means,
    vert=True,
    patch_artist=True,
    showfliers=True,
    flierprops=dict(marker=".", markersize=3, alpha=0.5),
    medianprops=dict(color="black", linewidth=1.5),
)
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.6)
ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
ax.set_title("Scenario Mean Spread (Normalized)", fontweight="bold")
ax.set_ylabel("Z-score")

fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(
    r"C:\Users\zamin\Downloads\Nazim\eda_4_scenario_variation.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="white",
)
plt.close()
print("Saved: eda_4_scenario_variation.png")
