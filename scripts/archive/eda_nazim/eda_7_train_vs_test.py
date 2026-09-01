"""EDA 7: Train vs Test Data Comparison - Clean Academic Style"""

import matplotlib

matplotlib.use("Agg")
import torch
import numpy as np
import matplotlib.pyplot as plt

TRAIN_PATH = r"C:\Users\zamin\Downloads\Nazim\Thesis\Nazim\ml_surrogates_thesis_final\code\data\train_data\dist_not_connected_10k_1pct\datalist_batch_1.pt"
TEST_PATH = r"C:\Users\zamin\Downloads\Nazim\Thesis\Nazim\ml_surrogates_thesis_final\code\data\TR-C_Benchmarks\point_net_transf_gat_8th_trial_lower_dropout\data_created_during_training\test_dl.pt"

train_list = torch.load(TRAIN_PATH, weights_only=False, map_location="cpu")
test_list = torch.load(TEST_PATH, weights_only=False, map_location="cpu")

feature_names = ["VOL_BASE", "CAP_BASE", "CAP_REDUC", "FREESPEED", "LENGTH"]
train_idx = [0, 1, 2, 3, 5]  # train feature indices for 5 shared features

train_x = np.vstack([train_list[i].x.numpy() for i in range(10)])
train_y = np.concatenate([train_list[i].y.numpy().flatten() for i in range(10)])
test_x = np.vstack([test_list[i].x.numpy() for i in range(10)])
test_y = np.concatenate([test_list[i].y.numpy().flatten() for i in range(10)])

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
    "Training Data (Raw) vs Test Data (Normalized)",
    fontsize=15,
    fontweight="bold",
    y=0.98,
)

for i in range(5):
    ax = axes[i // 3, i % 3]

    ax.hist(
        train_x[:, train_idx[i]],
        bins=80,
        density=True,
        alpha=0.5,
        color="#4878A8",
        label="Train (raw)",
        edgecolor="none",
    )

    ax2 = ax.twinx()
    ax2.hist(
        test_x[:, i],
        bins=80,
        density=True,
        alpha=0.5,
        color="#D66B6B",
        label="Test (norm.)",
        edgecolor="none",
    )

    ax.set_title(feature_names[i], fontweight="bold")
    ax.set_ylabel("Train density", fontsize=8, color="#4878A8")
    ax2.set_ylabel("Test density", fontsize=8, color="#D66B6B")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")

# Panel 6: Target y comparison
ax = axes[1, 2]
ax.hist(
    train_y,
    bins=100,
    density=True,
    alpha=0.5,
    color="#4878A8",
    label="Train y (raw)",
    edgecolor="none",
)
ax.hist(
    test_y,
    bins=100,
    density=True,
    alpha=0.5,
    color="#D66B6B",
    label="Test y (norm.)",
    edgecolor="none",
)
ax.set_title("Target (y)", fontweight="bold")
ax.set_xlabel("Value")
ax.set_ylabel("Density")
ax.legend(fontsize=8)

fig.tight_layout(rect=[0, 0.02, 1, 0.95])
fig.text(
    0.5,
    0.005,
    "Train: 6 raw features | Test: 5 normalized features (HWY_TYPE excluded, StandardScaler applied)",
    ha="center",
    fontsize=9,
    color="gray",
)

plt.savefig(
    r"C:\Users\zamin\Downloads\Nazim\eda_7_train_vs_test.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="white",
)
plt.close()
print("Saved: eda_7_train_vs_test.png")
