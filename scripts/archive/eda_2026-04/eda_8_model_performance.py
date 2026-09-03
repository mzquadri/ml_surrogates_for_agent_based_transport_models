"""EDA 8: Model Performance Charts - Clean Academic Style"""

import matplotlib

matplotlib.use("Agg")
import json
import numpy as np
import matplotlib.pyplot as plt

JSON_PATH = r"C:\Users\zamin\Downloads\Nazim\Thesis\Nazim\ml_surrogates_thesis_final\code\data\TR-C_Benchmarks\ALL_MODELS_COMPARISON\all_models_summary.json"
with open(JSON_PATH, "r") as f:
    data = json.load(f)

trials = data["trials"]
names, r2, mae, rmse, pearson, spearman, groups = [], [], [], [], [], [], []

for t in trials:
    num = t["trial_num"]
    names.append(f"T{num}")
    m = t["metrics"]
    r2.append(m["r2"])
    mae.append(m["mae"])
    rmse.append(m["rmse"])
    pearson.append(m["pearson"])
    spearman.append(m["spearman"])
    groups.append(t["split_group"])

x = np.arange(len(names))
group_colors = ["#4878A8" if g == "A" else "#D66B6B" for g in groups]

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

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(
    "Model Performance: 8 Trials Comparison", fontsize=16, fontweight="bold", y=0.98
)

metrics = [
    (r2, "R$^2$ Score", True),
    (mae, "MAE", False),
    (rmse, "RMSE", False),
    (pearson, "Pearson r", True),
    (spearman, "Spearman $\\rho$", True),
]

for idx, (vals, title, higher_better) in enumerate(metrics):
    ax = axes[idx // 3, idx % 3]
    bars = ax.bar(
        x,
        vals,
        color=group_colors,
        alpha=0.8,
        width=0.6,
        edgecolor="gray",
        linewidth=0.5,
    )

    best = np.argmax(vals) if higher_better else np.argmin(vals)
    bars[best].set_edgecolor("black")
    bars[best].set_linewidth(2)

    for i, (bar, v) in enumerate(zip(bars, vals)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + max(vals) * 0.01,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold" if i == best else "normal",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_title(title, fontweight="bold")

# Panel 6: Summary
ax = axes[1, 2]
ax.axis("off")
summary = (
    "Group A (blue): Trials 1-6\n"
    "  Split: 80/15/5, 50 test graphs\n"
    "  Best: T1 (R\u00b2=0.786)\n\n"
    "Group B (red): Trials 7-8\n"
    "  Split: 80/10/10, 100 test graphs\n"
    "  Best: T8 (R\u00b2=0.596)\n\n"
    "Trial 8 used for all UQ analysis\n"
    "(lower dropout=0.2, best Group B)"
)
ax.text(
    0.1,
    0.9,
    summary,
    transform=ax.transAxes,
    fontsize=11,
    va="top",
    family="serif",
    bbox=dict(boxstyle="round", facecolor="#f0f0f0", edgecolor="gray"),
)
ax.set_title("Summary", fontweight="bold")

from matplotlib.patches import Patch

fig.legend(
    handles=[
        Patch(facecolor="#4878A8", label="Group A (T1-T6)"),
        Patch(facecolor="#D66B6B", label="Group B (T7-T8)"),
    ],
    loc="lower center",
    ncol=2,
    fontsize=10,
    frameon=True,
)

fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(
    r"C:\Users\zamin\Downloads\Nazim\eda_8_model_performance.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="white",
)
plt.close()
print("Saved: eda_8_model_performance.png")
