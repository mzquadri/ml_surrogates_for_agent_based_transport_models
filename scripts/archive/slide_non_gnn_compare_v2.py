"""Cleaner non-GNN comparison chart for slide use.
Horizontal bar chart layout = no x-label overlap; one panel per metric.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib as mpl
import matplotlib.pyplot as plt
from plot_style import set_style, save_both

set_style()
OUT = "../../../document/figures/new"

# Light pastel TUM-aligned palette
T8_C   = "#DAD7CB"
DE_C   = "#98C6EA"
RF_C   = "#CDE5F2"
XGB_C  = "#F4B183"
MLP_C  = "#E8A5A5"
EDGE   = "#777777"

mpl.rcParams.update({"font.size": 11})

# Sort by R^2 descending so the strongest sits at the top of the chart
data = [
    ("XGBoost",       0.7414, 2.774, XGB_C),
    ("Deep Ensemble", 0.6841, 3.485, DE_C),
    ("Random Forest", 0.6612, 3.263, RF_C),
    ("T8 (primary UQ)", 0.5957, 3.957, T8_C),
    ("MLP",           0.4928, 3.883, MLP_C),
]
labels = [d[0] for d in data]
r2     = [d[1] for d in data]
mae    = [d[2] for d in data]
cols   = [d[3] for d in data]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

# Horizontal bars — strongest at top
ypos = list(range(len(labels)))[::-1]   # reverse so top = strongest

# Panel (a) — R^2
ax = axes[0]
bars = ax.barh(ypos, r2, color=cols, edgecolor=EDGE, linewidth=0.6, height=0.65)
for y, v in zip(ypos, r2):
    ax.text(v + 0.008, y, f"{v:.4f}", va="center", fontsize=11, fontweight="bold", color="#222")
ax.set_yticks(ypos); ax.set_yticklabels(labels)
ax.set_xlabel(r"Test $R^2$ (higher is better)")
ax.set_title("(a) Point accuracy")
ax.set_xlim(0, max(r2) * 1.18)

# Panel (b) — MAE
ax = axes[1]
bars = ax.barh(ypos, mae, color=cols, edgecolor=EDGE, linewidth=0.6, height=0.65)
for y, v in zip(ypos, mae):
    ax.text(v + 0.05, y, f"{v:.3f}", va="center", fontsize=11, fontweight="bold", color="#222")
ax.set_yticks(ypos); ax.set_yticklabels(labels)
ax.set_xlabel("MAE (veh/h, lower is better)")
ax.set_title("(b) Mean absolute error")
ax.set_xlim(0, max(mae) * 1.18)

fig.suptitle("All 5 evaluated models on the same scenario-level held-out test set",
             fontsize=12)
fig.tight_layout()
save_both(fig, OUT, "slide_non_gnn_compare")
print("Saved slide_non_gnn_compare.pdf and .png (horizontal bars, no overlap)")
