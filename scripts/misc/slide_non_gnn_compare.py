"""Generate a slide-friendly 2-panel comparison of all 5 evaluated models on
point-prediction metrics: R^2 (left) and MAE (right).

Models: T8, Deep Ensemble, Random Forest, XGBoost, MLP.
Values come from the user's saved JSON / NPZ artifacts as reported in
Section 5.11 and Table 5.8 of the thesis.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib as mpl
import matplotlib.pyplot as plt
from plot_style import set_style, save_both

set_style()
OUT = "../../../document/figures/new"

# Light pastel TUM-aligned palette (matches fig22 + fig28)
T8_C    = "#DAD7CB"       # TUM accent grey  (baseline)
DE_C    = "#98C6EA"       # TUM accent light blue (GNN ensemble)
RF_C    = "#CDE5F2"       # very light blue (tree #1)
XGB_C   = "#F4B183"       # soft pale orange (winner)
MLP_C   = "#E8A5A5"       # light coral (weakest)
EDGE    = "#777777"

mpl.rcParams.update({"font.size": 11})

models = ["T8\n(GNN, primary UQ)", "Deep Ensemble\n(GNN, 5 members)",
          "Random Forest\n(cuML)", "XGBoost\n(default-style)", "MLP\n(sklearn-style)"]
r2_vals  = [0.5957, 0.6841, 0.6612, 0.7414, 0.4928]
mae_vals = [3.957, 3.485, 3.263, 2.774, 3.883]
colors   = [T8_C, DE_C, RF_C, XGB_C, MLP_C]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))


def annotate(ax, bars, vals, fmt, dy_frac=0.015, fc=None):
    ymax = max(vals)
    dy = ymax * dy_frac
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + dy, fmt.format(v),
                ha="center", fontsize=10, fontweight="bold", color="#222222")


# Panel (a) — R²
ax = axes[0]
bars = ax.bar(models, r2_vals, color=colors, edgecolor=EDGE, linewidth=0.6)
annotate(ax, bars, r2_vals, "{:.4f}")
ax.set_ylabel(r"Test $R^2$")
ax.set_title("(a) Point accuracy across all evaluated models")
ax.set_ylim(0, max(r2_vals) * 1.18)
# Highlight XGBoost as the winner with a small annotation arrow
ax.annotate("strongest\noverall",
            xy=(3, r2_vals[3] + 0.012), xytext=(3, r2_vals[3] + 0.10),
            ha="center", fontsize=9, color="#555555",
            arrowprops=dict(arrowstyle="->", color="#888888", lw=0.9))

# Panel (b) — MAE
ax = axes[1]
bars = ax.bar(models, mae_vals, color=colors, edgecolor=EDGE, linewidth=0.6)
annotate(ax, bars, mae_vals, "{:.3f}")
ax.set_ylabel("MAE (veh/h) — lower is better")
ax.set_title("(b) Mean absolute error across all evaluated models")
ax.set_ylim(0, max(mae_vals) * 1.18)

fig.suptitle("All 5 evaluated models on the same scenario-level held-out test set "
             r"(3{,}163{,}500 nodes, same 5 features)",
             fontsize=12)
fig.tight_layout()
save_both(fig, OUT, "slide_non_gnn_compare")
print("Saved slide_non_gnn_compare.pdf and .png")
print("Values used:")
for m, r, mae in zip(models, r2_vals, mae_vals):
    print(f"  {m.replace(chr(10), ' '):<35} R^2={r:.4f}  MAE={mae:.3f}")
