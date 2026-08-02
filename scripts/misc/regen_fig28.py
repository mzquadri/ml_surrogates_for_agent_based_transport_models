"""Regenerate fig28_stratified_uq_quartiles.pdf with all 4 panels:
  (a) MAE per quartile
  (b) Spearman rho per quartile
  (c) mean |delta v| per quartile (shows what each quartile actually represents)
  (d) mean sigma per quartile (shows sigma rises but not enough to track |Delta v|)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from plot_style import set_style, save_both

set_style()
OUT = "../../../document/figures/new"
BASE = "../../data/TR-C_Benchmarks/"

# Unified thesis palette: all quartiles in light blue, Q4 highlighted in light orange
Q1_COLOR = "#A8C8E8"      # lighter blue (Q1, easiest)
Q2_COLOR = "#8AB6DD"      # medium-light blue (Q2)
Q3_COLOR = "#6CA4D3"      # medium blue (Q3)
Q4_COLOR = "#ED7D31"      # light orange (Q4, highlighted as hardest)
EDGE     = "#777777"
GUIDE    = "#999999"

mc = np.load(BASE + "point_net_transf_gat_8th_trial_lower_dropout/uq_results/mc_dropout_full_100graphs_mc30.npz")
y = mc["targets"]
yhat = mc["predictions"]
sigma = mc["uncertainties"]
err = np.abs(y - yhat)
abs_y = np.abs(y)
n_total = len(abs_y)
ranks = np.argsort(np.argsort(abs_y))
labels = ["Q1\n(smallest |Δv|)", "Q2", "Q3", "Q4\n(largest |Δv|)"]
boundaries = [0, n_total // 4, n_total // 2, 3 * n_total // 4, n_total]
colors = [Q1_COLOR, Q2_COLOR, Q3_COLOR, Q4_COLOR]

mae_by_q, rho_by_q, abs_y_by_q, sigma_by_q = [], [], [], []
for k in range(4):
    mask = (ranks >= boundaries[k]) & (ranks < boundaries[k + 1])
    e, s = err[mask], sigma[mask]
    mae_by_q.append(float(e.mean()))
    rho_by_q.append(float(spearmanr(s, e)[0]))
    abs_y_by_q.append(float(abs_y[mask].mean()))
    sigma_by_q.append(float(s.mean()))


def annotate_bars(ax, bars, vals, fmt, dy_frac=0.02):
    """Place a numerical label slightly above each bar."""
    ymax = max(vals) if max(vals) > 0 else 1
    dy = ymax * dy_frac
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + dy, fmt.format(v),
                ha="center", fontsize=10, fontweight="bold")


fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# Panel (a): MAE per quartile
ax = axes[0, 0]
bars = ax.bar(labels, mae_by_q, color=colors, edgecolor=EDGE, linewidth=0.6)
annotate_bars(ax, bars, mae_by_q, "{:.2f}")
ax.set_ylabel("MAE (veh/h)")
ax.set_title("(a) Prediction error by quartile")
ax.set_ylim(0, max(mae_by_q) * 1.18)

# Panel (b): Spearman rho per quartile
ax = axes[0, 1]
bars = ax.bar(labels, rho_by_q, color=colors, edgecolor=EDGE, linewidth=0.6)
annotate_bars(ax, bars, rho_by_q, "{:.3f}")
ax.axhline(0.4820, color=GUIDE, linestyle="--", linewidth=1.0,
           label=r"Aggregate $\rho = 0.4820$")
ax.set_ylabel(r"Spearman $\rho$ within quartile")
ax.set_title("(b) Uncertainty ranking quality by quartile")
ax.legend(fontsize=9, loc="upper right")
ax.set_ylim(0, max(rho_by_q) * 1.25)

# Panel (c): mean |delta v| per quartile
ax = axes[1, 0]
bars = ax.bar(labels, abs_y_by_q, color=colors, edgecolor=EDGE, linewidth=0.6)
annotate_bars(ax, bars, abs_y_by_q, "{:.2f}")
ax.set_ylabel(r"Mean $|\Delta v|$ (veh/h)")
ax.set_title(r"(c) Mean true $|\Delta v|$ per quartile")
ax.set_ylim(0, max(abs_y_by_q) * 1.18)

# Panel (d): mean sigma per quartile
ax = axes[1, 1]
bars = ax.bar(labels, sigma_by_q, color=colors, edgecolor=EDGE, linewidth=0.6)
annotate_bars(ax, bars, sigma_by_q, "{:.2f}")
ax.set_ylabel(r"Mean $\sigma$ (veh/h)")
ax.set_title(r"(d) Mean MC Dropout $\sigma$ per quartile")
ax.set_ylim(0, max(sigma_by_q) * 1.25)

fig.suptitle("Stratified UQ analysis (T8): how MAE, ranking quality, target magnitude, and uncertainty vary across quartiles",
             fontsize=12)
fig.tight_layout()
save_both(fig, OUT, "fig28_stratified_uq_quartiles")

# Print stats
print("Stratified UQ — verified values:")
print(f"{'Quartile':<6} {'n':>10} {'MAE':>8} {'rho':>8} {'mean|y|':>10} {'mean_sigma':>12}")
for k in range(4):
    n = (boundaries[k + 1] - boundaries[k])
    print(f"{['Q1', 'Q2', 'Q3', 'Q4'][k]:<6} {n:>10,} {mae_by_q[k]:>8.3f} {rho_by_q[k]:>8.4f} {abs_y_by_q[k]:>10.3f} {sigma_by_q[k]:>12.4f}")
print()
print("F28 regenerated with 4 panels.")
