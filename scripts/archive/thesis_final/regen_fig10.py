"""Regenerate fig10_pit_before_after_ts.pdf with updated suptitle (no 'temperature scaling')."""
import numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
from scipy.stats import norm

PRIMARY = "#5B9BD5"; SECOND = "#ED7D31"; GRID = "#E5E5E5"; TICKDARK = "#404040"
ALPHA = 0.7

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.labelcolor": "black", "xtick.labelsize": 9, "ytick.labelsize": 9,
    "xtick.color": TICKDARK, "ytick.color": TICKDARK,
    "legend.fontsize": 9, "legend.frameon": False,
    "axes.edgecolor": "#666666", "axes.linewidth": 0.6,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

ROOT = "C:/Users/zamin/Downloads/ml_surrogates_thesis_final/ml_surrogates_thesis_final"
OUT  = f"{ROOT}/document/figures/new"
BASE = f"{ROOT}/code/data/TR-C_Benchmarks/"

mc = np.load(f"{BASE}point_net_transf_gat_8th_trial_lower_dropout/uq_results/mc_dropout_full_100graphs_mc30.npz")
y, yhat, sigma = mc["targets"], mc["predictions"], mc["uncertainties"]
T = 2.887
rng = np.random.RandomState(42)
idx = rng.choice(len(y), 500_000, replace=False)

def pit(se):
    z = (y - yhat) / np.maximum(se, 1e-10)
    return norm.cdf(z)[idx]

pit_raw = pit(sigma); pit_ts = pit(sigma * T)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
for ax, p, label, ks in [
    (axes[0], pit_raw, "Raw $\\sigma$ ($T=1$)", 0.245),
    (axes[1], pit_ts,  f"After $\\sigma$-scaling ($T={T}$)", 0.104),
]:
    ax.hist(p, bins=20, color=PRIMARY if ks > 0.2 else SECOND, edgecolor="white",
            alpha=ALPHA, density=True)
    ax.axhline(1.0, color="#5A9E6F", linestyle="--", linewidth=1.0, alpha=0.7,
               label="Uniform (ideal)")
    ax.set_xlabel("PIT value $F(y)$")
    ax.set_title(f"{label}   KS $= {ks:.3f}$, mean(PIT) $= {p.mean():.3f}$",
                 color="black", fontsize=10.5)
    ax.legend(loc="upper right")
    ax.set_xlim(0, 1)

axes[0].set_ylabel("Density")
fig.suptitle("PIT histograms before and after regression $\\sigma$-scaling",
             fontsize=11, color="black")
fig.tight_layout()
fig.savefig(f"{OUT}/fig10_pit_before_after_ts.pdf")
fig.savefig(f"{OUT}/fig10_pit_before_after_ts.png", dpi=300)
plt.close(fig)
print("Saved fig10_pit_before_after_ts.pdf and .png with updated suptitle.")
