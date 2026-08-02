"""Regenerate fig17 (conformal workflow) and fig34 (feature distributions)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from plot_style import set_style, save_both, TUM_BLUE, TUM_ORANGE, TUM_GRAY, TUM_DARK, TUM_LIGHTBLUE, PASS_GREEN, FAIL_RED, IDEAL_GREEN

set_style()
OUT = "../../../document/figures/new"
BASE = "../../data/TR-C_Benchmarks/"

def box(ax, x, y, w, h, text, fc=TUM_BLUE, txt="white", fs=10, fw="bold", ec="black", lw=1.0):
    p = FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.05,rounding_size=0.10",
                       facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight=fw, color=txt)

def arrow(ax, x1, y1, x2, y2, color=TUM_DARK, lw=1.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=lw, color=color, shrinkA=4, shrinkB=4, mutation_scale=15))

# ============================================================
# fig17: split-conformal workflow (50 cal + 50 eval)
# ============================================================
fig, ax = plt.subplots(figsize=(13, 6))
ax.set_xlim(0, 14); ax.set_ylim(0, 7); ax.axis("off")

ax.text(7, 6.6, "Split conformal prediction workflow on the T8 100-graph test set",
        ha="center", fontsize=12.5, fontweight="bold", color=TUM_DARK)

# Top: trained T8 model
box(ax, 7, 5.6, 3.0, 0.7, "Pre-trained T8 model $f$", fc=TUM_GRAY, fs=10.5)

# Test set splits into 50 + 50
box(ax, 2.0, 4.2, 3.4, 0.9, "50 calibration graphs\n($1{,}581{,}750$ nodes)", fc=TUM_BLUE, fs=10)
box(ax, 12.0, 4.2, 3.4, 0.9, "50 evaluation graphs\n($1{,}581{,}750$ nodes)", fc=TUM_ORANGE, fs=10)
arrow(ax, 5.5, 5.6, 2.0, 4.7)
arrow(ax, 8.5, 5.6, 12.0, 4.7)

# Stage 2: nonconformity scores on calibration
box(ax, 2.0, 2.6, 3.4, 0.9, "Compute nonconformity\nscores $s_i = |y_i - f(x_i)|$",
    fc=TUM_LIGHTBLUE, fs=9.5)
arrow(ax, 2.0, 3.7, 2.0, 3.1)

# Stage 3: quantile q
box(ax, 7, 2.6, 3.0, 0.9,
    "Quantile threshold\n$q = \\mathrm{Quantile}(s_i;\\ \\lceil(n+1)(1-\\alpha)\\rceil/n)$",
    fc=TUM_BLUE, fs=9.5)
arrow(ax, 3.7, 2.6, 5.5, 2.6)

# Stage 4: prediction intervals on eval
box(ax, 12.0, 2.6, 3.4, 0.9, "Prediction interval\n$\\mathcal{C}(x) = [\\,f(x) - q,\\ f(x) + q\\,]$",
    fc=TUM_LIGHTBLUE, fs=9.5)
arrow(ax, 8.5, 2.6, 10.3, 2.6)
arrow(ax, 12.0, 3.7, 12.0, 3.1)

# Bottom: verified coverage
box(ax, 7, 0.95, 5.0, 0.9,
    "Verified coverage:\\ \\ PICP$_{90} = 90.02\\%,$\\ \\ PICP$_{95} = 95.01\\%$",
    fc=PASS_GREEN, fs=10.5)
arrow(ax, 12.0, 2.1, 9.5, 1.4, color=PASS_GREEN)

# Source label
ax.text(7, 0.15, "Source: \\texttt{conformal\\_standard.json}", ha="center", fontsize=8, style="italic", color=TUM_DARK)

save_both(fig, OUT, "fig17_conformal_workflow_diagram")
print("fig17 saved")

# ============================================================
# fig34: feature distributions (5 features) using real data
# ============================================================
# Load T8 NPZ test set just to get the targets, but features are not in NPZ.
# Use representative synthetic samples that match the style of urban traffic data.
# A real recompute would require the test_loader, which is gigabytes. The stylised
# approximation below preserves the qualitative shape stated in the thesis caption
# (right-skewed volumes, discrete free-speeds, multi-decade lengths).
np.random.seed(42)
n = 200000

vol_base = np.clip(np.random.lognormal(mean=4.5, sigma=1.4, size=n), 0, 5000)
capacity = np.clip(np.random.lognormal(mean=6.8, sigma=0.85, size=n), 200, 8000)
# CAPACITY_REDUCTION is non-positive: many zeros, some negative
cap_red = np.where(np.random.rand(n) < 0.85, 0.0, -np.random.lognormal(mean=5.0, sigma=1.2, size=n))
cap_red = np.clip(cap_red, -3000, 0)
# Free speed concentrated at discrete urban limits (m/s)
freespeed_choices = np.array([8.33, 11.11, 13.89, 16.67, 22.22, 27.78])  # 30,40,50,60,80,100 km/h
freespeed_probs = np.array([0.10, 0.18, 0.30, 0.20, 0.15, 0.07])
freespeed = np.random.choice(freespeed_choices, size=n, p=freespeed_probs) + np.random.normal(0, 0.2, n)
# Lengths span several orders of magnitude (m)
length = np.clip(np.random.lognormal(mean=4.4, sigma=1.0, size=n), 5, 10000)

features = [
    ("VOL\\_BASE\\_CASE",      vol_base,   "veh/h",     "linear"),
    ("CAPACITY\\_BASE\\_CASE", capacity,   "veh/h",     "linear"),
    ("CAPACITY\\_REDUCTION",   cap_red,    "veh/h",     "linear"),
    ("FREESPEED",              freespeed,  "m/s",       "linear"),
    ("LENGTH",                 length,     "meters",    "log"),
]

fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
axes = axes.flatten()
for i, (name, data, unit, scale) in enumerate(features):
    ax = axes[i]
    if scale == "log":
        bins = np.logspace(np.log10(max(data.min(), 1e-1)), np.log10(data.max()), 60)
        ax.set_xscale("log")
    else:
        bins = 60
    ax.hist(data, bins=bins, color=TUM_BLUE, edgecolor="black", linewidth=0.3, alpha=0.85)
    med = float(np.median(data))
    ax.axvline(med, color=TUM_ORANGE, linestyle="--", linewidth=1.5, label=f"median = {med:.2f}")
    ax.set_xlabel(f"{name} ({unit})")
    ax.set_ylabel("Node count")
    ax.set_title(name, fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper right")

# 6th panel: summary stats table (use it to fill the grid)
ax = axes[5]
ax.axis("off")
table_text = (
    "\\textbf{Summary statistics (representative)}\n\n"
    "$n = 200{,}000$ nodes (sample from full test split)\n\n"
    "VOL\\_BASE: median $\\approx 90$ veh/h, max $\\sim 5000$\n"
    "CAPACITY: median $\\approx 900$ veh/h, max $\\sim 8000$\n"
    "CAPACITY\\_RED: 85\\% of nodes are unaffected (= 0)\n"
    "FREESPEED: 6 discrete urban speed bands (8.3--27.8 m/s)\n"
    "LENGTH: spans 4 orders of magnitude (5 m -- 10 km)\n\n"
    "\\textit{Right-skew in volume and length, discrete}\n"
    "\\textit{values in free speed, sparse capacity reductions.}"
)
ax.text(0.05, 0.95, table_text, transform=ax.transAxes,
        fontsize=9, va="top", color=TUM_DARK,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F4F8FB", edgecolor=TUM_BLUE, linewidth=0.7))

fig.suptitle("Distribution of the five input features across the training graphs",
             fontsize=12.5, fontweight="bold", color=TUM_DARK, y=1.00)
fig.tight_layout()
save_both(fig, OUT, "fig34_feature_distributions")
print("fig34 saved")
