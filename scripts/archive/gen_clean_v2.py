"""TASK 1: Regenerate F12, F13, F15, F35, F29 with cleaner academic style."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.stats import norm

# ----- shared style -----
PRIMARY = "#4A90C4"
ACCENT  = "#E07B39"
SUCCESS = "#5A9E6F"
NEUTRAL = "#888888"
GRID    = "#EEEEEE"
ALPHA   = 0.55

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.7,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

OUT  = "../../../document/figures/new"
BASE = "../../data/TR-C_Benchmarks/"

def save_both(fig, stem):
    pdf = os.path.join(OUT, f"{stem}.pdf")
    png = os.path.join(OUT, f"{stem}.png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    print(f"  saved {stem}")

# ============================================================
# F12 — Temperature scaling (2 panels: ECE curve + reliability)
# ============================================================
mc = np.load(BASE + "point_net_transf_gat_8th_trial_lower_dropout/uq_results/mc_dropout_full_100graphs_mc30.npz")
y, yhat, sigma = mc["targets"], mc["predictions"], mc["uncertainties"]
err = np.abs(y - yhat)
T_OPT = 2.887

# ECE curve (subsample for speed)
rng = np.random.RandomState(42)
idx = rng.choice(len(err), 500_000, replace=False)
err_s, sig_s = err[idx], sigma[idx]

T_grid = np.linspace(0.5, 5.0, 50)
def ece_at(T):
    se = sig_s * T
    grid = np.linspace(0.05, 0.95, 19)
    diffs = []
    for p in grid:
        z = norm.ppf(0.5 + p/2)
        diffs.append(abs((err_s <= z*se).mean() - p))
    return np.mean(diffs)
ece_grid = [ece_at(T) for T in T_grid]

# Reliability
nominal = np.linspace(0.05, 0.95, 19)
def cov(se):
    return np.array([(err <= norm.ppf(0.5 + p/2) * se).mean() for p in nominal])
obs_raw = cov(sigma)
obs_ts  = cov(sigma * T_OPT)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ax = axes[0]
ax.plot(T_grid, ece_grid, "-", color=PRIMARY, linewidth=1.6)
ax.axvline(T_OPT, color=ACCENT, linestyle="--", linewidth=1.0, alpha=0.85)
ax.text(T_OPT + 0.1, max(ece_grid)*0.85, f"$T^* = {T_OPT}$", fontsize=9, color=ACCENT)
ax.set_xlabel("Temperature $T$")
ax.set_ylabel("Expected Calibration Error")
ax.set_title("(a) ECE optimisation")

ax = axes[1]
ax.plot([0, 1], [0, 1], "-", color=NEUTRAL, linewidth=0.7, alpha=0.6)
ax.plot(nominal, obs_raw, "-",  color=PRIMARY, linewidth=1.5, alpha=0.85, label="Before  ($T = 1$)")
ax.plot(nominal, obs_ts,  "-",  color=ACCENT,  linewidth=1.5, alpha=0.85, label=f"After  ($T = {T_OPT}$)")
ax.set_xlabel("Nominal coverage")
ax.set_ylabel("Observed coverage")
ax.set_title("(b) Reliability before / after")
ax.legend(loc="upper left", frameon=False)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_aspect("equal")

fig.tight_layout()
save_both(fig, "fig12_temperature_scaling_4panel")

# ============================================================
# F13 — Conformal coverage at 80, 90, 95 (3 bars per group)
# ============================================================
levels   = [80, 90, 95]
raw_cov  = [40.09, 48.55, 54.85]
std_cov  = [80.01, 90.02, 95.01]
adapt_cov= [80.20, 89.87, 95.00]

x = np.arange(len(levels))
w = 0.27
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(x - w, raw_cov,   w, color=PRIMARY, edgecolor="black", linewidth=0.4, alpha=ALPHA, label="Raw MC Dropout")
ax.bar(x,     std_cov,   w, color=ACCENT,  edgecolor="black", linewidth=0.4, alpha=ALPHA, label="Standard conformal")
ax.bar(x + w, adapt_cov, w, color=SUCCESS, edgecolor="black", linewidth=0.4, alpha=ALPHA, label="Adaptive conformal")
for i, lvl in enumerate(levels):
    ax.plot([i - 1.5*w, i + 1.5*w], [lvl, lvl], "--", color=NEUTRAL, linewidth=0.8, alpha=0.7)
ax.text(2.55, 95 + 1.2, "Nominal target", fontsize=8, color=NEUTRAL, ha="right", style="italic")
ax.set_xticks(x); ax.set_xticklabels([f"{l}\\%" for l in levels])
ax.set_xlabel("Nominal coverage")
ax.set_ylabel("Achieved coverage (\\%)")
ax.set_ylim(0, 105)
ax.legend(loc="upper left", frameon=False)
fig.tight_layout()
save_both(fig, "fig13_conformal_coverage_nominal_vs_achieved")

# ============================================================
# F15 — Conditional coverage by decile (2 clean lines)
# ============================================================
adapt_decile = json.load(open(BASE + "point_net_transf_gat_8th_trial_lower_dropout/uq_results/phase3_results/adaptive_conformal_decile.json"))
deciles = adapt_decile["deciles"]
d_idx = [d["decile"] for d in deciles]
std_by_d = [d["standard_coverage_pct"] for d in deciles]
adapt_by_d = [d["adaptive_coverage_pct"] for d in deciles]

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(d_idx, std_by_d,   "--", color=PRIMARY, linewidth=1.4, alpha=0.9, label="Standard conformal")
ax.plot(d_idx, adapt_by_d, "-",  color=ACCENT,  linewidth=1.4, alpha=0.9, label="Adaptive conformal")
# Mark D1 and D10
for arr, c in [(std_by_d, PRIMARY), (adapt_by_d, ACCENT)]:
    ax.plot([1, 10], [arr[0], arr[-1]], "o", color=c, markersize=4)
ax.axhline(90, color=NEUTRAL, linestyle=":", linewidth=0.8, alpha=0.6)
ax.text(10.2, 90, "90\\%", fontsize=8, color=NEUTRAL, va="center")
ax.set_xticks(d_idx); ax.set_xticklabels([f"D{i}" for i in d_idx])
ax.set_xlabel(r"Uncertainty decile (low $\sigma$ $\rightarrow$ high $\sigma$)")
ax.set_ylabel(r"Conditional coverage (\%)")
ax.set_ylim(55, 102)
ax.legend(loc="lower left", frameon=False)
fig.tight_layout()
save_both(fig, "fig15_conditional_coverage_by_decile")

# ============================================================
# F35 — Decision framework (3-box vertical flowchart)
# ============================================================
fig, ax = plt.subplots(figsize=(6, 5.4))
ax.set_xlim(0, 6); ax.set_ylim(0, 8); ax.axis("off")

def soft_box(ax, x, y, w, h, text, color, txt_color="black"):
    p = FancyBboxPatch((x - w/2, y - h/2), w, h,
                       boxstyle="round,pad=0.04,rounding_size=0.10",
                       facecolor=color, edgecolor=color, alpha=ALPHA, linewidth=1.2)
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center", fontsize=10, color=txt_color)

soft_box(ax, 3, 6.7, 5.0, 1.1,
         "Low uncertainty\n($\\sigma < \\tau_\\mathrm{low}$)\n$\\rightarrow$ Accept prediction",
         SUCCESS)
soft_box(ax, 3, 4.0, 5.0, 1.1,
         "Medium uncertainty\n($\\tau_\\mathrm{low} \\leq \\sigma < \\tau_\\mathrm{high}$)\n$\\rightarrow$ Flag for review",
         ACCENT)
soft_box(ax, 3, 1.3, 5.0, 1.1,
         "High uncertainty\n($\\sigma \\geq \\tau_\\mathrm{high}$)\n$\\rightarrow$ Abstain  /  request simulation",
         "#C86E6E")

# Arrows
for y1, y2 in [(6.15, 4.55), (3.45, 1.85)]:
    ax.annotate("", xy=(3, y2), xytext=(3, y1),
                arrowprops=dict(arrowstyle="->", lw=1.0, color="#444"))

save_both(fig, "fig35_policy_decision_framework")

# ============================================================
# F29 — PointNetTransfGAT architecture (Elena Natterer style)
# ============================================================
fig, ax = plt.subplots(figsize=(7, 2.5))
ax.set_xlim(0, 14); ax.set_ylim(0, 4); ax.axis("off")

names = ["Input graph", "PointNet Conv", "Transformer Attn.", "GAT layers", "Output $\\Delta v$"]
sub   = ["5 node features\n+ 2D pos.",
         "2 layers,\nS=30 dropout",
         "2 layers,\n4 heads",
         "2 layers,\n1 head",
         "per-node\nveh/h"]
xs = np.linspace(1.0, 13.0, 5)
bw, bh = 2.2, 1.0

for x, name, s in zip(xs, names, sub):
    rect = FancyBboxPatch((x - bw/2, 2.4 - bh/2), bw, bh,
                          boxstyle="round,pad=0.02,rounding_size=0.08",
                          facecolor="white", edgecolor=PRIMARY, linewidth=1.1)
    ax.add_patch(rect)
    ax.text(x, 2.4, name, ha="center", va="center", fontsize=10, color="black")
    ax.text(x, 1.35, s, ha="center", va="top", fontsize=8, color=NEUTRAL, style="italic")

# arrows between boxes
for i in range(len(xs) - 1):
    ax.annotate("", xy=(xs[i+1] - bw/2 - 0.05, 2.4),
                xytext=(xs[i] + bw/2 + 0.05, 2.4),
                arrowprops=dict(arrowstyle="->", lw=0.9, color="#666"))

save_both(fig, "fig29_pointnet_architecture")

print("Task 1 done.")
