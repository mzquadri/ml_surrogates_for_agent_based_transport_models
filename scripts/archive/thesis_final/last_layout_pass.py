"""Final cosmetic pass: F31, F29, F10, F22 — strict overlap control."""
import os, numpy as np
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch
from scipy.stats import norm

PRIMARY = "#5B9BD5"; SECOND = "#ED7D31"; TERTIARY = "#70AD47"
NEUTRAL = "#A5A5A5"; EDGE = "#888888"; GRID = "#E5E5E5"; TICKDARK = "#404040"
ALPHA = 0.7

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "normal",
    "axes.labelsize": 10, "axes.labelcolor": "black",
    "xtick.labelsize": 9, "ytick.labelsize": 9,
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

def save(fig, stem):
    fig.savefig(f"{OUT}/{stem}.pdf"); fig.savefig(f"{OUT}/{stem}.png", dpi=300); plt.close(fig)
    print(f"  saved {stem}")

# ============================================================
# FIX 1 — F31: features list as figure-level annotation BELOW both panels (no overlap)
# ============================================================
fig = plt.figure(figsize=(9, 5.0))
# Reserve a strip at the bottom for the features text; use GridSpec to split
gs = fig.add_gridspec(2, 2, height_ratios=[14, 1], hspace=0.05, wspace=0.05)

# Panel a
ax = fig.add_subplot(gs[0, 0])
ax.set_xlim(-0.5, 5.5); ax.set_ylim(-0.5, 4.5); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("(a) Original road network", fontsize=10.5)
intersections = {"A":(0.5,3.5),"B":(2.5,3.8),"C":(4.5,3.5),"D":(1.0,1.5),"E":(3.0,1.5),"F":(4.5,1.0)}
roads = [("A","B","r1"),("B","C","r2"),("A","D","r3"),("B","E","r4"),
         ("C","F","r5"),("D","E","r6"),("E","F","r7")]
for u, v, name in roads:
    ux, uy = intersections[u]; vx, vy = intersections[v]
    ax.annotate("", xy=(vx, vy), xytext=(ux, uy),
                arrowprops=dict(arrowstyle="->", lw=1.6, color=PRIMARY,
                                shrinkA=10, shrinkB=10, alpha=ALPHA))
    mx, my = (ux+vx)/2, (uy+vy)/2
    ax.text(mx + 0.08, my + 0.18, name, fontsize=8, color=PRIMARY, ha="center")
for name, (x, y) in intersections.items():
    ax.add_patch(Circle((x, y), 0.16, facecolor="black", edgecolor="black", zorder=3))
    ax.text(x, y, name, ha="center", va="center", fontsize=8, color="white", zorder=4)

# Panel b
ax = fig.add_subplot(gs[0, 1])
ax.set_xlim(-0.5, 5.5); ax.set_ylim(-0.5, 4.5); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("(b) Line graph representation", fontsize=10.5)
seg_pos = {n: ((intersections[u][0]+intersections[v][0])/2,
               (intersections[u][1]+intersections[v][1])/2)
           for (u, v, n) in roads}
adj = []
for i, (u1, v1, n1) in enumerate(roads):
    for u2, v2, n2 in roads[i+1:]:
        if {u1, v1} & {u2, v2}:
            adj.append((n1, n2))
for n1, n2 in adj:
    x1, y1 = seg_pos[n1]; x2, y2 = seg_pos[n2]
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-", lw=0.8, color=NEUTRAL,
                                shrinkA=8, shrinkB=8, alpha=0.6))
affected = {"r2", "r5"}
for name, (x, y) in seg_pos.items():
    fc = SECOND if name in affected else PRIMARY
    ax.add_patch(Circle((x, y), 0.20, facecolor=fc, edgecolor=EDGE, linewidth=0.5, alpha=ALPHA, zorder=3))
    ax.text(x, y, name, ha="center", va="center", fontsize=8, color="white", fontweight="bold", zorder=4)

# Bottom strip: features text spanning full figure width
ax_bot = fig.add_subplot(gs[1, :])
ax_bot.axis("off")
ax_bot.text(0.5, 0.5,
        "Each node carries 5 features:  volume, capacity, capacity reduction, free-flow speed, length    "
        "(orange in panel b = policy-affected segment)",
        transform=ax_bot.transAxes, ha="center", va="center", fontsize=8.5, color=TICKDARK,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=NEUTRAL, linewidth=0.5))

save(fig, "fig31_network_intro")

# ============================================================
# FIX 3 — F29: strict 7 x 2.8 in, no overflow
# ============================================================
fig, ax = plt.subplots(figsize=(7, 2.8))
ax.set_xlim(0, 7); ax.set_ylim(0, 4); ax.axis("off")

# Boxes: width 0.95 in, but in axes units (xlim=7) means 0.95 axes-width-units each
# Total: 6 boxes * 0.95 = 5.7, plus 5 gaps of 0.15 = 0.75 → 6.45 total → fits in 7 (0.275 margin each side)
boxes = [
    ("Input graph",          "5 features\nper node"),
    ("PointNet Conv x2",     "Local\ngeometry"),
    ("Transformer Conv x2",  "Self-attention"),
    ("GAT Conv x2",          "Graph\nattention"),
    ("MLP head",             "Regression"),
    ("Predicted $\\Delta v$", "veh/h\noutput"),
]
bw, bh = 0.95, 0.55
spacing = 0.21
xs = [0.275 + bw/2 + i * (bw + spacing) for i in range(6)]

for i, (x, (name, sub)) in enumerate(zip(xs, boxes)):
    border = SECOND if i == 5 else PRIMARY
    rect = FancyBboxPatch((x - bw/2, 2.0 - bh/2), bw, bh,
                          boxstyle="round,pad=0.015,rounding_size=0.04",
                          facecolor="white", edgecolor=border, linewidth=0.9)
    ax.add_patch(rect)
    ax.text(x, 2.0, name, ha="center", va="center", fontsize=8, color="black")
    sub_y = 1.10 if i % 2 == 0 else 2.95
    sub_va = "top" if i % 2 == 0 else "bottom"
    ax.text(x, sub_y, sub, ha="center", va=sub_va, fontsize=7, color=TICKDARK, style="italic")

for i in range(len(xs) - 1):
    ax.annotate("", xy=(xs[i+1] - bw/2 - 0.02, 2.0),
                xytext=(xs[i] + bw/2 + 0.02, 2.0),
                arrowprops=dict(arrowstyle="->", lw=0.6, color="black"))

ax.text(3.5, 0.30,
        "Dropout active in PointNet / Transformer / GAT layers during MC Dropout inference",
        ha="center", fontsize=7, color=NEUTRAL, style="italic")

fig.tight_layout(pad=0.2)
save(fig, "fig29_pointnet_architecture")

# ============================================================
# FIX 4 — F10 PIT histograms: ideal-uniform line, KS in upper-left clean box
# ============================================================
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
    (axes[1], pit_ts,  f"After $T = {T}$",       0.104),
]:
    ax.hist(p, bins=20, range=(0, 1), color=PRIMARY, edgecolor=EDGE, linewidth=0.5,
            alpha=ALPHA, density=True, label="PIT density")
    # Thinner uniform reference line
    ax.axhline(1.0, color="#7EBA82", linestyle="--", linewidth=1.0, alpha=0.85,
               label="Uniform (ideal)")
    ax.set_xlabel("PIT value $F(y)$")
    ax.set_title(f"{label}    mean(PIT) $= {p.mean():.3f}$",
                 color="black", fontsize=10.5)
    # Legend top-right (frameless, 9pt)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.set_xlim(0, 1)
    # KS box in upper-LEFT corner — well clear of any bars (which peak in the middle)
    ax.text(0.03, 0.96, f"KS = {ks:.3f}",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=9, color=SECOND,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#CCCCCC", linewidth=0.5))

axes[0].set_ylabel("Density")
fig.suptitle("PIT histograms before and after temperature scaling", fontsize=11, color="black")
fig.tight_layout()
save(fig, "fig10_pit_before_after_ts")

# ============================================================
# FIX 5 — F22 CQR R^2: trial names on x-axis, FAIL/PASS on a SECOND row below
# ============================================================
fig, ax = plt.subplots(figsize=(7.5, 5))
labels = ["T8", "T10-v1", "T10-v2", "T11"]
r2s = [0.596, 0.31, 0.41, 0.58]
status = ["BASELINE", "FAIL", "FAIL", "PASS"]
colors = [NEUTRAL, "#C66666", "#C66666", "#70AD47"]

x = np.arange(len(labels))
bars = ax.bar(x, r2s, color=colors, edgecolor=EDGE, linewidth=0.5, alpha=ALPHA, width=0.55)
for b, v in zip(bars, r2s):
    ax.text(b.get_x() + b.get_width()/2, v + 0.030, f"{v:.3f}",
            ha="center", fontsize=10)

# Primary tick row: trial names
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10, color="black")

# Secondary row: status labels at y = -0.10 in axes coords (well below the trial names)
status_colours = {"BASELINE": NEUTRAL, "FAIL": "#C66666", "PASS": "#70AD47"}
for xi, s in zip(x, status):
    ax.text(xi, -0.10, s, transform=ax.get_xaxis_transform(),
            ha="center", va="top",
            fontsize=9, color=status_colours[s], style="italic")

# Gate line
ax.axhline(0.57, color=NEUTRAL, linestyle="--", linewidth=1.0, alpha=0.8)
ax.text(3.45, 0.585, "Gate $R^2 \\geq 0.57$", color=TICKDARK, fontsize=8.5, ha="right")

ax.set_ylabel("$R^2$")
ax.set_title("CQR variants: $R^2$ gate evaluation")
ax.set_ylim(0, 0.7)

# Give room for the second-row annotations
plt.subplots_adjust(bottom=0.20)
save(fig, "fig22_cqr_r2_progression")

print("Done.")
