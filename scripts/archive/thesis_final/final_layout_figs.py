"""Final layout pass: 8 figure regenerations."""
import sys, os, glob, json
import numpy as np, torch
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch

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
# FIX 1 — F31 network intro, features text ABOVE right panel, no overflow
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))

# Panel a: original road network with intersections
ax = axes[0]
ax.set_xlim(-0.5, 5.5); ax.set_ylim(-0.5, 4.5); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("(a) Original road network", fontsize=10.5)
intersections = {"A":(0.5,3.5), "B":(2.5,3.8), "C":(4.5,3.5),
                 "D":(1.0,1.5), "E":(3.0,1.5), "F":(4.5,1.0)}
roads = [("A","B","r1"), ("B","C","r2"), ("A","D","r3"),
         ("B","E","r4"), ("C","F","r5"), ("D","E","r6"), ("E","F","r7")]
for u, v, name in roads:
    ux, uy = intersections[u]; vx, vy = intersections[v]
    ax.annotate("", xy=(vx, vy), xytext=(ux, uy),
                arrowprops=dict(arrowstyle="->", lw=1.6, color=PRIMARY, shrinkA=10, shrinkB=10, alpha=ALPHA))
    mx, my = (ux+vx)/2, (uy+vy)/2
    ax.text(mx + 0.08, my + 0.18, name, fontsize=8, color=PRIMARY, ha="center")
for name, (x, y) in intersections.items():
    ax.add_patch(Circle((x, y), 0.16, facecolor="black", edgecolor="black", zorder=3))
    ax.text(x, y, name, ha="center", va="center", fontsize=8, color="white", zorder=4)

# Panel b: line graph
ax = axes[1]
ax.set_xlim(-0.5, 5.5); ax.set_ylim(-0.5, 4.8); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("(b) Line graph representation", fontsize=10.5)

# Features text ABOVE the right panel (using axes coords for safe placement)
ax.text(0.5, 1.06,
        "Each node carries 5 features:\nvolume, capacity, capacity reduction, free-flow speed, length",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5, color=TICKDARK,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=NEUTRAL, linewidth=0.5))

seg_pos = {name: ((intersections[u][0]+intersections[v][0])/2,
                  (intersections[u][1]+intersections[v][1])/2)
           for (u, v, name) in roads}
adj = []
for i, (u1, v1, n1) in enumerate(roads):
    for u2, v2, n2 in roads[i+1:]:
        if {u1, v1} & {u2, v2}:
            adj.append((n1, n2))
for n1, n2 in adj:
    x1, y1 = seg_pos[n1]; x2, y2 = seg_pos[n2]
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-", lw=0.8, color=NEUTRAL, shrinkA=8, shrinkB=8, alpha=0.6))

affected = {"r2", "r5"}
for name, (x, y) in seg_pos.items():
    fc = SECOND if name in affected else PRIMARY
    ax.add_patch(Circle((x, y), 0.20, facecolor=fc, edgecolor=EDGE, linewidth=0.5, alpha=ALPHA, zorder=3))
    ax.text(x, y, name, ha="center", va="center", fontsize=8, color="white", fontweight="bold", zorder=4)

ax.text(0.02, -0.04, "Orange = policy-affected segment",
        transform=ax.transAxes, fontsize=7, color=SECOND, style="italic")

fig.tight_layout()
save(fig, "fig31_network_intro")

# ============================================================
# FIX 3 — F34 feature distributions, NO stats overlay
# ============================================================
DATA_DIR = f"{ROOT}/code/data/train_data/dist_not_connected_10k_1pct"
batches = sorted(glob.glob(f"{DATA_DIR}/datalist_batch_*.pt"))[:2]
all_x = []
sys.path.insert(0, f"{ROOT}/code/scripts")
try:
    from data_preprocessing.process_simulations_for_gnn import EdgeFeatures
except ImportError: pass
for b in batches:
    d = torch.load(b, map_location="cpu", weights_only=False)
    if isinstance(d, list):
        for g in d:
            all_x.append(g.x.numpy() if hasattr(g.x, "numpy") else np.asarray(g.x))
X = np.concatenate(all_x, axis=0)
USE = [0, 1, 2, 3, 5]
NAMES = ["VOL\\_BASE\\_CASE", "CAPACITY\\_BASE\\_CASE", "CAPACITY\\_REDUCTION", "FREESPEED", "LENGTH"]
UNITS = ["veh/h", "veh/h", "veh/h", "m/s", "m"]
raw = X[:, USE]

# Layout: 2 rows × 3 cols, last cell empty (NO stats box, just hide axis)
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
axes = axes.flatten()
for i, (name, unit) in enumerate(zip(NAMES, UNITS)):
    ax = axes[i]
    d = raw[:, i]
    ax.hist(d, bins=60, color=PRIMARY, edgecolor=EDGE, linewidth=0.3, alpha=ALPHA)
    med = float(np.median(d))
    ax.axvline(med, color=SECOND, linestyle="--", linewidth=1.4, alpha=0.85,
               label=f"median = {med:.2f}")
    ax.set_xlabel(f"{name} ({unit})")
    ax.set_ylabel("Node count")
    ax.set_title(name, fontsize=10)
    ax.legend(loc="upper right")

# Hide the 6th panel completely (no stats box, statistics now in companion table)
axes[5].axis("off")

fig.suptitle("Distributions of the five input features (raw units)",
             fontsize=11, color="black", y=1.0)
fig.tight_layout()
save(fig, "fig34_feature_distributions")

# ============================================================
# FIX 4 — F29 architecture diagram, smaller boxes, staggered subtext
# ============================================================
fig, ax = plt.subplots(figsize=(8, 2.8))
ax.set_xlim(0, 16); ax.set_ylim(0, 4); ax.axis("off")

boxes = [
    ("Input graph",          "Nodes = road segments,\n5 features per node"),
    ("PointNet Conv x2",     "Encodes local geometry\nfrom start/end points"),
    ("Transformer Conv x2",  "Self-attention\nacross network"),
    ("GAT Conv x2",          "Graph attention\nlayers"),
    ("MLP head",             "Final regression head"),
    ("Predicted $\\Delta v$","Per-node output (veh/h)"),
]
xs = np.linspace(1.5, 14.5, 6)
bw, bh = 2.2, 0.7

for i, (x, (name, sub)) in enumerate(zip(xs, boxes)):
    border = SECOND if i == 5 else PRIMARY
    rect = FancyBboxPatch((x - bw/2, 2.2 - bh/2), bw, bh,
                          boxstyle="round,pad=0.02,rounding_size=0.05",
                          facecolor="white", edgecolor=border, linewidth=1.0)
    ax.add_patch(rect)
    ax.text(x, 2.2, name, ha="center", va="center", fontsize=9, color="black")
    # Stagger sub-text alternately above and below
    sub_y = 2.95 if i % 2 == 0 else 1.45
    sub_va = "bottom" if i % 2 == 0 else "top"
    ax.text(x, sub_y, sub, ha="center", va=sub_va, fontsize=7.5, color=TICKDARK, style="italic")

for i in range(len(xs) - 1):
    ax.annotate("", xy=(xs[i+1] - bw/2 - 0.05, 2.2),
                xytext=(xs[i] + bw/2 + 0.05, 2.2),
                arrowprops=dict(arrowstyle="->", lw=0.7, color="black"))

# Dropout note in a single line at very bottom
ax.text(8, 0.25, "(dropout active in PointNet/Transformer/GAT layers during MC Dropout inference)",
        ha="center", fontsize=7.5, color=NEUTRAL, style="italic")

save(fig, "fig29_pointnet_architecture")

# ============================================================
# FIX 5 — F10 PIT histograms before/after TS (lighter colours)
# ============================================================
mc = np.load(f"{BASE}point_net_transf_gat_8th_trial_lower_dropout/uq_results/mc_dropout_full_100graphs_mc30.npz")
y, yhat, sigma = mc["targets"], mc["predictions"], mc["uncertainties"]
from scipy.stats import norm
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
            alpha=ALPHA, density=True)
    ax.axhline(1.0, color="#5A9E6F", linestyle="--", linewidth=1.0, alpha=0.7,
               label="Uniform (ideal)")
    ax.set_xlabel("PIT value $F(y)$")
    ax.set_title(f"{label}   KS $= {ks:.3f}$, mean(PIT) $= {p.mean():.3f}$",
                 color="black", fontsize=10.5)
    ax.legend(loc="upper center")
    ax.set_xlim(0, 1)
    # Highlight KS in muted orange via a small marker
    ax.text(0.5, 0.96, f"KS = {ks:.3f}", transform=ax.transAxes, ha="center", va="top",
            fontsize=9, color=SECOND,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor=SECOND, linewidth=0.5))

axes[0].set_ylabel("Density")
fig.suptitle("PIT histograms before and after regression $\\sigma$-scaling", fontsize=11, color="black")
fig.tight_layout()
save(fig, "fig10_pit_before_after_ts")

# ============================================================
# FIX 6 — F00 master summary (verify all 12 models, axes, etc.)
# ============================================================
import matplotlib.patches as mpatches
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

# Panel a: ALL 12 models, sorted by R^2 descending
models = [
    ("T1", 0.7860, "point"), ("T2", 0.5117, "point"), ("T3", 0.2246, "point"),
    ("T4", 0.2426, "point"), ("T5", 0.5553, "point"), ("T6", 0.5223, "point"),
    ("T7", 0.5471, "point"), ("T8", 0.5957, "point"),
    ("T9", 0.4991, "uq"),    ("T10", 0.4057, "uq"),   ("T11", 0.5835, "uq"),
    ("Deep Ens.", 0.6841, "ens"),
]
colour_for = {"point": PRIMARY, "uq": SECOND, "ens": TERTIARY}
sm = sorted(models, key=lambda m: m[1], reverse=True)
labels = [m[0] for m in sm]; vals = [m[1] for m in sm]
cols = [colour_for[m[2]] for m in sm]
ax = axes[0]
ax.barh(np.arange(len(labels)), vals, color=cols, alpha=ALPHA, edgecolor=EDGE, linewidth=0.4)
ax.set_yticks(np.arange(len(labels))); ax.set_yticklabels(labels)
ax.invert_yaxis()
ax.set_xlabel("Test $R^2$")
ax.set_title("(a) Point accuracy across all 12 models")
ax.set_xlim(0, 0.85)
handles = [mpatches.Patch(color=PRIMARY, alpha=ALPHA, label="Point prediction"),
           mpatches.Patch(color=SECOND,  alpha=ALPHA, label="UQ extension"),
           mpatches.Patch(color=TERTIARY, alpha=ALPHA, label="Ensemble")]
ax.legend(handles=handles, loc="lower right")

# Panel b: trade-off scatter
uqm = [
    ("T8 MC Dropout",      0.5857, 0.4820, 30,  PRIMARY),
    ("T8 MC avg, 5 seeds", 0.5865, 0.4908, 150, PRIMARY),
    ("Multi-model ens.",   0.5656, 0.4333, 5,   PRIMARY),
    ("T9 hetero.",         0.4991, 0.4797, 30,  SECOND),
    ("T11 CQR frozen",     0.5835, 0.2943, 1,   SECOND),
    ("Deep Ensemble",      0.6841, 0.3997, 5,   TERTIARY),
]
ax = axes[1]
for label, r2, rho, cost, col in uqm:
    s = 30 + 60 * np.log10(cost + 1)
    ax.scatter(r2, rho, s=s, color=col, alpha=ALPHA, edgecolor=EDGE, linewidth=0.4)
offsets = {
    "T8 MC Dropout":      (0.005,  0.005),
    "T8 MC avg, 5 seeds": (0.005, -0.012),
    "Multi-model ens.":   (-0.04,  0.012),
    "T9 hetero.":         (0.005, -0.012),
    "T11 CQR frozen":     (0.005,  0.005),
    "Deep Ensemble":      (-0.07, -0.018),
}
for label, r2, rho, cost, col in uqm:
    dx, dy = offsets[label]
    ax.text(r2 + dx, rho + dy, label, fontsize=8, color="black")
ax.set_xlabel("Test $R^2$")
ax.set_ylabel(r"Spearman $\rho$ ($\sigma$ vs $|err|$)")
ax.set_title("(b) UQ quality vs accuracy (size $\\propto$ inference cost)")
ax.set_xlim(0.40, 0.72); ax.set_ylim(0.27, 0.53)

# Panel c: k95 calibration
labels_k = ["Ideal\nGaussian", "T9 hetero.", "T8 + TS", "T8 MC", "Deep Ens."]
vals_k = [1.96, 2.84, 4.04, 11.66, 15.18]
cols_k = [TERTIARY, SECOND, "#9CC4E4", PRIMARY, NEUTRAL]
ax = axes[2]
ax.bar(labels_k, vals_k, color=cols_k, alpha=ALPHA, edgecolor=EDGE, linewidth=0.4)
ax.axhline(1.96, color=TERTIARY, linestyle=":", linewidth=0.8, alpha=0.7)
for i, v in enumerate(vals_k):
    ax.text(i, v + 0.30, f"{v:.2f}", ha="center", fontsize=9)
ax.set_ylabel(r"Empirical $k_{95}$ (lower better)")
ax.set_title("(c) Calibration of UQ methods")
ax.set_ylim(0, 17.5)

fig.tight_layout()
save(fig, "fig00_all_models_summary")

# ============================================================
# FIX 7 — F19 T9 decomposition (clearer)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6),
                                gridspec_kw={"width_ratios": [1.4, 1]})
labels = ["Aleatoric", "Epistemic", "Total"]
values = [4.657, 1.099, 4.823]
colors = [PRIMARY, SECOND, NEUTRAL]
bars = ax1.bar(labels, values, color=colors, edgecolor=EDGE, linewidth=0.5, alpha=ALPHA)
for b, v in zip(bars, values):
    ax1.text(b.get_x() + b.get_width()/2, v + 0.15, f"{v:.3f}", ha="center", fontsize=10)
ax1.set_ylabel("Mean uncertainty (vehicles per hour)")
ax1.set_title("(a) Mean uncertainty by component (Trial 9)", fontsize=10.5)
ax1.set_ylim(0, max(values) * 1.30)
ax1.text(0.5, -0.22, "Aleatoric to epistemic ratio: 4.24",
         transform=ax1.transAxes, ha="center", fontsize=9.5, color="black",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=NEUTRAL, linewidth=0.6))

sizes = [99.85, 0.15]
labels_pie = ["Aleatoric-dominant\n(99.85\\%)", "Epistemic-dominant\n(0.15\\%)"]
ax2.pie(sizes, labels=labels_pie, colors=[PRIMARY, SECOND], startangle=90,
        wedgeprops={"edgecolor": EDGE, "linewidth": 0.5, "alpha": ALPHA},
        textprops={"fontsize": 9.5}, labeldistance=1.15)
ax2.set_title("(b) Per-node dominant component", fontsize=10.5)

fig.suptitle("Trial 9 uncertainty decomposition", fontsize=11)
fig.tight_layout()
save(fig, "fig19_t9_uncertainty_decomposition")

# ============================================================
# FIX 8 — F22 CQR R^2 progression (no overlap)
# ============================================================
fig, ax = plt.subplots(figsize=(7.5, 5))
labels = ["T8", "T10-v1", "T10-v2", "T11"]
r2s = [0.596, 0.31, 0.41, 0.58]
status = ["BASELINE", "FAIL", "FAIL", "PASS"]
colors = [NEUTRAL, "#D88080", "#D88080", "#7EBA82"]

x = np.arange(len(labels))
bars = ax.bar(x, r2s, color=colors, edgecolor=EDGE, linewidth=0.5, alpha=ALPHA, width=0.55)
for b, v in zip(bars, r2s):
    ax.text(b.get_x() + b.get_width()/2, v + 0.030, f"{v:.3f}",
            ha="center", fontsize=10)

# Below-bar status text
for xi, s in zip(x, status):
    if s == "FAIL":
        ax.text(xi, -0.04, s, ha="center", va="top", fontsize=9, color="#C66666",
                fontweight="bold", transform=ax.get_xaxis_transform())
    elif s == "PASS":
        ax.text(xi, -0.04, s, ha="center", va="top", fontsize=9, color="#5A9E6F",
                fontweight="bold", transform=ax.get_xaxis_transform())

# Gate line
ax.axhline(0.57, color=NEUTRAL, linestyle="--", linewidth=1.0, alpha=0.8)
ax.text(3.45, 0.585, "Gate $R^2 \\geq 0.57$", color=TICKDARK, fontsize=8.5, ha="right")

ax.set_xticks(x); ax.set_xticklabels(labels, rotation=0)
ax.set_ylabel("$R^2$")
ax.set_title("CQR variants: $R^2$ gate evaluation")
ax.set_ylim(0, 0.7)
fig.tight_layout()
save(fig, "fig22_cqr_r2_progression")

# ============================================================
# FIX 9 — F35 horizontal flowchart, larger and clearer
# ============================================================
fig, ax = plt.subplots(figsize=(10, 4))
ax.set_xlim(0, 13); ax.set_ylim(-1.8, 4.2); ax.axis("off")

ax.text(6.5, 4.0, "Three-tier deployment policy based on uncertainty percentile",
        ha="center", fontsize=11, color="black")

def tier(ax, x, w, color, top, mid, bot, icon):
    rect = FancyBboxPatch((x - w/2, 0.6), w, 2.6,
                          boxstyle="round,pad=0.06,rounding_size=0.10",
                          facecolor=color, edgecolor=NEUTRAL, alpha=ALPHA, linewidth=0.7)
    ax.add_patch(rect)
    ax.text(x, 2.85, top, ha="center", va="center", fontsize=11.5, color="black", fontweight="bold")
    ax.text(x, 2.05, icon, ha="center", va="center", fontsize=18, color="black")
    ax.text(x, 1.40, mid, ha="center", va="center", fontsize=9.5, color="black")
    ax.text(x, 0.90, bot, ha="center", va="center", fontsize=9, color=TICKDARK, style="italic")

tier(ax, 2.5, 3.6, "#7EBA82", "ACCEPT",          "0\\% to 50\\% percentile",  "Use prediction directly", "OK")
tier(ax, 6.5, 3.6, "#F4A460", "FLAG FOR REVIEW", "50\\% to 90\\% percentile", "Send to expert verification", "!")
tier(ax, 10.5,3.6, "#D88080", "REJECT",          "90\\% to 100\\% percentile","Re-run MATSim simulation", "X")

# Bottom uncertainty axis
ax.annotate("", xy=(12.6, 0.0), xytext=(0.4, 0.0),
            arrowprops=dict(arrowstyle="->", lw=0.9, color="black"))
for tx, lbl in [(0.7, "0\\%"), (4.5, "50\\%"), (8.5, "90\\%"), (12.4, "100\\%")]:
    ax.plot([tx, tx], [-0.08, 0.08], color="black", linewidth=0.7)
    ax.text(tx, -0.30, lbl, ha="center", va="top", fontsize=8.5, color=TICKDARK)
ax.text(6.5, -1.05, "Increasing predicted uncertainty",
        ha="center", fontsize=10, color=TICKDARK, style="italic")

save(fig, "fig35_policy_decision_framework")

print("Done.")
