"""Final figure regeneration: F31, F34 (verified), F29, F19, F22, F35."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle

PRIMARY  = "#5B9BD5"
SECOND   = "#ED7D31"
TERTIARY = "#70AD47"
NEUTRAL  = "#A5A5A5"
GRID     = "#E5E5E5"
EDGE     = "#888888"
TICKDARK = "#404040"
ALPHA    = 0.7

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "normal",
    "axes.labelsize": 10,
    "axes.labelcolor": "black",
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

OUT = "../../../document/figures/new"

def save_both(fig, stem):
    fig.savefig(f"{OUT}/{stem}.pdf")
    fig.savefig(f"{OUT}/{stem}.png", dpi=300)
    plt.close(fig)
    print(f"  saved {stem}")

# ============================================================
# F31 — clean 2D network + line graph
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

# Panel a: original road network with intersections + directed segments
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
                arrowprops=dict(arrowstyle="->", lw=1.6, color=PRIMARY,
                                shrinkA=10, shrinkB=10, alpha=ALPHA))
    mx, my = (ux + vx) / 2, (uy + vy) / 2
    ax.text(mx + 0.08, my + 0.18, name, fontsize=8, color=PRIMARY, ha="center")

for name, (x, y) in intersections.items():
    ax.add_patch(Circle((x, y), 0.16, facecolor="black", edgecolor="black", zorder=3))
    ax.text(x, y, name, ha="center", va="center", fontsize=8, color="white", zorder=4)

# Panel b: line graph
ax = axes[1]
ax.set_xlim(-0.5, 5.5); ax.set_ylim(-1.7, 4.5); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("(b) Line graph representation", fontsize=10.5)

seg_pos = {name: ((intersections[u][0]+intersections[v][0])/2,
                  (intersections[u][1]+intersections[v][1])/2)
           for (u, v, name) in roads}

# Adjacency edges in line graph: two segments share an intersection
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

# Highlight r2 and r5 as policy-affected
affected = {"r2", "r5"}
for name, (x, y) in seg_pos.items():
    fc = SECOND if name in affected else PRIMARY
    ax.add_patch(Circle((x, y), 0.20, facecolor=fc, edgecolor=EDGE, linewidth=0.5, alpha=ALPHA, zorder=3))
    ax.text(x, y, name, ha="center", va="center", fontsize=8, color="white", fontweight="bold", zorder=4)

# Features box below
ax.add_patch(FancyBboxPatch((0.0, -1.5), 5.0, 0.95,
                             boxstyle="round,pad=0.04", facecolor="white",
                             edgecolor=NEUTRAL, linewidth=0.6))
ax.text(2.5, -0.85, "Each node carries: volume, capacity, capacity reduction,",
        ha="center", fontsize=8, color=TICKDARK)
ax.text(2.5, -1.20, "free-flow speed, length", ha="center", fontsize=8, color=TICKDARK)
ax.text(0.15, 0.05, "(orange = policy-affected segment)", fontsize=7, color=SECOND, style="italic")

fig.tight_layout()
save_both(fig, "fig31_network_intro")

# ============================================================
# F34 — verified feature distributions
# ============================================================
data = np.load("feature_data.npz")
vol = data["vol"]; cap = data["cap"]; cap_red = data["cap_red"]
freespeed = data["freespeed"]; length = data["length"]
stats = json.load(open("feature_stats.json"))

# LENGTH appears stored normalised; use it as-is but in original-scale display we'd convert.
# We just plot its distribution.

panels = [
    ("VOL\\_BASE\\_CASE", vol,        "veh/h", "linear"),
    ("CAPACITY\\_BASE\\_CASE", cap,   "veh/h", "linear"),
    ("CAPACITY\\_REDUCTION", cap_red, "veh/h", "linear"),
    ("FREESPEED", freespeed,          "m/s",   "linear"),
    ("LENGTH (normalised)", length,   "(z)",   "linear"),
]
fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
axes = axes.flatten()
for i, (name, d, unit, scale) in enumerate(panels):
    ax = axes[i]
    bins = 60
    ax.hist(d, bins=bins, color=PRIMARY, edgecolor=EDGE, linewidth=0.3, alpha=ALPHA)
    med = float(np.median(d))
    ax.axvline(med, color=SECOND, linestyle="--", linewidth=1.4, alpha=0.85,
               label=f"median = {med:.2f}")
    ax.set_xlabel(f"{name} ({unit})")
    ax.set_ylabel("Node count")
    ax.set_title(name, fontsize=10)
    ax.legend(loc="upper right")

# Stats summary panel
ax = axes[5]; ax.axis("off")
nz_pct_red = stats["CAPACITY_REDUCTION"]["nz"] / stats["CAPACITY_REDUCTION"]["total"] * 100
n_speed_bands = stats["FREESPEED"]["unique_speeds"]
total_n = stats["VOL_BASE_CASE"]["total"]
summary = (
    f"\\textbf{{Verified statistics}} (sample of $n = {total_n:,}$ nodes)\n\n"
    f"VOL\\_BASE\\_CASE:  mean $= {stats['VOL_BASE_CASE']['mean']:.1f}$,\n"
    f"  median $= {stats['VOL_BASE_CASE']['median']:.1f}$, max $= {stats['VOL_BASE_CASE']['mx']:.0f}$ veh/h\n\n"
    f"CAPACITY\\_BASE\\_CASE:  mean $= {stats['CAPACITY_BASE_CASE']['mean']:.0f}$,\n"
    f"  median $= {stats['CAPACITY_BASE_CASE']['median']:.0f}$, max $= {stats['CAPACITY_BASE_CASE']['mx']:.0f}$ veh/h\n\n"
    f"CAPACITY\\_REDUCTION:\n"
    f"  ${nz_pct_red:.1f}\\%$ zero values (no-policy segments)\n\n"
    f"FREESPEED:  ${n_speed_bands}$ discrete urban bands\n\n"
    f"LENGTH:  pre-normalised, range $[{stats['LENGTH']['mn']:.1f},\\ {stats['LENGTH']['mx']:.1f}]$"
)
ax.text(0.03, 0.97, summary, transform=ax.transAxes, fontsize=9, va="top", color="black",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=NEUTRAL, linewidth=0.6))

fig.suptitle(f"Distributions of the five input features (verified, $n = {total_n:,}$ nodes)",
             fontsize=11, color="black", y=1.0)
fig.tight_layout()
save_both(fig, "fig34_feature_distributions")

# Print the verification table
print()
print("=== VERIFIED FEATURE TABLE (for caption rewrite) ===")
print(f"  CAPACITY_REDUCTION zero fraction: {nz_pct_red:.1f}%")
print(f"  FREESPEED unique bands:           {n_speed_bands}")
print(f"  LENGTH range:                     [{stats['LENGTH']['mn']:.1f}, {stats['LENGTH']['mx']:.1f}]")
print(f"  Total nodes:                      {total_n:,}")

# ============================================================
# F29 — Elena Natterer-style horizontal pipeline
# ============================================================
fig, ax = plt.subplots(figsize=(9, 3))
ax.set_xlim(0, 18); ax.set_ylim(0, 4); ax.axis("off")

boxes = [
    ("Input graph",          "Nodes = road segments,\n5 features per node",       PRIMARY),
    ("PointNet Conv x2",     "Encodes local geometry\nfrom start/end points",     PRIMARY),
    ("Transformer Conv x2",  "Self-attention\nacross network",                    PRIMARY),
    ("GAT Conv x2",          "Graph attention\nlayers",                            PRIMARY),
    ("MLP head",             "Final regression head",                              PRIMARY),
    ("Predicted $\\Delta v$","Per-node output\n(veh/h)",                          SECOND),
]
xs = np.linspace(1.5, 16.5, 6)
bw, bh = 2.4, 1.1

for x, (name, sub, col) in zip(xs, boxes):
    rect = FancyBboxPatch((x - bw/2, 2.2 - bh/2), bw, bh,
                          boxstyle="round,pad=0.02,rounding_size=0.06",
                          facecolor="white", edgecolor=col, linewidth=1.0)
    ax.add_patch(rect)
    ax.text(x, 2.35, name, ha="center", va="center", fontsize=9.5, color="black")
    ax.text(x, 1.95, sub, ha="center", va="center", fontsize=7.5, color=TICKDARK, style="italic")

for i in range(len(xs) - 1):
    ax.annotate("", xy=(xs[i+1] - bw/2 - 0.05, 2.2),
                xytext=(xs[i] + bw/2 + 0.05, 2.2),
                arrowprops=dict(arrowstyle="->", lw=0.8, color="black"))

# Dropout-active dashed line under boxes 2-4
x_left  = xs[1] - bw/2
x_right = xs[3] + bw/2
y_d = 0.95
ax.plot([x_left, x_right], [y_d, y_d], "--", color=NEUTRAL, linewidth=0.9)
for x in [xs[1], xs[2], xs[3]]:
    ax.annotate("", xy=(x, 2.2 - bh/2 - 0.05), xytext=(x, y_d + 0.1),
                arrowprops=dict(arrowstyle="->", lw=0.6, color=NEUTRAL))
ax.text((x_left + x_right)/2, y_d - 0.4,
        "Dropout active during MC Dropout inference",
        ha="center", fontsize=8, color=NEUTRAL, style="italic")

save_both(fig, "fig29_pointnet_architecture")

# ============================================================
# F19 — T9 decomposition (clean English labels, no Greek in bars)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.4),
                                gridspec_kw={"width_ratios": [1.4, 1]})
labels = ["Aleatoric", "Epistemic", "Total"]
values = [4.657, 1.099, 4.823]
colors = [PRIMARY, SECOND, NEUTRAL]
bars = ax1.bar(labels, values, color=colors, edgecolor=EDGE, linewidth=0.5, alpha=ALPHA)
for b, v in zip(bars, values):
    ax1.text(b.get_x() + b.get_width()/2, v + 0.13, f"{v:.3f}",
             ha="center", fontsize=10)
ax1.set_ylabel("Mean uncertainty (vehicles per hour)")
ax1.set_title("(a) Mean uncertainty by component", fontsize=10.5)
ax1.set_ylim(0, max(values) * 1.25)

# Annotation box BELOW bars (over the x-axis area)
ax1.text(0.5, -0.20, "Aleatoric to epistemic ratio: 4.24",
         transform=ax1.transAxes, ha="center", fontsize=9.5, color="black",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                   edgecolor=NEUTRAL, linewidth=0.6))

# Pie
sizes = [99.85, 0.15]
labels_pie = [f"Aleatoric-dominant\n(99.85\\%)", f"Epistemic-dominant\n(0.15\\%)"]
ax2.pie(sizes, labels=labels_pie, colors=[PRIMARY, SECOND], startangle=90,
        wedgeprops={"edgecolor": EDGE, "linewidth": 0.5, "alpha": ALPHA},
        textprops={"fontsize": 9.5})
ax2.set_title("(b) Per-node dominant component", fontsize=10.5)

fig.suptitle("Trial 9 uncertainty decomposition", fontsize=11)
fig.tight_layout()
save_both(fig, "fig19_t9_uncertainty_decomposition")

# ============================================================
# F22 — CQR R^2 progression (no overlap, gate line clean)
# ============================================================
fig, ax = plt.subplots(figsize=(7, 4.5))
labels = ["T8", "T10-v1", "T10-v2", "T11"]
sublabels = ["MSE baseline", "CQR all", "CQR all,\nlower lr", "CQR frozen"]
r2s = [0.5957, 0.315, 0.4057, 0.5835]
colors = [NEUTRAL, SECOND, SECOND, TERTIARY]

x = np.arange(len(labels))
bars = ax.bar(x, r2s, color=colors, edgecolor=EDGE, linewidth=0.5, alpha=ALPHA, width=0.6)
for b, v in zip(bars, r2s):
    ax.text(b.get_x() + b.get_width()/2, v + 0.020, f"{v:.4f}",
            ha="center", fontsize=9.5)

# Gate line at 0.57 — clearly above T10v1, T10v2 and below T8, T11
ax.axhline(0.57, color=NEUTRAL, linestyle="--", linewidth=1.0, alpha=0.8)
ax.text(3.45, 0.58, "Gate $R^2 \\geq 0.57$", color=TICKDARK, fontsize=8.5, ha="right")

ax.set_xticks(x); ax.set_xticklabels(labels, rotation=0)
# Sub-labels under each tick
for xi, sl in zip(x, sublabels):
    ax.text(xi, -0.05, sl, ha="center", va="top", fontsize=8, color=TICKDARK,
            style="italic", transform=ax.get_xaxis_transform())

ax.set_ylabel("Test $R^2$ (midpoint)")
ax.set_title("CQR $R^2$ progression: only frozen-backbone training meets the gate")
ax.set_ylim(0, 0.78)
fig.tight_layout()
save_both(fig, "fig22_cqr_r2_progression")

# ============================================================
# F35 — horizontal 3-tier policy framework with axis
# ============================================================
fig, ax = plt.subplots(figsize=(9, 3.4))
ax.set_xlim(0, 12); ax.set_ylim(-1.5, 3.6); ax.axis("off")

ax.text(6, 3.3, "Three-tier deployment policy based on uncertainty percentile",
        ha="center", fontsize=11, color="black")

def tier_box(ax, x, w, color, top, mid, bot):
    rect = FancyBboxPatch((x - w/2, 0.6), w, 2.0,
                          boxstyle="round,pad=0.04,rounding_size=0.10",
                          facecolor=color, edgecolor=NEUTRAL, alpha=ALPHA, linewidth=0.7)
    ax.add_patch(rect)
    ax.text(x, 2.20, top, ha="center", va="center", fontsize=11, color="black", fontweight="bold")
    ax.text(x, 1.65, mid, ha="center", va="center", fontsize=9.5, color="black")
    ax.text(x, 1.00, bot, ha="center", va="center", fontsize=9, color=TICKDARK, style="italic")

tier_box(ax, 2.0, 3.6, TERTIARY, "ACCEPT",          "Bottom 50\\%",  "Use ML prediction directly")
tier_box(ax, 6.0, 3.0, SECOND,   "FLAG FOR REVIEW", "Middle 40\\%",  "Human verification")
tier_box(ax, 9.5, 1.8, "#C66666", "REJECT",          "Top 10\\%",     "Run MATSim simulation")

# Bottom axis
ax.plot([0.5, 11.0], [0, 0], color="black", linewidth=0.7)
for tx, lbl in [(0.5, "0\\%"), (3.75, "50\\%"), (8.6, "90\\%"), (11.0, "100\\%")]:
    ax.plot([tx, tx], [-0.10, 0.10], color="black", linewidth=0.7)
    ax.text(tx, -0.35, lbl, ha="center", va="top", fontsize=8.5, color=TICKDARK)
ax.text(5.75, -1.0, "Predicted uncertainty (low to high)",
        ha="center", fontsize=9.5, color=TICKDARK)

save_both(fig, "fig35_policy_decision_framework")

print("All final figures regenerated.")
