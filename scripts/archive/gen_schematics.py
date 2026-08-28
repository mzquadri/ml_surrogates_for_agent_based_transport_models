"""Regenerate 5 schematic figures using matplotlib (TUM style)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon
from matplotlib.lines import Line2D
from plot_style import set_style, save_both, TUM_BLUE, TUM_ORANGE, TUM_LIGHTBLUE, TUM_GRAY, TUM_GREEN, IDEAL_GREEN, FAIL_RED, PASS_GREEN, TUM_DARK

set_style()
OUT = "../../../document/figures/new"

# ============================================================
# Helper for boxes + arrows
# ============================================================
def box(ax, x, y, w, h, text, fc=TUM_BLUE, ec="black", txt_color="white", fontsize=10, fontweight="bold", alpha=1.0):
    p = FancyBboxPatch((x - w/2, y - h/2), w, h,
                       boxstyle="round,pad=0.04,rounding_size=0.10",
                       facecolor=fc, edgecolor=ec, linewidth=0.9, alpha=alpha)
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, color=txt_color)

def arrow(ax, x1, y1, x2, y2, color=TUM_DARK, lw=1.4, style="->,head_width=4,head_length=6"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=lw, color=color,
                                shrinkA=2, shrinkB=2,
                                mutation_scale=15))

# ============================================================
# F29: PointNetTransfGAT layer-by-layer architecture
# ============================================================
fig, ax = plt.subplots(figsize=(13, 6.5))
ax.set_xlim(0, 13); ax.set_ylim(0, 7); ax.axis("off")

# Inputs (left)
box(ax, 0.9, 5.2, 1.4, 0.9, "Node features\n$x \\in \\mathbb{R}^{N \\times 5}$", fc=TUM_GRAY, fontsize=8.5)
box(ax, 0.9, 3.5, 1.4, 0.9, "Pos. start\n$\\in \\mathbb{R}^{N \\times 2}$", fc=TUM_GRAY, fontsize=8.5)
box(ax, 0.9, 1.8, 1.4, 0.9, "Pos. end\n$\\in \\mathbb{R}^{N \\times 2}$", fc=TUM_GRAY, fontsize=8.5)

# Layer stack (centre-right)
xs = [3.0, 4.7, 6.4, 8.1, 9.8, 11.5]
ys_top = 5.0
labels = [
    ("PointNetConv-1\n5 → 512", "Local: $7\\to256$\nGlobal: $\\to512\\to512$"),
    ("PointNetConv-2\n512 → 128", "Local: $514\\to256$\nGlobal: $\\to512\\to128$"),
    ("TransformerConv-1\n128 → 256", "4 heads $\\times$ 64"),
    ("TransformerConv-2\n256 → 512", "4 heads $\\times$ 128"),
    ("GATConv (mid)\n512 → 64", "1 head"),
    ("GATConv (final)\n64 → 1", "Trials 2--8\n[Trial 1: Linear]"),
]
colors = [TUM_BLUE, TUM_BLUE, TUM_LIGHTBLUE, TUM_LIGHTBLUE, TUM_ORANGE, TUM_ORANGE]
for i, (x, (lbl, sub), c) in enumerate(zip(xs, labels, colors)):
    box(ax, x, ys_top, 1.55, 1.05, lbl, fc=c, fontsize=9, fontweight="bold")
    ax.text(x, ys_top - 1.1, sub, ha="center", va="top", fontsize=7.5, color=TUM_DARK,
            style="italic")

# Connect inputs to PointNetConv-1 and -2
arrow(ax, 1.6, 5.2, 2.25, 5.05)
arrow(ax, 1.6, 3.5, 2.25, 4.85)            # pos_start to PNC-1
arrow(ax, 1.6, 1.8, 3.95, 4.7)             # pos_end to PNC-2

# Layer chain arrows
for i in range(len(xs) - 1):
    arrow(ax, xs[i] + 0.78, ys_top, xs[i+1] - 0.78, ys_top, lw=1.6)

# Final output box
box(ax, 12.55, 2.5, 1.6, 1.0, "$\\Delta v$\nper node\n(veh/h)", fc=PASS_GREEN, fontsize=9)
arrow(ax, 11.5, 4.45, 12.45, 3.05)

# Dropout annotation (between PNC and Transformer)
ax.add_patch(Rectangle((2.2, 5.65), 6.65, 1.0, fill=False, edgecolor=FAIL_RED,
                       linestyle="--", linewidth=1.2))
ax.text(5.55, 6.4, "Dropout active here (training and MC inference)",
        ha="center", fontsize=8.5, color=FAIL_RED, fontweight="bold")

# Title and footer
ax.text(6.5, 6.85, "PointNetTransfGAT: layer-by-layer forward pass",
        ha="center", fontsize=12.5, fontweight="bold", color=TUM_DARK)
ax.text(6.5, 0.3, "Total trainable parameters: 1,416,768   |   Source: \\texttt{point\\_net\\_transf\\_gat.py}",
        ha="center", fontsize=8.5, color=TUM_DARK, style="italic")

save_both(fig, OUT, "fig29_pointnet_architecture")
print("F29 saved")

# ============================================================
# F30: End-to-end pipeline (MATSim -> GNN -> UQ -> decisions)
# ============================================================
fig, ax = plt.subplots(figsize=(13.5, 6))
ax.set_xlim(0, 14); ax.set_ylim(0, 6); ax.axis("off")

# Stage 1: MATSim
box(ax, 1.3, 4.5, 2.3, 1.4, "MATSim\nagent-based\nsimulation", fc=TUM_GRAY, fontsize=10)
ax.text(1.3, 3.55, "1{,}000 of 10{,}000\nscenarios (10\\%)", ha="center", fontsize=8, color=TUM_DARK, style="italic")
ax.text(1.3, 2.8, ">8 h per scenario", ha="center", fontsize=8, color=FAIL_RED, fontweight="bold")

# Stage 2: line-graph
box(ax, 4.0, 4.5, 2.0, 1.4, "Line-graph\nrepresentation\n(31{,}635 nodes)", fc=TUM_LIGHTBLUE, fontsize=9.5)
ax.text(4.0, 3.55, "5 features + 2D pos.", ha="center", fontsize=8, color=TUM_DARK, style="italic")

# Stage 3: GNN training (8 trials)
box(ax, 6.7, 4.5, 2.1, 1.4, "GNN training\n8 trials (T1--T8)\n+T9, T10, T11", fc=TUM_BLUE, fontsize=9.5)
ax.text(6.7, 3.55, "T8: $R^2 = 0.5957$", ha="center", fontsize=8, color=TUM_DARK, style="italic")
ax.text(6.7, 3.15, "DeepEns: $R^2 = 0.6841$", ha="center", fontsize=8, color=TUM_DARK, style="italic")

# Stage 4: UQ layered
box(ax, 9.6, 4.5, 2.4, 1.4, "Uncertainty\nquantification\n(layered)", fc=TUM_ORANGE, fontsize=9.5)
ax.text(9.6, 3.55, "MC Dropout, $\\rho = 0.4820$", ha="center", fontsize=8, color=TUM_DARK, style="italic")
ax.text(9.6, 3.15, "Conformal: PICP$_{95}$ = 95.01\\%", ha="center", fontsize=8, color=TUM_DARK, style="italic")
ax.text(9.6, 2.75, "TS: ECE $-90.5$\\%", ha="center", fontsize=8, color=TUM_DARK, style="italic")

# Stage 5: deployment
box(ax, 12.6, 4.5, 1.7, 1.4, "Policy\ndecision\nframework", fc=PASS_GREEN, fontsize=10)
ax.text(12.6, 3.55, "Accept / Flag / Reject", ha="center", fontsize=8, color=TUM_DARK, style="italic")

# Arrows
for x1, x2 in [(2.45, 3.0), (5.0, 5.65), (7.75, 8.4), (10.8, 11.75)]:
    arrow(ax, x1, 4.5, x2, 4.5, lw=2)

# Bottom band: "Seconds vs Hours"
ax.add_patch(FancyBboxPatch((0.3, 0.3), 13.4, 1.0,
                             boxstyle="round,pad=0.02",
                             facecolor="#F4F8FB", edgecolor=TUM_BLUE, linewidth=0.8))
ax.text(7.0, 0.85, "Surrogate evaluation: \\textbf{seconds per scenario} (vs >8 h for full MATSim simulation)",
        ha="center", fontsize=10.5, color=TUM_BLUE, fontweight="bold")

# Title
ax.text(7.0, 5.6, "Thesis pipeline: from agent-based simulation to uncertainty-aware traffic prediction",
        ha="center", fontsize=12.5, fontweight="bold", color=TUM_DARK)

save_both(fig, OUT, "fig30_data_flow_pipeline")
print("F30 saved")

# ============================================================
# F31: Road network as line graph
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5),
                         gridspec_kw={"width_ratios": [1, 1]})

# Panel 1: Original road network (intersections + roads)
ax = axes[0]
ax.set_xlim(-1, 6); ax.set_ylim(-1, 5); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("(a) Road network (intersections as nodes)", fontsize=11, fontweight="bold")

# Intersections
intersections = {"A": (0.5, 3.5), "B": (2.5, 3.8), "C": (4.5, 3.5),
                 "D": (1.0, 1.5), "E": (3.0, 1.5), "F": (4.8, 1.0),
                 "G": (2.5, 0.0)}
roads = [("A","B"), ("B","C"), ("A","D"), ("B","E"), ("C","F"),
         ("D","E"), ("E","F"), ("D","G"), ("E","G"), ("F","G")]

# Draw intersections (gray)
for name, (x, y) in intersections.items():
    ax.add_patch(Circle((x, y), 0.18, facecolor=TUM_GRAY, edgecolor="black", zorder=3))
    ax.text(x, y, name, ha="center", va="center", fontsize=9, fontweight="bold", color="white", zorder=4)

# Draw roads (TUM blue)
for u, v in roads:
    ux, uy = intersections[u]; vx, vy = intersections[v]
    ax.plot([ux, vx], [uy, vy], "-", color=TUM_BLUE, linewidth=2, zorder=2)
    # midpoint label
    mx, my = (ux+vx)/2, (uy+vy)/2
    ax.text(mx + 0.05, my + 0.15, f"{u}{v}", fontsize=7.5, color=TUM_BLUE, ha="center")

ax.text(2.5, -0.7, "Intersections = nodes (gray)\nRoad segments = edges (blue)",
        ha="center", fontsize=9, style="italic", color=TUM_DARK)

# Panel 2: Line graph (segments as nodes)
ax = axes[1]
ax.set_xlim(-1, 6); ax.set_ylim(-1, 5); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("(b) Line graph $L(G)$ used in this thesis", fontsize=11, fontweight="bold")

# Compute segment midpoints as new nodes
seg_nodes = {}
for (u, v) in roads:
    ux, uy = intersections[u]; vx, vy = intersections[v]
    seg_nodes[f"{u}{v}"] = ((ux+vx)/2, (uy+vy)/2)

# Edges in line graph: two segments are adjacent if they share an intersection
adjacency = []
for i, (u1, v1) in enumerate(roads):
    for u2, v2 in roads[i+1:]:
        if {u1, v1} & {u2, v2}:
            adjacency.append((f"{u1}{v1}", f"{u2}{v2}"))

# Draw edges (line-graph adjacency)
for s1, s2 in adjacency:
    x1, y1 = seg_nodes[s1]; x2, y2 = seg_nodes[s2]
    ax.plot([x1, x2], [y1, y2], "-", color=TUM_LIGHTBLUE, linewidth=0.8, alpha=0.6, zorder=1)

# Draw segment nodes (TUM orange) - colored by feature value
import matplotlib.cm as cm
np.random.seed(7)
feat_vals = np.random.rand(len(seg_nodes))
for (sname, (x, y)), v in zip(seg_nodes.items(), feat_vals):
    fc = TUM_ORANGE if v > 0.7 else TUM_BLUE
    ax.add_patch(Circle((x, y), 0.16, facecolor=fc, edgecolor="black", zorder=3))
    ax.text(x, y, sname, ha="center", va="center", fontsize=6.5, color="white", fontweight="bold", zorder=4)

# Feature panel (right)
ax.add_patch(FancyBboxPatch((4.4, 2.2), 1.7, 2.3, boxstyle="round,pad=0.03",
                             facecolor="white", edgecolor=TUM_DARK, linewidth=0.8))
ax.text(5.25, 4.25, "Per-node features", ha="center", fontsize=8, fontweight="bold", color=TUM_DARK)
features = ["VOL\\_BASE\\_CASE", "CAPACITY\\_BASE", "CAPACITY\\_RED.", "FREESPEED", "LENGTH"]
for i, f in enumerate(features):
    ax.text(4.55, 3.85 - 0.32*i, f"\\textbullet{{}} {f}", fontsize=7, color=TUM_DARK)

# Highlight orange = policy-affected
ax.text(2.5, -0.55, "Each segment = node\nOrange: $\\mathrm{CAPACITY\\_REDUCTION} \\neq 0$",
        ha="center", fontsize=9, style="italic", color=TUM_DARK)

fig.suptitle("Line-graph transformation: predicting per-segment $\\Delta v$ on the Paris road network",
             fontsize=12.5, fontweight="bold", color=TUM_DARK, y=0.99)
fig.tight_layout()
save_both(fig, OUT, "fig31_network_intro")
print("F31 saved")

# ============================================================
# F32: Node-level vs graph-level prediction
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Panel (a): node-level predictions (scatter with sigma error bars)
ax = axes[0]
np.random.seed(42)
n = 100
nodes = np.arange(n)
y_true = np.random.normal(0, 8, n)
y_pred = y_true + np.random.normal(0, 3, n)
sig = np.abs(np.random.normal(2, 1, n))

ax.errorbar(nodes, y_pred, yerr=sig, fmt='o', color=TUM_BLUE, ecolor=TUM_LIGHTBLUE,
            elinewidth=0.6, markersize=2.5, capsize=0, alpha=0.6, label="$\\hat{y}_i \\pm \\sigma_i$")
ax.plot(nodes, y_true, "x", color=TUM_ORANGE, markersize=4, alpha=0.7, label="True $y_i$")
ax.set_xlabel("Node index $i$ (sample of 100 nodes from one graph)")
ax.set_ylabel("$\\Delta v$ (veh/h)")
ax.set_title("(a) Node-level: per-segment prediction with $\\sigma$", fontsize=10.5)
ax.legend(loc="upper right", fontsize=9)
ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")

# Panel (b): graph-level (one MAE per scenario, with error bar)
ax = axes[1]
n_graphs = 20
np.random.seed(7)
mae_per_graph = np.random.gamma(8, 0.5, n_graphs)
sigma_mean_per_graph = np.random.gamma(3, 0.4, n_graphs)
gx = np.arange(n_graphs)
ax.bar(gx, mae_per_graph, color=TUM_BLUE, edgecolor="black", linewidth=0.4, alpha=0.85,
       yerr=sigma_mean_per_graph, ecolor=TUM_ORANGE, capsize=2)
ax.axhline(np.mean(mae_per_graph), color=TUM_ORANGE, linestyle="--",
           linewidth=1.4, label=f"Mean MAE = {np.mean(mae_per_graph):.2f}")
ax.set_xlabel("Test scenario index (20 of 100)")
ax.set_ylabel("MAE per scenario (veh/h)")
ax.set_title("(b) Graph-level: per-scenario MAE with $\\bar{\\sigma}$ error bar", fontsize=10.5)
ax.legend(loc="upper right", fontsize=9)

# Footer with full counts
fig.suptitle("Two evaluation granularities for T8: $100 \\times 31{,}635 = 3{,}163{,}500$ node-level predictions",
             fontsize=12, fontweight="bold", color=TUM_DARK, y=1.0)
fig.tight_layout()
save_both(fig, OUT, "fig32_node_vs_graph")
print("F32 saved")

# ============================================================
# F33: MC Dropout S=30 stochastic forward passes
# ============================================================
fig, ax = plt.subplots(figsize=(13, 5.8))
ax.set_xlim(0, 14); ax.set_ylim(0, 6); ax.axis("off")

# Single input
box(ax, 1.0, 3.0, 1.5, 1.4, "Input\ngraph $G$", fc=TUM_GRAY, fontsize=10)

# Trained model copies (S=30, but show 5 + dots)
ys = [5.2, 4.3, 3.4, 2.5, 1.6, 0.7]
labels = ["Pass 1", "Pass 2", "Pass 3", "...", "Pass 29", "Pass 30"]
for y, lbl in zip(ys, labels):
    fc = TUM_BLUE if "..." not in lbl else "white"
    txt_color = "white" if "..." not in lbl else TUM_DARK
    ec = "black" if "..." not in lbl else "white"
    box(ax, 4.5, y, 2.0, 0.65,
        f"{lbl}: $f_{{\\hat{{w}}_s}}(G)$" if "..." not in lbl else "$\\vdots$",
        fc=fc, ec=ec, txt_color=txt_color, fontsize=9, alpha=1.0)
    arrow(ax, 1.8, 3.0, 3.55, y, color=TUM_LIGHTBLUE, lw=0.8)

# Stack of outputs
ys_out = [5.2, 4.3, 3.4, 2.5, 1.6, 0.7]
labels_out = ["$\\hat{y}_1$", "$\\hat{y}_2$", "$\\hat{y}_3$", "...", "$\\hat{y}_{29}$", "$\\hat{y}_{30}$"]
for y, lbl in zip(ys_out, labels_out):
    fc = TUM_LIGHTBLUE if "..." not in lbl else "white"
    txt_color = "black" if "..." not in lbl else TUM_DARK
    ec = "black" if "..." not in lbl else "white"
    box(ax, 7.5, y, 1.3, 0.65, lbl, fc=fc, ec=ec, txt_color=txt_color, fontsize=9.5)
    arrow(ax, 5.55, y, 6.85, y, color=TUM_DARK, lw=1)

# Aggregate to mean + std
box(ax, 10.5, 4.0, 2.0, 1.0, "Mean $\\hat{y}$\n$ = \\frac{1}{30}\\sum \\hat{y}_s$", fc=TUM_ORANGE, fontsize=10)
box(ax, 10.5, 2.0, 2.0, 1.0, "Std $\\sigma$\n$ = \\sqrt{\\mathrm{Var}(\\hat{y}_s)}$", fc=TUM_ORANGE, fontsize=10)

# Aggregation arrows from sample stack
for y in ys_out:
    if y > 3.0:
        arrow(ax, 8.2, y, 9.6, 4.0, color=TUM_DARK, lw=0.7)
    else:
        arrow(ax, 8.2, y, 9.6, 2.0, color=TUM_DARK, lw=0.7)

# Final per-node prediction with uncertainty
box(ax, 13.2, 3.0, 1.5, 1.4, "$(\\hat{y},\\,\\sigma)$\nper node", fc=PASS_GREEN, fontsize=10)
arrow(ax, 11.55, 4.0, 12.7, 3.3, lw=1.4)
arrow(ax, 11.55, 2.0, 12.7, 2.7, lw=1.4)

# Annotation about dropout
ax.text(4.5, 5.85, "Same trained network, different dropout masks", ha="center",
        fontsize=9.5, fontweight="bold", color=FAIL_RED, style="italic")
ax.text(7.0, 0.1, "T8: $S = 30$ passes × 100 test graphs × 31{,}635 nodes = 94.9M forward evaluations  (~228 min on T4 GPU)",
        ha="center", fontsize=8.5, color=TUM_DARK, style="italic")

# Title
ax.text(7.0, 5.6, "MC Dropout inference: $S = 30$ stochastic forward passes per input graph",
        ha="center", fontsize=12.5, fontweight="bold", color=TUM_DARK)

save_both(fig, OUT, "fig33_mc_dropout_inference_diagram")
print("F33 saved")
