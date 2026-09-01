import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

PT_PATH = (
    r"C:\Users\zamin\Downloads\Nazim\Thesis\Nazim"
    r"\ml_surrogates_thesis_final\code\data\train_data"
    r"\dist_not_connected_10k_1pct\datalist_batch_1.pt"
)
graphs = torch.load(PT_PATH, weights_only=False, map_location="cpu")
g = graphs[0]
ei = g.edge_index.numpy()  # [2, 59851]
num_nodes = int(g.num_nodes)  # 31635

BLUE = "#4878A8"
GREEN = "#5DA573"
RED = "#D66B6B"
GOLD = "#D4A843"
GREY = "#888888"
CREAM = "#F5F0E8"
BG = "#FFFFFF"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "figure.facecolor": BG,
        "axes.facecolor": BG,
    }
)
OUT = r"C:\Users\zamin\Downloads\Nazim"

# Degree of each node
degree = np.bincount(ei[0], minlength=num_nodes)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE:  top row = degree histogram (full width)
#          bottom row = two explanation panels side by side
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor(BG)
fig.suptitle(
    "Attribute 4 of 7 :   edge_index   [2 x 59,851]   --   Road-to-road connections",
    fontsize=15,
    fontweight="bold",
    color="#222222",
    y=0.98,
)

# ── TOP: degree distribution histogram ───────────────────────────────────────
ax_hist = fig.add_axes([0.07, 0.62, 0.88, 0.30])
ax_hist.set_facecolor(BG)
ax_hist.spines["top"].set_visible(False)
ax_hist.spines["right"].set_visible(False)

deg_vals, deg_counts = np.unique(degree, return_counts=True)
ax_hist.bar(deg_vals, deg_counts, color=GREEN, alpha=0.80, edgecolor="white", width=0.7)
mean_deg = np.mean(degree)
ax_hist.axvline(mean_deg, color=RED, linewidth=2.2, linestyle="--")
ax_hist.annotate(
    f"MEAN = {mean_deg:.2f} connections",
    xy=(mean_deg, deg_counts.max() * 0.75),
    xytext=(mean_deg + 0.5, deg_counts.max() * 0.85),
    fontsize=10,
    color=RED,
    fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=RED, lw=1.5),
)

ax_hist.set_xlabel(
    "X AXIS  -->  Degree  (number of connections each road has)\n"
    '"How many other roads is this road connected to?"',
    fontsize=10,
    labelpad=10,
    color="#333333",
)
ax_hist.set_ylabel(
    "Y AXIS  -->  Number of roads\n(out of 31,635)\nthat have this many connections",
    fontsize=10,
    labelpad=10,
    color="#333333",
)
ax_hist.set_title(
    f"Total connections: 59,851     |     Most roads have 1-4 connections",
    fontsize=9.5,
    color="#666666",
    style="italic",
)

# ── BOTTOM LEFT: what is edge_index? ─────────────────────────────────────────
ax_left = fig.add_axes([0.04, 0.04, 0.44, 0.52])
ax_left.axis("off")
ax_left.set_xlim(0, 10)
ax_left.set_ylim(0, 10)
ax_left.set_facecolor(BG)

ax_left.text(
    5,
    9.6,
    "What is edge_index?",
    ha="center",
    fontsize=13,
    fontweight="bold",
    color="#222222",
)
ax_left.text(
    5,
    9.0,
    "It stores which roads are connected to which.\n"
    "Row 0 = SOURCE road index\n"
    "Row 1 = DESTINATION road index",
    ha="center",
    va="top",
    fontsize=10,
    color="#333333",
)

# Draw the matrix visually
headers = ["Row 0\n(source)", "Row 1\n(dest)"]
example_data = [
    [0, 19],
    [1, 11091],
    [1, 11092],
    [2, 5],
    ["...", "..."],
]
col_x = [2.5, 7.0]
row_start_y = 7.4

# column headers
for hx, h in zip(col_x, headers):
    rect = mpatches.FancyBboxPatch(
        (hx - 1.1, row_start_y + 0.05),
        2.2,
        0.7,
        boxstyle="round,pad=0.05",
        linewidth=1.5,
        edgecolor=GREEN,
        facecolor="#D4EDDA",
        clip_on=False,
    )
    ax_left.add_patch(rect)
    ax_left.text(
        hx,
        row_start_y + 0.42,
        h,
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color="#1a4a2e",
    )

for i, row in enumerate(example_data):
    ry = row_start_y - 0.72 * (i + 1)
    bg = "#F0F8F0" if i % 2 == 0 else BG
    for hx, val in zip(col_x, row):
        rect = mpatches.FancyBboxPatch(
            (hx - 1.1, ry - 0.28),
            2.2,
            0.55,
            boxstyle="round,pad=0.02",
            linewidth=0.8,
            edgecolor=GREY,
            facecolor=bg,
            clip_on=False,
        )
        ax_left.add_patch(rect)
        ax_left.text(
            hx,
            ry,
            str(val),
            ha="center",
            va="center",
            fontsize=10,
            color="#222222",
            fontweight="bold" if str(val) != "..." else "normal",
        )

# Interpretation label
ax_left.text(
    5,
    2.7,
    "Reading row by row:\n"
    "  Road 0  is connected to  Road 19\n"
    "  Road 1  is connected to  Road 11091\n"
    "  Road 1  is connected to  Road 11092\n"
    "  Road 2  is connected to  Road 5\n"
    "  ...  (59,851 pairs total)",
    ha="center",
    va="top",
    fontsize=9.5,
    color="#333333",
    bbox=dict(boxstyle="round,pad=0.4", facecolor=CREAM, edgecolor=GREY, linewidth=1.2),
)

# ── BOTTOM RIGHT: what does "connected" mean in real life? ───────────────────
ax_right = fig.add_axes([0.54, 0.04, 0.42, 0.52])
ax_right.axis("off")
ax_right.set_xlim(0, 10)
ax_right.set_ylim(0, 10)
ax_right.set_facecolor(BG)

ax_right.text(
    5,
    9.6,
    'What does "connected" mean?',
    ha="center",
    fontsize=13,
    fontweight="bold",
    color="#222222",
)

# Draw intersection diagram
# Roads as rectangles/lines meeting at a point
cx, cy = 5, 6.2  # intersection center

# 4 roads meeting at intersection
road_ends = [(5, 8.2), (7.8, 6.2), (5, 4.2), (2.2, 6.2)]
road_colors = [BLUE, GOLD, GREEN, RED]
road_names = ["Road A", "Road B", "Road C", "Road D"]

for (ex, ey), rc, rn in zip(road_ends, road_colors, road_names):
    ax_right.annotate(
        "",
        xy=(cx, cy),
        xytext=(ex, ey),
        arrowprops=dict(arrowstyle="->", color=rc, lw=3.5, mutation_scale=20),
    )
    offset = [(0, 0.4), (0.5, 0), (0, -0.4), (-0.5, 0)]
    ox, oy = offset[road_ends.index((ex, ey))]
    ax_right.text(
        ex + ox,
        ey + oy,
        rn,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=rc,
    )

# intersection dot
ax_right.plot(cx, cy, "o", color="#333333", markersize=14, zorder=10)
ax_right.text(
    cx + 0.5, cy + 0.3, "Intersection", fontsize=8.5, color="#333333", fontweight="bold"
)

ax_right.text(
    5,
    3.2,
    "Roads A, B, C, D all meet at one intersection.\n"
    "In edge_index this means:\n"
    "  A -- B,  A -- C,  A -- D\n"
    "  B -- C,  B -- D,  C -- D\n\n"
    'All 4 roads are "connected" because they\n'
    "share the same intersection.\n\n"
    "This is the LINE GRAPH structure:\n"
    "Roads = nodes,  Intersections = edges.",
    ha="center",
    va="top",
    fontsize=9.5,
    color="#333333",
    bbox=dict(boxstyle="round,pad=0.4", facecolor=CREAM, edgecolor=GREY, linewidth=1.2),
)

plt.savefig(
    f"{OUT}\\detail_4_edge_index_fixed.png", dpi=150, bbox_inches="tight", facecolor=BG
)
plt.close()
print("Saved detail_4_edge_index_fixed.png")
