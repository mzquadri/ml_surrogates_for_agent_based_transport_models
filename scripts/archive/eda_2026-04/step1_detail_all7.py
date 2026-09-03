import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import torch

# ── Load data ─────────────────────────────────────────────────────────────────
PT_PATH = (
    r"C:\Users\zamin\Downloads\Nazim\Thesis\Nazim"
    r"\ml_surrogates_thesis_final\code\data\train_data"
    r"\dist_not_connected_10k_1pct\datalist_batch_1.pt"
)

print("Loading .pt file …")
graphs = torch.load(PT_PATH, weights_only=False, map_location="cpu")
g = graphs[0]  # use graph 0 for everything
print("Loaded. Graph 0 keys:", g.keys())

# ── Palette & style ───────────────────────────────────────────────────────────
BLUE = "#4878A8"
RED = "#D66B6B"
GREEN = "#5DA573"
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
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

OUT = r"C:\Users\zamin\Downloads\Nazim"

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — num_nodes
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(BG)
ax.axis("off")

ax.text(
    0.5,
    0.90,
    "Attribute 1 of 7 :  num_nodes",
    ha="center",
    va="center",
    transform=ax.transAxes,
    fontsize=17,
    fontweight="bold",
    color="#222222",
)

# big number
ax.text(
    0.5,
    0.60,
    "31,635",
    ha="center",
    va="center",
    transform=ax.transAxes,
    fontsize=72,
    fontweight="bold",
    color=BLUE,
)

ax.text(
    0.5,
    0.42,
    "road segments  inside Paris",
    ha="center",
    va="center",
    transform=ax.transAxes,
    fontsize=16,
    color="#444444",
)

# analogy boxes
boxes = [
    ("Shape of data", "Just a single integer\n(not a tensor)", BLUE),
    (
        "What it means",
        "Paris road network has\n31,635 individual road\nsegments (edges in graph)",
        GREEN,
    ),
    (
        "Why it matters",
        "Every other tensor in this\ngraph has 31,635 rows —\none row = one road segment",
        GOLD,
    ),
]
for i, (title, body, clr) in enumerate(boxes):
    x0 = 0.08 + i * 0.31
    rect = mpatches.FancyBboxPatch(
        (x0, 0.05),
        0.28,
        0.28,
        transform=ax.transAxes,
        boxstyle="round,pad=0.02",
        linewidth=2,
        edgecolor=clr,
        facecolor=CREAM,
        clip_on=False,
    )
    ax.add_patch(rect)
    ax.text(
        x0 + 0.14,
        0.30,
        title,
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        color=clr,
    )
    ax.text(
        x0 + 0.14,
        0.16,
        body,
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#333333",
    )

plt.tight_layout()
plt.savefig(
    f"{OUT}\\detail_1_num_nodes.png", dpi=150, bbox_inches="tight", facecolor=BG
)
plt.close()
print("Saved detail_1_num_nodes.png")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — x  [31635, 6]
# ═══════════════════════════════════════════════════════════════════════════════
x_np = g.x.numpy()  # [31635, 6]

feat_info = [
    ("VOL_BASE_CASE", "Base traffic volume\n(vehicles / hour)", BLUE, 0),
    ("CAPACITY_BASE", "Road capacity\n(vehicles / hour)", GREEN, 1),
    ("CAPACITY_REDUCTION", "Capacity reduced in\nthis scenario (veh/hr)", RED, 2),
    ("FREESPEED", "Speed limit\n(m/s → ~km/h)", GOLD, 3),
    ("HIGHWAY_TYPE", "Road type\n(categorical  −1 to 9)", GREY, 4),
    ("LENGTH", "Road segment length\n(metres)", BLUE, 5),
]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.patch.set_facecolor(BG)
fig.suptitle(
    "Attribute 2 of 7 :  x   [31,635 × 6]  —  Features of each road segment",
    fontsize=15,
    fontweight="bold",
    color="#222222",
    y=1.01,
)

for ax, (name, desc, clr, col) in zip(axes.flat, feat_info):
    vals = x_np[:, col]
    if name == "HIGHWAY_TYPE":
        unique, counts = np.unique(vals, return_counts=True)
        ax.bar(unique, counts, color=clr, alpha=0.75, edgecolor="white", width=0.7)
        ax.set_xlabel("Road type code", fontsize=9)
        ax.set_ylabel("Number of roads", fontsize=9)
    else:
        ax.hist(vals, bins=60, color=clr, alpha=0.75, edgecolor="white", linewidth=0.3)
        ax.set_xlabel("Value", fontsize=9)
        ax.set_ylabel("Number of roads", fontsize=9)
        ax.axvline(
            np.mean(vals),
            color="#222222",
            linewidth=1.4,
            linestyle="--",
            label=f"Mean = {np.mean(vals):.1f}",
        )
        ax.legend(fontsize=8, frameon=False)

    ax.set_facecolor(BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(
        f"Col {col} :  {name}", fontsize=10, fontweight="bold", color=clr, pad=4
    )
    ax.text(
        0.97,
        0.95,
        desc,
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=8,
        color="#666666",
        style="italic",
    )

plt.tight_layout()
plt.savefig(
    f"{OUT}\\detail_2_x_features.png", dpi=150, bbox_inches="tight", facecolor=BG
)
plt.close()
print("Saved detail_2_x_features.png")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — y  [31635, 1]
# ═══════════════════════════════════════════════════════════════════════════════
y_np = g.y.numpy().flatten()  # [31635]

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.patch.set_facecolor(BG)
fig.suptitle(
    "Attribute 3 of 7 :  y   [31,635 × 1]  —  Target : Δ car volume per road",
    fontsize=14,
    fontweight="bold",
    color="#222222",
)

# left: full histogram
ax = axes[0]
neg = y_np[y_np < -0.5]
zero = y_np[(y_np >= -0.5) & (y_np <= 0.5)]
pos = y_np[y_np > 0.5]

ax.hist(neg, bins=60, color=RED, alpha=0.80, label=f"Negative  (n={len(neg):,})")
ax.hist(zero, bins=30, color=GREY, alpha=0.60, label=f"≈ Zero     (n={len(zero):,})")
ax.hist(pos, bins=60, color=GREEN, alpha=0.80, label=f"Positive  (n={len(pos):,})")
ax.axvline(0, color="#222222", linewidth=1.5, linestyle="--")
ax.set_xlabel("Δ car volume  (veh / hr)", fontsize=11)
ax.set_ylabel("Number of road segments", fontsize=11)
ax.set_title("Full distribution", fontsize=12, fontweight="bold")
ax.legend(fontsize=9, frameon=False)
ax.set_facecolor(BG)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# right: pie of positive / near-zero / negative
ax2 = axes[1]
sizes = [len(neg), len(zero), len(pos)]
labels = [
    f"Traffic\ndecreased\n({len(neg):,} roads)",
    f"No change\n({len(zero):,} roads)",
    f"Traffic\nincreased\n({len(pos):,} roads)",
]
colors = [RED, GREY, GREEN]
wedges, texts, autotexts = ax2.pie(
    sizes,
    labels=labels,
    colors=colors,
    autopct="%1.1f%%",
    startangle=90,
    textprops={"fontsize": 9},
    wedgeprops={"edgecolor": "white", "linewidth": 1.5},
)
for at in autotexts:
    at.set_fontsize(9)
    at.set_fontweight("bold")
ax2.set_title("Road segments by traffic change", fontsize=12, fontweight="bold")

# annotation box
note = (
    f"Range : [{y_np.min():.1f},  {y_np.max():.1f}]  veh/hr\n"
    f"Mean  : {y_np.mean():.2f}   |   Std : {y_np.std():.2f}\n"
    "Positive = more cars after scenario\n"
    "Negative = fewer cars after scenario"
)
fig.text(
    0.5,
    -0.02,
    note,
    ha="center",
    va="top",
    fontsize=9,
    color="#555555",
    style="italic",
    bbox=dict(boxstyle="round,pad=0.4", facecolor=CREAM, edgecolor=GREY),
)

plt.tight_layout()
plt.savefig(f"{OUT}\\detail_3_y_target.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved detail_3_y_target.png")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 4 — edge_index  [2, 59851]
# ═══════════════════════════════════════════════════════════════════════════════
ei = g.edge_index.numpy()  # [2, 59851]

# degree (how many connections each road has)
num_nodes = int(g.num_nodes)
degree = np.bincount(ei[0], minlength=num_nodes)

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.patch.set_facecolor(BG)
fig.suptitle(
    "Attribute 4 of 7 :  edge_index   [2 × 59,851]  —  Road-to-road connections",
    fontsize=14,
    fontweight="bold",
    color="#222222",
)

# left: degree distribution
ax = axes[0]
deg_vals, deg_counts = np.unique(degree, return_counts=True)
ax.bar(deg_vals, deg_counts, color=GREEN, alpha=0.80, edgecolor="white")
ax.set_xlabel("Number of connections (degree)", fontsize=11)
ax.set_ylabel("Number of road segments", fontsize=11)
ax.set_title(
    "How many roads does each road connect to?", fontsize=11, fontweight="bold"
)
ax.axvline(
    np.mean(degree),
    color=RED,
    linewidth=1.8,
    linestyle="--",
    label=f"Mean degree = {np.mean(degree):.2f}",
)
ax.legend(fontsize=9, frameon=False)
ax.set_facecolor(BG)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# right: explanation diagram (text + boxes)
ax2 = axes[1]
ax2.axis("off")
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)

ax2.text(
    5,
    9.3,
    "What does edge_index mean?",
    ha="center",
    fontsize=12,
    fontweight="bold",
    color="#222222",
)

# draw 3 "road" boxes and arrows between them
road_positions = [(2, 6), (5, 8), (8, 6), (5, 4)]
road_labels = ["Road A", "Road B", "Road C", "Road D"]
rclr = [BLUE, GREEN, GOLD, RED]

for (rx, ry), lbl, rc in zip(road_positions, road_labels, rclr):
    rect = mpatches.FancyBboxPatch(
        (rx - 1, ry - 0.5),
        2,
        1,
        boxstyle="round,pad=0.1",
        linewidth=2,
        edgecolor=rc,
        facecolor=CREAM,
    )
    ax2.add_patch(rect)
    ax2.text(
        rx, ry, lbl, ha="center", va="center", fontsize=10, fontweight="bold", color=rc
    )

# arrows: A-B, B-C, A-D, C-D
for (x1, y1), (x2, y2) in [
    (road_positions[0], road_positions[1]),
    (road_positions[1], road_positions[2]),
    (road_positions[0], road_positions[3]),
    (road_positions[2], road_positions[3]),
]:
    ax2.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="<->", color=GREY, lw=1.5, connectionstyle="arc3,rad=0.1"
        ),
    )

ax2.text(
    5,
    2.8,
    "edge_index row 0 = source road\n"
    "edge_index row 1 = destination road\n\n"
    'Two roads are "connected" if they\nshare an intersection in Paris\n\n'
    "59,851 such connections exist",
    ha="center",
    va="center",
    fontsize=9.5,
    color="#333333",
    bbox=dict(boxstyle="round,pad=0.5", facecolor=CREAM, edgecolor=GREY),
)

plt.tight_layout()
plt.savefig(
    f"{OUT}\\detail_4_edge_index.png", dpi=150, bbox_inches="tight", facecolor=BG
)
plt.close()
print("Saved detail_4_edge_index.png")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 5 — pos  [31635, 3, 2]
# ═══════════════════════════════════════════════════════════════════════════════
pos_np = g.pos.numpy()  # [31635, 3, 2]

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
fig.patch.set_facecolor(BG)
fig.suptitle(
    "Attribute 5 of 7 :  pos   [31,635 × 3 × 2]  —  GPS coordinates of each road",
    fontsize=14,
    fontweight="bold",
    color="#222222",
)

# left: road network map (draw start→end for each road as a line)
ax = axes[0]
ax.set_facecolor(BG)
starts = pos_np[:, 0, :]  # [31635, 2]  lon,lat of start
ends = pos_np[:, 1, :]  # [31635, 2]  lon,lat of end

# Plot as line segments (vectorised via LineCollection)
from matplotlib.collections import LineCollection

segments = np.stack([starts, ends], axis=1)  # [31635, 2, 2]
lc = LineCollection(segments, colors=BLUE, linewidths=0.3, alpha=0.5)
ax.add_collection(lc)
ax.set_xlim(starts[:, 0].min() - 0.005, starts[:, 0].max() + 0.005)
ax.set_ylim(starts[:, 1].min() - 0.005, starts[:, 1].max() + 0.005)
ax.set_xlabel("Longitude  (degrees East)", fontsize=10)
ax.set_ylabel("Latitude  (degrees North)", fontsize=10)
ax.set_title(
    "All 31,635 roads plotted\n(start → end points)", fontsize=11, fontweight="bold"
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# right: explanation of the 3 points per road
ax2 = axes[1]
ax2.axis("off")
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)

ax2.text(
    5,
    9.4,
    "What are the 3 points per road?",
    ha="center",
    fontsize=12,
    fontweight="bold",
    color="#222222",
)

# draw a road with 3 annotated points
road_y = 6.5
ax2.annotate(
    "",
    xy=(8, road_y),
    xytext=(2, road_y),
    arrowprops=dict(arrowstyle="->", color=BLUE, lw=2.5),
)

pts = [
    (2, road_y, "index 0\nSTART point\n(lon, lat)", BLUE),
    (5, road_y, "index 2\nMIDpoint\n(lon, lat)", GOLD),
    (8, road_y, "index 1\nEND point\n(lon, lat)", RED),
]

for px, py, lbl, clr in pts:
    ax2.plot(px, py, "o", color=clr, markersize=14, zorder=5)
    ax2.text(
        px,
        py - 0.8,
        lbl,
        ha="center",
        va="top",
        fontsize=9,
        color=clr,
        fontweight="bold",
    )

ax2.text(
    5,
    4.0,
    "pos[:, 0, :] = start  (lon, lat)\n"
    "pos[:, 1, :] = end    (lon, lat)\n"
    "pos[:, 2, :] = midpoint (lon, lat)\n\n"
    "Coordinate system: WGS-84 / lon-lat\n"
    "Paris range:\n"
    "  Longitude : 2.27 → 2.42°E\n"
    "  Latitude  : 48.82 → 48.90°N",
    ha="center",
    va="center",
    fontsize=10,
    color="#333333",
    bbox=dict(boxstyle="round,pad=0.5", facecolor=CREAM, edgecolor=GREY),
)

plt.tight_layout()
plt.savefig(f"{OUT}\\detail_5_pos.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved detail_5_pos.png")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 6 — mode_stats_diff  [6, 3]
# ═══════════════════════════════════════════════════════════════════════════════
ms = g.mode_stats_diff.numpy()  # [6, 3]

modes = ["Car", "Public\nTransit", "Bike", "Walk", "Freight", "Ride-\nhailing"]
metrics = ["Trips\n(count)", "Distance\n(metres)", "Duration\n(seconds)"]

fig, axes = plt.subplots(1, 3, figsize=(15, 6))
fig.patch.set_facecolor(BG)
fig.suptitle(
    "Attribute 6 of 7 :  mode_stats_diff   [6 × 3]  —  Absolute change in transport use",
    fontsize=14,
    fontweight="bold",
    color="#222222",
)

colors_modes = [BLUE, GREEN, GOLD, RED, GREY, "#9B59B6"]

for j, (ax, metric) in enumerate(zip(axes, metrics)):
    vals = ms[:, j]
    bars = ax.barh(
        modes, vals, color=colors_modes, alpha=0.80, edgecolor="white", height=0.55
    )
    ax.axvline(0, color="#333333", linewidth=1.2, linestyle="--")
    ax.set_xlabel(f"Change in {metric.split(chr(10))[0]}", fontsize=10)
    ax.set_title(metric, fontsize=11, fontweight="bold")
    ax.set_facecolor(BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # format x-axis with scientific notation for large numbers
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

fig.text(
    0.5,
    -0.03,
    "Each value = total change across ALL trips in Paris for this scenario\n"
    "(scenario − base case).  Large negative numbers = large reduction in that mode.",
    ha="center",
    fontsize=9,
    color="#555555",
    style="italic",
)

plt.tight_layout()
plt.savefig(
    f"{OUT}\\detail_6_mode_stats_diff.png", dpi=150, bbox_inches="tight", facecolor=BG
)
plt.close()
print("Saved detail_6_mode_stats_diff.png")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 7 — mode_stats_diff_perc  [6, 3]
# ═══════════════════════════════════════════════════════════════════════════════
msp = g.mode_stats_diff_perc.numpy()  # [6, 3]

fig, axes = plt.subplots(1, 3, figsize=(15, 6))
fig.patch.set_facecolor(BG)
fig.suptitle(
    "Attribute 7 of 7 :  mode_stats_diff_perc   [6 × 3]  —  % change in transport use",
    fontsize=14,
    fontweight="bold",
    color="#222222",
)

for j, (ax, metric) in enumerate(zip(axes, metrics)):
    vals = msp[:, j]
    ax.barh(modes, vals, color=colors_modes, alpha=0.80, edgecolor="white", height=0.55)
    ax.axvline(0, color="#333333", linewidth=1.2, linestyle="--")
    ax.set_xlabel(f"% change in {metric.split(chr(10))[0]}", fontsize=10)
    ax.set_title(metric, fontsize=11, fontweight="bold")
    ax.set_facecolor(BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # add % labels
    for i, v in enumerate(vals):
        ax.text(
            v + (0.5 if v >= 0 else -0.5),
            i,
            f"{v:.1f}%",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=8,
            color="#333333",
        )

fig.text(
    0.5,
    -0.03,
    "Same data as mode_stats_diff but expressed as percentage change.\n"
    "−100% means that mode dropped to zero for this extreme scenario.",
    ha="center",
    fontsize=9,
    color="#555555",
    style="italic",
)

plt.tight_layout()
plt.savefig(
    f"{OUT}\\detail_7_mode_stats_diff_perc.png",
    dpi=150,
    bbox_inches="tight",
    facecolor=BG,
)
plt.close()
print("Saved detail_7_mode_stats_diff_perc.png")

print("\n✓ All 7 detail plots saved.")
