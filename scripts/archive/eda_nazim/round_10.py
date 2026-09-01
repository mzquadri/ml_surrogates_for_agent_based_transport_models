import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.collections import LineCollection
import numpy as np
import torch

# ── Load data ─────────────────────────────────────────────────────────────────
PT = (
    r"C:\Users\zamin\Downloads\Nazim\Thesis\Nazim\ml_surrogates_thesis_final"
    r"\code\data\train_data\dist_not_connected_10k_1pct\datalist_batch_1.pt"
)
data = torch.load(PT, weights_only=False, map_location="cpu")
g = data[0]
pos = g.pos.numpy()  # [31635, 3, 2]

start = pos[:, 0, :]  # lon, lat of start node
end = pos[:, 1, :]  # lon, lat of end node
mid = pos[:, 2, :]  # lon, lat of midpoint
n_roads = pos.shape[0]

# ── Palette ───────────────────────────────────────────────────────────────────
BG = "#FFFFFF"
C_ROAD = "#2166AC"
C_START = "#2166AC"
C_END = "#B2182B"
C_MID = "#D08020"

plt.rcParams.update(
    {
        "font.family": "serif",
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)

fig = plt.figure(figsize=(20, 9))
fig.patch.set_facecolor(BG)

# ── Title block ───────────────────────────────────────────────────────────────
fig.text(
    0.5,
    0.990,
    "Road Network Spatial Layout  —  pos   [31,635 \u00d7 3 \u00d7 2]",
    ha="center",
    va="top",
    fontsize=20,
    fontweight="bold",
    color="#0D0D0D",
)
fig.text(
    0.5,
    0.952,
    "31,635 directed road segments  |  WGS-84 lon/lat  |  "
    "Paris metropolitan area  |  1,000 disruption scenarios",
    ha="center",
    va="top",
    fontsize=11,
    color="#555555",
    style="italic",
)
fig.add_artist(
    mlines.Line2D(
        [0.04, 0.96],
        [0.932, 0.932],
        transform=fig.transFigure,
        color="#C8C8C8",
        linewidth=0.9,
    )
)

gs = fig.add_gridspec(
    1,
    2,
    width_ratios=[2.3, 1.0],
    wspace=0.05,
    left=0.06,
    right=0.97,
    top=0.885,
    bottom=0.095,
)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])


# ─────────────────────────────────────────────────────────────────────────────
# Panel (a)  Road network map — draw every road as a line segment
# ─────────────────────────────────────────────────────────────────────────────
def style_ax(ax):
    for sp in ax.spines.values():
        sp.set_color("#BBBBBB")
        sp.set_linewidth(0.85)
        sp.set_visible(True)
    ax.set_facecolor(BG)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="both", length=3, color="#BBBBBB", direction="in")


style_ax(ax1)
ax1.yaxis.grid(True, color="#F0F0F0", linewidth=0.6, zorder=0)
ax1.xaxis.grid(True, color="#F0F0F0", linewidth=0.6, zorder=0)

# Draw lines start → end for all road segments
segments = np.stack([start, end], axis=1)  # [31635, 2, 2]
lc = LineCollection(
    segments,
    linewidths=0.35,
    alpha=0.20,
    colors=C_ROAD,
    zorder=3,
)
ax1.add_collection(lc)

# Midpoints as tiny dots — show density
ax1.scatter(
    mid[:, 0],
    mid[:, 1],
    s=0.5,
    c=C_ROAD,
    alpha=0.30,
    zorder=4,
    linewidths=0,
)

LON_MIN, LON_MAX = 2.195, 2.500
LAT_MIN, LAT_MAX = 48.750, 48.933
ax1.set_xlim(LON_MIN, LON_MAX)
ax1.set_ylim(LAT_MIN, LAT_MAX)

# Equal aspect for geographic accuracy
ax1.set_aspect(1 / np.cos(np.radians(48.85)))

ax1.set_xlabel("Longitude  (\u00b0E)", fontsize=12, color="#333333", labelpad=8)
ax1.set_ylabel("Latitude  (\u00b0N)", fontsize=12, color="#333333", labelpad=8)
ax1.set_title(
    "(a)  Paris Road Network  —  31,635 Directed Segments",
    fontsize=13.5,
    fontweight="bold",
    color="#111111",
    loc="left",
    pad=8,
)

lon_range_str = f"{mid[:, 0].min():.3f}\u00b0 \u2013 {mid[:, 0].max():.3f}\u00b0 E"
lat_range_str = f"{mid[:, 1].min():.3f}\u00b0 \u2013 {mid[:, 1].max():.3f}\u00b0 N"
stats_txt = (
    f"Road segments = {n_roads:,}\n"
    f"Longitude     = {lon_range_str}\n"
    f"Latitude      = {lat_range_str}\n"
    f"Coord system  = WGS-84\n"
    f"Points / road = 3  (start, end, mid)"
)
ax1.text(
    0.986,
    0.972,
    stats_txt,
    transform=ax1.transAxes,
    ha="right",
    va="top",
    fontsize=9.5,
    color="#222222",
    linespacing=1.65,
    bbox=dict(
        boxstyle="round,pad=0.42",
        facecolor="white",
        edgecolor="#BBBBBB",
        linewidth=1.0,
        alpha=0.96,
    ),
)

# ─────────────────────────────────────────────────────────────────────────────
# Panel (b)  Schematic: 3-point structure per road
# ─────────────────────────────────────────────────────────────────────────────
ax2.set_facecolor(BG)
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis("off")

ax2.text(
    5,
    9.55,
    "(b)  Structure of  pos  Attribute",
    ha="center",
    va="top",
    fontsize=13.5,
    fontweight="bold",
    color="#111111",
)
ax2.text(
    5,
    9.00,
    "Each road segment stores 3 GPS coordinates:",
    ha="center",
    va="top",
    fontsize=10.5,
    color="#444444",
    style="italic",
)

# Road line
y_road = 6.8
x0, x1 = 1.0, 9.0
xm = (x0 + x1) / 2.0
ax2.annotate(
    "",
    xy=(x1 - 0.05, y_road),
    xytext=(x0, y_road),
    arrowprops=dict(arrowstyle="-|>", color=C_ROAD, lw=2.8),
)

# Points
DOT = 250
ax2.scatter(
    [x0], [y_road], s=DOT, c=C_START, zorder=6, edgecolors="white", linewidths=1.8
)
ax2.scatter(
    [x1], [y_road], s=DOT, c=C_END, zorder=6, edgecolors="white", linewidths=1.8
)
ax2.scatter(
    [xm], [y_road], s=DOT, c=C_MID, zorder=6, edgecolors="white", linewidths=1.8
)


def pt_label(ax, x, y, index_str, name_str, sub_str, color, above=True):
    dy = 0.62 if above else -0.62
    dy2 = 0.25 if above else -0.25
    va_top = "bottom" if above else "top"
    ax.text(
        x,
        y + dy + (0.32 if above else 0),
        index_str,
        ha="center",
        va=va_top,
        fontsize=9,
        color=color,
        fontfamily="monospace",
    )
    ax.text(
        x,
        y + dy,
        name_str,
        ha="center",
        va=va_top,
        fontsize=11,
        color=color,
        fontweight="bold",
    )
    ax.text(
        x,
        y + (dy2 if above else -dy2),
        sub_str,
        ha="center",
        va="top" if above else "bottom",
        fontsize=8.5,
        color=color,
        style="italic",
    )


pt_label(ax2, x0, y_road, "pos[:, 0, :]", "START", "(lon, lat)", C_START)
pt_label(ax2, x1, y_road, "pos[:, 1, :]", "END", "(lon, lat)", C_END)
pt_label(ax2, xm, y_road, "pos[:, 2, :]", "MIDPOINT", "(lon, lat)", C_MID, above=False)

# Tensor shape box
ax2.text(
    5,
    4.10,
    "Tensor shape:   [31,635  \u00d7  3  \u00d7  2]",
    ha="center",
    va="center",
    fontsize=11.5,
    color="#1A1A1A",
    bbox=dict(
        boxstyle="round,pad=0.50",
        facecolor="#EEF3FA",
        edgecolor="#2166AC",
        linewidth=1.5,
        alpha=0.97,
    ),
)
ax2.text(
    5,
    3.30,
    "31,635  road segments\n"
    "\u00d7   3  points   (start,  end,  midpoint)\n"
    "\u00d7   2  values   (longitude,  latitude)",
    ha="center",
    va="top",
    fontsize=10,
    color="#333333",
    linespacing=1.75,
)

# Usage note
ax2.text(
    5,
    1.15,
    "Model uses midpoint coordinates\n"
    "to encode the spatial position\n"
    "of each road segment as a node.",
    ha="center",
    va="center",
    fontsize=9.5,
    color="#444444",
    style="italic",
    bbox=dict(
        boxstyle="round,pad=0.42",
        facecolor="#FFFCF0",
        edgecolor="#D08020",
        linewidth=1.1,
        alpha=0.95,
    ),
)

# ── Save ─────────────────────────────────────────────────────────────────────
OUT = r"C:\Users\zamin\Downloads\Nazim\round_10_pos_map.png"
plt.savefig(OUT, dpi=200, bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved:", OUT)
