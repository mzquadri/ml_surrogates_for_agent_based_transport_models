import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import torch
from collections import Counter

# ── Load data ─────────────────────────────────────────────────────────────────
PT = (
    r"C:\Users\zamin\Downloads\Nazim\Thesis\Nazim\ml_surrogates_thesis_final"
    r"\code\data\train_data\dist_not_connected_10k_1pct\datalist_batch_1.pt"
)
data = torch.load(PT, weights_only=False, map_location="cpu")

# Use first graph (same network topology across all 50 graphs)
g = data[0]
n_nodes = g.x.shape[0]  # 31,635
edge_index = g.edge_index  # [2, num_edges]
n_edges = edge_index.shape[1]

src = edge_index[0].numpy()
dst = edge_index[1].numpy()

out_deg_c = Counter(src)
in_deg_c = Counter(dst)

in_deg_arr = np.array([in_deg_c.get(i, 0) for i in range(n_nodes)])
out_deg_arr = np.array([out_deg_c.get(i, 0) for i in range(n_nodes)])
total_deg = in_deg_arr + out_deg_arr

mean_deg = float(total_deg.mean())
median_deg = float(np.median(total_deg))
max_deg = int(total_deg.max())
zero_deg = int((total_deg == 0).sum())

# ── Palette ───────────────────────────────────────────────────────────────────
BG = "#FFFFFF"
C_TOTAL = "#2166AC"  # cobalt blue
C_IN = "#1A7034"  # forest green
C_OUT = "#B2182B"  # crimson
GRAY = "#888888"
CREAM = "#F7F4EF"

plt.rcParams.update(
    {
        "font.family": "serif",
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    }
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={"wspace": 0.40})
fig.patch.set_facecolor(BG)

# ── Title block ───────────────────────────────────────────────────────────────
fig.text(
    0.5,
    0.990,
    "Node Degree Distribution  —  Road Network Connectivity",
    ha="center",
    va="top",
    fontsize=20,
    fontweight="bold",
    color="#0D0D0D",
)
fig.text(
    0.5,
    0.952,
    f"31,635 road segments  |  {n_edges:,} directed edges  "
    f"|  Paris traffic network  |  1,000 disruption scenarios",
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


def style_ax(ax):
    for sp in ax.spines.values():
        sp.set_color("#BBBBBB")
        sp.set_linewidth(0.85)
        sp.set_visible(True)
    ax.set_facecolor(BG)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="both", length=3, color="#BBBBBB", direction="in")


# ─────────────────────────────────────────────────────────────────────────────
# Panel (a)  Total degree distribution
# ─────────────────────────────────────────────────────────────────────────────
style_ax(ax1)
ax1.yaxis.grid(True, color="#EEEEEE", linewidth=0.8, zorder=0)

# Degree values typically 0-12 for road networks — use integer bins
max_show = min(max_deg, 14)
deg_vals = np.arange(0, max_show + 1)
counts = np.array([(total_deg == d).sum() for d in deg_vals])

bars = ax1.bar(
    deg_vals,
    counts,
    color=C_TOTAL,
    alpha=0.85,
    edgecolor="white",
    linewidth=0.5,
    zorder=3,
    width=0.7,
)

# Annotate top of each bar with count
for d, c in zip(deg_vals, counts):
    if c > 0:
        ax1.text(
            d,
            c + counts.max() * 0.012,
            f"{c:,}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#333333",
        )

ax1.axvline(
    mean_deg, color="#B2182B", lw=2.0, ls="--", zorder=6, label=f"Mean = {mean_deg:.2f}"
)
ax1.axvline(
    median_deg,
    color="#1A7034",
    lw=2.0,
    ls=":",
    zorder=6,
    label=f"Median = {median_deg:.1f}",
)

ax1.set_xticks(deg_vals)
ax1.set_xlabel("Node Degree  (in + out)", fontsize=12, color="#333333", labelpad=8)
ax1.set_ylabel("Number of Road Segments", fontsize=12, color="#333333", labelpad=8)
ax1.set_title(
    "(a)  Total Degree Distribution",
    fontsize=13.5,
    fontweight="bold",
    color="#111111",
    loc="left",
    pad=8,
)
ax1.set_xlim(-0.6, max_show + 0.6)
ax1.set_ylim(0, counts.max() * 1.18)

# Stats box
stats_txt = (
    f"Nodes  = {n_nodes:,}\n"
    f"Edges  = {n_edges:,}\n"
    f"Mean degree   = {mean_deg:.2f}\n"
    f"Median degree = {median_deg:.1f}\n"
    f"Max degree    = {max_deg}\n"
    f"Isolated nodes = {zero_deg}"
)
ax1.text(
    0.985,
    0.970,
    stats_txt,
    transform=ax1.transAxes,
    ha="right",
    va="top",
    fontsize=9.5,
    color="#222222",
    linespacing=1.65,
    bbox=dict(
        boxstyle="round,pad=0.40",
        facecolor="white",
        edgecolor="#BBBBBB",
        linewidth=1.0,
        alpha=0.95,
    ),
)

leg1 = ax1.legend(fontsize=10, loc="upper left", framealpha=0.95, edgecolor="#CCCCCC")
leg1.get_frame().set_linewidth(0.8)

# ─────────────────────────────────────────────────────────────────────────────
# Panel (b)  In-degree vs Out-degree grouped bars
# ─────────────────────────────────────────────────────────────────────────────
style_ax(ax2)
ax2.yaxis.grid(True, color="#EEEEEE", linewidth=0.8, zorder=0)

max_show2 = min(max(in_deg_arr.max(), out_deg_arr.max()), 9)
deg_vals2 = np.arange(0, max_show2 + 1)
in_counts = np.array([(in_deg_arr == d).sum() for d in deg_vals2])
out_counts = np.array([(out_deg_arr == d).sum() for d in deg_vals2])

w = 0.34
x = deg_vals2
ax2.bar(
    x - w / 2,
    in_counts,
    width=w,
    color=C_IN,
    alpha=0.85,
    edgecolor="white",
    linewidth=0.5,
    zorder=3,
    label="In-degree",
)
ax2.bar(
    x + w / 2,
    out_counts,
    width=w,
    color=C_OUT,
    alpha=0.85,
    edgecolor="white",
    linewidth=0.5,
    zorder=3,
    label="Out-degree",
)

ax2.set_xticks(deg_vals2)
ax2.set_xlabel("Degree Value", fontsize=12, color="#333333", labelpad=8)
ax2.set_ylabel("Number of Road Segments", fontsize=12, color="#333333", labelpad=8)
ax2.set_title(
    "(b)  In-degree vs Out-degree",
    fontsize=13.5,
    fontweight="bold",
    color="#111111",
    loc="left",
    pad=8,
)
ax2.set_xlim(-0.6, max_show2 + 0.6)
ax2.set_ylim(0, max(in_counts.max(), out_counts.max()) * 1.18)

# Interpretation annotation
ax2.text(
    0.985,
    0.970,
    "In = Out  →  balanced network\n(undirected-style road graph)",
    transform=ax2.transAxes,
    ha="right",
    va="top",
    fontsize=9.5,
    color="#333333",
    style="italic",
    bbox=dict(
        boxstyle="round,pad=0.35",
        facecolor=CREAM,
        edgecolor="#BBBBBB",
        linewidth=1.0,
        alpha=0.95,
    ),
)

leg2 = ax2.legend(fontsize=10, loc="upper left", framealpha=0.95, edgecolor="#CCCCCC")
leg2.get_frame().set_linewidth(0.8)

# ── Layout & save ─────────────────────────────────────────────────────────────
plt.subplots_adjust(top=0.858, bottom=0.110, left=0.070, right=0.970)
OUT = r"C:\Users\zamin\Downloads\Nazim\round_09_degree_dist.png"
plt.savefig(OUT, dpi=200, bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved:", OUT)
