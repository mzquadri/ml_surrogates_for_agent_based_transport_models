import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

# ── Academic palette ─────────────────────────────────────────────────────────
CLR_FILE = "#4878A8"  # steel blue  — top level: the .pt file
CLR_GRAPH = "#5DA573"  # sage green  — second level: one graph
CLR_ATTR = "#F5F0E8"  # warm cream  — attribute boxes background
CLR_BORDER = "#888888"  # grey border
CLR_ARROW = "#555555"
CLR_BG = "#FFFFFF"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
    }
)

fig, ax = plt.subplots(figsize=(13, 8))
fig.patch.set_facecolor(CLR_BG)
ax.set_facecolor(CLR_BG)
ax.set_xlim(0, 13)
ax.set_ylim(0, 8)
ax.axis("off")

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(
    6.5,
    7.6,
    "Structure of one Training File  (.pt)",
    ha="center",
    va="center",
    fontsize=15,
    fontweight="bold",
    color="#222222",
)

# ── Level 1: The .pt file ────────────────────────────────────────────────────
file_box = FancyBboxPatch(
    (0.4, 6.1),
    12.2,
    0.9,
    boxstyle="round,pad=0.1",
    linewidth=2,
    edgecolor=CLR_FILE,
    facecolor="#D6E4F0",
)
ax.add_patch(file_box)
ax.text(
    6.5,
    6.55,
    "datalist_batch_1.pt   —   one file on disk  (~125 MB)",
    ha="center",
    va="center",
    fontsize=12,
    color="#1a3a5c",
    fontweight="bold",
)

# ── Arrow: file → graphs ──────────────────────────────────────────────────────
ax.annotate(
    "",
    xy=(6.5, 5.65),
    xytext=(6.5, 6.1),
    arrowprops=dict(arrowstyle="->", color=CLR_ARROW, lw=1.8),
)
ax.text(6.8, 5.87, "50 graphs inside", fontsize=9.5, color="#555555", style="italic")

# ── Level 2: "50 graphs" band ────────────────────────────────────────────────
graph_box = FancyBboxPatch(
    (0.4, 4.85),
    12.2,
    0.75,
    boxstyle="round,pad=0.1",
    linewidth=2,
    edgecolor="#3d8b5e",
    facecolor="#D4EDDA",
)
ax.add_patch(graph_box)
ax.text(
    6.5,
    5.225,
    "Graph 0,  Graph 1,  Graph 2,  …,  Graph 49"
    "        ←  each graph = one transport scenario in Paris",
    ha="center",
    va="center",
    fontsize=11,
    color="#1a4a2e",
    fontweight="bold",
)

# ── Arrow: one graph zoomed ───────────────────────────────────────────────────
ax.annotate(
    "",
    xy=(6.5, 4.5),
    xytext=(6.5, 4.85),
    arrowprops=dict(arrowstyle="->", color=CLR_ARROW, lw=1.8),
)
ax.text(6.8, 4.67, "zoom into one graph", fontsize=9.5, color="#555555", style="italic")

# ── Level 3: Attributes of one graph ─────────────────────────────────────────
attrs = [
    # (label,               shape_text,         explanation,                         color_accent)
    ("num_nodes", "31,635", "Total road segments in Paris", "#4878A8"),
    ("x", "[31635 × 6]", "6 features per road segment", "#4878A8"),
    ("y", "[31635 × 1]", "Target: Δ car volume (veh/hr)", "#D66B6B"),
    ("edge_index", "[2 × 59,851]", "Road-to-road connections", "#5DA573"),
    ("pos", "[31635 × 3 × 2]", "GPS coordinates (lon, lat)", "#D4A843"),
    ("mode_stats_diff", "[6 × 3]", "Δ trips by transport mode (abs)", "#888888"),
    ("mode_stats_diff_perc", "[6 × 3]", "Δ trips by transport mode (%)", "#888888"),
]

n = len(attrs)
col_w = 12.2 / n  # width of each attribute column
y_top = 4.45
y_bot = 0.35
box_h = y_top - y_bot

for i, (name, shape, expl, accent) in enumerate(attrs):
    x0 = 0.4 + i * col_w
    # outer box
    rect = FancyBboxPatch(
        (x0 + 0.05, y_bot),
        col_w - 0.1,
        box_h,
        boxstyle="round,pad=0.05",
        linewidth=1.5,
        edgecolor=accent,
        facecolor=CLR_ATTR,
    )
    ax.add_patch(rect)

    # colored top strip
    strip = FancyBboxPatch(
        (x0 + 0.05, y_top - 0.55),
        col_w - 0.1,
        0.5,
        boxstyle="round,pad=0.04",
        linewidth=0,
        edgecolor="none",
        facecolor=accent,
        alpha=0.18,
    )
    ax.add_patch(strip)

    cx = x0 + col_w / 2  # center x of this column

    # attribute name
    ax.text(
        cx,
        y_top - 0.28,
        name,
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color=accent,
    )

    # shape
    ax.text(
        cx,
        y_top - 0.83,
        shape,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#222222",
    )

    # explanation (wrapped manually)
    words = expl.split()
    lines, cur = [], []
    for w in words:
        cur.append(w)
        if len(" ".join(cur)) > 14:
            lines.append(" ".join(cur[:-1]))
            cur = [w]
    lines.append(" ".join(cur))
    for j, ln in enumerate(lines):
        ax.text(
            cx,
            y_top - 1.35 - j * 0.38,
            ln,
            ha="center",
            va="center",
            fontsize=8.2,
            color="#444444",
        )

# ── Bottom note ───────────────────────────────────────────────────────────────
ax.text(
    6.5,
    0.12,
    "There are 20 such files (batch_1 to batch_20)  →  20 × 50 = 1,000 training graphs total",
    ha="center",
    va="center",
    fontsize=9,
    color="#666666",
    style="italic",
)

plt.tight_layout()
plt.savefig(
    r"C:\Users\zamin\Downloads\Nazim\step1_pt_structure.png",
    dpi=150,
    bbox_inches="tight",
    facecolor=CLR_BG,
)
print("Saved: step1_pt_structure.png")
