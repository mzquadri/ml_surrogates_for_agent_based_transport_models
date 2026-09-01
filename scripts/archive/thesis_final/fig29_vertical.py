"""Regenerate fig29_pointnet_architecture as a VERTICAL flow diagram.
5 inches wide, 7 inches tall, 6 boxes top-to-bottom, single line of 11pt text per box.
"""
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

PRIMARY = "#5B9BD5"; SECOND = "#ED7D31"; NEUTRAL = "#A5A5A5"
EDGE = "#888888"; GRID = "#E5E5E5"; TICKDARK = "#404040"

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "font.size": 11, "axes.titlesize": 11, "axes.titleweight": "normal",
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

OUT = "C:/Users/zamin/Downloads/ml_surrogates_thesis_final/ml_surrogates_thesis_final/document/figures/new"

fig, ax = plt.subplots(figsize=(5, 7))
ax.set_xlim(0, 5); ax.set_ylim(0, 7); ax.axis("off")

stages = [
    "Input Graph",
    "PointNet Convolution (x2)",
    "Transformer Convolution (x2)",
    "GAT Convolution (x2)",
    "GATConv / Linear head",
    "Output: change in traffic volume",
]

bw = 3.0      # box width (axes units; figure width is 5)
bh = 0.62     # box height
x_centre = 2.0  # leaves 0.5 in margin left and 1.5 in for the right-side annotation

# Vertical positions: top to bottom, evenly spaced
y_centres = [6.4, 5.4, 4.4, 3.4, 2.4, 1.4]

for i, (y, text) in enumerate(zip(y_centres, stages)):
    border = SECOND if i == 5 else PRIMARY
    rect = FancyBboxPatch((x_centre - bw/2, y - bh/2), bw, bh,
                          boxstyle="round,pad=0.04,rounding_size=0.08",
                          facecolor="white", edgecolor=border, linewidth=1.0)
    ax.add_patch(rect)
    ax.text(x_centre, y, text, ha="center", va="center", fontsize=11, color="black")

# Downward arrows between adjacent boxes
for i in range(len(y_centres) - 1):
    y_top = y_centres[i] - bh/2
    y_bot = y_centres[i+1] + bh/2
    ax.annotate("", xy=(x_centre, y_bot + 0.02),
                xytext=(x_centre, y_top - 0.02),
                arrowprops=dict(arrowstyle="->", lw=0.9, color="black"))

# Right-side annotation pointing to stages 2-4
y_top_anno = y_centres[1]   # stage 2 (PointNet Conv)
y_bot_anno = y_centres[3]   # stage 4 (GAT Conv)
y_mid_anno = (y_top_anno + y_bot_anno) / 2
x_anno = 4.05

# Vertical bracket on the right
ax.plot([x_anno, x_anno], [y_top_anno + bh/2, y_bot_anno - bh/2], color=NEUTRAL, linewidth=0.8)
ax.plot([x_anno - 0.06, x_anno], [y_top_anno + bh/2, y_top_anno + bh/2], color=NEUTRAL, linewidth=0.8)
ax.plot([x_anno - 0.06, x_anno], [y_bot_anno - bh/2, y_bot_anno - bh/2], color=NEUTRAL, linewidth=0.8)

# Annotation text to the right of bracket — wrapped to 3 short lines
ax.text(x_anno + 0.08, y_mid_anno,
        "Dropout active\nduring MC Dropout\ninference (stages 2--4)",
        ha="left", va="center", fontsize=8.5, color=NEUTRAL, style="italic")

fig.tight_layout(pad=0.25)
fig.savefig(f"{OUT}/fig29_pointnet_architecture.pdf")
fig.savefig(f"{OUT}/fig29_pointnet_architecture.png", dpi=300)
plt.close(fig)
print("saved fig29_pointnet_architecture (vertical)")
