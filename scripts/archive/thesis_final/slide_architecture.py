"""Slide-friendly PointNetTransfGAT architecture diagram (landscape, 16:9).

Designed for a presentation slide — larger fonts, more breathing room, parameter
counts shown per block, and explicit input/output labels. Architecture facts
come from Zamin's thesis Table 3.2 and Section 3.2; the underlying architecture
was originally proposed by Natterer et al. (2025), which the slide caption
should reference.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Soft, projector-friendly TUM-aligned palette
INPUT_COLOR  = "#DAD7CB"   # TUM accent grey
BLOCK_COLOR  = "#98C6EA"   # TUM accent light blue
OUTPUT_COLOR = "#F4B183"   # soft pale orange
EDGE         = "#555555"
TEXT_DARK    = "#1F1F1F"
ANNO_COLOR   = "#666666"

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "font.size": 14,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

OUT = "C:/Users/zamin/Downloads/ml_surrogates_thesis_final/ml_surrogates_thesis_final/document/figures/new"

# 16:9 landscape canvas
fig, ax = plt.subplots(figsize=(13, 6.5))
ax.set_xlim(0, 13)
ax.set_ylim(0, 6.5)
ax.axis("off")


def draw_block(x_center, y_center, width, height, text_lines, fill, border):
    rect = FancyBboxPatch(
        (x_center - width/2, y_center - height/2),
        width, height,
        boxstyle="round,pad=0.05,rounding_size=0.12",
        facecolor=fill, edgecolor=border, linewidth=1.4,
    )
    ax.add_patch(rect)
    if isinstance(text_lines, str):
        ax.text(x_center, y_center, text_lines,
                ha="center", va="center",
                fontsize=13, color=TEXT_DARK)
    else:
        title, *rest = text_lines
        n = 1 + len(rest)
        # title on top, sub-lines underneath
        line_h = height / (n + 0.4)
        ax.text(x_center, y_center + line_h*0.5, title,
                ha="center", va="center",
                fontsize=13.5, fontweight="bold", color=TEXT_DARK)
        for k, sub in enumerate(rest):
            ax.text(x_center, y_center + line_h*0.5 - line_h*(k+1), sub,
                    ha="center", va="center",
                    fontsize=11, color=ANNO_COLOR, style="italic")


def draw_arrow(x_from, x_to, y, dx_pad=0.08):
    arr = FancyArrowPatch(
        (x_from + dx_pad, y), (x_to - dx_pad, y),
        arrowstyle="-|>", mutation_scale=14,
        color=EDGE, linewidth=1.2,
    )
    ax.add_patch(arr)


# ---- Layout (5 columns of stages, plus input + output bookends)
y_main = 3.6
block_w = 1.65
block_h = 1.35

# Block centres along the x-axis
xs = [0.95, 3.05, 5.20, 7.35, 9.50, 11.85]

# Block contents: (lines, fill, border)
blocks = [
    (["Input Graph",
      "31,635 nodes",
      "5 features / node"],                         INPUT_COLOR,  EDGE),
    (["PointNetConv  ×2",
      "local geometry",
      "from coords"],                                BLOCK_COLOR,  EDGE),
    (["TransformerConv ×2",
      "long-range",
      "attention"],                                  BLOCK_COLOR,  EDGE),
    (["GATConv  ×2",
      "node-level",
      "embedding (64-d)"],                           BLOCK_COLOR,  EDGE),
    (["Output head",
      "GATConv (T2--T8)",
      "or Linear (T1)"],                             BLOCK_COLOR,  EDGE),
    (["Output",
      "Δv per segment",
      "(veh/h)"],                                    OUTPUT_COLOR, EDGE),
]

for x, (lines, fill, border) in zip(xs, blocks):
    draw_block(x, y_main, block_w, block_h, lines, fill, border)

# Arrows between consecutive blocks
for i in range(len(xs) - 1):
    draw_arrow(xs[i] + block_w/2, xs[i+1] - block_w/2, y_main)

# ---- Input feature list under the input block
input_features = [
    "VOL_BASE_CASE",
    "CAPACITY_BASE_CASE",
    "CAPACITY_REDUCTION",
    "FREESPEED",
    "LENGTH",
]
ax.text(xs[0], y_main - block_h/2 - 0.45,
        "5 input features:",
        ha="center", va="top", fontsize=10.5,
        color=TEXT_DARK, fontweight="bold")
for k, feat in enumerate(input_features):
    ax.text(xs[0], y_main - block_h/2 - 0.75 - k*0.30,
            feat, ha="center", va="top", fontsize=10,
            color=ANNO_COLOR, family="monospace")

# ---- Dropout-active bracket above stages 2 and 3 (PointNetConv + TransformerConv)
bracket_x_left  = xs[1] - block_w/2
bracket_x_right = xs[2] + block_w/2
bracket_y       = y_main + block_h/2 + 0.35

ax.plot([bracket_x_left, bracket_x_right], [bracket_y, bracket_y],
        color=ANNO_COLOR, linewidth=1.0)
ax.plot([bracket_x_left, bracket_x_left], [bracket_y, bracket_y - 0.10],
        color=ANNO_COLOR, linewidth=1.0)
ax.plot([bracket_x_right, bracket_x_right], [bracket_y, bracket_y - 0.10],
        color=ANNO_COLOR, linewidth=1.0)
ax.text((bracket_x_left + bracket_x_right) / 2, bracket_y + 0.15,
        "Dropout active here — used by MC Dropout at inference",
        ha="center", va="bottom", fontsize=10.5,
        color=ANNO_COLOR, style="italic")

# ---- Trainable parameter total at bottom right
ax.text(xs[-1], y_main - block_h/2 - 0.55,
        "Total: ≈ 1,416,768 trainable parameters",
        ha="center", va="top", fontsize=10.5,
        color=TEXT_DARK, fontweight="bold")
ax.text(xs[-1], y_main - block_h/2 - 0.95,
        "(frozen in Trial 9 / Trial 11;\nonly the 134-parameter head trains)",
        ha="center", va="top", fontsize=9.5,
        color=ANNO_COLOR, style="italic")

# ---- Source caption (bottom-left)
ax.text(0.15, 0.25,
        "Architecture: PointNetTransfGAT, adapted from Natterer et al. (2025).",
        ha="left", va="bottom", fontsize=10,
        color=ANNO_COLOR, style="italic")

fig.tight_layout(pad=0.35)
fig.savefig(f"{OUT}/slide_architecture.pdf")
fig.savefig(f"{OUT}/slide_architecture.png", dpi=300)
plt.close(fig)
print("Saved slide_architecture.pdf and slide_architecture.png")
