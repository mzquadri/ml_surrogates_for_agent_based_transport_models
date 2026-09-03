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
vals = graphs[0].x[:, 4].numpy()  # HIGHWAY_TYPE

GREY = "#888888"
BLUE = "#4878A8"
RED = "#D66B6B"
CREAM = "#F5F0E8"
BG = "#FFFFFF"

plt.rcParams.update(
    {"font.family": "serif", "figure.facecolor": BG, "axes.facecolor": BG}
)

# Road type meanings
type_labels = {
    -1: "Unknown",
    0: "Motorway",
    1: "Trunk",
    2: "Primary",
    3: "Secondary",
    4: "Tertiary",
    5: "Residential",
    6: "Living Street",
    7: "Service",
    8: "Unclassified",
    9: "Other",
}

unique, counts = np.unique(vals.astype(int), return_counts=True)
labels = [f"{int(u)}\n{type_labels.get(int(u), '')}" for u in unique]

# color each bar by "importance" of road
bar_colors = [
    "#AAAAAA",
    "#D66B6B",
    "#E8945A",
    "#D4A843",
    "#8BBF6A",
    "#5DA573",
    "#4878A8",
    "#6B9EC7",
    "#9B8DC7",
    "#C78D9B",
    "#888888",
]

fig = plt.figure(figsize=(13, 7))
fig.patch.set_facecolor(BG)

# ── Bar chart ─────────────────────────────────────────────────────────────────
ax = fig.add_axes([0.07, 0.20, 0.56, 0.63])
ax.set_facecolor(BG)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

bars = ax.bar(
    range(len(unique)),
    counts,
    color=bar_colors[: len(unique)],
    alpha=0.85,
    edgecolor="white",
    width=0.65,
)

# count labels on top of each bar
for bar, c in zip(bars, counts):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 100,
        f"{c:,}",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#333333",
    )

ax.set_xticks(range(len(unique)))
ax.set_xticklabels(labels, fontsize=9, color="#333333")

ax.set_xlabel(
    "X AXIS  --  Road ka type  (har number = ek alag qism ki sadak)",
    fontsize=11,
    labelpad=10,
    color="#333333",
)
ax.set_ylabel(
    "Y AXIS  --  Kitni sadken hain\niss type ki\n(out of 31,635)",
    fontsize=11,
    labelpad=10,
    color="#333333",
)
ax.set_title(
    "Round 6 / 12   --   Feature 5 :   HIGHWAY_TYPE\n"
    '"Yeh sadak kis type ki hai?  (gali, badi sadak, motorway...)"',
    fontsize=13,
    fontweight="bold",
    color="#222222",
    pad=10,
)

# ── Right panel ───────────────────────────────────────────────────────────────
ax2 = fig.add_axes([0.67, 0.05, 0.31, 0.88])
ax2.axis("off")
ax2.set_facecolor(BG)

r1 = mpatches.FancyBboxPatch(
    (0.02, 0.55),
    0.96,
    0.42,
    transform=ax2.transAxes,
    boxstyle="round,pad=0.03",
    linewidth=1.8,
    edgecolor=GREY,
    facecolor=CREAM,
    clip_on=False,
)
ax2.add_patch(r1)
ax2.text(
    0.5,
    0.95,
    "Har code ka matlab:",
    ha="center",
    va="top",
    transform=ax2.transAxes,
    fontsize=11,
    fontweight="bold",
    color="#444444",
)
ax2.text(
    0.5,
    0.87,
    "-1  Unknown  (pata nahi)\n"
    " 0  Motorway  (Peripherique)\n"
    " 1  Trunk  (express road)\n"
    " 2  Primary  (badi sadak)\n"
    " 3  Secondary  (medium sadak)\n"
    " 4  Tertiary  (chhoti sadak)\n"
    " 5  Residential  (mohalla)\n"
    " 6  Living Street  (walk zone)\n"
    " 7  Service  (parking/entry)\n"
    " 8  Unclassified\n"
    " 9  Other",
    ha="center",
    va="top",
    transform=ax2.transAxes,
    fontsize=9.5,
    color="#222222",
    linespacing=1.55,
    family="monospace",
)

r2 = mpatches.FancyBboxPatch(
    (0.02, 0.25),
    0.96,
    0.27,
    transform=ax2.transAxes,
    boxstyle="round,pad=0.03",
    linewidth=1.8,
    edgecolor=BLUE,
    facecolor=CREAM,
    clip_on=False,
)
ax2.add_patch(r2)
ax2.text(
    0.5,
    0.50,
    "Chart mein kya dikh raha hai?",
    ha="center",
    va="top",
    transform=ax2.transAxes,
    fontsize=11,
    fontweight="bold",
    color=BLUE,
)
ax2.text(
    0.5,
    0.42,
    "Residential (5) sabse zyaada\nhain -- Paris mein\nmohalle ki sadken\nsabse common hain.",
    ha="center",
    va="top",
    transform=ax2.transAxes,
    fontsize=10,
    color="#222222",
    linespacing=1.5,
)

r3 = mpatches.FancyBboxPatch(
    (0.02, 0.03),
    0.96,
    0.19,
    transform=ax2.transAxes,
    boxstyle="round,pad=0.03",
    linewidth=1.8,
    edgecolor=RED,
    facecolor="#FFF4F4",
    clip_on=False,
)
ax2.add_patch(r3)
ax2.text(
    0.5,
    0.20,
    "Important note:",
    ha="center",
    va="top",
    transform=ax2.transAxes,
    fontsize=11,
    fontweight="bold",
    color=RED,
)
ax2.text(
    0.5,
    0.12,
    "Yeh feature MODEL\nko nahi diya jata.\nSirf reference ke liye hai.",
    ha="center",
    va="top",
    transform=ax2.transAxes,
    fontsize=10,
    color="#222222",
    linespacing=1.5,
)

plt.savefig(
    r"C:\Users\zamin\Downloads\Nazim\round_06_highway_type.png",
    dpi=150,
    bbox_inches="tight",
    facecolor=BG,
)
plt.close()
print("Saved.")
