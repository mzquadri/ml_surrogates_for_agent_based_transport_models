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
vals = graphs[0].x[:, 5].numpy()  # LENGTH (metres)

BLUE = "#4878A8"
GREEN = "#5DA573"
CREAM = "#F5F0E8"
BG = "#FFFFFF"

plt.rcParams.update(
    {"font.family": "serif", "figure.facecolor": BG, "axes.facecolor": BG}
)

fig = plt.figure(figsize=(13, 7))
fig.patch.set_facecolor(BG)

# ── Histogram ─────────────────────────────────────────────────────────────────
ax = fig.add_axes([0.07, 0.18, 0.56, 0.65])
ax.set_facecolor(BG)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.hist(vals, bins=80, color=BLUE, alpha=0.82, edgecolor="white", linewidth=0.3)

mean_val = np.mean(vals)
ax.axvline(mean_val, color="#CC0000", linewidth=2.2, linestyle="--", zorder=5)
ax.text(
    mean_val + 20,
    ax.get_ylim()[1] * 0.80,
    f"Mean = {mean_val:.1f} m",
    fontsize=10.5,
    color="#CC0000",
    fontweight="bold",
)

# reference lines
refs = [(50, "50m"), (100, "100m"), (500, "500m")]
for xv, lbl in refs:
    ax.axvline(xv, color="#BBBBBB", linewidth=1.0, linestyle=":", zorder=3)
    ax.text(
        xv, ax.get_ylim()[1] * 0.96, lbl, ha="center", fontsize=8.5, color="#888888"
    )

ax.set_xlabel(
    "X AXIS  --  Sadak ki lambai  (metres)", fontsize=11, labelpad=10, color="#333333"
)
ax.set_ylabel(
    "Y AXIS  --  Kitni sadken hain\nuss lambai ki\n(out of 31,635)",
    fontsize=11,
    labelpad=10,
    color="#333333",
)
ax.set_title(
    'Round 7 / 12   --   Feature 6 :   LENGTH\n"Yeh road segment kitna lamba hai?"',
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
    (0.02, 0.68),
    0.96,
    0.29,
    transform=ax2.transAxes,
    boxstyle="round,pad=0.03",
    linewidth=1.8,
    edgecolor=BLUE,
    facecolor=CREAM,
    clip_on=False,
)
ax2.add_patch(r1)
ax2.text(
    0.5,
    0.95,
    "X Axis kya hai?",
    ha="center",
    va="top",
    transform=ax2.transAxes,
    fontsize=11,
    fontweight="bold",
    color=BLUE,
)
ax2.text(
    0.5,
    0.87,
    "Road segment ki lambai\nmetres mein.\n\n"
    "4 m    =  bahut chhoti gali\n"
    "91 m   =  average Paris segment\n"
    "500 m  =  lamba road segment\n"
    "2568 m =  sabse lamba segment",
    ha="center",
    va="top",
    transform=ax2.transAxes,
    fontsize=10,
    color="#222222",
    linespacing=1.5,
)

r2 = mpatches.FancyBboxPatch(
    (0.02, 0.36),
    0.96,
    0.29,
    transform=ax2.transAxes,
    boxstyle="round,pad=0.03",
    linewidth=1.8,
    edgecolor=GREEN,
    facecolor=CREAM,
    clip_on=False,
)
ax2.add_patch(r2)
ax2.text(
    0.5,
    0.63,
    "Chart mein kya dikh raha hai?",
    ha="center",
    va="top",
    transform=ax2.transAxes,
    fontsize=11,
    fontweight="bold",
    color=GREEN,
)
ax2.text(
    0.5,
    0.55,
    "Zyaadatar sadken\n0-200 metre ke beech hain.\n\n"
    "Chart left side pe jhukta\nhai -- matlab Paris mein\nchhote chhote segments\nzyaada hain.\n\n"
    "Lamba tail right mein =\nchand badi sadken hain.",
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
    0.30,
    transform=ax2.transAxes,
    boxstyle="round,pad=0.03",
    linewidth=1.8,
    edgecolor="#CC0000",
    facecolor="#FFF4F4",
    clip_on=False,
)
ax2.add_patch(r3)
ax2.text(
    0.5,
    0.31,
    "Real Example:",
    ha="center",
    va="top",
    transform=ax2.transAxes,
    fontsize=11,
    fontweight="bold",
    color="#CC0000",
)
ax2.text(
    0.5,
    0.23,
    "Ek city block = ~80-100m\n\n"
    "Mean = 91.6m ka matlab:\nParis ki ek average sadak\nlagbhag ek city block\njitni lambi hai.\n\n"
    "Peripherique ka ek segment\n= 500m+ tak ho sakta hai.",
    ha="center",
    va="top",
    transform=ax2.transAxes,
    fontsize=10,
    color="#222222",
    linespacing=1.5,
)

plt.savefig(
    r"C:\Users\zamin\Downloads\Nazim\round_07_length.png",
    dpi=150,
    bbox_inches="tight",
    facecolor=BG,
)
plt.close()
print("Saved.")
