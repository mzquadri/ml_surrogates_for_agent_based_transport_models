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
vals = graphs[0].x[:, 2].numpy()  # CAPACITY_REDUCTION

RED = "#D66B6B"
BLUE = "#4878A8"
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

ax.hist(vals, bins=80, color=RED, alpha=0.80, edgecolor="white", linewidth=0.3)

mean_val = np.mean(vals)
ax.axvline(mean_val, color="#CC0000", linewidth=2.2, linestyle="--", zorder=5)
ax.axvline(0, color="#333333", linewidth=1.5, linestyle=":", zorder=5)

# annotations
ymax = ax.get_ylim()[1]
ax.text(
    0 + 30,
    ymax * 0.90,
    "Zero\n(koi change\nnahi)",
    fontsize=9,
    color="#333333",
    va="top",
)
ax.text(
    mean_val - 50,
    ymax * 0.72,
    f"Mean = {mean_val:.1f}",
    fontsize=10.5,
    color="#CC0000",
    fontweight="bold",
    ha="right",
)

ax.set_xlabel(
    "X AXIS  --  Capacity kitni GHAI  (vehicles per hour)\n"
    "Hamesha zero ya negative  --  capacity sirf ghatti hai, badhti nahi",
    fontsize=11,
    labelpad=10,
    color="#333333",
)
ax.set_ylabel(
    "Y AXIS  --  Kitni sadken hain\njinka capacity utna ghata\n(out of 31,635)",
    fontsize=11,
    labelpad=10,
    color="#333333",
)
ax.set_title(
    "Round 4 / 12   --   Feature 3 :   CAPACITY_REDUCTION\n"
    '"Is scenario mein, is sadak ki capacity kitni kam ki gayi?"',
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
    edgecolor=RED,
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
    color=RED,
)
ax2.text(
    0.5,
    0.87,
    "Capacity mein kami\n(hamesha 0 ya negative)\n\n"
    "0     =  sadak bilkul theek hai\n"
    "         koi change nahi\n"
    "-500  =  500 veh/hr capacity\n"
    "         kam kar di gayi\n"
    "-4800 =  road lagbhag band",
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
    edgecolor=BLUE,
    facecolor=CREAM,
    clip_on=False,
)
ax2.add_patch(r2)
ax2.text(
    0.5,
    0.63,
    "Chart ko kaise parhein?",
    ha="center",
    va="top",
    transform=ax2.transAxes,
    fontsize=11,
    fontweight="bold",
    color=BLUE,
)
ax2.text(
    0.5,
    0.55,
    "Chart mein dekho --\nSabse unchi bar = 0 par\n\n"
    "Matlab: is scenario mein\nzyaadatar sadkon ki\ncapacity NAHI ghayi.\n\n"
    "Sirf chhoti kuch sadkon\npar reduction hua hai.",
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
    "Ek sadak ki capacity\n"
    "pehle thi: 1,000 veh/hr\n\n"
    "CAPACITY_REDUCTION = -300\n\n"
    "Matlab: ek lane band kar\n"
    "di — ab sirf 700 veh/hr\n"
    "ja sakti hain us sadak se.",
    ha="center",
    va="top",
    transform=ax2.transAxes,
    fontsize=10,
    color="#222222",
    linespacing=1.5,
)

plt.savefig(
    r"C:\Users\zamin\Downloads\Nazim\round_04_capacity_reduction.png",
    dpi=150,
    bbox_inches="tight",
    facecolor=BG,
)
plt.close()
print("Saved.")
