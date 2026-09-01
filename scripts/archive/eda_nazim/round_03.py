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
vals = graphs[0].x[:, 1].numpy()  # CAPACITY_BASE

GREEN = "#5DA573"
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

ax.hist(vals, bins=80, color=GREEN, alpha=0.80, edgecolor="white", linewidth=0.3)

mean_val = np.mean(vals)
ax.axvline(mean_val, color="#CC0000", linewidth=2.2, linestyle="--", zorder=5)
ax.text(
    mean_val + 120,
    ax.get_ylim()[1] * 0.82,
    f"Mean = {mean_val:.0f} veh/hr",
    fontsize=10.5,
    color="#CC0000",
    fontweight="bold",
)

ax.set_xlabel(
    "X AXIS  --  Road ki capacity  (vehicles per hour)",
    fontsize=11,
    labelpad=10,
    color="#333333",
)
ax.set_ylabel(
    "Y AXIS  --  Kitni sadken hain\niss capacity ke saath\n(out of 31,635)",
    fontsize=11,
    labelpad=10,
    color="#333333",
)
ax.set_title(
    "Round 3 / 12   --   Feature 2 :   CAPACITY_BASE\n"
    '"Is sadak par ek ghante mein zyaada se zyaada kitni gaariyan aa sakti hain?"',
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
    edgecolor=GREEN,
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
    color=GREEN,
)
ax2.text(
    0.5,
    0.87,
    "Road ki maximum capacity --\nkitni gaariyan handle kar\nskti hai ek ghante mein.\n\n"
    "0      =  koi gari nahi ja sakti\n"
    "1,000  =  ek hazaar gari/ghanta\n"
    "14,400 =  motorway (bahut badi)",
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
    "Y Axis kya hai?",
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
    "Kitni sadken hain jinka\ncapacity woh value hai.\n\n"
    "Ek unchi bar ka matlab:\nbahut saari sadken hain\niss capacity level par.",
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
    "VOL se farq kya hai?",
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
    "VOL  = abhi kitni gaariyan\n"
    "         hain  (actual traffic)\n\n"
    "CAPACITY = zyaada se zyaada\n"
    "              kitni aa sakti hain\n\n"
    "Agar VOL > CAPACITY\n"
    "--> Road jam ho gayi!",
    ha="center",
    va="top",
    transform=ax2.transAxes,
    fontsize=10,
    color="#222222",
    linespacing=1.5,
)

plt.savefig(
    r"C:\Users\zamin\Downloads\Nazim\round_03_capacity_base.png",
    dpi=150,
    bbox_inches="tight",
    facecolor=BG,
)
plt.close()
print("Saved.")
