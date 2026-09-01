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
vals = graphs[0].x[:, 3].numpy()  # FREESPEED  (m/s)

GOLD = "#D4A843"
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

ax.hist(vals, bins=60, color=GOLD, alpha=0.85, edgecolor="white", linewidth=0.3)

mean_val = np.mean(vals)
ax.axvline(mean_val, color="#CC0000", linewidth=2.2, linestyle="--", zorder=5)
ax.text(
    mean_val + 0.3,
    ax.get_ylim()[1] * 0.80,
    f"Mean = {mean_val:.1f} m/s\n= {mean_val * 3.6:.0f} km/h",
    fontsize=10.5,
    color="#CC0000",
    fontweight="bold",
)

# mark common speed limits on x axis
common_speeds = [
    (4.17, "15\nkm/h"),
    (8.33, "30\nkm/h"),
    (13.89, "50\nkm/h"),
    (33.33, "120\nkm/h"),
]
for ms, label in common_speeds:
    ax.axvline(ms, color="#AAAAAA", linewidth=1.0, linestyle=":", zorder=3)
    ax.text(
        ms, -ax.get_ylim()[1] * 0.13, label, ha="center", fontsize=8.5, color="#666666"
    )

ax.set_xlabel(
    "X AXIS  --  Speed limit  (metres per second, m/s)\n"
    "Upar  diye hain km/h mein bhi  --  1 m/s = 3.6 km/h",
    fontsize=11,
    labelpad=22,
    color="#333333",
)
ax.set_ylabel(
    "Y AXIS  --  Kitni sadken hain\niss speed limit ke saath\n(out of 31,635)",
    fontsize=11,
    labelpad=10,
    color="#333333",
)
ax.set_title(
    "Round 5 / 12   --   Feature 4 :   FREESPEED\n"
    '"Is sadak par gaadi kitni tez ja sakti hai?"',
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
    edgecolor=GOLD,
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
    color=GOLD,
)
ax2.text(
    0.5,
    0.87,
    "Speed limit -- m/s mein\n(km/h mein convert karo\n x 3.6 se)\n\n"
    "4.2 m/s  =  15 km/h  (gali)\n"
    "8.3 m/s  =  30 km/h  (Paris street)\n"
    "13.9 m/s =  50 km/h  (badi sadak)\n"
    "33.3 m/s = 120 km/h  (motorway)",
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
    0.55,
    "Do bade peaks hain:\n\n"
    "1. ~8.3 m/s (30 km/h)\n"
    "   Paris ki zyaadatar sadken\n"
    "   30 km/h wali hain\n\n"
    "2. ~13.9 m/s (50 km/h)\n"
    "   Badi boulevards aur\n"
    "   main roads",
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
    "Mean ka matlab:",
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
    f"Mean = {mean_val:.1f} m/s\n"
    f"     = {mean_val * 3.6:.0f} km/h\n\n"
    "Paris ne 2021 mein\nshehr ki zyaadatar sadkon\nko 30 km/h kar diya.\n"
    "Yeh data usi policy\nko reflect karta hai.",
    ha="center",
    va="top",
    transform=ax2.transAxes,
    fontsize=10,
    color="#222222",
    linespacing=1.5,
)

plt.savefig(
    r"C:\Users\zamin\Downloads\Nazim\round_05_freespeed.png",
    dpi=150,
    bbox_inches="tight",
    facecolor=BG,
)
plt.close()
print("Saved.")
