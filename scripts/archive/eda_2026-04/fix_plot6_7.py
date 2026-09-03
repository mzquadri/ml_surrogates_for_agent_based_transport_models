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
g = graphs[0]
ms = g.mode_stats_diff.numpy()  # [6, 3]
msp = g.mode_stats_diff_perc.numpy()  # [6, 3]

BLUE = "#4878A8"
RED = "#D66B6B"
GREEN = "#5DA573"
GOLD = "#D4A843"
GREY = "#888888"
PURP = "#9B59B6"
TEAL = "#2E86AB"
CREAM = "#F5F0E8"
BG = "#FFFFFF"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "figure.facecolor": BG,
        "axes.facecolor": BG,
    }
)
OUT = r"C:\Users\zamin\Downloads\Nazim"

modes = ["Car", "Public Transit", "Bike", "Walk", "Freight", "Ride-hailing"]
metrics = ["Trips\n(total count)", "Distance\n(metres)", "Duration\n(seconds)"]
metric_units = ["trips", "metres", "seconds"]
colors_modes = [BLUE, GREEN, GOLD, RED, GREY, PURP]

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 6 — mode_stats_diff  (absolute)
# ═══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor(BG)
fig.suptitle(
    "Attribute 6 of 7 :   mode_stats_diff   [6 x 3]   --   Absolute change in transport use\n"
    "(scenario  minus  base case,  summed across ALL trips in Paris)",
    fontsize=14,
    fontweight="bold",
    color="#222222",
    y=0.99,
)

# Top row: 3 bar charts (one per metric)
for j in range(3):
    ax = fig.add_axes([0.07 + j * 0.325, 0.52, 0.28, 0.38])
    ax.set_facecolor(BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    vals = ms[:, j]
    bars = ax.barh(
        modes, vals, color=colors_modes, alpha=0.82, edgecolor="white", height=0.55
    )
    ax.axvline(0, color="#222222", linewidth=1.5, linestyle="--")
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    ax.set_title(metrics[j], fontsize=12, fontweight="bold", pad=6)
    ax.set_xlabel(
        f"X AXIS  -->  Change in {metric_units[j]}\n"
        "(negative = less,  positive = more,  0 = no change)",
        fontsize=8.5,
        labelpad=8,
        color="#333333",
    )
    ax.set_ylabel(
        "Y AXIS  -->  Transport mode", fontsize=8.5, labelpad=6, color="#333333"
    )

    # value labels on bars
    for bar, v in zip(bars, vals):
        label = f"{v / 1e6:.2f}M" if abs(v) > 1e5 else f"{v:.0f}"
        ax.text(
            v,
            bar.get_y() + bar.get_height() / 2,
            f"  {label}" if v >= 0 else f"{label}  ",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=7.5,
            color="#333333",
        )

# Bottom: explanation panel (full width)
ax_exp = fig.add_axes([0.04, 0.03, 0.92, 0.42])
ax_exp.axis("off")
ax_exp.set_xlim(0, 18)
ax_exp.set_ylim(0, 6)
ax_exp.set_facecolor(BG)

ax_exp.text(
    9,
    5.7,
    "How to read these charts?",
    ha="center",
    fontsize=14,
    fontweight="bold",
    color="#222222",
)

# --- Y axis explanation ---
rect1 = mpatches.FancyBboxPatch(
    (0.1, 3.0),
    5.2,
    2.4,
    boxstyle="round,pad=0.1",
    linewidth=1.8,
    edgecolor=GREEN,
    facecolor=CREAM,
    clip_on=False,
)
ax_exp.add_patch(rect1)
ax_exp.text(
    2.7,
    5.2,
    "Y AXIS  =  Transport Mode",
    ha="center",
    fontsize=11,
    fontweight="bold",
    color=GREEN,
)
ax_exp.text(
    2.7,
    4.7,
    "6 rows, one per travel mode:\n"
    "  Car           = private car trips\n"
    "  Public Transit = metro, bus, RER\n"
    "  Bike          = cycling\n"
    "  Walk          = on foot\n"
    "  Freight       = trucks, delivery\n"
    "  Ride-hailing  = Uber, taxi",
    ha="center",
    va="top",
    fontsize=9.5,
    color="#222222",
)

# --- X axis explanation ---
rect2 = mpatches.FancyBboxPatch(
    (6.2, 3.0),
    5.2,
    2.4,
    boxstyle="round,pad=0.1",
    linewidth=1.8,
    edgecolor=BLUE,
    facecolor=CREAM,
    clip_on=False,
)
ax_exp.add_patch(rect2)
ax_exp.text(
    8.8,
    5.2,
    "X AXIS  =  Change (absolute)",
    ha="center",
    fontsize=11,
    fontweight="bold",
    color=BLUE,
)
ax_exp.text(
    8.8,
    4.7,
    "3 columns = 3 types of measurement:\n"
    "  Trips    = total number of journeys made\n"
    "  Distance = total km/m travelled\n"
    "  Duration = total time spent travelling\n\n"
    "Each value = (scenario - base case)\n"
    "summed over all trips in Paris.",
    ha="center",
    va="top",
    fontsize=9.5,
    color="#222222",
)

# --- Example ---
rect3 = mpatches.FancyBboxPatch(
    (12.4, 3.0),
    5.4,
    2.4,
    boxstyle="round,pad=0.1",
    linewidth=1.8,
    edgecolor=RED,
    facecolor=CREAM,
    clip_on=False,
)
ax_exp.add_patch(rect3)
ax_exp.text(
    15.1, 5.2, "Real Example", ha="center", fontsize=11, fontweight="bold", color=RED
)

val_car_trips = ms[0, 0]
ax_exp.text(
    15.1,
    4.7,
    f"Car row, Trips column = {val_car_trips:.2e}\n\n"
    f"This means:  in this scenario,\n"
    f"the total number of car trips\n"
    f"across all of Paris changed by\n"
    f"{val_car_trips / 1e6:.2f} million trips\n"
    f"compared to the base case.",
    ha="center",
    va="top",
    fontsize=9.5,
    color="#222222",
)

# Bottom note
ax_exp.text(
    9,
    0.3,
    "NOTE: These numbers are CITY-WIDE totals (all of Paris combined)."
    "  They are NOT per-road values.\n"
    "mode_stats_diff is a global summary of how the whole transport system changed.",
    ha="center",
    va="center",
    fontsize=9,
    color="#666666",
    style="italic",
)

plt.savefig(
    f"{OUT}\\detail_6_mode_stats_diff_fixed.png",
    dpi=150,
    bbox_inches="tight",
    facecolor=BG,
)
plt.close()
print("Saved detail_6_mode_stats_diff_fixed.png")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 7 — mode_stats_diff_perc  (percentage)
# ═══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor(BG)
fig.suptitle(
    "Attribute 7 of 7 :   mode_stats_diff_perc   [6 x 3]   --   % change in transport use\n"
    "(same data as attribute 6, but expressed as percentage change)",
    fontsize=14,
    fontweight="bold",
    color="#222222",
    y=0.99,
)

for j in range(3):
    ax = fig.add_axes([0.07 + j * 0.325, 0.52, 0.28, 0.38])
    ax.set_facecolor(BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    vals = msp[:, j]
    bars = ax.barh(
        modes, vals, color=colors_modes, alpha=0.82, edgecolor="white", height=0.55
    )
    ax.axvline(0, color="#222222", linewidth=1.5, linestyle="--")

    ax.set_title(metrics[j], fontsize=12, fontweight="bold", pad=6)
    ax.set_xlabel(
        f"X AXIS  -->  % change in {metric_units[j]}\n"
        "(-100% = completely gone,  0% = no change,  +50% = 50% more)",
        fontsize=8.5,
        labelpad=8,
        color="#333333",
    )
    ax.set_ylabel(
        "Y AXIS  -->  Transport mode", fontsize=8.5, labelpad=6, color="#333333"
    )

    # value labels -- place them OUTSIDE the bar, with enough gap
    x_range = vals.max() - vals.min() if vals.max() != vals.min() else 1
    offset = x_range * 0.03
    for bar, v in zip(bars, vals):
        ax.text(
            v + (offset if v >= 0 else -offset),
            bar.get_y() + bar.get_height() / 2,
            f"{v:.1f}%",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=8.5,
            color="#333333",
            fontweight="bold",
        )

    # give extra margin so labels don't clip
    x_min, x_max = ax.get_xlim()
    ax.set_xlim(x_min - x_range * 0.15, x_max + x_range * 0.15)

# Bottom explanation panel
ax_exp = fig.add_axes([0.04, 0.03, 0.92, 0.42])
ax_exp.axis("off")
ax_exp.set_xlim(0, 18)
ax_exp.set_ylim(0, 6)
ax_exp.set_facecolor(BG)

ax_exp.text(
    9,
    5.7,
    "How to read the percentage chart?",
    ha="center",
    fontsize=14,
    fontweight="bold",
    color="#222222",
)

boxes_pct = [
    (
        0.1,
        "-100%  means...",
        RED,
        "That transport mode has ZERO trips\nin this scenario.\n\n"
        "Example: Car Trips = -100%\n"
        "--> No cars at all in this scenario.\n"
        "(This is an extreme simulation.)",
    ),
    (
        6.2,
        "0%  means...",
        GREY,
        "No change at all.\nExactly same number of trips\nas in the base case.\n\n"
        "Example: Bike Distance = 0%\n"
        "--> Cyclists travelled same total\n    distance as before.",
    ),
    (
        12.4,
        "Positive %  means...",
        GREEN,
        "That mode INCREASED.\n\n"
        "Example: Walk Trips = +30%\n"
        "--> 30% more walking trips\n"
        "    than in the base case.\n"
        "    (People switched from car\n"
        "     to walking.)",
    ),
]

for x0, title, clr, body in boxes_pct:
    rect = mpatches.FancyBboxPatch(
        (x0, 0.6),
        5.2,
        4.8,
        boxstyle="round,pad=0.1",
        linewidth=1.8,
        edgecolor=clr,
        facecolor=CREAM,
        clip_on=False,
    )
    ax_exp.add_patch(rect)
    ax_exp.text(
        x0 + 2.6, 5.2, title, ha="center", fontsize=11, fontweight="bold", color=clr
    )
    ax_exp.text(
        x0 + 2.6, 4.7, body, ha="center", va="top", fontsize=9.5, color="#222222"
    )

ax_exp.text(
    9,
    0.2,
    "Difference from attribute 6:  Attribute 6 gives raw numbers (e.g., -2 million trips)."
    "  Attribute 7 gives percentages (e.g., -95%).\n"
    "Both say the same thing, just in different units.",
    ha="center",
    va="center",
    fontsize=9,
    color="#666666",
    style="italic",
)

plt.savefig(
    f"{OUT}\\detail_7_mode_stats_diff_perc_fixed.png",
    dpi=150,
    bbox_inches="tight",
    facecolor=BG,
)
plt.close()
print("Saved detail_7_mode_stats_diff_perc_fixed.png")
