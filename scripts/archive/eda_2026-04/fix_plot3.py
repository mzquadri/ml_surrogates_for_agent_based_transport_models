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
y_np = g.y.numpy().flatten()  # [31635]

BLUE = "#4878A8"
RED = "#D66B6B"
GREEN = "#5DA573"
GOLD = "#D4A843"
GREY = "#888888"
CREAM = "#F5F0E8"
BG = "#FFFFFF"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "figure.facecolor": BG,
        "axes.facecolor": BG,
    }
)
OUT = r"C:\Users\zamin\Downloads\Nazim"

fig = plt.figure(figsize=(16, 8))
fig.patch.set_facecolor(BG)
fig.suptitle(
    "Attribute 3 of 7 :   y   [31,635 x 1]   --   The TARGET value",
    fontsize=16,
    fontweight="bold",
    color="#222222",
    y=1.01,
)

# ── Left panel: big histogram ─────────────────────────────────────────────────
ax = fig.add_axes([0.06, 0.18, 0.54, 0.70])
ax.set_facecolor(BG)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

neg = y_np[y_np < -0.5]
zero = y_np[(y_np >= -0.5) & (y_np <= 0.5)]
pos = y_np[y_np > 0.5]

ax.hist(
    neg,
    bins=60,
    color=RED,
    alpha=0.82,
    label=f"Traffic DECREASED  ({len(neg):,} roads)",
)
ax.hist(
    zero, bins=20, color=GREY, alpha=0.60, label=f"No change  ({len(zero):,} roads)"
)
ax.hist(
    pos,
    bins=60,
    color=GREEN,
    alpha=0.82,
    label=f"Traffic INCREASED  ({len(pos):,} roads)",
)
ax.axvline(0, color="#111111", linewidth=2.0, linestyle="--", zorder=5)

# annotate the zero line
ax.text(
    1,
    ax.get_ylim()[1] * 0.95,
    "<-- zero line\n(no change)",
    fontsize=9,
    color="#111111",
    va="top",
)

ax.set_xlabel(
    "X AXIS  -->  Delta car volume  (delta veh/hr)\n"
    '"How many MORE or FEWER cars per hour on this road after the scenario?"\n'
    "Negative = fewer cars     |     Zero = no change     |     Positive = more cars",
    fontsize=10,
    labelpad=12,
    color="#333333",
)
ax.set_ylabel(
    "Y AXIS  -->  Number of road segments\n"
    "(out of 31,635 total)\nthat have this delta value",
    fontsize=10,
    labelpad=10,
    color="#333333",
)
ax.set_title(
    f"Range: [{y_np.min():.1f},  {y_np.max():.1f}]  veh/hr     "
    f"Mean = {y_np.mean():.2f}  veh/hr",
    fontsize=10,
    color="#666666",
    style="italic",
)
ax.legend(fontsize=10, frameon=False, loc="upper left")

# ── Right panel: explanation ──────────────────────────────────────────────────
ax2 = fig.add_axes([0.63, 0.05, 0.35, 0.88])
ax2.axis("off")
ax2.set_facecolor(BG)

ax2.text(
    0.5,
    0.99,
    "How to read this chart?",
    ha="center",
    va="top",
    fontsize=13,
    fontweight="bold",
    color="#222222",
    transform=ax2.transAxes,
)

# --- Three explanation boxes ---
boxes = [
    # (y_start, height, color, title, body)
    (
        0.70,
        0.26,
        RED,
        "RED bars  (left side, negative values)",
        "These roads have FEWER cars after\nthe scenario.\n\n"
        "Example:  y = -30  means\n"
        "  --> 30 fewer cars per hour on that road.\n"
        "  --> Maybe a road was closed nearby,\n"
        "       so drivers took a different route.",
    ),
    (
        0.38,
        0.28,
        GREY,
        "GREY bars  (near zero)",
        "These roads are NOT affected.\n\n"
        "Example:  y = 0  means\n"
        "  --> Exact same number of cars as before.\n"
        "  --> This road is far from the scenario change\n"
        "       so traffic did not shift here.",
    ),
    (
        0.05,
        0.28,
        GREEN,
        "GREEN bars  (right side, positive values)",
        "These roads have MORE cars after\nthe scenario.\n\n"
        "Example:  y = +25  means\n"
        "  --> 25 extra cars per hour on that road.\n"
        "  --> Drivers diverted here because\n"
        "       another road was restricted.",
    ),
]

for y0, h, clr, title, body in boxes:
    rect = mpatches.FancyBboxPatch(
        (0.01, y0),
        0.98,
        h,
        transform=ax2.transAxes,
        boxstyle="round,pad=0.02",
        linewidth=1.8,
        edgecolor=clr,
        facecolor=CREAM,
        clip_on=False,
    )
    ax2.add_patch(rect)
    ax2.text(
        0.50,
        y0 + h - 0.02,
        title,
        ha="center",
        va="top",
        transform=ax2.transAxes,
        fontsize=9.5,
        fontweight="bold",
        color=clr,
    )
    ax2.text(
        0.50,
        y0 + h - 0.07,
        body,
        ha="center",
        va="top",
        transform=ax2.transAxes,
        fontsize=9.0,
        color="#222222",
    )

plt.savefig(
    f"{OUT}\\detail_3_y_target_fixed.png", dpi=150, bbox_inches="tight", facecolor=BG
)
plt.close()
print("Saved detail_3_y_target_fixed.png")
