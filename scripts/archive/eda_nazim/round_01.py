import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

BLUE = "#4878A8"
GREEN = "#5DA573"
GOLD = "#D4A843"
GREY = "#888888"
CREAM = "#F5F0E8"
BG = "#FFFFFF"

plt.rcParams.update(
    {
        "font.family": "serif",
        "figure.facecolor": BG,
        "axes.facecolor": BG,
    }
)

fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor(BG)
ax.axis("off")
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(
    5,
    9.5,
    "Attribute 1 / 7   --   num_nodes",
    ha="center",
    va="center",
    fontsize=16,
    fontweight="bold",
    color="#222222",
)

# ── Big number ────────────────────────────────────────────────────────────────
ax.text(
    5,
    7.8,
    "31,635",
    ha="center",
    va="center",
    fontsize=80,
    fontweight="bold",
    color=BLUE,
)

ax.text(
    5,
    6.6,
    "road segments in Paris",
    ha="center",
    va="center",
    fontsize=15,
    color="#444444",
)

# ── Divider line ──────────────────────────────────────────────────────────────
ax.plot([1, 9], [5.9, 5.9], color="#CCCCCC", linewidth=1.2)

# ── Three explanation boxes ───────────────────────────────────────────────────
box_data = [
    (
        1.3,
        BLUE,
        "Kya hai yeh?",
        'Paris ki sadkon ko\nchhote tukdon mein\nkaat diya gaya hai.\nHar tukda ek\n"road segment" hai.',
    ),
    (
        4.15,
        GREEN,
        "Example:",
        "Ek lambi sadak jo\n3 crossings se guzre\nusse 3 alag\nsegments mein\nbaanta gaya hai.",
    ),
    (
        7.0,
        GOLD,
        "Kyun important hai?",
        "Har doosri cheez\n(x, y, pos) mein\n31,635 rows hain.\nEk row = ek\nroad segment.",
    ),
]

for cx, clr, title, body in box_data:
    rect = mpatches.FancyBboxPatch(
        (cx - 1.3, 1.2),
        2.6,
        4.4,
        boxstyle="round,pad=0.15",
        linewidth=2,
        edgecolor=clr,
        facecolor=CREAM,
    )
    ax.add_patch(rect)
    ax.text(
        cx,
        5.3,
        title,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=clr,
    )
    ax.plot([cx - 1.1, cx + 1.1], [4.85, 4.85], color=clr, linewidth=0.8, alpha=0.5)
    ax.text(
        cx,
        3.1,
        body,
        ha="center",
        va="center",
        fontsize=10.5,
        color="#333333",
        linespacing=1.6,
    )

# ── Bottom note ───────────────────────────────────────────────────────────────
ax.text(
    5,
    0.7,
    "num_nodes ek simple integer hai -- koi graph ya chart nahi hota iske liye.",
    ha="center",
    va="center",
    fontsize=9.5,
    color="#888888",
    style="italic",
)

plt.tight_layout()
plt.savefig(
    r"C:\Users\zamin\Downloads\Nazim\round_01_num_nodes.png",
    dpi=150,
    bbox_inches="tight",
    facecolor=BG,
)
plt.close()
print("Saved round_01_num_nodes.png")
