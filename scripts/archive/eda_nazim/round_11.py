import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import torch

# ── Load data ─────────────────────────────────────────────────────────────────
PT = (
    r"C:\Users\zamin\Downloads\Nazim\Thesis\Nazim\ml_surrogates_thesis_final"
    r"\code\data\train_data\dist_not_connected_10k_1pct\datalist_batch_1.pt"
)
data = torch.load(PT, weights_only=False, map_location="cpu")

# mode_stats_diff: [6, 3] per graph  — rows = 6 modes, cols = [Trips, Distance(m), Duration(s)]
all_msd = np.stack([d.mode_stats_diff.numpy() for d in data])  # [50, 6, 3]
mean_msd = all_msd.mean(axis=0)  # [6, 3]

# Transport mode labels (row order confirmed from data inspection)
MODES = ["Car", "Public\nTransit", "Bike", "Walk", "Freight", "Ride-\nhailing"]
MODES_CLEAN = ["Car", "Public Transit", "Bike", "Walk", "Freight", "Ride-hailing"]
N_MODES = 6

# Mode colours — one per mode (ColorBrewer-inspired, visually distinct)
MODE_COLORS = [
    "#2166AC",  # Car           — cobalt blue
    "#1A7034",  # Public Transit — forest green
    "#9B6B00",  # Bike          — amber
    "#B2182B",  # Walk          — crimson
    "#555555",  # Freight       — charcoal
    "#6A3D9A",  # Ride-hailing  — purple
]

# Column titles and units
COL_TITLES = ["Trips", "Distance", "Duration"]
COL_SUBTITLES = ["(total count)", "(metres)", "(seconds)"]
COL_XLABEL = [
    "\u0394 Trips  (\u00d710\u2076)",
    "\u0394 Distance  (\u00d710\u2076 m)",
    "\u0394 Duration  (s)",
]
DIVISORS = [1e6, 1e6, 1.0]

# ── Palette & style ───────────────────────────────────────────────────────────
BG = "#FFFFFF"
CREAM = "#F7F4EF"

plt.rcParams.update(
    {
        "font.family": "serif",
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10.5,
    }
)

fig, axes = plt.subplots(1, 3, figsize=(22, 8.5), gridspec_kw={"wspace": 0.42})
fig.patch.set_facecolor(BG)

# ── Title block ───────────────────────────────────────────────────────────────
fig.text(
    0.5,
    0.990,
    "Modal Transport Impact  —  mode_stats_diff   [6 \u00d7 3]",
    ha="center",
    va="top",
    fontsize=20,
    fontweight="bold",
    color="#0D0D0D",
)
fig.text(
    0.5,
    0.952,
    "Absolute change in city-wide transport use  "
    "(disruption scenario \u2212 base case,  summed across all trips in Paris)  "
    "|  Mean across 50 disruption scenarios",
    ha="center",
    va="top",
    fontsize=11,
    color="#555555",
    style="italic",
)
fig.add_artist(
    mlines.Line2D(
        [0.04, 0.96],
        [0.928, 0.928],
        transform=fig.transFigure,
        color="#C8C8C8",
        linewidth=0.9,
    )
)

plt.subplots_adjust(top=0.870, bottom=0.115, left=0.090, right=0.975)


def style_ax(ax):
    for sp in ax.spines.values():
        sp.set_color("#BBBBBB")
        sp.set_linewidth(0.85)
        sp.set_visible(True)
    ax.set_facecolor(BG)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="both", length=3, color="#BBBBBB", direction="in")


y_pos = np.arange(N_MODES)  # 0..5 bottom-to-top

for col_idx, ax in enumerate(axes):
    style_ax(ax)
    ax.xaxis.grid(True, color="#EEEEEE", linewidth=0.8, zorder=0)

    vals = mean_msd[:, col_idx] / DIVISORS[col_idx]  # [6]

    # Horizontal bars
    bars = ax.barh(
        y_pos,
        vals,
        height=0.60,
        color=MODE_COLORS,
        alpha=0.88,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )

    # Zero reference line
    ax.axvline(0, color="#888888", lw=1.0, ls="--", zorder=5, alpha=0.7)

    # Value annotations on each bar
    x_range = vals.max() - vals.min() if vals.max() != vals.min() else 1.0
    for i, v in enumerate(vals):
        offset = x_range * 0.018
        ha = "left" if v >= 0 else "right"
        xpos = v + offset if v >= 0 else v - offset
        fmt = (
            f"{v:+.1f}"
            if col_idx == 2
            else f"{v:+.2f}M"
            if abs(v) >= 0.001
            else f"{v * 1000:+.1f}K"
        )
        # For walk (row 3, col 0): value is -18514 → -0.019M, show as K
        raw = mean_msd[i, col_idx]
        if col_idx == 0:
            if abs(raw) < 1e5:
                fmt = f"{raw / 1000:+.1f}K"
            else:
                fmt = f"{raw / 1e6:+.2f}M"
        elif col_idx == 1:
            fmt = f"{raw / 1e6:+.1f}M"
        else:
            fmt = f"{raw:+.1f} s"

        ax.text(
            xpos,
            i,
            fmt,
            ha=ha,
            va="center",
            fontsize=9,
            color="#111111",
            fontweight="bold",
            zorder=6,
        )

    # Y-axis mode labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(MODES, fontsize=10.5)
    ax.set_ylim(-0.55, N_MODES - 0.45)

    # Extend xlim slightly so annotations don't clip
    pad = abs(vals).max() * 0.22
    xmin = min(vals.min() - pad, 0 - pad * 0.1)
    xmax = max(vals.max() + pad, 0 + pad * 0.1)
    ax.set_xlim(xmin, xmax)

    ax.set_xlabel(COL_XLABEL[col_idx], fontsize=11, color="#333333", labelpad=8)
    ax.set_title(
        COL_TITLES[col_idx],
        fontsize=14,
        fontweight="bold",
        color="#111111",
        loc="center",
        pad=4,
    )
    ax.text(
        0.5,
        1.045,
        COL_SUBTITLES[col_idx],
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#555555",
        style="italic",
        transform=ax.transAxes,
    )

    # Hide y-axis label for cols 1 and 2
    if col_idx > 0:
        ax.set_yticklabels([])

# ── Legend strip (mode → colour) ─────────────────────────────────────────────
from matplotlib.patches import Patch

handles = [
    Patch(facecolor=c, label=m, alpha=0.88) for c, m in zip(MODE_COLORS, MODES_CLEAN)
]
fig.legend(
    handles=handles,
    loc="lower center",
    ncol=6,
    fontsize=10,
    framealpha=0.95,
    edgecolor="#CCCCCC",
    bbox_to_anchor=(0.5, 0.005),
    handlelength=1.2,
    handleheight=0.9,
)

# ── Note at bottom right ──────────────────────────────────────────────────────
fig.text(
    0.975,
    0.012,
    "Values = city-wide aggregate  (all of Paris combined).  "
    "Negative = fewer trips/km/time than base case.",
    ha="right",
    va="bottom",
    fontsize=8.5,
    color="#777777",
    style="italic",
)

OUT = r"C:\Users\zamin\Downloads\Nazim\round_11_mode_stats_diff.png"
plt.savefig(OUT, dpi=200, bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved:", OUT)
