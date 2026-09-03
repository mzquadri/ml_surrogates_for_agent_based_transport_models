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

# mode_stats_diff_perc: [6, 3] per graph
all_msdp = np.stack([d.mode_stats_diff_perc.numpy() for d in data])  # [50, 6, 3]
mean_msdp = all_msdp.mean(axis=0)  # [6, 3]

MODES = ["Car", "Public\nTransit", "Bike", "Walk", "Freight", "Ride-\nhailing"]
MODES_CLEAN = ["Car", "Public Transit", "Bike", "Walk", "Freight", "Ride-hailing"]
N_MODES = 6

MODE_COLORS = [
    "#2166AC",
    "#1A7034",
    "#9B6B00",
    "#B2182B",
    "#555555",
    "#6A3D9A",
]

COL_TITLES = ["Trips", "Distance", "Duration"]
COL_SUBTITLES = ["(total count)", "(metres)", "(seconds)"]

# ── Palette & style ───────────────────────────────────────────────────────────
BG = "#FFFFFF"

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
    "Modal Transport Impact  —  mode_stats_diff_perc   [6 \u00d7 3]",
    ha="center",
    va="top",
    fontsize=20,
    fontweight="bold",
    color="#0D0D0D",
)
fig.text(
    0.5,
    0.952,
    "Percentage change in city-wide transport use  "
    "(disruption scenario \u2212 base case)  "
    "|  Mean across 50 disruption scenarios  "
    "|  \u22120% = no change,  \u2212100% = completely gone",
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


y_pos = np.arange(N_MODES)

# Column-specific x limits
# Trips & Distance: all ≈ -100%, show [-105, 3] to see bars clearly
# Duration: small values, auto
XLIMS = [(-105, 5), (-105, 5), None]

for col_idx, ax in enumerate(axes):
    style_ax(ax)
    ax.xaxis.grid(True, color="#EEEEEE", linewidth=0.8, zorder=0)

    vals = mean_msdp[:, col_idx]  # already in %

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

    # Value annotations
    x_range = (
        XLIMS[col_idx][1] - XLIMS[col_idx][0]
        if XLIMS[col_idx]
        else (vals.max() - vals.min() or 1.0)
    )
    for i, v in enumerate(vals):
        if col_idx in (0, 1):
            # All ≈ -100%; annotate on left inside bar
            ax.text(
                v + x_range * 0.015,
                i,
                f"{v:.2f}%",
                ha="left",
                va="center",
                fontsize=9,
                color="white",
                fontweight="bold",
                zorder=6,
            )
        else:
            # Duration: small values, annotate to the right/left of bar
            offset = abs(x_range) * 0.04 if x_range else 0.01
            ha = "left" if v >= 0 else "right"
            xpos = v + offset if v >= 0 else v - offset
            ax.text(
                xpos,
                i,
                f"{v:+.3f}%",
                ha=ha,
                va="center",
                fontsize=9,
                color="#111111",
                fontweight="bold",
                zorder=6,
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(MODES, fontsize=10.5)
    ax.set_ylim(-0.55, N_MODES - 0.45)

    if XLIMS[col_idx]:
        ax.set_xlim(*XLIMS[col_idx])
    else:
        pad = max(abs(vals).max() * 0.35, 0.05)
        ax.set_xlim(vals.min() - pad, vals.max() + pad)

    ax.set_xlabel("\u0394  (%)", fontsize=11, color="#333333", labelpad=8)
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

    if col_idx > 0:
        ax.set_yticklabels([])

    # Annotation for trips/distance panels
    if col_idx in (0, 1):
        ax.text(
            0.50,
            0.035,
            "All modes \u2248 \u221299.99%\n(near-total modal suppression)",
            ha="center",
            va="bottom",
            transform=ax.transAxes,
            fontsize=9,
            color="#555555",
            style="italic",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="#FFF8F0",
                edgecolor="#BBBBBB",
                linewidth=0.9,
                alpha=0.95,
            ),
        )

    # Duration panel note
    if col_idx == 2:
        ax.text(
            0.5,
            0.035,
            "Mixed effects:\nFreight & Walk longer (+)\nCar, PT, Ride-hailing shorter (\u2212)",
            ha="center",
            va="bottom",
            transform=ax.transAxes,
            fontsize=9,
            color="#444444",
            style="italic",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="#F0F6FF",
                edgecolor="#BBBBBB",
                linewidth=0.9,
                alpha=0.95,
            ),
        )

# ── Legend ────────────────────────────────────────────────────────────────────
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

fig.text(
    0.975,
    0.012,
    "\u2212100% = mode completely absent in scenario.   "
    "Duration % can be positive (longer trips) or negative (shorter trips).",
    ha="right",
    va="bottom",
    fontsize=8.5,
    color="#777777",
    style="italic",
)

OUT = r"C:\Users\zamin\Downloads\Nazim\round_12_mode_stats_diff_perc.png"
plt.savefig(OUT, dpi=200, bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved:", OUT)
