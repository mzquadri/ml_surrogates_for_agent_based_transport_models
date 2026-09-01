import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from scipy.stats import gaussian_kde
import torch

# ── Load data ─────────────────────────────────────────────────────────────────
PT = (
    r"C:\Users\zamin\Downloads\Nazim\Thesis\Nazim\ml_surrogates_thesis_final"
    r"\code\data\train_data\dist_not_connected_10k_1pct\datalist_batch_1.pt"
)
data = torch.load(PT, weights_only=False, map_location="cpu")
all_y = np.concatenate([d.y.numpy().flatten() for d in data])

# ── Statistics ────────────────────────────────────────────────────────────────
n_total = len(all_y)
n_zero = int((all_y == 0).sum())
pct_zero = 100 * n_zero / n_total
mean_y = float(all_y.mean())
median_y = float(np.median(all_y))
std_y = float(all_y.std())
p25, p75 = np.percentile(all_y, [25, 75])
iqr = p75 - p25

# ── Palette & style ───────────────────────────────────────────────────────────
BG = "#FFFFFF"
C_BAR = "#2166AC"
C_KDE = "#0D2D55"
C_MEAN = "#B2182B"
C_MED = "#1A7034"
GRAY = "#888888"

plt.rcParams.update(
    {
        "font.family": "serif",
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    }
)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    figsize=(20, 8),
    gridspec_kw={"wspace": 0.40},
)
fig.patch.set_facecolor(BG)

# ── Title block ───────────────────────────────────────────────────────────────
fig.text(
    0.5,
    0.990,
    "Target Variable Distribution  —  \u0394 Traffic Flow per Road Segment",
    ha="center",
    va="top",
    fontsize=20,
    fontweight="bold",
    color="#0D0D0D",
)
fig.text(
    0.5,
    0.952,
    f"50 graphs  |  31,635 road segments per graph  |  {n_total:,} total observations"
    f"  |  1,000 disruption scenarios",
    ha="center",
    va="top",
    fontsize=11,
    color="#555555",
    style="italic",
)
fig.add_artist(
    mlines.Line2D(
        [0.04, 0.96],
        [0.932, 0.932],
        transform=fig.transFigure,
        color="#C8C8C8",
        linewidth=0.9,
    )
)


def style_ax(ax):
    for sp in ax.spines.values():
        sp.set_color("#BBBBBB")
        sp.set_linewidth(0.85)
        sp.set_visible(True)
    ax.set_facecolor(BG)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="both", length=3, color="#BBBBBB", direction="in")


# ─────────────────────────────────────────────────────────────────────────────
# Panel (a)  Full distribution  [-65, 65]  —  log y-scale
# ─────────────────────────────────────────────────────────────────────────────
CLIP_FULL = 65
y_full = all_y[(all_y >= -CLIP_FULL) & (all_y <= CLIP_FULL)]
pct_full = 100 * len(y_full) / n_total

style_ax(ax1)
ax1.yaxis.grid(True, which="both", color="#EEEEEE", linewidth=0.8, zorder=0)

ax1.hist(
    y_full,
    bins=100,
    range=(-CLIP_FULL, CLIP_FULL),
    color=C_BAR,
    alpha=0.82,
    edgecolor="white",
    linewidth=0.35,
    zorder=3,
)
ax1.set_yscale("log")
ax1.set_xlim(-CLIP_FULL, CLIP_FULL)

ax1.axvline(
    mean_y, color=C_MEAN, lw=2.0, ls="--", zorder=6, label=f"Mean = {mean_y:.2f} veh/hr"
)
ax1.axvline(
    median_y,
    color=C_MED,
    lw=2.0,
    ls=":",
    zorder=6,
    label=f"Median = {median_y:.2f} veh/hr",
)
ax1.axvline(0, color="#777777", lw=1.0, ls="-", zorder=5, alpha=0.6)

ax1.set_xlabel(
    "\u0394 Traffic Flow  [veh/hr]", fontsize=12, color="#333333", labelpad=8
)
ax1.set_ylabel("Frequency  (log scale)", fontsize=12, color="#333333", labelpad=8)
ax1.set_title(
    f"(a)  Full Distribution  \u0394Flow \u2208 [\u2212{CLIP_FULL}, {CLIP_FULL}] veh/hr",
    fontsize=13.5,
    fontweight="bold",
    color="#111111",
    loc="left",
    pad=8,
)
ax1.text(
    0.985,
    0.970,
    f"{pct_full:.1f}% of data shown",
    transform=ax1.transAxes,
    ha="right",
    va="top",
    fontsize=9,
    color=GRAY,
    style="italic",
)

# Stats box
stats_txt = (
    f"n = {n_total:,}\n"
    f"Mean   = {mean_y:+.3f} veh/hr\n"
    f"Median = {median_y:+.3f} veh/hr\n"
    f"Std    = {std_y:.3f} veh/hr\n"
    f"IQR    = [{p25:.2f}, {p75:.2f}]\n"
    f"Zero (y\u22600) = {pct_zero:.1f}%"
)
ax1.text(
    0.015,
    0.970,
    stats_txt,
    transform=ax1.transAxes,
    ha="left",
    va="top",
    fontsize=9.5,
    color="#222222",
    linespacing=1.65,
    bbox=dict(
        boxstyle="round,pad=0.40",
        facecolor="white",
        edgecolor="#BBBBBB",
        linewidth=1.0,
        alpha=0.95,
    ),
)
leg1 = ax1.legend(fontsize=10, loc="center right", framealpha=0.95, edgecolor="#CCCCCC")
leg1.get_frame().set_linewidth(0.8)


# ─────────────────────────────────────────────────────────────────────────────
# Panel (b)  Non-zero values only  [-15, 15]  —  density + smooth KDE
# ─────────────────────────────────────────────────────────────────────────────
CLIP_ZOOM = 15
# Exclude exact zeros so density shows continuous shape
y_nz = all_y[(all_y != 0) & (all_y >= -CLIP_ZOOM) & (all_y <= CLIP_ZOOM)]
pct_nz = 100 * len(y_nz) / n_total

style_ax(ax2)
ax2.yaxis.grid(True, color="#EEEEEE", linewidth=0.8, zorder=0)

counts2, bins2, _ = ax2.hist(
    y_nz,
    bins=70,
    range=(-CLIP_ZOOM, CLIP_ZOOM),
    density=True,
    color=C_BAR,
    alpha=0.72,
    edgecolor="white",
    linewidth=0.35,
    zorder=3,
)

# KDE with generous bandwidth for smooth, interpretable curve
kde = gaussian_kde(y_nz, bw_method=0.8)
kde_x = np.linspace(-CLIP_ZOOM, CLIP_ZOOM, 600)
ax2.plot(
    kde_x,
    kde(kde_x),
    color=C_KDE,
    linewidth=2.4,
    zorder=6,
    label="KDE (non-zero, bw=0.8)",
)

ax2.axvline(
    mean_y, color=C_MEAN, lw=2.0, ls="--", zorder=7, label=f"Mean = {mean_y:.2f}"
)
ax2.axvline(
    median_y, color=C_MED, lw=2.0, ls=":", zorder=7, label=f"Median = {median_y:.2f}"
)
ax2.axvline(0, color="#777777", lw=1.0, ls="-", zorder=5, alpha=0.6)

ax2.set_xlim(-CLIP_ZOOM, CLIP_ZOOM)
ax2.set_xlabel(
    "\u0394 Traffic Flow  [veh/hr]", fontsize=12, color="#333333", labelpad=8
)
ax2.set_ylabel("Probability Density", fontsize=12, color="#333333", labelpad=8)
ax2.set_title(
    f"(b)  Non-zero Values  \u0394Flow \u2208 [\u2212{CLIP_ZOOM}, {CLIP_ZOOM}] veh/hr",
    fontsize=13.5,
    fontweight="bold",
    color="#111111",
    loc="left",
    pad=8,
)
ax2.text(
    0.985,
    0.035,
    f"{pct_nz:.1f}% of data shown  (zeros excluded)",
    transform=ax2.transAxes,
    ha="right",
    va="bottom",
    fontsize=9,
    color=GRAY,
    style="italic",
)

# Zero-mass annotation — left side, clean
top_density = counts2.max()
ax2.text(
    -14.2,
    top_density * 0.92,
    f"{pct_zero:.1f}% of all segments\nhave  \u0394flow = 0\n(excluded from this panel)",
    ha="left",
    va="top",
    fontsize=9.5,
    color="#333333",
    bbox=dict(
        boxstyle="round,pad=0.35",
        facecolor="white",
        edgecolor="#BBBBBB",
        linewidth=1.0,
        alpha=0.95,
    ),
    zorder=8,
)

leg2 = ax2.legend(fontsize=10, loc="upper right", framealpha=0.95, edgecolor="#CCCCCC")
leg2.get_frame().set_linewidth(0.8)

# ── Layout & save ─────────────────────────────────────────────────────────────
plt.subplots_adjust(top=0.858, bottom=0.110, left=0.070, right=0.970)
OUT = r"C:\Users\zamin\Downloads\Nazim\round_08_y_distribution.png"
plt.savefig(OUT, dpi=200, bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved:", OUT)
