import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# ── Load data ─────────────────────────────────────────────────────────────────
PT_PATH = (
    r"C:\Users\zamin\Downloads\Nazim\Thesis\Nazim"
    r"\ml_surrogates_thesis_final\code\data\train_data"
    r"\dist_not_connected_10k_1pct\datalist_batch_1.pt"
)
graphs = torch.load(PT_PATH, weights_only=False, map_location="cpu")
x_np = graphs[0].x.numpy()  # [31635, 6]

# ── Palette ───────────────────────────────────────────────────────────────────
COLORS = ["#4878A8", "#5DA573", "#D66B6B", "#D4A843", "#888888", "#6B9EC7"]
BG = "#FFFFFF"
CREAM = "#F7F4EF"
RED = "#CC2222"

# Per-feature gradient colormaps (light → dark of each feature color)
CMAPS = {
    "#4878A8": plt.cm.Blues,
    "#5DA573": plt.cm.Greens,
    "#D66B6B": plt.cm.Reds,
    "#D4A843": plt.cm.YlOrBr,
    "#6B9EC7": plt.cm.Blues,
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    }
)

# ── Feature metadata ──────────────────────────────────────────────────────────
features = [
    dict(
        col=0,
        name="VOL_BASE_CASE",
        color=COLORS[0],
        subtitle="How many vehicles actually travel on this road per hour  (base-case / normal conditions)",
        x_desc="X Axis  \u2192  Base Traffic Volume",
        x_unit="Unit : vehicles per hour  [veh/hr]     Range : 0  to  1,596 veh/hr",
        examples=[
            "0 veh/hr      \u2192  pedestrian zone, no cars at all",
            "50 veh/hr     \u2192  quiet residential street",
            "500 veh/hr    \u2192  busy main boulevard",
            "1,596 veh/hr  \u2192  most heavily trafficked road in dataset",
        ],
        bar_type="hist",
        bins=70,
        clip_pct=99,
        mean_side="right",
        ex_corner="TR",
    ),
    dict(
        col=1,
        name="CAPACITY_BASE",
        color=COLORS[1],
        subtitle="Maximum number of vehicles this road can physically handle per hour  (capacity, not actual traffic)",
        x_desc="X Axis  \u2192  Road Capacity",
        x_unit="Unit : vehicles per hour  [veh/hr]     Range : 0  to  14,400 veh/hr",
        examples=[
            "200 veh/hr     \u2192  narrow alley  (1 slow lane)",
            "1,000 veh/hr   \u2192  typical Paris inner-city street",
            "3,600 veh/hr   \u2192  major boulevard  (multiple lanes)",
            "14,400 veh/hr  \u2192  motorway / Peripherique",
        ],
        bar_type="hist",
        bins=70,
        clip_pct=None,
        mean_side="right",
        ex_corner="TR",
    ),
    dict(
        col=2,
        name="CAPACITY_REDUCTION",
        color=COLORS[2],
        subtitle="How much capacity was removed from this road in this scenario  (always zero or negative)",
        x_desc="X Axis  \u2192  Capacity Reduction Applied in This Scenario",
        x_unit="Unit : vehicles per hour  [veh/hr]     Range : -4,800  to  0     (zero = road unchanged)",
        examples=[
            "0 veh/hr       \u2192  road is fully open, no change at all",
            "-300 veh/hr    \u2192  one lane closed  (e.g. construction)",
            "-1,000 veh/hr  \u2192  major restriction, half the road blocked",
            "-4,800 veh/hr  \u2192  road almost entirely closed",
        ],
        bar_type="hist",
        bins=70,
        clip_pct=None,
        x_lim=(-620, 30),
        mean_side="left",
        ex_corner="TL",
    ),
    dict(
        col=3,
        name="FREESPEED",
        color=COLORS[3],
        subtitle="Speed limit of this road segment  (multiply by 3.6 to convert from m/s to km/h)",
        x_desc="X Axis  \u2192  Speed Limit",
        x_unit="Unit : metres per second  [m/s]     Range : 0  to  33.3 m/s     (33.3 m/s = 120 km/h)",
        examples=[
            "4.2 m/s  = 15 km/h   \u2192  pedestrian zone / living street",
            "8.3 m/s  = 30 km/h   \u2192  typical Paris inner-city street  (most common)",
            "13.9 m/s = 50 km/h   \u2192  main boulevard / arterial road",
            "33.3 m/s = 120 km/h  \u2192  motorway  (Peripherique)",
        ],
        bar_type="hist",
        bins=25,
        clip_pct=None,
        mean_side="right",
        ex_corner="TR",
    ),
    dict(
        col=4,
        name="HIGHWAY_TYPE",
        color=COLORS[4],
        subtitle="Road category code  (each integer = one type of road)     NOTE : this feature is NOT passed to the model",
        x_desc="X Axis  \u2192  Road Type Code  (integer label for each road category)",
        x_unit="",
        examples=[
            " 0  Motorway     \u2192  Peripherique, highest speed road",
            " 5  Residential  \u2192  most common, neighbourhood streets",
            " 7  Service      \u2192  parking access, back lanes",
            "-1  Unknown      \u2192  unclassified / missing road type",
        ],
        bar_type="bar",
        ex_corner="TR",
    ),
    dict(
        col=5,
        name="LENGTH",
        color=COLORS[5],
        subtitle="Physical length of this road segment in metres  (long roads are split into segments at every junction)",
        x_desc="X Axis  \u2192  Road Segment Length",
        x_unit="Unit : metres  [m]     Range : 4.2 m  to  2,568 m",
        examples=[
            "10 m    \u2192  tiny connector road or alley",
            "91 m    \u2192  average Paris segment  (approx. one city block)",
            "300 m   \u2192  longer stretch between two junctions",
            "2,568 m \u2192  longest segment in dataset  (motorway stretch)",
        ],
        bar_type="hist",
        bins=70,
        clip_pct=99,
        mean_side="right",
        ex_corner="TR",
    ),
]

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(32, 22))
fig.patch.set_facecolor(BG)

fig.suptitle(
    "Feature Matrix   x   [31,635 x 6]   --   All 6 Input Features of Each Road Segment in Paris",
    fontsize=22,
    fontweight="bold",
    color="#1a1a1a",
    y=0.990,
)

plt.subplots_adjust(
    top=0.950, bottom=0.055, left=0.07, right=0.975, hspace=0.34, wspace=0.38
)

# ── Draw each subplot ─────────────────────────────────────────────────────────
for idx, feat in enumerate(features):
    row, col_pos = divmod(idx, 3)
    ax = axes[row, col_pos]
    ax.set_facecolor(BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    vals = x_np[:, feat["col"]]
    color = feat["color"]

    # ── Pre-compute examples corner (shared by mean box + examples box) ────────
    corner = feat.get("ex_corner", "TR")
    if corner == "TR":
        ex_x, ex_ha, ex_y, ex_va = 0.98, "right", 0.55, "top"
    else:
        ex_x, ex_ha, ex_y, ex_va = 0.02, "left", 0.55, "top"
    if feat["bar_type"] == "bar":
        ex_y = 0.48

    # ── HISTOGRAM ─────────────────────────────────────────────────────────────
    if feat["bar_type"] == "hist":
        plot_vals = vals.copy()

        # Clip to percentile
        if feat.get("clip_pct"):
            pct_val = np.percentile(vals, feat["clip_pct"])
            plot_vals = plot_vals[plot_vals <= pct_val]

        # FIX 1: clip to x_lim BEFORE binning → uniform bar widths
        if "x_lim" in feat:
            lo, hi = feat["x_lim"]
            plot_vals = plot_vals[(plot_vals >= lo) & (plot_vals <= hi)]

        # FIX 3: gradient colourmap (light → dark of feature colour)
        cmap = CMAPS.get(color, plt.cm.Blues)
        counts_h, edges_h = np.histogram(plot_vals, bins=feat["bins"])
        n_bins = len(counts_h)
        bin_colors = [cmap(0.28 + 0.72 * i / max(n_bins - 1, 1)) for i in range(n_bins)]

        ax.bar(
            edges_h[:-1],
            counts_h,
            width=np.diff(edges_h),
            color=bin_colors,
            alpha=0.92,
            edgecolor="white",
            linewidth=0.4,
            align="edge",
        )

        if "x_lim" in feat:
            ax.set_xlim(feat["x_lim"])

        if feat.get("clip_pct"):
            pct_val = np.percentile(vals, feat["clip_pct"])
            ax.text(
                0.99,
                0.01,
                f"Showing values up to {pct_val:.0f}  ({feat['clip_pct']}th percentile) -- a few extreme values omitted for clarity",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8.5,
                color="#999999",
                style="italic",
            )

        # Mean line
        mean_val = np.mean(vals)
        mean_unit = (
            feat["x_unit"].split("[")[1].split("]")[0].strip()
            if "[" in feat["x_unit"]
            else ""
        )
        ax.axvline(mean_val, color=RED, linewidth=2.4, linestyle="--", zorder=6)

        # FIX 4: mean box just above examples box (same x, y = ex_y + small gap)
        ax.text(
            ex_x,
            ex_y + 0.025,
            f"Mean = {mean_val:.1f}  [{mean_unit}]",
            transform=ax.transAxes,
            ha=ex_ha,
            va="bottom",
            fontsize=11.5,
            color=RED,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="#FFF0F0",
                edgecolor=RED,
                linewidth=1.4,
                alpha=0.96,
            ),
        )

    # ── BAR CHART (HIGHWAY_TYPE) ───────────────────────────────────────────────
    else:
        type_names = {
            -1: "Unknown",
            0: "Motorway",
            1: "Trunk",
            2: "Primary",
            3: "Secondary",
            4: "Tertiary",
            5: "Residential",
            6: "Living St.",
            7: "Service",
            8: "Unclassified",
            9: "Other",
        }
        unique, counts = np.unique(vals.astype(int), return_counts=True)
        tick_labels = [f"{int(u)}  --  {type_names.get(int(u), '')}" for u in unique]

        bars = ax.bar(
            range(len(unique)),
            counts,
            color=color,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
            width=0.60,
        )
        ax.set_xticks(range(len(unique)))
        ax.set_xticklabels(
            tick_labels,
            fontsize=10.5,
            color="#333333",
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )

        for b, c in zip(bars, counts):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + max(counts) * 0.013,
                f"{c:,}",
                ha="center",
                va="bottom",
                fontsize=9.5,
                color="#333333",
                fontweight="bold",
            )

        ax.text(
            ex_x,
            ex_y + 0.025,
            "This feature is EXCLUDED\nfrom model training",
            transform=ax.transAxes,
            ha=ex_ha,
            va="bottom",
            fontsize=11,
            color="#AA3333",
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.45",
                facecolor="#FFF0F0",
                edgecolor="#AA3333",
                linewidth=1.4,
                alpha=0.97,
            ),
        )

    # ── Title (1 bold line) ───────────────────────────────────────────────────
    ax.set_title(
        f"Feature {feat['col'] + 1} / 6   :   {feat['name']}",
        fontsize=15,
        fontweight="bold",
        color=color,
        loc="left",
        pad=8,
    )

    # ── Subtitle — INSIDE axes at top, white background ───────────────────────
    ax.text(
        0.01,
        0.988,
        feat["subtitle"],
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color="#444444",
        style="italic",
        clip_on=True,
        zorder=10,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="none",
            alpha=0.85,
        ),
    )

    # ── Y axis ────────────────────────────────────────────────────────────────
    ax.set_ylabel(
        "Y Axis  \u2192  Number of Road Segments\n(out of 31,635 total road segments in Paris)",
        fontsize=11.5,
        labelpad=10,
        color="#333333",
        linespacing=1.6,
    )

    # ── X axis ────────────────────────────────────────────────────────────────
    xlabel_str = feat["x_desc"]
    if feat["x_unit"]:
        xlabel_str += "\n" + feat["x_unit"]
    ax.set_xlabel(
        xlabel_str, fontsize=11.5, labelpad=14, color="#333333", linespacing=1.6
    )

    # ── Examples box — inside axes ────────────────────────────────────────────
    ex_text = "Examples :\n" + "\n".join(feat["examples"])

    ax.text(
        ex_x,
        ex_y,
        ex_text,
        transform=ax.transAxes,
        ha=ex_ha,
        va=ex_va,
        fontsize=10.5,
        color="#222222",
        linespacing=1.6,
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor=CREAM,
            edgecolor=color,
            linewidth=1.4,
            alpha=0.97,
        ),
    )


# ── Save ─────────────────────────────────────────────────────────────────────
out_path = r"C:\Users\zamin\Downloads\Nazim\round_all6_features_HD.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved:", out_path)
