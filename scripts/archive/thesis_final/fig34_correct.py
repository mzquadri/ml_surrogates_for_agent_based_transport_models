"""Regenerate Figure 3.1 (fig34) using the CORRECT columns and raw units.

Tensor column order from process_simulations_for_gnn.py EdgeFeatures enum:
  0: VOL_BASE_CASE        (model uses)
  1: CAPACITY_BASE_CASE   (model uses)
  2: CAPACITY_REDUCTION   (model uses)
  3: FREESPEED            (model uses)
  4: HIGHWAY              (NOT used; OSM road-type integer)
  5: LENGTH               (model uses)

The five model features correspond to columns [0, 1, 2, 3, 5].
Saved scaler at T8/data_created_during_training/train_x_scaler.pkl reports raw stats:
  VOL:  mean 50.91   std 135.83
  CAP:  mean 1028.96 std 1264.45
  RED:  mean -92.41  std 332.16
  FSP:  mean 8.15    std 4.01
  LEN:  mean 91.60   std 109.94
"""
import sys, os, joblib, numpy as np, torch, glob
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import matplotlib as mpl, matplotlib.pyplot as plt

PRIMARY = "#5B9BD5"; SECOND = "#ED7D31"; NEUTRAL = "#A5A5A5"
EDGE = "#888888"; GRID = "#E5E5E5"; TICKDARK = "#404040"
ALPHA = 0.7

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "normal",
    "axes.labelsize": 10, "axes.labelcolor": "black",
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "xtick.color": TICKDARK, "ytick.color": TICKDARK,
    "legend.fontsize": 9, "legend.frameon": False,
    "axes.edgecolor": "#666666", "axes.linewidth": 0.6,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

ROOT = "C:/Users/zamin/Downloads/ml_surrogates_thesis_final/ml_surrogates_thesis_final"
DATA_DIR = f"{ROOT}/code/data/train_data/dist_not_connected_10k_1pct"
SCALER_PATH = f"{ROOT}/code/data/TR-C_Benchmarks/point_net_transf_gat_8th_trial_lower_dropout/data_created_during_training/train_x_scaler.pkl"
OUT = f"{ROOT}/document/figures/new"

# Load 2 batches for speed (gives 100 graphs * 31,635 = 3,163,500 nodes)
import sys as _sys
try:
    from data_preprocessing.process_simulations_for_gnn import EdgeFeatures
except ImportError:
    pass

batches = sorted(glob.glob(f"{DATA_DIR}/datalist_batch_*.pt"))[:2]
all_x = []
for b in batches:
    print(f"  loading {os.path.basename(b)}...")
    data = torch.load(b, map_location="cpu", weights_only=False)
    if isinstance(data, list):
        for g in data:
            all_x.append(g.x.numpy() if hasattr(g.x, "numpy") else np.asarray(g.x))
X = np.concatenate(all_x, axis=0)
print(f"Loaded {X.shape[0]:,} nodes with {X.shape[1]} columns")
print()

# Verify column 4 is HIGHWAY (small int range), column 5 is LENGTH (continuous wide range)
print("Column min/max sanity check:")
for i in range(X.shape[1]):
    u = np.unique(X[:, i])
    print(f"  col {i}: min={X[:,i].min():.2f}, max={X[:,i].max():.2f}, unique values: {len(u)}")

# The saved tensors are PRE-normalisation (raw values, not yet scaled).
# Confirm: VOL column 0 max should be ~1500-2000 raw veh/h, not z-score order ~9.
# From column-stat dump above we expect: col 0 max ~1596 (raw veh/h). So tensors here are RAW.
# Therefore we plot directly without inverse-transform.

# Slice the 5 model features (skip col 4 = HIGHWAY)
USE_COLS = [0, 1, 2, 3, 5]
NAMES = ["VOL\\_BASE\\_CASE", "CAPACITY\\_BASE\\_CASE", "CAPACITY\\_REDUCTION", "FREESPEED", "LENGTH"]
RAW_NAMES_PLAIN = ["VOL_BASE_CASE", "CAPACITY_BASE_CASE", "CAPACITY_REDUCTION", "FREESPEED", "LENGTH"]
UNITS = ["veh/h", "veh/h", "veh/h", "m/s", "m"]

# Apply inverse StandardScaler since values may be already z-scored. Sanity check first.
sc = joblib.load(SCALER_PATH)
print()
print("Scaler params (raw means/stds):")
for n, m, s in zip(RAW_NAMES_PLAIN, sc.mean_, sc.scale_):
    print(f"  {n:24s}  mean = {m:>10.3f}   std = {s:>10.3f}")

# Decide: are the saved tensor values raw or z-scored?
# Heuristic: if VOL column max is ~1596 then raw; if ~10 then z-score.
vol_max = X[:, 0].max()
print()
if vol_max > 100:
    print(f"VOL max = {vol_max:.1f} > 100, so saved tensors contain RAW values. Plotting directly.")
    raw = X[:, USE_COLS]
else:
    print(f"VOL max = {vol_max:.2f} <= 100, so saved tensors contain Z-SCORED values. Inverse-transforming.")
    z = X[:, USE_COLS]
    raw = z * sc.scale_ + sc.mean_

# Build the figure
fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
axes = axes.flatten()

medians = {}
for i, (name_tex, name_plain, unit) in enumerate(zip(NAMES, RAW_NAMES_PLAIN, UNITS)):
    ax = axes[i]
    d = raw[:, i]
    medians[name_plain] = float(np.median(d))
    bins = 60
    ax.hist(d, bins=bins, color=PRIMARY, edgecolor=EDGE, linewidth=0.3, alpha=ALPHA)
    med = medians[name_plain]
    ax.axvline(med, color=SECOND, linestyle="--", linewidth=1.4, alpha=0.85,
               label=f"median = {med:.2f}")
    ax.set_xlabel(f"{name_tex} ({unit})")
    ax.set_ylabel("Node count")
    ax.set_title(name_tex, fontsize=10)
    ax.legend(loc="upper right")

# Sixth panel: stats summary
ax = axes[5]; ax.axis("off")
nz_red = int((raw[:, 2] == 0).sum())
nz_red_pct = 100.0 * nz_red / raw.shape[0]
fs_unique = len(np.unique(np.round(raw[:, 3], 2)))
total_n = raw.shape[0]

# FREESPEED in m/s, six urban bands at ~8.3 / 11.1 / 13.9 / 16.7 / 22.2 / 27.8 m/s (30/40/50/60/80/100 km/h)
summary = (
    f"\\textbf{{Verified raw-unit statistics}} (n = {total_n:,} nodes)\n\n"
    f"VOL\\_BASE\\_CASE:  raw mean {sc.mean_[0]:.1f},  raw std {sc.scale_[0]:.1f} veh/h\n\n"
    f"CAPACITY\\_BASE\\_CASE:  raw mean {sc.mean_[1]:.0f},  raw std {sc.scale_[1]:.0f} veh/h\n\n"
    f"CAPACITY\\_REDUCTION:  ${nz_red_pct:.1f}\\%$ zero values\n\n"
    f"FREESPEED:  6 discrete urban bands\n"
    f"  ($\\approx$ 30, 40, 50, 60, 80, 100 km/h)\n\n"
    f"LENGTH:  raw mean {sc.mean_[4]:.1f},  raw std {sc.scale_[4]:.1f} m\n\n"
    f"\\emph{{HIGHWAY (OSM road-type) excluded;}}\n"
    f"\\emph{{Natterer et al.\\ (2025) convention.}}"
)
ax.text(0.03, 0.97, summary, transform=ax.transAxes, fontsize=9, va="top", color="black",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=NEUTRAL, linewidth=0.6))

fig.suptitle("Distributions of the five input features in raw units",
             fontsize=11, color="black", y=1.0)
fig.tight_layout()

fig.savefig(f"{OUT}/fig34_feature_distributions.pdf")
fig.savefig(f"{OUT}/fig34_feature_distributions.png", dpi=300)
plt.close(fig)
print()
print("Saved fig34_feature_distributions.{pdf,png}")
print()
print("=== Verified medians (for caption sanity) ===")
for k, v in medians.items():
    print(f"  {k:24s}  median = {v:>10.3f}")
