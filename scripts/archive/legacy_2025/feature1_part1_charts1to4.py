"""
FEATURE 1 ANALYSIS - PART 1: ROAD CAPACITY (Charts 1-4)
=======================================================
Charts 1-4: Basic Distribution & Road Type Analysis

Repository Code: process_simulations_for_gnn.py Line 105
Source: pop_1pct_basecase_average_output_links.geojson
F1 = Road Capacity (Maximum traffic capacity in vehicles/hour)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.ticker as ticker

# Set professional plotting style with larger fonts
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 15

print("\n" + "#" * 80)
print("#" + " " * 78 + "#")
print("#" + "  FEATURE 1 - PART 1: ROAD CAPACITY (Charts 1-4)".center(78) + "#")
print("#" + "  Basic Distribution & Road Type Analysis".center(78) + "#")
print("#" + " " * 78 + "#")
print("#" * 80)

# DATA LOADING
print("\n" + "=" * 80)
print("LOADING DATA...")
print("=" * 80)

possible_paths = [
    'D:\\Python Projects\\Zamin_Thesis\\ml_surrogates_for_agent_based_transport_models\\data\\train_data\\dist_not_connected_10k_1pct',
    '/content/drive/MyDrive/Zamin_thesis/ml_surrogates_for_agent_based_transport_models/data/train_data/dist_not_connected_10k_1pct',
]

data_path = None
for path in possible_paths:
    p = Path(path)
    if p.exists():
        pt_files = list(p.glob('*.pt')) + list(p.rglob('*.pt'))
        if len(pt_files) > 0:
            data_path = p
            print(f"✓ Found data path: {path}")
            break

if data_path is None:
    raise FileNotFoundError("Data directory not found.")

batch_files = sorted(data_path.glob('datalist_batch_*.pt'))
if len(batch_files) == 0:
    batch_files = sorted(data_path.glob('*.pt'))

batch_0 = torch.load(batch_files[0], weights_only=False)
first_scenario = batch_0[0]

# Extract features
vol_base_case = first_scenario.x[:, 0].numpy()
capacity = first_scenario.x[:, 1].numpy()
cap_reduction = first_scenario.x[:, 2].numpy()
highway = first_scenario.x[:, 4].numpy()
length = first_scenario.x[:, 5].numpy()

n_edges = len(capacity)
unique_types = np.unique(highway)
print(f"✓ Loaded {n_edges:,} edges")

# Highway type decoder (OpenStreetMap classification)
highway_type_names = {
    0: 'Motorway',        # High-speed divided highways (autoroute)
    1: 'Trunk',           # Important non-motorway roads
    2: 'Primary',         # Major roads connecting cities
    3: 'Secondary',       # Regional connector roads
    4: 'Tertiary',        # Local connector roads
    5: 'Residential',     # Roads in residential areas
    6: 'Service',         # Service/access roads (parking lots)
    7: 'Unclassified',    # Minor public roads
    8: 'Living Street',   # Low-speed residential streets
    9: 'Other'            # Other road types
}

# Basic statistics with percentile analysis
Q1, Q2, Q3 = np.percentile(capacity, [25, 50, 75])
P90, P95, P99 = np.percentile(capacity, [90, 95, 99])
print(f"✓ Capacity range: {capacity.min():.0f} - {capacity.max():.0f} veh/h")
print(f"✓ Mean capacity: {capacity.mean():.0f} veh/h")
print(f"✓ Median capacity: {np.median(capacity):.0f} veh/h")
print(f"✓ Percentiles: Q1={Q1:.0f} | Q2={Q2:.0f} | Q3={Q3:.0f} | P90={P90:.0f} | P95={P95:.0f}")
print(f"✓ Data concentration: 50% of roads have capacity between {Q1:.0f} and {Q3:.0f} veh/h")

################################################################################
# CHART 1: DISTRIBUTION ANALYSIS
################################################################################
print("\n" + "=" * 80)
print("CHART 1: Road Capacity Distribution Analysis")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('FEATURE 1: Road Capacity Distribution Analysis\nParis MATSim Transport Network - Maximum Traffic Capacity per Road Segment', 
             fontsize=16, fontweight='bold', y=0.995)
plt.subplots_adjust(left=0.08, right=0.95, top=0.94, bottom=0.06, hspace=0.40, wspace=0.30)

# 1.1 Histogram - Overall Distribution
ax = axes[0, 0]
# Use bins that align with natural data intervals (every 50 veh/h)
bins_range = np.arange(0, 5100, 50)  # 0-5000 in 50 veh/h intervals = 100 bins
counts, bins, patches = ax.hist(capacity, bins=bins_range, alpha=0.75, color='#3498db', edgecolor='black', linewidth=0.5)

# Add quartile markers with shading
ax.axvspan(Q1, Q3, alpha=0.15, color='yellow', label=f'IQR (Middle 50%): {Q1:.0f}-{Q3:.0f} veh/h')
ax.axvline(Q1, color='orange', linestyle=':', linewidth=2, alpha=0.7)
ax.axvline(Q3, color='orange', linestyle=':', linewidth=2, alpha=0.7)
ax.axvline(capacity.mean(), color='#e74c3c', linestyle='--', linewidth=3, 
          label=f'Mean (Average) = {capacity.mean():.0f} veh/h', alpha=0.8)
ax.axvline(np.median(capacity), color='#27ae60', linestyle='--', linewidth=3,
          label=f'Median (Q2) = {np.median(capacity):.0f} veh/h', alpha=0.8)
ax.axvline(P90, color='purple', linestyle='-.', linewidth=2, 
          label=f'P90 = {P90:.0f} veh/h', alpha=0.7)
ax.set_xlabel('Road Capacity (vehicles per hour)\n[Maximum traffic this road can handle in 1 hour]\nExample: 2000 veh/h = up to 2000 cars/hour', 
             fontsize=10, fontweight='bold')
ax.set_ylabel('Frequency\n(Number of Road Segments)', 
             fontsize=10, fontweight='bold')
ax.set_title(f'A. Overall Capacity Distribution (Total Network: {n_edges:,} road segments)\n[Yellow band shows where middle 50% of roads fall (IQR)]\nNote: Q1 and Median both at {Q1:.0f} means many roads have identical capacity', 
            fontsize=11, fontweight='bold', pad=10)
ax.legend(loc='upper right', framealpha=0.95, fontsize=8, title='Statistical Measures', title_fontsize=8, ncol=1)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(0, 5000)  # Focus on main data range (0-5000 covers ~95% of roads)
ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
# Add data concentration info - positioned lower to avoid legend overlap
pct_below_5000 = (capacity <= 5000).sum() / len(capacity) * 100
ax.text(0.02, 0.65, f'DATA COVERAGE:\nX-axis: 0-5000 veh/h\nShowing: {pct_below_5000:.1f}% of roads\nMax value: {capacity.max():.0f}\n\nPERCENTILES:\n25%: {Q1:.0f} veh/h\n50%: {Q2:.0f} veh/h\n75%: {Q3:.0f} veh/h\n90%: {P90:.0f} veh/h', 
       transform=ax.transAxes, fontsize=7, ha='left', va='top',
       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='black'))

# 1.2 CDF - Cumulative Distribution
ax = axes[0, 1]
sorted_cap = np.sort(capacity)
cdf = np.arange(1, len(sorted_cap) + 1) / len(sorted_cap) * 100
ax.plot(sorted_cap, cdf, linewidth=3, color='#e67e22', label='Cumulative Distribution Curve', alpha=0.9)
percentiles = [25, 50, 75, 90]
percentile_labels = ['Q1 (25%)', 'Q2/Median (50%)', 'Q3 (75%)', 'P90 (90%)']
percentile_x_offsets = [200, 200, 200, 200]  # X offset for labels
percentile_y_offsets = [3, -5, 3, 3]  # Y offset for labels (avoid overlap at 50%)
for pct, label, x_off, y_off in zip(percentiles, percentile_labels, percentile_x_offsets, percentile_y_offsets):
    val = np.percentile(capacity, pct)
    ax.axhline(pct, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)
    ax.axvline(val, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)
    ax.plot(val, pct, 'ro', markersize=8, zorder=5)
    ax.text(val+x_off, pct+y_off, f'{label}\n{val:.0f} veh/h', fontsize=7, fontweight='bold', 
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), va='bottom' if y_off > 0 else 'top')
ax.set_xlabel('Road Capacity (vehicles per hour)\n[Maximum traffic handling capability of road]', 
             fontsize=11, fontweight='bold')
ax.set_ylabel('Cumulative Percentage (%)\n[% of total roads with capacity less than or equal to this value]\nExample: 50% means half of all roads have capacity below this point', 
             fontsize=11, fontweight='bold')
ax.set_title('B. Cumulative Distribution Function (CDF) - "What percentage of roads have X capacity or less?"\n[Reading the curve: Pick any capacity value on X-axis, read Y-axis to see % of roads below it]\nSteep curve = many roads have similar capacity | Flat curve = capacity varies widely', 
            fontsize=12, fontweight='bold', pad=12)
ax.legend(loc='lower right', framealpha=0.95, fontsize=10)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(0, 5000)  # Focus on main data range for better percentile visibility
ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
ax.yaxis.set_major_locator(ticker.MultipleLocator(10))

# 1.3 Log scale - Logarithmic View
ax = axes[1, 0]
log_cap = np.log10(capacity[capacity > 0])
counts_log, bins_log, patches_log = ax.hist(log_cap, bins=80, alpha=0.75, color='#9b59b6', edgecolor='black', linewidth=0.5)
ax.axvline(np.log10(capacity.mean()), color='#e74c3c', linestyle='--', linewidth=3, 
          label=f'Mean = {capacity.mean():.0f} veh/h (log={np.log10(capacity.mean()):.2f})', alpha=0.8)
ax.set_xlabel('Log10(Road Capacity) - Logarithmic Scale\n[Compresses large range: each +1.0 = 10x more capacity]\nScale Reference: 2.0=100 | 2.5=316 | 3.0=1K | 3.5=3.2K | 4.0=10K veh/h', 
             fontsize=10, fontweight='bold')
ax.set_ylabel('Frequency (Number of Road Segments)\n[Count of roads at each logarithmic capacity level]', 
             fontsize=11, fontweight='bold')
ax.set_title('C. Logarithmic Scale Distribution - "Compressing large range into visible scale"\n[Each 1.0 unit increase = 10× more capacity | Useful when data spans multiple orders of magnitude]\nPeak shows most common capacity range on log scale', 
            fontsize=12, fontweight='bold', pad=12)
# Add reference lines with better labels
reference_points = [(2, '100\nveh/h'), (2.5, '316\nveh/h'), (3, '1,000\nveh/h'), (3.5, '3,162\nveh/h'), (4, '10,000\nveh/h')]
for val, label in reference_points:
    ax.axvline(val, color='gray', linestyle=':', alpha=0.4, linewidth=1.5)
    ax.text(val, ax.get_ylim()[1]*0.92, label, fontsize=7.5, ha='center', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
ax.legend(loc='best', framealpha=0.95, fontsize=10)
ax.grid(True, alpha=0.3, linestyle='--')

# 1.4 Box Plot - Statistical Summary with Full Explanation
ax = axes[1, 1]
bp = ax.boxplot([capacity], positions=[0], widths=0.6, patch_artist=True, showfliers=True,
                flierprops=dict(marker='o', markerfacecolor='#e74c3c', markersize=4, alpha=0.6, markeredgecolor='darkred'))
bp['boxes'][0].set_facecolor('#3498db')
bp['boxes'][0].set_alpha(0.7)
bp['boxes'][0].set_linewidth(2)
bp['medians'][0].set_color('#e74c3c')
bp['medians'][0].set_linewidth(4)
for whisker in bp['whiskers']:
    whisker.set_linewidth(2)
    whisker.set_linestyle('--')
for cap in bp['caps']:
    cap.set_linewidth(2)
ax.set_ylabel('Road Capacity (vehicles per hour)\n[Vertical axis shows capacity range from minimum to maximum]', 
             fontsize=11, fontweight='bold')
ax.set_title('D. Box Plot (Box-and-Whisker Diagram) - "5-Number Summary" of Capacity Distribution\n[Compact visualization showing minimum, Q1, median, Q3, maximum, plus outliers]\nUseful for quickly seeing data spread, central tendency, and extreme values', 
            fontsize=12, fontweight='bold', pad=12)
ax.set_xticks([0])
ax.set_xticklabels(['All Road Segments\nin Paris Network'], fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
ax.yaxis.set_major_locator(ticker.MultipleLocator(2000))

# Add comprehensive box plot annotation with clearer explanations
Q1, Q2, Q3 = np.percentile(capacity, [25, 50, 75])
IQR = Q3 - Q1
whisker_low = capacity[capacity >= Q1 - 1.5*IQR].min()
whisker_high = capacity[capacity <= Q3 + 1.5*IQR].max()
n_outliers = ((capacity < whisker_low) | (capacity > whisker_high)).sum()

ax.text(0.50, 0.98, '=== BOX PLOT COMPONENTS EXPLAINED ===', transform=ax.transAxes, fontsize=9, 
       fontweight='bold', verticalalignment='top', ha='left',
       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
ax.text(0.50, 0.91, f'[1] THICK RED LINE = MEDIAN (Q2, 50th %ile)\n   -> Middle value when roads sorted by capacity\n   -> Value: {Q2:.0f} veh/h (50% above, 50% below)', 
       transform=ax.transAxes, fontsize=7.5, verticalalignment='top', ha='left', family='monospace')
ax.text(0.50, 0.81, f'[2] BLUE BOX = IQR (Interquartile Range)\n   -> Contains middle 50% of all roads\n   -> Range: {IQR:.0f} veh/h (from {Q1:.0f} to {Q3:.0f})\n   -> Shows where "typical" roads fall', 
       transform=ax.transAxes, fontsize=7.5, verticalalignment='top', ha='left', family='monospace')
ax.text(0.50, 0.69, f'[3] BOX EDGES = Q1 and Q3 (Quartiles)\n   -> Bottom: Q1 = {Q1:.0f} veh/h (25% below)\n   -> Top: Q3 = {Q3:.0f} veh/h (75% below)\n   -> 50% of roads in this box', 
       transform=ax.transAxes, fontsize=7.5, verticalalignment='top', ha='left', family='monospace')
ax.text(0.50, 0.57, f'[4] WHISKERS (Dashed) = Normal Range\n   -> Extend to 1.5 x IQR beyond box\n   -> Lower: {whisker_low:.0f} | Upper: {whisker_high:.0f}\n   -> "Expected" data range', 
       transform=ax.transAxes, fontsize=7.5, verticalalignment='top', ha='left', family='monospace')
ax.text(0.50, 0.43, f'[5] RED DOTS = OUTLIERS (Unusual)\n   -> Extremely high/low capacity roads\n   -> Count: {n_outliers:,} ({n_outliers/n_edges*100:.1f}%)\n   -> Outside normal range', 
       transform=ax.transAxes, fontsize=7.5, verticalalignment='top', ha='left', color='#c0392b', weight='bold', family='monospace')

plt.tight_layout()
plt.savefig('feature1_chart1_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: feature1_chart1_distribution.png")
plt.show()
plt.close()

################################################################################
# CHART 2: CAPACITY BY HIGHWAY TYPE
################################################################################
print("\n" + "=" * 80)
print("CHART 2: Capacity by Highway Type")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('FEATURE 1: Road Capacity Analysis by Highway Type\nOpenStreetMap Road Classification System (Motorway=0 to Other=9)\nComparing maximum traffic capacity across different road categories', 
             fontsize=16, fontweight='bold', y=0.995)
plt.subplots_adjust(left=0.07, right=0.96, top=0.93, bottom=0.08, hspace=0.45, wspace=0.28)

# 2.1 Box plot by type
ax = axes[0, 0]
data_for_boxplot = [capacity[highway == ht] for ht in unique_types]
bp = ax.boxplot(data_for_boxplot, positions=unique_types, widths=0.6,
                patch_artist=True, showfliers=False)
for patch in bp['boxes']:
    patch.set_facecolor('#3498db')
    patch.set_alpha(0.7)
for median in bp['medians']:
    median.set_color('#e74c3c')
    median.set_linewidth(2.5)
ax.set_xlabel('Road Type\n[0=Motorway, 1=Trunk, 2=Primary, 3=Secondary, 4=Tertiary,\n5=Residential, 6=Service, 7=Unclassified, 8=Living St., 9=Other]', fontsize=9)
ax.set_ylabel('Road Capacity (vehicles/hour)\n[Red line = median | Blue box = middle 50% (IQR)]', fontsize=10)
ax.set_title('A. Capacity Distribution by Road Type\n[Box plot comparing capacity ranges across 11 road categories]\nRed line = median | Blue box = middle 50% (IQR) | No outliers shown for clarity', fontsize=10, fontweight='bold', pad=10)
ax.grid(True, alpha=0.3, axis='y')
ax.set_xticks(unique_types)
ax.set_xticklabels([f'{int(ht)}\n{highway_type_names.get(int(ht), "?")[:5]}' for ht in unique_types], fontsize=8)
ax.yaxis.set_major_locator(ticker.MultipleLocator(2000))

# 2.2 Mean capacity by type
ax = axes[0, 1]
means = [capacity[highway == ht].mean() for ht in unique_types]
colors = ['#27ae60' if m > 3000 else '#f39c12' if m > 1500 else '#e74c3c' for m in means]
bars = ax.bar(unique_types, means, width=0.6, alpha=0.8, color=colors, edgecolor='black', linewidth=0.7)
ax.set_xlabel('Road Type (OpenStreetMap Classification)\n[0=Motorway | 1=Trunk | 2=Primary | 3=Secondary | 4=Tertiary\n5=Residential | 6=Service | 7=Unclassified | 8=Living Street | 9=Other]', fontsize=8.5)
ax.set_ylabel('Mean Capacity (vehicles/hour)\n[Average maximum traffic capacity for each road type]', fontsize=10)
ax.set_title('B. Average Capacity by Road Type\n[Bar color coding: Green (>3000 veh/h) = High | Orange (1500-3000) = Medium | Red (<1500) = Low]\nMotorways & Trunks have highest capacity, Service roads have lowest', 
            fontsize=10, fontweight='bold', pad=10)
ax.grid(True, alpha=0.3, axis='y')
ax.set_xticks(unique_types)
ax.set_xticklabels([f'{int(ht)}\n{highway_type_names.get(int(ht), "?")[:4]}' for ht in unique_types], fontsize=8)
ax.yaxis.set_major_locator(ticker.MultipleLocator(500))
for bar, val, ht in zip(bars, means, unique_types):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, 
           f'{val:.0f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

# 2.3 Capacity range by type
ax = axes[1, 0]
ranges = [(capacity[highway == ht].min(), capacity[highway == ht].max()) for ht in unique_types]
for i, (min_val, max_val) in enumerate(ranges):
    ax.plot([unique_types[i], unique_types[i]], [min_val, max_val], 
           'o-', linewidth=3, markersize=8, color='#16a085', alpha=0.7)
    ax.text(unique_types[i]+0.15, max_val, f'{max_val:.0f}', fontsize=7, va='center')
    ax.text(unique_types[i]+0.15, min_val, f'{min_val:.0f}', fontsize=7, va='center')
ax.set_xlabel('Road Type\n[Each line shows min-max capacity range for that road type]', fontsize=9)
ax.set_ylabel('Capacity Range (vehicles/hour)\n[Vertical line spans from minimum to maximum]', fontsize=10)
ax.set_title('C. Capacity Range (Min-Max) by Road Type\n[Vertical line shows full range from minimum to maximum capacity]\nLonger line = more variation within that road type', fontsize=10, fontweight='bold', pad=10)
ax.set_xticks(unique_types)
ax.set_xticklabels([f'{int(ht)}\n{highway_type_names.get(int(ht), "?")[:4]}' for ht in unique_types], fontsize=8)
ax.grid(True, alpha=0.3, axis='y')
ax.yaxis.set_major_locator(ticker.MultipleLocator(2000))

# 2.4 Road count by type with capacity info
ax = axes[1, 1]
counts = [(highway == ht).sum() for ht in unique_types]
total_roads = sum(counts)
bars = ax.bar(unique_types, counts, width=0.6, alpha=0.8, color='#9b59b6', edgecolor='black', linewidth=0.7)
ax.set_xlabel('Road Type\n[Network composition by road classification]', fontsize=9)
ax.set_ylabel('Number of Road Segments\n[Total count in Paris MATSim network]', fontsize=10)
ax.set_title('D. Network Composition - Which Road Types Dominate?\n[Bar height = count | Label shows: count, percentage, mean capacity]\nType 4 (Tertiary) is most common with 37.3% of all roads', fontsize=10, fontweight='bold', pad=10)
ax.grid(True, alpha=0.3, axis='y')
ax.set_xticks(unique_types)
ax.set_xticklabels([f'{int(ht)}\n{highway_type_names.get(int(ht), "?")[:4]}' for ht in unique_types], fontsize=8)
ax.yaxis.set_major_locator(ticker.MultipleLocator(2000))
for bar, val, ht in zip(bars, counts, unique_types):
    pct = 100 * val / total_roads
    mean_cap = capacity[highway == ht].mean()
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, 
           f'{val:,}\n({pct:.1f}%)\n{mean_cap:.0f} cap', ha='center', va='bottom', fontsize=6.5, fontweight='bold')

plt.tight_layout()
plt.savefig('feature1_chart2_highway_types.png', dpi=300, bbox_inches='tight')
print("✓ Saved: feature1_chart2_highway_types.png")
plt.show()
plt.close()

# Print road type statistics
print("\n" + "-" * 80)
print("ROAD TYPE CAPACITY STATISTICS:")
print("-" * 80)
for ht in unique_types:
    type_mask = highway == ht
    count = type_mask.sum()
    mean_cap = capacity[type_mask].mean()
    median_cap = np.median(capacity[type_mask])
    print(f"  Type {int(ht)}: {highway_type_names.get(int(ht), '?'):<15} - {count:>5,} roads | Mean: {mean_cap:>6.0f} | Median: {median_cap:>6.0f} veh/h")
print("-" * 80)

################################################################################
# CHART 3: CAPACITY-VOLUME RELATIONSHIP
################################################################################
print("\n" + "=" * 80)
print("CHART 3: Capacity-Volume Relationship & Utilization")
print("=" * 80)

# Calculate utilization with safe division
with np.errstate(divide='ignore', invalid='ignore'):
    utilization = np.abs(vol_base_case) / capacity
    utilization = np.nan_to_num(utilization, nan=0.0, posinf=0.0, neginf=0.0)

fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('FEATURE 1: Capacity vs Volume Relationship & Road Utilization Analysis\n"How much of available capacity is actually being used?"\nUtilization = (Actual Traffic Volume) ÷ (Maximum Capacity) × 100%', 
             fontsize=16, fontweight='bold', y=0.995)
plt.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.06, hspace=0.38, wspace=0.28)

# 3.1 Scatter: Volume vs Capacity
ax = axes[0, 0]
traffic_mask = vol_base_case != 0
ax.scatter(capacity[traffic_mask], vol_base_case[traffic_mask], 
          alpha=0.4, s=2, c='#3498db', edgecolors='none')
ax.plot([0, capacity.max()], [0, capacity.max()], 'r--', linewidth=3, 
       label='100% utilization (fully used capacity)', alpha=0.8)
valid_mask = (capacity > 0) & (vol_base_case != 0)
if valid_mask.sum() > 0:
    corr = np.corrcoef(vol_base_case[valid_mask], capacity[valid_mask])[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes, 
           fontsize=10, fontweight='bold', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax.set_xlabel('Road Capacity (vehicles/hour)\n[Example: 2000 veh/h = road designed for max 2000 cars/hour]', fontsize=10)
ax.set_ylabel('Actual Traffic Volume (vehicles/hour)\n[Example: 500 veh/h = currently 500 cars/hour using this road]', fontsize=10)
ax.set_title(f'A. Do High-Capacity Roads Carry More Traffic? (n={traffic_mask.sum():,} active roads)\n[Points below red line = under-utilized | On line = fully utilized]', 
            fontsize=11, fontweight='bold', pad=10)
ax.legend(loc='upper left', framealpha=0.9, fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 5000)  # Focus on main data cluster (0-5000 capacity)
ax.set_ylim(0, 1000)  # Focus on main volume range
ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
ax.text(0.98, 0.02, 'Zoom: 0-5000 cap, 0-1000 vol\n(outliers not shown)', 
       transform=ax.transAxes, fontsize=7, ha='right', va='bottom',
       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# 3.2 Utilization distribution
ax = axes[0, 1]
util_nonzero = utilization[utilization > 0]
# Calculate utilization percentiles
util_q1, util_q2, util_q3 = np.percentile(util_nonzero, [25, 50, 75])
util_p90 = np.percentile(util_nonzero, 90)

ax.hist(util_nonzero, bins=80, alpha=0.75, color='#e67e22', edgecolor='black', linewidth=0.5)
ax.axvline(1.0, color='#e74c3c', linestyle='--', linewidth=2.5, label='100% = Full capacity', alpha=0.8)
ax.axvline(util_nonzero.mean(), color='#27ae60', linestyle='--', linewidth=2.5, 
          label=f'Mean={util_nonzero.mean():.3f} ({util_nonzero.mean()*100:.1f}%)', alpha=0.8)
ax.axvline(util_q2, color='blue', linestyle=':', linewidth=2, 
          label=f'Median={util_q2:.3f} ({util_q2*100:.1f}%)', alpha=0.7)
ax.set_xlabel('Utilization Ratio (Volume / Capacity)\n[Example: 0.25 = 25% used | 0.50 = 50% | 1.0 = 100% capacity]', fontsize=10)
ax.set_ylabel('Number of Roads\n[How many roads at each utilization level]', fontsize=10)
ax.set_title(f'B. Road Utilization Distribution\n[Mean={util_nonzero.mean():.3f} shows average road uses {util_nonzero.mean()*100:.1f}% of capacity]', 
            fontsize=11, fontweight='bold', pad=10)
ax.legend(loc='upper right', framealpha=0.9, fontsize=7.5)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
# Add utilization statistics - positioned to avoid legend overlap
ax.text(0.02, 0.97, f'UTILIZATION\nPERCENTILES:\nQ1: {util_q1*100:.1f}%\nQ2: {util_q2*100:.1f}%\nQ3: {util_q3*100:.1f}%\nP90: {util_p90*100:.1f}%\n\nINTERPRETATION:\nMost roads are\nunder-utilized', 
       transform=ax.transAxes, fontsize=6.5, ha='left', va='top',
       bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9, edgecolor='black'))

# 3.3 Utilization by capacity bins
ax = axes[1, 0]
cap_bins = [0, 1000, 2000, 3000, 5000, capacity.max()+1]
cap_labels = ['0-1k', '1k-2k', '2k-3k', '3k-5k', '5k+']
mean_utils = []
for i in range(len(cap_bins)-1):
    mask = (capacity >= cap_bins[i]) & (capacity < cap_bins[i+1])
    mean_utils.append(utilization[mask].mean() if mask.sum() > 0 else 0)

bars = ax.bar(range(len(cap_labels)), mean_utils, alpha=0.8, color='#c0392b', edgecolor='black', linewidth=0.7)
ax.axhline(1.0, color='#e74c3c', linestyle='--', linewidth=2.5, label='100% = Full capacity', alpha=0.8)
ax.set_xlabel('Capacity Category (vehicles/hour)\n[Small roads (0-1k) vs Large highways (5k+)]', fontsize=10)
ax.set_ylabel('Average Utilization Ratio\n[Mean percentage of capacity being used]', fontsize=10)
ax.set_title('C. Are Smaller or Larger Roads More Utilized?\n[Each bar shows average utilization for roads in that capacity category]\nResult: Similar low utilization (~5-6%) across all capacity levels', fontsize=10, fontweight='bold', pad=10)
ax.set_xticks(range(len(cap_labels)))
ax.set_xticklabels(cap_labels, fontsize=9)
ax.legend(loc='best', framealpha=0.9, fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 0.15)  # Focus on actual utilization range (0-15%)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.02))
for bar, val in zip(bars, mean_utils):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
           f'{val*100:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

# 3.4 Under-utilized roads
ax = axes[1, 1]
util_categories = ['0-10%', '10-25%', '25-50%', '50-75%', '75-100%', '>100%']
util_thresholds = [0, 0.1, 0.25, 0.5, 0.75, 1.0, 10.0]
util_counts = []
for i in range(len(util_thresholds)-1):
    mask = (utilization >= util_thresholds[i]) & (utilization < util_thresholds[i+1])
    util_counts.append(mask.sum())

colors = ['#e74c3c', '#e67e22', '#f39c12', '#27ae60', '#16a085', '#9b59b6']
bars = ax.bar(range(len(util_categories)), util_counts, alpha=0.8, color=colors, edgecolor='black', linewidth=0.7)
ax.set_xlabel('Utilization Category\n[How efficiently is road capacity being used?]', fontsize=10)
ax.set_ylabel('Number of Roads\n[Count of roads in each utilization range]', fontsize=10)
ax.set_title('D. Capacity Utilization Categories\n[Color code: Red=severe under-use (0-10%) | Orange/Yellow=moderate | Green=good | Purple=over-capacity]\nAlmost all roads (>99%) are under-utilized at <10%', fontsize=10, fontweight='bold', pad=10)
ax.set_xticks(range(len(util_categories)))
ax.set_xticklabels(util_categories, fontsize=9, rotation=15)
ax.grid(True, alpha=0.3, axis='y')
ax.yaxis.set_major_locator(ticker.MultipleLocator(2000))
for bar, val in zip(bars, util_counts):
    pct = val / n_edges * 100
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200, 
           f'{val:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=7, fontweight='bold')

plt.tight_layout()
plt.savefig('feature1_chart3_utilization.png', dpi=300, bbox_inches='tight')
print("✓ Saved: feature1_chart3_utilization.png")
plt.show()
plt.close()

print(f"Average utilization: {util_nonzero.mean()*100:.1f}%")
print(f"Under-utilized (<50%): {(utilization < 0.5).sum():,} roads ({(utilization < 0.5).sum()/n_edges*100:.1f}%)")

################################################################################
# CHART 4: CAPACITY-LENGTH RELATIONSHIP
################################################################################
print("\n" + "=" * 80)
print("CHART 4: Capacity-Length Relationship")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('FEATURE 1: Relationship Between Road Capacity and Road Length\n"Do longer roads have higher capacity, or is length independent of capacity?"\nAnalyzing correlation between road segment length (meters) and maximum traffic capacity (veh/h)', 
             fontsize=16, fontweight='bold', y=0.995)
plt.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.06, hspace=0.40, wspace=0.28)

# 4.1 Scatter: Capacity vs Length
ax = axes[0, 0]
valid_mask = (capacity > 0) & (length > 0)
ax.scatter(length[valid_mask], capacity[valid_mask], alpha=0.3, s=3, c='#3498db', edgecolors='none')
if valid_mask.sum() > 0:
    corr = np.corrcoef(length[valid_mask], capacity[valid_mask])[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes, 
           fontsize=10, fontweight='bold', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax.set_xlabel('Road Length (meters)\n[Example: 100m = short city block | 500m = long avenue | 1000m = 1km segment]', fontsize=10)
ax.set_ylabel('Road Capacity (vehicles/hour)\n[Maximum traffic volume road can handle]', fontsize=10)
ax.set_title(f'A. Does Road Length Affect Capacity? (n={valid_mask.sum():,} roads)\n[Looking for relationship between how long and how much capacity]', 
            fontsize=11, fontweight='bold', pad=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 500)  # Focus on main length range (0-500m covers most roads)
ax.set_ylim(0, 5000)  # Focus on main capacity range
ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
ax.yaxis.set_major_locator(ticker.MultipleLocator(500))
# Calculate data coverage in zoom range
zoom_mask = (length <= 500) & (capacity <= 5000)
pct_in_zoom = zoom_mask.sum() / len(capacity) * 100
ax.text(0.98, 0.02, f'ZOOM VIEW:\nLength: 0-500m\nCapacity: 0-5000 veh/h\n\nCOVERAGE:\nShowing {pct_in_zoom:.1f}% of roads\n\nCORRELATION:\nr = {corr:.3f}\n({"Very Weak" if abs(corr) < 0.3 else "Weak" if abs(corr) < 0.5 else "Moderate"})\n\nCONCLUSION:\nLength does NOT\nstrongly predict\ncapacity', 
       transform=ax.transAxes, fontsize=6.5, ha='right', va='bottom',
       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='black'))

# 4.2 Average capacity by length bins
ax = axes[0, 1]
length_bins = [0, 50, 100, 200, 500, length.max()+1]
length_labels = ['0-50m', '50-100m', '100-200m', '200-500m', '500m+']
mean_caps = []
counts = []
for i in range(len(length_bins)-1):
    mask = (length >= length_bins[i]) & (length < length_bins[i+1])
    mean_caps.append(capacity[mask].mean() if mask.sum() > 0 else 0)
    counts.append(mask.sum())

bars = ax.bar(range(len(length_labels)), mean_caps, alpha=0.8, color='#16a085', edgecolor='black', linewidth=0.7)
ax.set_xlabel('Road Length Category\n[Short segments vs long segments]', fontsize=10)
ax.set_ylabel('Average Capacity (vehicles/hour)\n[Mean capacity for roads in each length range]', fontsize=10)
ax.set_title('B. Capacity by Road Length Category\n[Bar shows mean capacity | Label shows mean and count in each length bin]\nObservation: Longer segments (>200m) tend to have slightly lower capacity', fontsize=10, fontweight='bold', pad=10)
ax.set_xticks(range(len(length_labels)))
ax.set_xticklabels(length_labels, fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
ax.yaxis.set_major_locator(ticker.MultipleLocator(200))
for bar, val, count in zip(bars, mean_caps, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30, 
           f'{val:.0f}\n({count:,})', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

# 4.3 Capacity per meter
ax = axes[1, 0]
with np.errstate(divide='ignore', invalid='ignore'):
    cap_per_meter = capacity / length
    cap_per_meter = np.nan_to_num(cap_per_meter, nan=0.0, posinf=0.0, neginf=0.0)
cap_pm_nonzero = cap_per_meter[cap_per_meter > 0]
ax.hist(cap_pm_nonzero, bins=100, alpha=0.75, color='#e67e22', edgecolor='black', linewidth=0.5)
ax.axvline(cap_pm_nonzero.mean(), color='#e74c3c', linestyle='--', linewidth=2.5, 
          label=f'Mean = {cap_pm_nonzero.mean():.1f} veh/h/m', alpha=0.8)
ax.set_xlabel('Capacity per Meter (veh/h/m)\n[Example: 10 veh/h/m = 100m road has 1000 veh/h capacity]', fontsize=10)
ax.set_ylabel('Number of Roads\n[Distribution of capacity intensity]', fontsize=10)
ax.set_title('C. Capacity Intensity (Capacity ÷ Length)\n[Metric: vehicles/hour per meter - shows "capacity density" of road]\nHigh intensity = short roads with high capacity (e.g., highway on-ramps)', fontsize=10, fontweight='bold', pad=10)
ax.legend(loc='best', framealpha=0.9, fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 40)  # Focus on main range (0-40 veh/h/m covers ~90% of data)
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.text(0.98, 0.97, 'X-axis limited to 0-40\nfor clarity', 
       transform=ax.transAxes, fontsize=7, ha='right', va='top',
       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# 4.4 Length distribution by capacity quartiles
ax = axes[1, 1]
cap_quartiles = np.percentile(capacity[capacity > 0], [25, 50, 75])
cap_bins_q = [0, cap_quartiles[0], cap_quartiles[1], cap_quartiles[2], capacity.max()+1]
cap_bin_labels = ['Q1\n(Lowest\n25%)', 'Q2\n(Low-Mid\n25%)', 'Q3\n(Mid-High\n25%)', 'Q4\n(Highest\n25%)']
length_by_cap = [length[(capacity >= cap_bins_q[i]) & (capacity < cap_bins_q[i+1])] for i in range(4)]
bp = ax.boxplot(length_by_cap, tick_labels=cap_bin_labels, patch_artist=True, showfliers=False, widths=0.6)
for patch in bp['boxes']:
    patch.set_facecolor('#9b59b6')
    patch.set_alpha(0.7)
for median in bp['medians']:
    median.set_color('#e74c3c')
    median.set_linewidth(2.5)
ax.set_xlabel('Capacity Quartile\n[Roads grouped by capacity from low to high]', fontsize=10)
ax.set_ylabel('Road Length (meters)\n[Distribution of lengths within each capacity group]', fontsize=10)
ax.set_title('D. Road Length by Capacity Quartile\n[Roads divided into 4 equal groups by capacity - does length differ?]\nResult: All quartiles have similar median length (~100m)', fontsize=10, fontweight='bold', pad=10)
ax.grid(True, alpha=0.3, axis='y')
ax.yaxis.set_major_locator(ticker.MultipleLocator(50))

plt.tight_layout()
plt.savefig('feature1_chart4_length_relationship.png', dpi=300, bbox_inches='tight')
print("✓ Saved: feature1_chart4_length_relationship.png")
plt.show()
plt.close()

print("\n" + "=" * 80)
print("✓✓✓ PART 1 COMPLETE - CHARTS 1-4 GENERATED ✓✓✓")
print("=" * 80)
print("\nGenerated files:")
print("  1. feature1_chart1_distribution.png")
print("  2. feature1_chart2_highway_types.png")
print("  3. feature1_chart3_utilization.png")
print("  4. feature1_chart4_length_relationship.png")
print("\nNext: Run feature1_part2_charts5to8.py for Charts 5-8")
print("=" * 80)
