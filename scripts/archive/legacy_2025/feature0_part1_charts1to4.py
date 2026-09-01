"""
FEATURE 0 ANALYSIS - PART 1: BASIC STATISTICS (Charts 1-4)
============================================================
- Chart 1: Distribution Analysis
- Chart 2: Negative Values Check
- Chart 3: Zero Traffic Analysis
- Chart 4: Temporal Variance (Static Validation)

Repository Code: process_simulations_for_gnn.py Line 104
Source: pop_1pct_basecase_average_output_links.geojson
F0 = Baseline traffic volume WITHOUT policy intervention
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from scipy import stats
import matplotlib.ticker as ticker

# Set professional plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

print("\n" + "#" * 80)
print("#" + " " * 78 + "#")
print("#" + "  FEATURE 0 - PART 1: BASIC STATISTICS (Charts 1-4)".center(78) + "#")
print("#" + "  Paris MATSim Network Analysis".center(78) + "#")
print("#" + " " * 78 + "#")
print("#" * 80)

# DATA LOADING
print("\n" + "=" * 80)
print("LOADING DATA...")
print("=" * 80)

possible_paths = [
    'D:\\Python Projects\\Zamin_Thesis\\ml_surrogates_for_agent_based_transport_models\\data\\train_data\\dist_not_connected_10k_1pct',
    '/content/drive/MyDrive/Zamin_thesis/ml_surrogates_for_agent_based_transport_models/data/train_data/dist_not_connected_10k_1pct',
    '/content/drive/MyDrive/Zamin_thesis/ml_surrogates_for_agent_based_transport_models/data/train_data',
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
    raise FileNotFoundError("Data directory not found. Update possible_paths list.")

batch_files = sorted(data_path.glob('datalist_batch_*.pt'))
if len(batch_files) == 0:
    batch_files = sorted(data_path.glob('*.pt'))

print(f"✓ Found {len(batch_files)} batch files")
print(f"✓ Loading first batch: {batch_files[0].name}")

batch_0 = torch.load(batch_files[0], weights_only=False)
first_scenario = batch_0[0]

# Extract features
vol_base_case = first_scenario.x[:, 0].numpy()
capacity = first_scenario.x[:, 1].numpy()
cap_reduction = first_scenario.x[:, 2].numpy()
highway = first_scenario.x[:, 4].numpy()
length = first_scenario.x[:, 5].numpy()

n_edges = len(vol_base_case)
zeros = (vol_base_case == 0).sum()
negatives = (vol_base_case < 0).sum()

# Highway type decoder (OpenStreetMap classification)
highway_types = {
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

print(f"\n✓ Network Size: {n_edges:,} edges")
print(f"✓ Zero traffic: {zeros:,} ({zeros/n_edges*100:.1f}%)")
print(f"✓ Negative traffic: {negatives:,} ({negatives/n_edges*100:.1f}%)")
print(f"✓ Range: {vol_base_case.min():.1f} to {vol_base_case.max():.1f} veh/h")

################################################################################
# CHART 1: DISTRIBUTION ANALYSIS
################################################################################
print("\n" + "=" * 80)
print("CHART 1: Distribution Analysis")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(17, 13))
fig.suptitle('FEATURE 0: Baseline Traffic Volume Distribution\nParis MATSim Network (31,635 edges)', 
             fontsize=15, fontweight='bold', y=0.995)
plt.subplots_adjust(left=0.07, right=0.96, top=0.94, bottom=0.06, hspace=0.38, wspace=0.28)

# 1.1 Histogram - All values
axes[0, 0].hist(vol_base_case, bins=100, alpha=0.75, color='#3498db', edgecolor='black', linewidth=0.5)
axes[0, 0].set_xlabel('Baseline Traffic Volume (vehicles/hour)\n[Example: 500 veh/h = 500 cars pass through that road per hour | Range: 0 to {:.0f}]'.format(vol_base_case.max()), fontsize=10)
axes[0, 0].set_ylabel('Frequency (Number of Road Segments)\n[Example: Height of 2000 = 2000 roads have that traffic volume]', fontsize=10)
axes[0, 0].set_title(f'A. Distribution: All {n_edges:,} Road Segments\n({zeros:,} zero-traffic roads = {zeros/n_edges*100:.2f}% of network)', 
                     fontsize=11, fontweight='bold', pad=10)
axes[0, 0].axvline(vol_base_case.mean(), color='#e74c3c', linestyle='--', linewidth=2.5, 
                   label=f'Mean = {vol_base_case.mean():.2f} veh/h')
axes[0, 0].axvline(np.median(vol_base_case), color='#27ae60', linestyle='--', linewidth=2.5, 
                   label=f'Median = {np.median(vol_base_case):.2f} veh/h')
axes[0, 0].legend(loc='upper right', framealpha=0.9, fontsize=9)
axes[0, 0].grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
axes[0, 0].set_xlim(0, vol_base_case.max()+50)
axes[0, 0].xaxis.set_major_locator(ticker.MultipleLocator(200))
axes[0, 0].xaxis.set_minor_locator(ticker.MultipleLocator(100))

# 1.2 Histogram - Non-zero values
vol_nonzero = vol_base_case[vol_base_case != 0]
axes[0, 1].hist(vol_nonzero, bins=100, alpha=0.75, color='#e67e22', edgecolor='black', linewidth=0.5)
axes[0, 1].set_xlabel('Baseline Traffic Volume (vehicles/hour)\n[Active Roads Only - Example: 200 veh/h = 200 cars/hour on that specific road]', fontsize=10)
axes[0, 1].set_ylabel('Frequency (Number of Road Segments)\n[How many roads have each traffic level]', fontsize=10)
axes[0, 1].set_title(f'B. Active Roads Distribution (n={len(vol_nonzero):,})\nMean = {vol_nonzero.mean():.2f} veh/h | Std Dev = {vol_nonzero.std():.2f} veh/h', 
                     fontsize=11, fontweight='bold', pad=10)
axes[0, 1].axvline(vol_nonzero.mean(), color='#e74c3c', linestyle='--', linewidth=2.5, 
                   label=f'Mean = {vol_nonzero.mean():.2f} veh/h')
axes[0, 1].axvline(np.median(vol_nonzero), color='#27ae60', linestyle='--', linewidth=2.5, 
                   label=f'Median = {np.median(vol_nonzero):.2f} veh/h')
axes[0, 1].legend(loc='upper right', framealpha=0.9, fontsize=9)
axes[0, 1].grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
axes[0, 1].xaxis.set_major_locator(ticker.MultipleLocator(200))
axes[0, 1].xaxis.set_minor_locator(ticker.MultipleLocator(100))

# 1.3 Log scale
vol_positive = vol_base_case[vol_base_case > 0]
log_values = np.log10(vol_positive + 1)
axes[1, 0].hist(log_values, bins=80, alpha=0.75, color='#16a085', edgecolor='black', linewidth=0.5)
axes[1, 0].set_xlabel('Log10(Traffic Volume + 1) - Logarithmic Scale\n[Example: 0=1 veh/h | 1=10 veh/h | 2=100 veh/h | 3=1000 veh/h]', fontsize=10)
axes[1, 0].set_ylabel('Frequency (Number of Road Segments)\n[How many roads fall in each traffic magnitude range]', fontsize=10)
axes[1, 0].set_title(f'C. Logarithmic Scale View (n={len(vol_positive):,} active roads)\n[Compresses wide range 1-1596 veh/h to see distribution pattern clearly]', 
                     fontsize=11, fontweight='bold', pad=10)
axes[1, 0].grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
# Add reference lines for magnitude orders
for mag, label in [(0, '1'), (1, '10'), (2, '100'), (3, '1000')]:
    if mag <= log_values.max():
        axes[1, 0].axvline(mag, color='red', linestyle=':', alpha=0.4, linewidth=1.5)
        axes[1, 0].text(mag, axes[1, 0].get_ylim()[1]*0.95, f'{label}\nveh/h', 
                       ha='center', va='top', fontsize=8, color='red', fontweight='bold')
axes[1, 0].xaxis.set_major_locator(ticker.MultipleLocator(0.5))

# 1.4 Box plot with detailed annotations
box_data = [vol_base_case, vol_nonzero, vol_positive]
bp = axes[1, 1].boxplot(box_data, 
                         tick_labels=['All Roads\n(n={:,})\nIncl. zeros'.format(n_edges), 
                                    'Non-Zero\n(n={:,})\nActive only'.format(len(vol_nonzero)), 
                                    'Positive\n(n={:,})\nNo negatives'.format(len(vol_positive))], 
                         showfliers=True, patch_artist=True,
                         boxprops=dict(facecolor='#3498db', alpha=0.7, linewidth=1.5),
                         medianprops=dict(color='#e74c3c', linewidth=3),
                         whiskerprops=dict(linewidth=1.5, color='#2c3e50'),
                         capprops=dict(linewidth=1.5, color='#2c3e50'),
                         flierprops=dict(marker='o', markerfacecolor='red', markersize=2, alpha=0.3))

# Add explanatory text annotations
axes[1, 1].text(0.02, 0.98, 'BOX PLOT COMPONENTS:', transform=axes[1, 1].transAxes,
               fontsize=9, fontweight='bold', va='top', ha='left',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
axes[1, 1].text(0.02, 0.92, '• Red Line = MEDIAN (50th percentile)\n  Half roads above, half below this value',
               transform=axes[1, 1].transAxes, fontsize=7.5, va='top', ha='left')
axes[1, 1].text(0.02, 0.84, '• Blue Box = IQR (Interquartile Range)\n  Contains middle 50% of all roads',
               transform=axes[1, 1].transAxes, fontsize=7.5, va='top', ha='left')
axes[1, 1].text(0.02, 0.76, '• Box Bottom = Q1 (25th percentile)\n  25% of roads below this traffic level',
               transform=axes[1, 1].transAxes, fontsize=7.5, va='top', ha='left')
axes[1, 1].text(0.02, 0.68, '• Box Top = Q3 (75th percentile)\n  75% of roads below this traffic level',
               transform=axes[1, 1].transAxes, fontsize=7.5, va='top', ha='left')
axes[1, 1].text(0.02, 0.60, '• Whiskers = Extend to min/max\n  within 1.5×IQR from box edges',
               transform=axes[1, 1].transAxes, fontsize=7.5, va='top', ha='left')
axes[1, 1].text(0.02, 0.52, '• Red Dots = OUTLIERS\n  Extreme values beyond whiskers',
               transform=axes[1, 1].transAxes, fontsize=7.5, va='top', ha='left')

axes[1, 1].set_ylabel('Baseline Traffic Volume (vehicles/hour)\n[Vertical spread shows traffic variability | Wider box = more variable traffic]', fontsize=10)
axes[1, 1].set_title('D. Box Plot Statistical Summary - Compare Traffic Distributions\n[Shows median, spread, and outliers for each road category]', 
                     fontsize=11, fontweight='bold', pad=10)
axes[1, 1].grid(True, alpha=0.3, axis='y', linestyle=':', linewidth=0.5)
axes[1, 1].yaxis.set_major_locator(ticker.MultipleLocator(200))
axes[1, 1].yaxis.set_minor_locator(ticker.MultipleLocator(100))

plt.tight_layout()
plt.savefig('feature0_chart1_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: feature0_chart1_distribution.png")
plt.show()  # Display in Colab
plt.close()

################################################################################
# CHART 2: NEGATIVE VALUES ANALYSIS
################################################################################
print("\n" + "=" * 80)
print("CHART 2: Negative Values Analysis")
print("=" * 80)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('FEATURE 0: Negative Values Check - Directional Encoding Validation', 
             fontsize=14, fontweight='bold')
plt.subplots_adjust(left=0.07, right=0.96, top=0.90, bottom=0.10, wspace=0.20)

# 2.1 Scatter plot
neg_mask = vol_base_case < 0
pos_mask = vol_base_case >= 0
axes[0].scatter(capacity[neg_mask], vol_base_case[neg_mask], alpha=0.6, s=20, 
                c='#e74c3c', label=f'Negative Values: {neg_mask.sum():,} roads ({neg_mask.sum()/n_edges*100:.2f}%)', edgecolors='black', linewidth=0.5)
axes[0].scatter(capacity[pos_mask], vol_base_case[pos_mask], alpha=0.4, s=10, 
                c='#3498db', label=f'Positive/Zero Values: {pos_mask.sum():,} roads ({pos_mask.sum()/n_edges*100:.2f}%)', edgecolors='none')
axes[0].axhline(0, color='black', linestyle='-', linewidth=2.5, label='Zero Reference Line', alpha=0.8)
axes[0].set_xlabel('Road Capacity (vehicles/hour)\n[Example: 2000 veh/h capacity = road can handle max 2000 cars/hour]', fontsize=11)
axes[0].set_ylabel('Baseline Traffic Volume (vehicles/hour)\n[Example: -500 = traffic in opposite direction | +500 = normal direction | 0 = empty]', fontsize=11)
axes[0].set_title('A. Traffic Volume vs Road Capacity\n[Check for directional encoding: negative values = bidirectional network]', fontsize=12, fontweight='bold', pad=10)
axes[0].legend(loc='best', framealpha=0.9, fontsize=9)
axes[0].grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
axes[0].xaxis.set_major_locator(ticker.MultipleLocator(1000))
axes[0].yaxis.set_major_locator(ticker.MultipleLocator(200))

# 2.2 Histogram comparison
bins = np.linspace(vol_base_case.min(), vol_base_case.max(), 100)
axes[1].hist(vol_base_case[neg_mask], bins=bins, alpha=0.7, color='#e74c3c', 
             label=f'Negative: {neg_mask.sum()} roads', edgecolor='black', linewidth=0.5)
axes[1].hist(vol_base_case[pos_mask], bins=bins, alpha=0.7, color='#3498db', 
             label=f'Positive/Zero: {pos_mask.sum():,} roads', edgecolor='black', linewidth=0.5)
axes[1].axvline(0, color='black', linestyle='-', linewidth=2.5, label='Zero Reference', alpha=0.8)
axes[1].set_xlabel('Baseline Traffic Volume (vehicles/hour)\n[Example: Left of zero (<0) = opposite direction | Right of zero (>0) = normal flow]', fontsize=11)
axes[1].set_ylabel('Frequency (Number of Road Segments)\n[Bar height = how many roads have that traffic volume]', fontsize=11)
axes[1].set_title('B. Distribution Comparison: Negative vs Positive Traffic\n[Overlapping histograms show value spread]', fontsize=12, fontweight='bold', pad=10)
axes[1].legend(loc='best', framealpha=0.9, fontsize=9)
axes[1].grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
axes[1].xaxis.set_major_locator(ticker.MultipleLocator(200))

plt.tight_layout()
plt.savefig('feature0_chart2_negative_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: feature0_chart2_negative_analysis.png")
plt.show()  # Display in Colab
plt.close()

print(f"\nResult: {negatives:,} negative values ({negatives/n_edges*100:.2f}%)")
if negatives == 0:
    print("✓ Network uses DIRECTIONAL links (separate edge per direction)")

################################################################################
# CHART 3: ZERO TRAFFIC ANALYSIS
################################################################################
print("\n" + "=" * 80)
print("CHART 3: Zero Traffic Analysis")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(17, 14))
fig.suptitle('FEATURE 0: Zero Traffic Analysis - Why 24% Roads Empty?', 
             fontsize=14, fontweight='bold', y=0.995)
plt.subplots_adjust(left=0.07, right=0.96, top=0.93, bottom=0.07, hspace=0.40, wspace=0.25)

zero_mask = vol_base_case == 0
nonzero_mask = vol_base_case > 0

# 3.1 Capacity distribution
axes[0, 0].hist(capacity[zero_mask], bins=60, alpha=0.6, color='gray', 
                label=f'Zero Traffic: {zero_mask.sum():,} roads (Mean capacity = {capacity[zero_mask].mean():.0f} veh/h)', 
                edgecolor='black', linewidth=0.5)
axes[0, 0].hist(capacity[nonzero_mask], bins=60, alpha=0.7, color='#27ae60', 
                label=f'Has Traffic: {nonzero_mask.sum():,} roads (Mean capacity = {capacity[nonzero_mask].mean():.0f} veh/h)', 
                edgecolor='black', linewidth=0.5)
axes[0, 0].set_xlabel('Road Capacity (vehicles/hour)\n[Example: 1000 veh/h = road designed to handle max 1000 vehicles/hour]', fontsize=10)
axes[0, 0].set_ylabel('Frequency (Number of Road Segments)\n[How many roads have each capacity level]', fontsize=10)
axes[0, 0].set_title(f'A. Capacity Distribution: Empty vs Active Roads\n[Difference in mean: {capacity[nonzero_mask].mean() - capacity[zero_mask].mean():.0f} veh/h]', 
                     fontsize=11, fontweight='bold', pad=10)
axes[0, 0].legend(loc='best', framealpha=0.9, fontsize=8)
axes[0, 0].grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
axes[0, 0].xaxis.set_major_locator(ticker.MultipleLocator(1000))

# 3.2 Highway type counts with full names
unique_highway_types = np.unique(highway)
zero_counts = [((highway == ht) & zero_mask).sum() for ht in unique_highway_types]
nonzero_counts = [((highway == ht) & nonzero_mask).sum() for ht in unique_highway_types]

x = np.arange(len(unique_highway_types))
width = 0.35
axes[0, 1].bar(x - width/2, zero_counts, width, label=f'Zero traffic ({sum(zero_counts):,} roads)', color='gray', alpha=0.7, edgecolor='black')
axes[0, 1].bar(x + width/2, nonzero_counts, width, label=f'Has traffic ({sum(nonzero_counts):,} roads)', color='#27ae60', alpha=0.7, edgecolor='black')
axes[0, 1].set_xlabel('Road Type (OpenStreetMap Classification)\n[0=Motorway | 1=Trunk | 2=Primary | 3=Secondary | 4=Tertiary | 5=Residential | 6=Service | 7=Unclass. | 8=Living St. | 9=Other]', fontsize=9)
axes[0, 1].set_ylabel('Number of Roads\n[Count of road segments in Paris network]', fontsize=10)
axes[0, 1].set_title('B. Traffic Distribution by Road Type\n[Compare major highways vs local streets - which types are more utilized?]', fontsize=11, fontweight='bold', pad=10)
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels([f'{int(ht)}\n{highway_types.get(int(ht), "Unknown")[:4]}' for ht in unique_highway_types], fontsize=8)
axes[0, 1].legend(loc='best', framealpha=0.9, fontsize=9)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 3.3 Zero percentage by type with detailed labels
zero_pcts = []
for ht in unique_highway_types:
    type_mask = highway == ht
    type_zeros = (type_mask & zero_mask).sum()
    type_total = type_mask.sum()
    zero_pcts.append(100 * type_zeros / type_total if type_total > 0 else 0)

colors = ['#e74c3c' if pct > 50 else '#f39c12' if pct > 20 else '#27ae60' for pct in zero_pcts]
bars = axes[1, 0].bar(unique_highway_types, zero_pcts, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
axes[1, 0].set_xlabel('Road Type Code\n[Full names: 0=Motorway, 1=Trunk, 2=Primary, 3=Secondary, 4=Tertiary,\n5=Residential, 6=Service, 7=Unclassified, 8=Living Street, 9=Other]', fontsize=9)
axes[1, 0].set_ylabel('Zero Traffic Percentage (%)\n[What % of each road type is unused in simulation]', fontsize=10)
axes[1, 0].set_title('C. Road Utilization Rate by Type\n[Red bar (>50% empty) = poorly utilized | Green bar (<20% empty) = well utilized]', 
                     fontsize=11, fontweight='bold', pad=10)
axes[1, 0].grid(True, alpha=0.3, axis='y')
axes[1, 0].axhline(50, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='50% threshold (critical)')
axes[1, 0].axhline(20, color='orange', linestyle='--', alpha=0.5, linewidth=1.5, label='20% threshold (warning)')
axes[1, 0].legend(loc='upper right', framealpha=0.9, fontsize=8)
axes[1, 0].set_xticks(unique_highway_types)
axes[1, 0].set_xticklabels([f'{int(ht)}\n{highway_types.get(int(ht), "Unknown")[:4]}' for ht in unique_highway_types], fontsize=8)
# Add percentage labels on bars
for bar, pct in zip(bars, zero_pcts):
    if pct > 5:  # Only show label if bar is visible
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                       f'{pct:.1f}%', ha='center', va='bottom', fontsize=7, fontweight='bold')

# 3.4 Length distribution
axes[1, 1].hist(length[zero_mask], bins=60, alpha=0.6, color='gray', 
                label=f'Zero Traffic: Mean = {length[zero_mask].mean():.1f}m | Median = {np.median(length[zero_mask]):.1f}m', 
                edgecolor='black', linewidth=0.5)
axes[1, 1].hist(length[nonzero_mask], bins=60, alpha=0.7, color='#27ae60', 
                label=f'Has Traffic: Mean = {length[nonzero_mask].mean():.1f}m | Median = {np.median(length[nonzero_mask]):.1f}m', 
                edgecolor='black', linewidth=0.5)
axes[1, 1].set_xlabel('Road Segment Length (meters)\n[Example: 100m = road edge is 100 meters long (1 city block = 80-100m)]', fontsize=10)
axes[1, 1].set_ylabel('Frequency (Number of Road Segments)\n[How many roads have each length]', fontsize=10)
axes[1, 1].set_title('D. Road Length Distribution Comparison\n[Are longer roads more likely to have traffic?]', fontsize=11, fontweight='bold', pad=10)
axes[1, 1].legend(loc='best', framealpha=0.9, fontsize=8)
axes[1, 1].grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
axes[1, 1].xaxis.set_major_locator(ticker.MultipleLocator(100))

plt.tight_layout()
plt.savefig('feature0_chart3_zeros_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: feature0_chart3_zeros_analysis.png")
plt.show()  # Display in Colab
plt.close()

# Print road type legend
print("\n" + "-" * 80)
print("ROAD TYPE DEFINITIONS (OpenStreetMap Classification):")
print("-" * 80)
for code, name in highway_types.items():
    count = (highway == code).sum()
    zero_count = ((highway == code) & zero_mask).sum()
    zero_pct = 100 * zero_count / count if count > 0 else 0
    print(f"  Type {code}: {name:15s} - {count:5,} roads ({zero_count:5,} empty = {zero_pct:5.1f}%)")
print("-" * 80)

################################################################################
# CHART 4: TEMPORAL VARIANCE (STATIC VALIDATION)
################################################################################
print("\n" + "=" * 80)
print("CHART 4: Temporal Variance Check (Static Feature Validation)")
print("=" * 80)
print("Loading 10 scenarios for variance analysis...")

# Load 10 scenarios
n_scenarios = min(10, len(batch_0))
vol_scenarios = []
for i in range(n_scenarios):
    vol_scenarios.append(batch_0[i].x[:, 0].numpy())

vol_scenarios = np.array(vol_scenarios)  # Shape: (n_scenarios, n_edges)

# Calculate variance across scenarios
temporal_variance = np.var(vol_scenarios, axis=0)
temporal_mean = np.mean(vol_scenarios, axis=0)
temporal_std = np.std(vol_scenarios, axis=0)
# Calculate CV with safe division (avoid division by zero warning)
with np.errstate(divide='ignore', invalid='ignore'):
    cv = temporal_std / np.abs(temporal_mean)
    cv = np.nan_to_num(cv, nan=0.0, posinf=0.0, neginf=0.0)  # Convert NaN/Inf to 0

print(f"\nTemporal Variance Statistics:")
print(f"  Mean variance: {temporal_variance.mean():.6f}")
print(f"  Max variance: {temporal_variance.max():.6f}")
print(f"  Edges with variance > 0: {(temporal_variance > 0).sum()} ({(temporal_variance > 0).sum()/n_edges*100:.4f}%)")

fig, axes = plt.subplots(2, 2, figsize=(16, 13))
fig.suptitle(f'FEATURE 0: Temporal Variance Check - Static Feature Validation\nAnalyzing {n_scenarios} Scenarios', 
             fontsize=14, fontweight='bold', y=0.995)
plt.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.06, hspace=0.30, wspace=0.22)

# 4.1 Variance distribution
axes[0, 0].hist(temporal_variance, bins=100, alpha=0.75, color='#9b59b6', edgecolor='black', linewidth=0.5)
axes[0, 0].set_xlabel('Variance Across {} Scenarios (veh²/h²)\n[Example: 0 = traffic identical in all scenarios (static) | >0 = varies between scenarios]'.format(n_scenarios), fontsize=10)
axes[0, 0].set_ylabel('Frequency (Number of Road Segments)\n[How many roads have each variance level]', fontsize=10)
axes[0, 0].set_title(f'A. Variance Distribution (Should be ≈0 for Static Feature)\nMean = {temporal_variance.mean():.8f} | Max = {temporal_variance.max():.8f} | Non-zero = {(temporal_variance > 0).sum()}', 
                     fontsize=11, fontweight='bold', pad=10)
axes[0, 0].axvline(temporal_variance.mean(), color='#e74c3c', linestyle='--', linewidth=2.5, 
                   label=f'Mean Variance = {temporal_variance.mean():.8f}', alpha=0.8)
axes[0, 0].axvline(0, color='#27ae60', linestyle='-', linewidth=2, 
                   label='Zero (Perfect Static)', alpha=0.8)
axes[0, 0].legend(loc='best', framealpha=0.9, fontsize=9)
axes[0, 0].grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

# 4.2 Sample edges across scenarios
sample_indices = np.random.choice(n_edges, size=min(50, n_edges), replace=False)
for idx in sample_indices:
    axes[0, 1].plot(range(n_scenarios), vol_scenarios[:, idx], alpha=0.3, linewidth=1, color='#3498db')
axes[0, 1].set_xlabel('Scenario Index (Different Policy Scenarios)\n[Example: Scenario 0, 1, 2... each tests different policy | Total {} scenarios]'.format(n_scenarios), fontsize=10)
axes[0, 1].set_ylabel('Baseline Traffic Volume (vehicles/hour)\n[Example: Flat line at 300 = that road always has 300 veh/h regardless of policy]', fontsize=10)
axes[0, 1].set_title(f'B. Temporal Consistency Check: {min(50, n_edges)} Random Road Segments\n[Flat horizontal lines = Static | Varying lines = Dynamic]', 
                     fontsize=11, fontweight='bold', pad=10)
axes[0, 1].grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
axes[0, 1].xaxis.set_major_locator(ticker.MultipleLocator(1))
axes[0, 1].set_xlim(-0.5, n_scenarios-0.5)

# 4.3 Coefficient of Variation
axes[1, 0].hist(cv[cv > 0], bins=100, alpha=0.75, color='#e67e22', edgecolor='black', linewidth=0.5)
axes[1, 0].set_xlabel('Coefficient of Variation (CV = Std Dev / Mean)\n[Example: CV=0.05 means 5% variation | CV=0 = perfectly static | CV>0.1 = significant change]', fontsize=10)
axes[1, 0].set_ylabel('Frequency (Number of Road Segments)\n[How many roads have each CV level]', fontsize=10)
axes[1, 0].set_title(f'C. Relative Variability Analysis\nMean CV = {cv[cv > 0].mean():.8f} | Roads with CV > 0: {(cv > 0).sum():,}', 
                     fontsize=11, fontweight='bold', pad=10)
axes[1, 0].axvline(0.1, color='red', linestyle='--', linewidth=2, 
                  label='CV = 0.1 (10% variation threshold)', alpha=0.6)
axes[1, 0].legend(loc='best', framealpha=0.9, fontsize=9)
axes[1, 0].grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

# 4.4 Max - Min difference
diff = vol_scenarios.max(axis=0) - vol_scenarios.min(axis=0)
axes[1, 1].hist(diff, bins=100, alpha=0.75, color='#16a085', edgecolor='black', linewidth=0.5)
axes[1, 1].set_xlabel('Range of Values (Max - Min) Across {} Scenarios (veh/h)\n[Example: Range=50 means traffic varies by 50 veh/h between scenarios | 0=static]'.format(n_scenarios), fontsize=10)
axes[1, 1].set_ylabel('Frequency (Number of Road Segments)\n[How many roads have each variation range]', fontsize=10)
axes[1, 1].set_title(f'D. Absolute Variation Range per Road Segment\nMean Range = {diff.mean():.8f} | Max Range = {diff.max():.8f} | Zero Range = {(diff == 0).sum():,}', 
                     fontsize=11, fontweight='bold', pad=10)
axes[1, 1].axvline(0, color='#27ae60', linestyle='-', linewidth=2.5, 
                  label='Zero (Perfect Static Feature)', alpha=0.8)
axes[1, 1].axvline(diff.mean(), color='#e74c3c', linestyle='--', linewidth=2, 
                  label=f'Mean = {diff.mean():.8f}', alpha=0.8)
axes[1, 1].legend(loc='best', framealpha=0.9, fontsize=9)
axes[1, 1].grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

plt.tight_layout()
plt.savefig('feature0_chart4_temporal_variance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: feature0_chart4_temporal_variance.png")
plt.show()  # Display in Colab
plt.close()

if temporal_variance.max() < 1e-10:
    print("\n✓✓✓ CONFIRMED: F0 is STATIC - identical across all scenarios ✓✓✓")
else:
    print(f"\n⚠ Warning: Some variation detected (max variance = {temporal_variance.max():.10f})")

print("\n" + "=" * 80)
print("✓✓✓ PART 1 COMPLETE - Charts 1-4 Generated Successfully ✓✓✓")
print("=" * 80)
print("\nGenerated Charts:")
print("  1. feature0_chart1_distribution.png - Traffic volume distribution")
print("  2. feature0_chart2_negative_analysis.png - Directional encoding check")
print("  3. feature0_chart3_zeros_analysis.png - Zero traffic analysis")
print("  4. feature0_chart4_temporal_variance.png - Static validation")
print("\n✓ Charts displayed inline above (Colab)")
print("✓ PNG files saved in current directory")
print("\nNext: Run feature0_part2_charts5to8.py for Network Analysis")
