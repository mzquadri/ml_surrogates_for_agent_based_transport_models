"""
FEATURE 1 ANALYSIS - PART 2: ROAD CAPACITY (Charts 5-8)
=======================================================
Charts 5-8: Outliers, Network Stats & Policy Analysis

Run after feature1_part1_charts1to4.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.ticker as ticker

# Set professional plotting style
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
print("#" + "  FEATURE 1 - PART 2: ROAD CAPACITY (Charts 5-8)".center(78) + "#")
print("#" + "  Outliers, Network Stats & Policy Analysis".center(78) + "#")
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

# Highway type decoder
highway_type_names = {
    0: 'Motorway', 1: 'Trunk', 2: 'Primary', 3: 'Secondary', 4: 'Tertiary',
    5: 'Residential', 6: 'Service', 7: 'Unclassified', 8: 'Living Street', 9: 'Other'
}

# Calculate utilization
with np.errstate(divide='ignore', invalid='ignore'):
    utilization = np.abs(vol_base_case) / capacity
    utilization = np.nan_to_num(utilization, nan=0.0, posinf=0.0, neginf=0.0)

################################################################################
# CHART 5: OUTLIERS & EXTREME VALUES
################################################################################
print("\n" + "=" * 80)
print("CHART 5: Capacity Outliers & Extreme Values")
print("=" * 80)

# Identify outliers
Q1, Q3 = np.percentile(capacity, [25, 75])
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers_low = capacity < lower_bound
outliers_high = capacity > upper_bound
outliers = outliers_low | outliers_high

print(f"Outlier analysis: {outliers.sum():,} outliers ({outliers.sum()/n_edges*100:.1f}%)")
print(f"  Low outliers: {outliers_low.sum():,} | High outliers: {outliers_high.sum():,}")

fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('FEATURE 1: Capacity Outliers & Extreme Values Analysis\nIdentifying Roads with Unusual Capacity Values\nUsing 1.5×IQR Method (Interquartile Range)', 
             fontsize=16, fontweight='bold', y=0.995)
plt.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.06, hspace=0.40, wspace=0.28)

# 5.1 Outlier identification
ax = axes[0, 0]
ax.scatter(range(n_edges), capacity, alpha=0.3, s=1, c='lightgray', label='Normal', edgecolors='none')
ax.scatter(np.where(outliers_high)[0], capacity[outliers_high], alpha=0.7, s=8, c='#e74c3c', 
          label=f'High outliers ({outliers_high.sum():,})', edgecolors='none')
if outliers_low.sum() > 0:
    ax.scatter(np.where(outliers_low)[0], capacity[outliers_low], alpha=0.7, s=8, c='#f39c12', 
              label=f'Low outliers ({outliers_low.sum():,})', edgecolors='none')
ax.axhline(upper_bound, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.7, label=f'Upper bound = {upper_bound:.0f}')
ax.axhline(lower_bound, color='#f39c12', linestyle='--', linewidth=2, alpha=0.7, label=f'Lower bound = {lower_bound:.0f}')
ax.set_xlabel('Road Index\n[Each point = one road segment | Roads ordered by network index]', fontsize=10, fontweight='bold')
ax.set_ylabel('Road Capacity (vehicles/hour)\n[Vertical axis shows capacity value]', fontsize=10, fontweight='bold')
ax.set_title('A. Outlier Detection Using Statistical Method\n[Red dots = unusually HIGH capacity (above Q3 + 1.5×IQR)]\n[Orange = unusually LOW (below Q1 - 1.5×IQR) | Gray = normal range]', fontsize=10, fontweight='bold', pad=10)
ax.legend(loc='upper right', framealpha=0.95, fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_locator(ticker.MultipleLocator(2000))

# 5.2 High-capacity roads by type
ax = axes[0, 1]
top_10pct = np.percentile(capacity, 90)
high_cap_mask = capacity >= top_10pct
high_cap_by_type = [(highway[high_cap_mask] == ht).sum() for ht in unique_types]
bars = ax.bar(unique_types, high_cap_by_type, alpha=0.8, color='#e74c3c', edgecolor='black', linewidth=0.7)
ax.set_xlabel('Road Type\n[OpenStreetMap classification: 0=Motorway to 9=Other]', fontsize=10, fontweight='bold')
ax.set_ylabel('Number of High-Capacity Roads\n[Count in top 10% capacity (>{:.0f} veh/h)]'.format(top_10pct), fontsize=10, fontweight='bold')
ax.set_title('B. High-Capacity Roads Distribution by Type\n[Top 10% = roads with capacity >{:.0f} veh/h]\n[Label shows count and % of that road type in top 10%]'.format(top_10pct), fontsize=10, fontweight='bold', pad=10)
ax.set_xticks(unique_types)
ax.set_xticklabels([f'{int(ht)}\n{highway_type_names.get(int(ht), "?")[:4]}' for ht in unique_types], fontsize=8)
ax.grid(True, alpha=0.3, axis='y')
for bar, val, ht in zip(bars, high_cap_by_type, unique_types):
    total_of_type = (highway == ht).sum()
    pct = 100 * val / total_of_type if total_of_type > 0 else 0
    if val > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, 
               f'{val:,}\n({pct:.0f}%)', ha='center', va='bottom', fontsize=7, fontweight='bold')

# 5.3 Zero-capacity roads
ax = axes[1, 0]
zero_cap_mask = capacity == 0
zero_by_type = [(highway[zero_cap_mask] == ht).sum() for ht in unique_types]
total_by_type = [(highway == ht).sum() for ht in unique_types]
with np.errstate(divide='ignore', invalid='ignore'):
    zero_pct = [100 * z / t if t > 0 else 0 for z, t in zip(zero_by_type, total_by_type)]
colors = ['#e74c3c' if p > 50 else '#f39c12' if p > 20 else '#27ae60' for p in zero_pct]
bars = ax.bar(unique_types, zero_pct, alpha=0.8, color=colors, edgecolor='black', linewidth=0.7)
ax.set_xlabel('Road Type\n[Analyzing data completeness across road categories]', fontsize=10, fontweight='bold')
ax.set_ylabel('Percentage with Zero Capacity (%)\n[% of roads with capacity = 0]', fontsize=10, fontweight='bold')
ax.set_title(f'C. Zero-Capacity Roads by Type (Total: {zero_cap_mask.sum():,} roads = {zero_cap_mask.sum()/n_edges*100:.1f}%)\n[Color code: Red (>50%) = most roads missing data | Orange (20-50%) | Green (<20%) = good]\n[Label shows percentage and count for each road type]', 
            fontsize=10, fontweight='bold', pad=10)
ax.set_xticks(unique_types)
ax.set_xticklabels([f'{int(ht)}\n{highway_type_names.get(int(ht), "?")[:4]}' for ht in unique_types], fontsize=8)
ax.grid(True, alpha=0.3, axis='y')
ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
for bar, val, count in zip(bars, zero_pct, zero_by_type):
    if val > 1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
               f'{val:.0f}%\n({count:,})', ha='center', va='bottom', fontsize=7, fontweight='bold')

# 5.4 Capacity variance by type
ax = axes[1, 1]
std_by_type = [capacity[highway == ht].std() for ht in unique_types]
cv_by_type = [capacity[highway == ht].std() / capacity[highway == ht].mean() 
              if capacity[highway == ht].mean() > 0 else 0 for ht in unique_types]
ax2 = ax.twinx()
bars1 = ax.bar(unique_types - 0.2, std_by_type, width=0.4, alpha=0.8, color='#3498db', 
              edgecolor='black', linewidth=0.7, label='Std Dev')
line1 = ax2.plot(unique_types, cv_by_type, 'o-', color='#e74c3c', linewidth=2.5, markersize=8, 
                label='Coeff. of Variation', alpha=0.8)
ax.set_xlabel('Road Type\n[Measuring consistency of capacity within each road category]', fontsize=10, fontweight='bold')
ax.set_ylabel('Standard Deviation (veh/h)\n[Blue bars = absolute spread]', fontsize=10, color='#3498db', fontweight='bold')
ax2.set_ylabel('Coefficient of Variation (std/mean)\n[Red line = relative variability]', fontsize=10, color='#e74c3c', fontweight='bold')
ax.set_title('D. Capacity Variability Analysis by Road Type\n[High std dev = wide range of capacities | High CoV = inconsistent relative to mean]\n[Helps identify which road types have standardized vs varied capacity]', 
            fontsize=10, fontweight='bold', pad=10)
ax.set_xticks(unique_types)
ax.set_xticklabels([f'{int(ht)}\n{highway_type_names.get(int(ht), "?")[:4]}' for ht in unique_types], fontsize=8)
ax.tick_params(axis='y', labelcolor='#3498db')
ax2.tick_params(axis='y', labelcolor='#e74c3c')
ax.legend(loc='upper left', framealpha=0.9, fontsize=9)
ax2.legend(loc='upper right', framealpha=0.9, fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('feature1_chart5_outliers.png', dpi=300, bbox_inches='tight')
print("✓ Saved: feature1_chart5_outliers.png")
plt.show()
plt.close()

################################################################################
# CHART 6: NETWORK CAPACITY STATISTICS
################################################################################
print("\n" + "=" * 80)
print("CHART 6: Network Capacity Statistics")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('FEATURE 1: Network-Level Capacity Statistics\nAnalyzing Overall Distribution and Concentration of Road Capacity\nGini Coefficient, Lorenz Curve, and Capacity Inequality Metrics', 
             fontsize=16, fontweight='bold', y=0.995)
plt.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.06, hspace=0.42, wspace=0.30)

# 6.1 Lorenz curve for capacity inequality
ax = axes[0, 0]
sorted_cap = np.sort(capacity[capacity > 0])
cum_cap = np.cumsum(sorted_cap)
cum_cap_pct = cum_cap / cum_cap[-1] * 100
cum_roads_pct = np.arange(1, len(sorted_cap) + 1) / len(sorted_cap) * 100
ax.plot(cum_roads_pct, cum_cap_pct, linewidth=2.5, color='#3498db', label='Actual Distribution')
ax.plot([0, 100], [0, 100], 'r--', linewidth=2.5, label='Perfect Equality', alpha=0.7)
ax.fill_between(cum_roads_pct, cum_cap_pct, cum_roads_pct, alpha=0.3, color='lightcoral')

# Calculate Gini coefficient using trapezoid (with fallback for older numpy)
try:
    gini = 1 - 2 * np.trapezoid(cum_cap_pct / 100, cum_roads_pct / 100)
except AttributeError:
    gini = 1 - 2 * np.trapz(cum_cap_pct / 100, cum_roads_pct / 100)

ax.text(0.05, 0.95, f'Gini Coefficient: {gini:.3f}\n(0=perfect equality | 1=max inequality)', 
       transform=ax.transAxes, fontsize=10, fontweight='bold', 
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), verticalalignment='top')
ax.set_xlabel('Cumulative Percentage of Roads (sorted by capacity)\n[X-axis: roads ordered from lowest to highest capacity]', fontsize=10, fontweight='bold')
ax.set_ylabel('Cumulative Percentage of Total Network Capacity\n[Y-axis: what % of total capacity accumulated so far]', fontsize=10, fontweight='bold')
ax.set_title('A. Lorenz Curve - Measuring Capacity Inequality\n[Closer to diagonal red line = more equal distribution]\n[Large area between curves (pink) = high inequality]', 
            fontsize=10, fontweight='bold', pad=10)
ax.legend(loc='best', framealpha=0.9, fontsize=9)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.yaxis.set_major_locator(ticker.MultipleLocator(10))

# 6.2 Summary statistics table
ax = axes[0, 1]
ax.axis('off')
summary_data = [
    ['CAPACITY STATISTICS', '', ''],
    ['', '', ''],
    ['Total Network Capacity', f'{capacity.sum():,.0f}', 'veh/h'],
    ['Mean Capacity', f'{capacity.mean():.0f}', 'veh/h'],
    ['Median Capacity', f'{np.median(capacity):.0f}', 'veh/h'],
    ['Std Deviation', f'{capacity.std():.0f}', 'veh/h'],
    ['', '', ''],
    ['Min Capacity', f'{capacity.min():.0f}', 'veh/h'],
    ['Max Capacity', f'{capacity.max():.0f}', 'veh/h'],
    ['Range', f'{capacity.max() - capacity.min():.0f}', 'veh/h'],
    ['', '', ''],
    ['25th Percentile (Q1)', f'{np.percentile(capacity, 25):.0f}', 'veh/h'],
    ['75th Percentile (Q3)', f'{np.percentile(capacity, 75):.0f}', 'veh/h'],
    ['90th Percentile', f'{np.percentile(capacity, 90):.0f}', 'veh/h'],
    ['', '', ''],
    ['Zero-Capacity Roads', f'{(capacity == 0).sum():,}', f'({(capacity == 0).sum()/n_edges*100:.1f}%)'],
    ['Gini Coefficient', f'{gini:.3f}', '(inequality)'],
    ['Coeff. of Variation', f'{capacity.std()/capacity.mean():.3f}', '(variability)'],
]
table = ax.table(cellText=summary_data, cellLoc='left', loc='center',
                colWidths=[0.5, 0.25, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.2)
for i in [0, 1, 6, 10, 14]:
    for j in range(3):
        table[(i, j)].set_facecolor('#3498db')
        table[(i, j)].set_text_props(weight='bold', color='white')
ax.set_title('B. Comprehensive Capacity Statistics Table\n[Complete statistical summary: central tendency, spread, percentiles]\n[Blue rows = category headers | White rows = actual values]', 
            fontsize=10, fontweight='bold', pad=10)

# 6.3 Capacity concentration
ax = axes[1, 0]
bins = [0, 20, 40, 60, 80, 100]
bin_labels = ['Top 20%', '20-40%', '40-60%', '60-80%', 'Bottom 20%']
cap_by_quantile = []
road_counts = []
for i in range(len(bins)-1):
    lower = np.percentile(capacity, bins[i])
    upper = np.percentile(capacity, bins[i+1])
    mask = (capacity >= lower) & (capacity <= upper)
    cap_by_quantile.append(capacity[mask].sum())
    road_counts.append(mask.sum())

total_cap = capacity.sum()
cap_pct = [100 * c / total_cap for c in cap_by_quantile]
bars = ax.bar(range(len(bin_labels)), cap_pct, alpha=0.8, color='#27ae60', edgecolor='black', linewidth=0.7)
ax.axhline(20, color='#e74c3c', linestyle='--', linewidth=2, label='Equal share (20%)', alpha=0.7)
ax.set_xlabel('Road Capacity Quantile\n[Roads divided into 5 equal-sized groups: highest 20% to lowest 20%]', fontsize=10, fontweight='bold')
ax.set_ylabel('Percentage of Total Network Capacity\n[What % of total capacity this group provides]', fontsize=10, fontweight='bold')
ax.set_title('C. Capacity Concentration Analysis by Quantile\n[Red dashed line = equal share (20%) | Bars above line = over-represented]\n[Shows if high-capacity roads dominate total network capacity]', 
            fontsize=10, fontweight='bold', pad=10)
ax.set_xticks(range(len(bin_labels)))
ax.set_xticklabels(bin_labels, fontsize=9)
ax.legend(loc='best', framealpha=0.9, fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
for bar, val, count in zip(bars, cap_pct, road_counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
           f'{val:.1f}%\n({count:,} roads)', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

# 6.4 Cumulative capacity vs roads
ax = axes[1, 1]
sorted_indices = np.argsort(capacity)[::-1]
sorted_cap_desc = capacity[sorted_indices]
cum_cap_desc = np.cumsum(sorted_cap_desc)
cum_cap_pct_desc = cum_cap_desc / cum_cap_desc[-1] * 100
cum_roads_desc = np.arange(1, n_edges + 1)
cum_roads_pct_desc = cum_roads_desc / n_edges * 100
ax.plot(cum_roads_pct_desc, cum_cap_pct_desc, linewidth=2.5, color='#e67e22')
ax.axhline(50, color='gray', linestyle=':', alpha=0.5)
ax.axhline(80, color='gray', linestyle=':', alpha=0.5)
# Find how many roads for 50% and 80% capacity
idx_50 = np.where(cum_cap_pct_desc >= 50)[0][0]
idx_80 = np.where(cum_cap_pct_desc >= 80)[0][0]
roads_for_50 = cum_roads_pct_desc[idx_50]
roads_for_80 = cum_roads_pct_desc[idx_80]
ax.axvline(roads_for_50, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.7)
ax.axvline(roads_for_80, color='#9b59b6', linestyle='--', linewidth=2, alpha=0.7)
ax.plot(roads_for_50, 50, 'ro', markersize=10)
ax.plot(roads_for_80, 80, 'mo', markersize=10)
ax.text(roads_for_50+1, 48, f'{roads_for_50:.1f}% roads\nprovide 50% capacity', fontsize=8, fontweight='bold')
ax.text(roads_for_80+1, 78, f'{roads_for_80:.1f}% roads\nprovide 80% capacity', fontsize=8, fontweight='bold')
ax.set_xlabel('Cumulative Percentage of Roads (sorted highest to lowest)\n[X-axis: starting with highest-capacity roads and adding lower ones]', fontsize=10, fontweight='bold')
ax.set_ylabel('Cumulative Percentage of Total Network Capacity\n[Y-axis: % of total capacity accumulated]', fontsize=10, fontweight='bold')
ax.set_title('D. Capacity Accumulation Curve - Network Efficiency\n[Key insight: What % of roads provide 50% and 80% of capacity?]\n[Steep curve = capacity concentrated in few roads]', 
            fontsize=10, fontweight='bold', pad=10)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.yaxis.set_major_locator(ticker.MultipleLocator(10))

plt.tight_layout()
plt.savefig('feature1_chart6_network_stats.png', dpi=300, bbox_inches='tight')
print("✓ Saved: feature1_chart6_network_stats.png")
plt.show()
plt.close()

print(f"\nNetwork Capacity Insights:")
print(f"  • Gini coefficient: {gini:.3f} (capacity inequality)")
print(f"  • Top 20% roads provide {cap_pct[0]:.1f}% of total capacity")
print(f"  • {roads_for_50:.1f}% of roads provide 50% of network capacity")
print(f"  • {roads_for_80:.1f}% of roads provide 80% of network capacity")

################################################################################
# LOAD SCENARIOS FOR CHARTS 7-8
################################################################################
print("\n" + "=" * 80)
print("LOADING SCENARIOS FOR POLICY ANALYSIS...")
print("=" * 80)

all_scenarios = []
batch_count = 0
max_batches = min(5, len(batch_files))  # Load up to 5 batches to find scenarios with reduction
print(f"Scanning {max_batches} batch(es) to find policy scenarios...")

for batch_file in batch_files[:max_batches]:
    try:
        batch = torch.load(batch_file, weights_only=False)
        if isinstance(batch, list):
            all_scenarios.extend(batch)
            batch_count += 1
            print(f"  ✓ Loaded batch {batch_count}: {batch_file.name} ({len(batch)} scenarios)")
    except Exception as e:
        print(f"  Warning: Could not load {batch_file.name}: {e}")

n_scenarios = len(all_scenarios)
print(f"\n✓ Total loaded: {n_scenarios} scenarios from {batch_count} batch(es)")

# Find a scenario with actual capacity reduction for better visualization
print("\nSearching for scenario with capacity reduction data...")
scenario_with_reduction = None
scenarios_checked = 0

for idx, scenario in enumerate(all_scenarios):
    scenarios_checked += 1
    temp_reduction = scenario.x[:, 2].numpy()
    reduction_count = (temp_reduction > 0).sum()
    
    # Print every 50th scenario for progress
    if scenarios_checked % 50 == 0:
        print(f"  Checked {scenarios_checked}/{n_scenarios} scenarios...")
    
    if reduction_count > 0:
        scenario_with_reduction = scenario
        print(f"\n✓ FOUND! Scenario {idx} has {reduction_count:,} roads with capacity reduction ({reduction_count/n_edges*100:.1f}%)")
        print(f"  Reduction range: {temp_reduction[temp_reduction>0].min():.1f} - {temp_reduction[temp_reduction>0].max():.1f} veh/h")
        print(f"  Mean reduction: {temp_reduction[temp_reduction>0].mean():.1f} veh/h")
        # Use this scenario's data for Charts 7-8
        cap_reduction = temp_reduction
        break

if scenario_with_reduction is None:
    print(f"\n[!] WARNING: All {scenarios_checked} scenarios are BASELINE (no capacity reduction found)")
    print("    Charts 7-8 will show warning messages instead of actual policy analysis")
    print("    NOTE: This is expected if your data only contains baseline scenarios")
else:
    print(f"✓ Using scenario {idx} for detailed policy analysis (Charts 7-8)")

################################################################################
# CHART 7: CAPACITY & POLICY (SINGLE SCENARIO ANALYSIS)
################################################################################
print("\n" + "=" * 80)
print("CHART 7: Capacity & Policy Interaction")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('FEATURE 1: Capacity & Policy Interaction Analysis\nHow Road Capacity Relates to Traffic Volume and Policy Targeting\nExamining Relationships Between Capacity, Utilization, and Reduction', 
             fontsize=16, fontweight='bold', y=0.995)
plt.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.06, hspace=0.38, wspace=0.28)

# 7.1 Capacity vs Baseline Volume correlation
ax = axes[0, 0]
valid_mask = (capacity > 0) & (vol_base_case != 0)
ax.scatter(capacity[valid_mask], vol_base_case[valid_mask], alpha=0.4, s=3, c='#3498db', edgecolors='none')
if valid_mask.sum() > 1:
    corr = np.corrcoef(capacity[valid_mask], vol_base_case[valid_mask])[0, 1]
    z = np.polyfit(capacity[valid_mask], vol_base_case[valid_mask], 1)
    p = np.poly1d(z)
    cap_range = np.linspace(capacity[valid_mask].min(), capacity[valid_mask].max(), 100)
    ax.plot(cap_range, p(cap_range), "r--", linewidth=2.5, alpha=0.7, label=f'Trend (r={corr:.3f})')
    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes, 
           fontsize=10, fontweight='bold', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax.set_xlabel('Road Capacity (vehicles/hour)\n[Maximum design capacity of road]', fontsize=10, fontweight='bold')
ax.set_ylabel('Baseline Traffic Volume (vehicles/hour)\n[Actual traffic currently using road]', fontsize=10, fontweight='bold')
ax.set_title(f'A. Capacity-Volume Correlation Analysis (n={valid_mask.sum():,} roads)\n[Question: Do high-capacity roads carry proportionally more traffic?]\n[Red trend line shows overall relationship]',
            fontsize=10, fontweight='bold', pad=10)
ax.legend(loc='best', framealpha=0.9, fontsize=9)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(ticker.MultipleLocator(2000))

# 7.2 Capacity by utilization level
ax = axes[0, 1]
util_bins = [0, 0.1, 0.25, 0.5, 0.75, 1.0, 10.0]
util_labels = ['0-10%', '10-25%', '25-50%', '50-75%', '75-100%', '>100%']
cap_by_util = []
for i in range(len(util_bins)-1):
    mask = (utilization >= util_bins[i]) & (utilization < util_bins[i+1])
    cap_by_util.append(capacity[mask])

bp = ax.boxplot([c for c in cap_by_util if len(c) > 0], 
               tick_labels=[util_labels[i] for i in range(len(cap_by_util)) if len(cap_by_util[i]) > 0],
               patch_artist=True, showfliers=False, widths=0.6)
for patch in bp['boxes']:
    patch.set_facecolor('#e67e22')
    patch.set_alpha(0.7)
for median in bp['medians']:
    median.set_color('#e74c3c')
    median.set_linewidth(2.5)
ax.set_xlabel('Utilization Category\n[Groups: 0-10%, 10-25%, 25-50%, 50-75%, 75-100%, >100%]', fontsize=10, fontweight='bold')
ax.set_ylabel('Road Capacity Distribution (veh/h)\n[Box plot shows capacity range in each group]', fontsize=10, fontweight='bold')
ax.set_title('B. Capacity Distribution by Utilization Level\n[Do heavily-used roads have different capacity than lightly-used roads?]\n[Red line = median | Box = middle 50% (IQR)]', 
            fontsize=10, fontweight='bold', pad=10)
ax.grid(True, alpha=0.3, axis='y')
ax.yaxis.set_major_locator(ticker.MultipleLocator(1000))

# 7.3 Capacity reduction impact (or baseline capacity distribution if no reduction)
ax = axes[1, 0]
reduction_mask = cap_reduction > 0
targeted_cap = capacity[reduction_mask]
untargeted_cap = capacity[~reduction_mask]
if len(targeted_cap) > 0 and len(untargeted_cap) > 0:
    bp = ax.boxplot([untargeted_cap, targeted_cap], 
                   tick_labels=['Not Targeted', 'Targeted for\nReduction'],
                   patch_artist=True, showfliers=False, widths=0.6)
    bp['boxes'][0].set_facecolor('#27ae60')
    bp['boxes'][1].set_facecolor('#e74c3c')
    for box in bp['boxes']:
        box.set_alpha(0.8)
        box.set_edgecolor('black')
        box.set_linewidth(1.5)
    for median in bp['medians']:
        median.set_color('white')
        median.set_linewidth=3
    # Add statistical comparison
    mean_untargeted = untargeted_cap.mean()
    mean_targeted = targeted_cap.mean()
    ax.text(0.5, 0.97, f'Mean: Not Targeted={mean_untargeted:.0f} | Targeted={mean_targeted:.0f} veh/h',
           transform=ax.transAxes, ha='center', fontsize=9, 
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.set_ylabel('Road Capacity (vehicles/hour)\n[Baseline capacity before policy intervention]', fontsize=10, fontweight='bold')
    ax.set_title(f'C. Capacity Comparison: Targeted vs Untargeted Roads\n[Green = not targeted ({(~reduction_mask).sum():,} roads) | Red = targeted for reduction ({reduction_mask.sum():,} roads)]\n[Do policies target high-capacity or low-capacity roads?]', 
                fontsize=10, fontweight='bold', pad=10)
else:
    # BASELINE ANALYSIS: Show capacity distribution by bins
    cap_bins = [0, 500, 1000, 2000, 3000, 5000, capacity.max()+1]
    bin_labels = ['0-500', '500-1k', '1k-2k', '2k-3k', '3k-5k', '5k+']
    cap_by_bin = []
    for i in range(len(cap_bins)-1):
        mask = (capacity >= cap_bins[i]) & (capacity < cap_bins[i+1])
        cap_by_bin.append(capacity[mask])
    
    # Filter out empty bins
    cap_by_bin_filtered = [c for c in cap_by_bin if len(c) > 0]
    labels_filtered = [bin_labels[i] for i in range(len(cap_by_bin)) if len(cap_by_bin[i]) > 0]
    
    bp = ax.boxplot(cap_by_bin_filtered, tick_labels=labels_filtered,
                   patch_artist=True, showfliers=False, widths=0.6)
    colors = ['#3498db', '#27ae60', '#f39c12', '#e74c3c', '#9b59b6', '#e67e22']
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.2)
    for median in bp['medians']:
        median.set_color('white')
        median.set_linewidth(3)
    
    ax.set_ylabel('Road Capacity Distribution (veh/h)\n[Box = middle 50% | Line = median]', fontsize=10, fontweight='bold')
    ax.set_title('C. Baseline Capacity Distribution by Category\n[Showing capacity spread across different road capacity ranges]\n[BASELINE SCENARIO - No policy interventions applied]', 
                fontsize=10, fontweight='bold', pad=10)
    ax.set_xlabel('Capacity Category (vehicles/hour)', fontsize=10, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.yaxis.set_major_locator(ticker.MultipleLocator(500))

# 7.4 Capacity by highway type with reduction overlay
ax = axes[1, 1]
mean_cap_by_type = [capacity[highway == ht].mean() for ht in unique_types]
mean_reduction_by_type = [cap_reduction[highway == ht].mean() for ht in unique_types]
ax2 = ax.twinx()
bars = ax.bar(unique_types, mean_cap_by_type, alpha=0.7, color='#3498db', 
             edgecolor='black', linewidth=0.7, label='Mean Capacity')
line = ax2.plot(unique_types, mean_reduction_by_type, 'o-', color='#e74c3c', 
               linewidth=2.5, markersize=8, label='Mean Reduction', alpha=0.8)
ax.set_xlabel('Road Type\n[Comparing capacity and reduction across road categories]', fontsize=10, fontweight='bold')
ax.set_ylabel('Mean Capacity (veh/h)\n[Blue bars - left axis]', fontsize=10, color='#3498db', fontweight='bold')
ax2.set_ylabel('Mean Capacity Reduction (veh/h)\n[Red line - right axis]', fontsize=10, color='#e74c3c', fontweight='bold')
ax.set_title('D. Capacity & Reduction Patterns by Road Type\n[Are high-capacity road types also heavily reduced?]\n[Dual axis: bars = average capacity | line = average reduction when targeted]', 
            fontsize=10, fontweight='bold', pad=10)
ax.set_xticks(unique_types)
ax.set_xticklabels([f'{int(ht)}\n{highway_type_names.get(int(ht), "?")[:4]}' for ht in unique_types], fontsize=8)
ax.tick_params(axis='y', labelcolor='#3498db')
ax2.tick_params(axis='y', labelcolor='#e74c3c')
ax.legend(loc='upper left', framealpha=0.9, fontsize=9)
ax2.legend(loc='upper right', framealpha=0.9, fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('feature1_chart7_policy_interaction.png', dpi=300, bbox_inches='tight')
print("✓ Saved: feature1_chart7_policy_interaction.png")
plt.show()
plt.close()

################################################################################
# CHART 8: CAPACITY REDUCTION TARGETING
################################################################################
print("\n" + "=" * 80)
print("CHART 8: Capacity Reduction Targeting Analysis")
print("=" * 80)

# Diagnostic output
cap_red_nonzero = cap_reduction[cap_reduction > 0]
print(f"Roads with capacity reduction: {len(cap_red_nonzero):,} ({len(cap_red_nonzero)/n_edges*100:.2f}%)")
if len(cap_red_nonzero) > 0:
    print(f"Capacity reduction range: {cap_red_nonzero.min():.1f} - {cap_red_nonzero.max():.1f} veh/h")
    print(f"Mean reduction: {cap_red_nonzero.mean():.1f} veh/h")
else:
    print("[!] WARNING: No roads have capacity reduction in this scenario!")

fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('FEATURE 1: Capacity Reduction Targeting Analysis\nUnderstanding Which Roads Get Capacity Reduced and By How Much\nAnalyzing Policy Targeting Patterns and Utilization Impact', 
             fontsize=16, fontweight='bold', y=0.995)
plt.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.06, hspace=0.40, wspace=0.28)

# 8.1 Scatter: Capacity vs Reduction (or baseline histogram)
ax = axes[0, 0]
reduction_mask = cap_reduction > 0
if reduction_mask.sum() > 0:
    max_reduction = cap_reduction[reduction_mask].max()
    # Separate scatter for visual clarity
    ax.scatter(capacity[~reduction_mask], cap_reduction[~reduction_mask], 
              c='lightgray', s=1, alpha=0.2, label=f'No reduction ({(~reduction_mask).sum():,} roads)')
    ax.scatter(capacity[reduction_mask], cap_reduction[reduction_mask], 
              c='purple', s=5, alpha=0.7, label=f'With reduction ({reduction_mask.sum():,} roads)')
    ax.set_ylim(-max_reduction*0.05, max_reduction*1.1)
    if reduction_mask.sum() > 1:
        with np.errstate(invalid='ignore'):
            corr = np.corrcoef(capacity[reduction_mask], cap_reduction[reduction_mask])[0, 1]
            if not np.isnan(corr):
                ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes, 
                       fontsize=10, fontweight='bold', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.set_xlabel('Road Capacity (vehicles/hour)\n[X-axis: original baseline capacity before policy]', fontsize=10, fontweight='bold')
    ax.set_ylabel('Capacity Reduction (vehicles/hour)\n[Y-axis: amount of capacity removed]', fontsize=10, fontweight='bold')
    ax.set_title(f'A. Capacity Reduction Scatter Plot\n[Purple dots = roads with reduction | Gray = no reduction]\n[Pattern reveals if policy targets high-capacity or low-capacity roads]', 
                fontsize=10, fontweight='bold', pad=10)
    ax.legend(loc='best', framealpha=0.9, fontsize=9)
else:
    # BASELINE ANALYSIS: Show capacity histogram with focus on distribution
    cap_valid = capacity[capacity > 0]
    ax.hist(cap_valid, bins=50, alpha=0.7, color='#3498db', edgecolor='black', linewidth=0.5)
    ax.axvline(np.median(cap_valid), color='#e74c3c', linestyle='--', linewidth=2.5, 
              label=f'Median = {np.median(cap_valid):.0f} veh/h', alpha=0.8)
    ax.axvline(cap_valid.mean(), color='#27ae60', linestyle='--', linewidth=2.5,
              label=f'Mean = {cap_valid.mean():.0f} veh/h', alpha=0.8)
    Q1, Q3 = np.percentile(cap_valid, [25, 75])
    ax.axvspan(Q1, Q3, alpha=0.2, color='yellow', label=f'IQR: {Q1:.0f}-{Q3:.0f}')
    ax.set_xlabel('Road Capacity (vehicles/hour)\n[Distribution of baseline capacity across all roads]', fontsize=10, fontweight='bold')
    ax.set_ylabel('Number of Roads\n[Frequency count]', fontsize=10, fontweight='bold')
    ax.set_title('A. Baseline Capacity Distribution\n[Histogram showing how road capacities are distributed]\n[BASELINE SCENARIO - Most roads have low-to-medium capacity]', 
                fontsize=10, fontweight='bold', pad=10)
    ax.legend(loc='best', framealpha=0.9, fontsize=9)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(ticker.MultipleLocator(2000))

# 8.2 Targeting rates by capacity bins (or baseline road count distribution)
ax = axes[0, 1]
cap_bins_for_targeting = [0, 1000, 2000, 3000, 5000, capacity.max()+1]
targeting_labels = ['0-1k', '1k-2k', '2k-3k', '3k-5k', '5k+']
targeting_rates = []
n_targeted_list = []
roads_per_bin = []
for i in range(len(cap_bins_for_targeting)-1):
    mask = (capacity >= cap_bins_for_targeting[i]) & (capacity < cap_bins_for_targeting[i+1])
    targeted = (cap_reduction[mask] > 0).sum()
    total = mask.sum()
    rate = 100 * targeted / total if total > 0 else 0
    targeting_rates.append(rate)
    n_targeted_list.append(targeted)
    roads_per_bin.append(total)

if max(targeting_rates) > 0:
    colors = ['#e74c3c' if r < 5 else '#e67e22' if r < 20 else '#27ae60' for r in targeting_rates]
    bars = ax.bar(range(len(targeting_labels)), targeting_rates, alpha=0.85, color=colors, 
                 edgecolor='black', linewidth=1.2)
    ax.set_ylim(0, max(targeting_rates) * 1.2)
    # Add value labels on bars
    for bar, val, n_targeted, total_in_bin in zip(bars, targeting_rates, n_targeted_list, 
                                                   [mask.sum() for i in range(len(cap_bins_for_targeting)-1) 
                                                    for mask in [(capacity >= cap_bins_for_targeting[i]) & 
                                                                (capacity < cap_bins_for_targeting[i+1])]]):
        if val > 0.5:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(targeting_rates)*0.02, 
                   f'{val:.1f}%\n{n_targeted:,}/{total_in_bin:,}', 
                   ha='center', va='bottom', fontsize=7.5, fontweight='bold')
else:
    # BASELINE ANALYSIS: Show road count distribution by capacity category
    colors = ['#3498db', '#27ae60', '#f39c12', '#e67e22', '#e74c3c']
    bars = ax.bar(range(len(targeting_labels)), roads_per_bin, alpha=0.8, color=colors,
                 edgecolor='black', linewidth=1.2)
    # Add percentage labels
    total_roads = sum(roads_per_bin)
    for bar, count in zip(bars, roads_per_bin):
        pct = (count / total_roads) * 100
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(roads_per_bin)*0.02,
                   f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_xlabel('Capacity Category (vehicles/hour)\n[Five bins from small roads (0-1K) to large highways (5K+)]', fontsize=10, fontweight='bold')
    ax.set_ylabel('Number of Roads\n[Count of roads in each capacity category]', fontsize=10, fontweight='bold')
    ax.set_title('B. Road Count Distribution by Capacity Category\n[BASELINE SCENARIO - Shows how roads are distributed across capacity ranges]\n[Most roads fall in lower capacity categories]', 
                fontsize=10, fontweight='bold', pad=10)
ax.set_xticks(range(len(targeting_labels)))
ax.set_xticklabels(targeting_labels, fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# 8.3 Utilization: before vs after reduction (or baseline capacity-volume relationship)
ax = axes[1, 0]
with np.errstate(divide='ignore', invalid='ignore'):
    util_before = vol_base_case / capacity
    cap_after = capacity - cap_reduction
    util_after = vol_base_case / cap_after
    util_before = np.nan_to_num(util_before, nan=0, posinf=0, neginf=0)
    util_after = np.nan_to_num(util_after, nan=0, posinf=0, neginf=0)

util_before_nonzero = util_before[(util_before > 0) & (util_before < 5)]
util_after_nonzero = util_after[(util_after > 0) & (util_after < 5) & (cap_reduction > 0)]

if len(util_after_nonzero) > 0:
    bp = ax.boxplot([util_before_nonzero, util_after_nonzero], 
                   tick_labels=['Before\nReduction', 'After\nReduction'],
                   patch_artist=True, showfliers=False, widths=0.5)
    bp['boxes'][0].set_facecolor('#3498db')
    bp['boxes'][1].set_facecolor('#e74c3c')
    for box in bp['boxes']:
        box.set_alpha(0.7)
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2.5)
    ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='100% utilization', alpha=0.7)
    ax.legend(loc='best', framealpha=0.9, fontsize=9)
    ax.set_ylabel('Utilization Ratio (Volume / Capacity)\n[0-1 = under-capacity | 1.0 = at capacity | >1 = over-capacity]', fontsize=10, fontweight='bold')
    ax.set_title('C. Utilization Impact: Before vs After Capacity Reduction\n[Blue = before reduction | Red = after reduction | Red dashed = 100% utilization]\n[Shows if reducing capacity pushes roads toward congestion]', 
                fontsize=10, fontweight='bold', pad=10)
else:
    # BASELINE ANALYSIS: Scatter plot of capacity vs volume
    valid_mask = (capacity > 0) & (vol_base_case != 0)
    ax.scatter(capacity[valid_mask], np.abs(vol_base_case[valid_mask]), alpha=0.4, s=3, c='#3498db')
    # Add diagonal reference lines
    max_val = min(capacity[valid_mask].max(), 10000)  # Limit for visibility
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='100% utilization', alpha=0.7)
    ax.plot([0, max_val], [0, max_val*0.5], 'g--', linewidth=1.5, label='50% utilization', alpha=0.6)
    ax.plot([0, max_val], [0, max_val*0.25], 'y--', linewidth=1.5, label='25% utilization', alpha=0.5)
    ax.set_xlabel('Road Capacity (vehicles/hour)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Baseline Traffic Volume (vehicles/hour)', fontsize=10, fontweight='bold')
    ax.set_title('C. Baseline Capacity vs Volume Relationship\n[BASELINE SCENARIO - Most points below 50% line = under-utilized]\n[Points above red line would indicate over-capacity roads]', 
                fontsize=10, fontweight='bold', pad=10)
    ax.legend(loc='best', framealpha=0.9, fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
ax.yaxis.set_major_locator(ticker.MultipleLocator(500 if reduction_mask.sum() == 0 else 0.1))

# 8.4 Mean reduction by capacity bins
ax = axes[1, 1]
reduction_by_bin = []
counts_by_bin = []
for i in range(len(cap_bins_for_targeting)-1):
    mask = (capacity >= cap_bins_for_targeting[i]) & (capacity < cap_bins_for_targeting[i+1]) & (cap_reduction > 0)
    mean_red = cap_reduction[mask].mean() if mask.sum() > 0 else 0
    reduction_by_bin.append(mean_red)
    counts_by_bin.append(mask.sum())

if max(reduction_by_bin) > 0:
    # Create gradient colors based on reduction intensity
    max_reduction = max(reduction_by_bin)
    colors = ['#%02x%02x%02x' % (int(192 - (r/max_reduction)*100), int(57 - (r/max_reduction)*20), int(43)) 
              if r > 0 else '#cccccc' for r in reduction_by_bin]
    bars = ax.bar(range(len(targeting_labels)), reduction_by_bin, alpha=0.85, color=colors, 
                 edgecolor='black', linewidth=1.2)
    ax.set_ylim(0, max(reduction_by_bin) * 1.2)
    # Add value labels with percentage of capacity
    for idx, (bar, val, count) in enumerate(zip(bars, reduction_by_bin, counts_by_bin)):
        if val > 0:
            # Calculate what % of capacity is being reduced
            mask = (capacity >= cap_bins_for_targeting[idx]) & (capacity < cap_bins_for_targeting[idx+1]) & (cap_reduction > 0)
            if mask.sum() > 0:
                avg_cap_in_bin = capacity[mask].mean()
                reduction_pct = (val / avg_cap_in_bin) * 100 if avg_cap_in_bin > 0 else 0
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(reduction_by_bin)*0.03, 
                       f'{val:.0f} veh/h\n({reduction_pct:.0f}% of capacity)\n{count:,} roads', 
                       ha='center', va='bottom', fontsize=7, fontweight='bold')
    ax.set_xlabel('Capacity Category (vehicles/hour)\n[Five capacity bins from low to high]', fontsize=10, fontweight='bold')
    ax.set_ylabel('Mean Capacity Reduction (veh/h)\n[Average reduction amount when road is targeted]', fontsize=10, fontweight='bold')
    ax.set_title('D. Reduction Intensity by Capacity Category\n[When a road IS targeted, how aggressive is the reduction?]\n[Label shows: mean reduction amount, count of targeted roads]', 
                fontsize=10, fontweight='bold', pad=10)
else:
    # BASELINE ANALYSIS: Show average capacity per bin
    avg_cap_per_bin = []
    for i in range(len(cap_bins_for_targeting)-1):
        mask = (capacity >= cap_bins_for_targeting[i]) & (capacity < cap_bins_for_targeting[i+1])
        avg_cap_per_bin.append(capacity[mask].mean() if mask.sum() > 0 else 0)
    
    colors = ['#3498db', '#27ae60', '#f39c12', '#e67e22', '#e74c3c']
    bars = ax.bar(range(len(targeting_labels)), avg_cap_per_bin, alpha=0.8, color=colors,
                 edgecolor='black', linewidth=1.2)
    # Add value labels
    for bar, val, count in zip(bars, avg_cap_per_bin, roads_per_bin):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(avg_cap_per_bin)*0.02,
                   f'{val:.0f} veh/h\n({count:,} roads)', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    ax.set_xlabel('Capacity Category (vehicles/hour)\n[Five capacity bins from low to high]', fontsize=10, fontweight='bold')
    ax.set_ylabel('Average Capacity (veh/h)\n[Mean capacity value in each category]', fontsize=10, fontweight='bold')
    ax.set_title('D. Average Capacity by Category\n[BASELINE SCENARIO - Shows typical capacity for each road category]\n[Higher bins have higher average capacity as expected]', 
                fontsize=10, fontweight='bold', pad=10)
ax.set_xticks(range(len(targeting_labels)))
ax.set_xticklabels(targeting_labels, fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('feature1_chart8_reduction_targeting.png', dpi=300, bbox_inches='tight')
print("✓ Saved: feature1_chart8_reduction_targeting.png")
plt.show()
plt.close()

print("\n" + "=" * 80)
print("✓✓✓ PART 2 COMPLETE - CHARTS 5-8 GENERATED ✓✓✓")
print("=" * 80)
print("\nGenerated files:")
print("  5. feature1_chart5_outliers.png")
print("  6. feature1_chart6_network_stats.png")
print("  7. feature1_chart7_policy_interaction.png")
print("  8. feature1_chart8_reduction_targeting.png")
if scenario_with_reduction is None:
    print("\n[i] NOTE: Charts 7-8 show BASELINE ANALYSIS (no policy scenarios found in data)")
    print("    - Charts display useful baseline insights instead of policy comparisons")
    print("    - This is expected if your dataset contains only baseline scenarios")
print("\nNext: Run feature1_part3_charts9to12.py for final Charts 9-12")
print("=" * 80)
