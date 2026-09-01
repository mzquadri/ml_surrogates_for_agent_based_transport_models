"""
FEATURE 1 ANALYSIS - PART 3: ROAD CAPACITY (Charts 9-12)
=======================================================
Charts 9-12: Advanced Analysis & Summary

Run after feature1_part2_charts5to8.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.ticker as ticker
from scipy import stats

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
print("#" + "  FEATURE 1 - PART 3: ROAD CAPACITY (Charts 9-12)".center(78) + "#")
print("#" + "  Advanced Analysis & Summary".center(78) + "#")
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
vol_base_case = first_scenario.x[:, 0].numpy()  # F0: Volume
capacity = first_scenario.x[:, 1].numpy()        # F1: Capacity
cap_reduction = first_scenario.x[:, 2].numpy()   # F2: Capacity Reduction
free_speed = first_scenario.x[:, 3].numpy()      # F3: Free Speed
highway = first_scenario.x[:, 4].numpy()         # F4: Highway Type
length = first_scenario.x[:, 5].numpy()          # F5: Length

n_edges = len(capacity)
unique_types = np.unique(highway)
print(f"✓ Loaded {n_edges:,} edges with 6 features")

# Highway type decoder
highway_type_names = {
    0: 'Motorway', 1: 'Trunk', 2: 'Primary', 3: 'Secondary', 4: 'Tertiary',
    5: 'Residential', 6: 'Service', 7: 'Unclassified', 8: 'Living Street', 9: 'Other'
}

# Calculate derived metrics
with np.errstate(divide='ignore', invalid='ignore'):
    utilization = np.abs(vol_base_case) / capacity
    utilization = np.nan_to_num(utilization, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Travel time estimation (length / speed)
    travel_time = length / (free_speed * 1000 / 3600)  # Convert to hours
    travel_time = np.nan_to_num(travel_time, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Flow efficiency (volume per unit capacity per unit length)
    flow_efficiency = np.abs(vol_base_case) / (capacity * length)
    flow_efficiency = np.nan_to_num(flow_efficiency, nan=0.0, posinf=0.0, neginf=0.0)

################################################################################
# CHART 9: FEATURE CORRELATION & RELATIONSHIPS
################################################################################
print("\n" + "=" * 80)
print("CHART 9: Feature Correlation & Relationships")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('FEATURE 1: Multi-Feature Correlation & Relationship Analysis\nHow Road Capacity Relates to Other Network Features\nExamining Volume, Speed, Length, and Highway Type Interactions', 
             fontsize=16, fontweight='bold', y=0.995)
plt.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.06, hspace=0.40, wspace=0.28)

# 9.1 Correlation heatmap
ax = axes[0, 0]
features_for_corr = np.column_stack([
    vol_base_case, capacity, free_speed, length, utilization
])
feature_names = ['Volume\n(F0)', 'Capacity\n(F1)', 'Free Speed\n(F3)', 
                'Length\n(F5)', 'Utilization\n(derived)']

# Remove any infinite or nan values
valid_mask = np.all(np.isfinite(features_for_corr), axis=1)
corr_matrix = np.corrcoef(features_for_corr[valid_mask].T)

im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(len(feature_names)))
ax.set_yticks(range(len(feature_names)))
ax.set_xticklabels(feature_names, fontsize=9, rotation=45, ha='right')
ax.set_yticklabels(feature_names, fontsize=9)

# Add correlation values
for i in range(len(feature_names)):
    for j in range(len(feature_names)):
        text_color = 'white' if abs(corr_matrix[i, j]) > 0.5 else 'black'
        ax.text(j, i, f'{corr_matrix[i, j]:.2f}', 
               ha='center', va='center', color=text_color, fontweight='bold', fontsize=9)

plt.colorbar(im, ax=ax, label='Correlation Coefficient\n[-1 = perfect negative | 0 = no correlation | +1 = perfect positive]')
ax.set_title('A. Feature Correlation Heatmap\n[Red = negative correlation | Blue = positive correlation]\n[Key: Which features move together vs independently?]', 
            fontsize=10, fontweight='bold', pad=10)

# 9.2 Capacity vs Free Speed relationship
ax = axes[0, 1]
valid_mask = (capacity > 0) & (free_speed > 0)
ax.scatter(free_speed[valid_mask], capacity[valid_mask], alpha=0.4, s=3, c='#3498db', edgecolors='none')
if valid_mask.sum() > 100:
    corr = np.corrcoef(free_speed[valid_mask], capacity[valid_mask])[0, 1]
    z = np.polyfit(free_speed[valid_mask], capacity[valid_mask], 1)
    p = np.poly1d(z)
    speed_range = np.linspace(free_speed[valid_mask].min(), free_speed[valid_mask].max(), 100)
    ax.plot(speed_range, p(speed_range), "r--", linewidth=2.5, alpha=0.7, label=f'Trend (r={corr:.3f})')
    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes, 
           fontsize=10, fontweight='bold', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
           verticalalignment='top')
ax.set_xlabel('Free Flow Speed (km/h)\n[Design speed limit of road]', fontsize=10, fontweight='bold')
ax.set_ylabel('Road Capacity (vehicles/hour)\n[Maximum vehicle capacity]', fontsize=10, fontweight='bold')
ax.set_title(f'B. Capacity vs Free Speed Analysis (n={valid_mask.sum():,} roads)\n[Question: Do faster roads have higher capacity?]\n[Expectation: Positive correlation (highways = fast + high capacity)]', 
            fontsize=10, fontweight='bold', pad=10)
if valid_mask.sum() > 100:
    ax.legend(loc='best', framealpha=0.9, fontsize=9)
ax.grid(True, alpha=0.3)

# 9.3 Capacity vs Length relationship by highway type
ax = axes[1, 0]
# Select top 4 highway types by count
type_counts = [(ht, (highway == ht).sum()) for ht in unique_types]
type_counts.sort(key=lambda x: x[1], reverse=True)
top_4_types = [t[0] for t in type_counts[:4]]
colors_map = {top_4_types[0]: '#e74c3c', top_4_types[1]: '#3498db', 
             top_4_types[2]: '#27ae60', top_4_types[3]: '#f39c12'}

for ht in top_4_types:
    mask = (highway == ht) & (capacity > 0) & (length > 0) & (length < np.percentile(length[length > 0], 95))
    if mask.sum() > 50:
        ax.scatter(length[mask], capacity[mask], alpha=0.5, s=5, 
                  c=colors_map[ht], label=f'{highway_type_names.get(int(ht), "?")[:10]} (n={mask.sum():,})',
                  edgecolors='none')

ax.set_xlabel('Road Length (meters)\n[Physical length of road segment]', fontsize=10, fontweight='bold')
ax.set_ylabel('Road Capacity (vehicles/hour)\n[Maximum vehicle capacity]', fontsize=10, fontweight='bold')
ax.set_title('C. Capacity vs Length by Highway Type (Top 4 types)\n[Do longer roads have different capacity? Does it vary by road type?]\n[Each color = different road type | Points = individual road segments]', 
            fontsize=10, fontweight='bold', pad=10)
ax.legend(loc='best', framealpha=0.9, fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)
ax.set_xlim(left=0)

# 9.4 Capacity distribution by utilization category
ax = axes[1, 1]
util_categories = ['0-25%\n(Under-utilized)', '25-50%\n(Light)', 
                  '50-75%\n(Moderate)', '75-100%\n(Heavy)', '>100%\n(Over-capacity)']
util_bins = [0, 0.25, 0.5, 0.75, 1.0, 100]
cap_by_util_cat = []

for i in range(len(util_bins)-1):
    mask = (utilization >= util_bins[i]) & (utilization < util_bins[i+1]) & (capacity > 0)
    cap_by_util_cat.append(capacity[mask])

# Filter out empty categories
cap_filtered = [c for c in cap_by_util_cat if len(c) > 0]
labels_filtered = [util_categories[i] for i in range(len(cap_by_util_cat)) if len(cap_by_util_cat[i]) > 0]

if len(cap_filtered) > 0:
    bp = ax.boxplot(cap_filtered, tick_labels=labels_filtered,
                   patch_artist=True, showfliers=False, widths=0.6)
    colors = ['#27ae60', '#3498db', '#f39c12', '#e67e22', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.2)
    for median in bp['medians']:
        median.set_color('white')
        median.set_linewidth(3)

ax.set_xlabel('Utilization Category\n[Groups based on % of capacity currently used]', fontsize=10, fontweight='bold')
ax.set_ylabel('Road Capacity Distribution (veh/h)\n[Box = middle 50% | White line = median]', fontsize=10, fontweight='bold')
ax.set_title('D. Capacity by Utilization Level\n[Do heavily-utilized roads have systematically different capacity?]\n[Color code: Green (light) to Red (heavy/over-capacity)]', 
            fontsize=10, fontweight='bold', pad=10)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('feature1_chart9_correlations.png', dpi=300, bbox_inches='tight')
print("✓ Saved: feature1_chart9_correlations.png")
plt.show()
plt.close()

################################################################################
# CHART 10: CAPACITY EFFICIENCY METRICS
################################################################################
print("\n" + "=" * 80)
print("CHART 10: Capacity Efficiency Metrics")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('FEATURE 1: Road Capacity Efficiency & Performance Analysis\nEvaluating How Effectively Road Capacity is Utilized\nFlow Efficiency, Travel Time, and Network Performance Metrics', 
             fontsize=16, fontweight='bold', y=0.995)
plt.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.06, hspace=0.40, wspace=0.28)

# 10.1 Flow efficiency by highway type
ax = axes[0, 0]
flow_eff_by_type = []
type_labels = []
for ht in unique_types:
    mask = (highway == ht) & (flow_efficiency > 0) & np.isfinite(flow_efficiency)
    if mask.sum() > 10:
        flow_eff_by_type.append(flow_efficiency[mask])
        type_labels.append(f'{int(ht)}\n{highway_type_names.get(int(ht), "?")[:6]}')

if len(flow_eff_by_type) > 0:
    bp = ax.boxplot(flow_eff_by_type, tick_labels=type_labels,
                   patch_artist=True, showfliers=False, widths=0.6)
    for patch in bp['boxes']:
        patch.set_facecolor('#3498db')
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.2)
    for median in bp['medians']:
        median.set_color('#e74c3c')
        median.set_linewidth(2.5)

ax.set_xlabel('Road Type\n[OpenStreetMap classification]', fontsize=10, fontweight='bold')
ax.set_ylabel('Flow Efficiency (volume / capacity / length)\n[Higher = more efficient use of capacity per meter]', fontsize=10, fontweight='bold')
ax.set_title('A. Flow Efficiency Distribution by Road Type\n[Measures traffic flow per unit capacity per unit length]\n[Red line = median | Box = middle 50%]', 
            fontsize=10, fontweight='bold', pad=10)
ax.grid(True, alpha=0.3, axis='y')

# 10.2 Capacity utilization histogram
ax = axes[0, 1]
util_valid = utilization[(utilization > 0) & (utilization < 2)]  # Focus on 0-200%
ax.hist(util_valid, bins=50, alpha=0.7, color='#27ae60', edgecolor='black', linewidth=0.5)
ax.axvline(1.0, color='#e74c3c', linestyle='--', linewidth=3, label='100% utilization (at capacity)', alpha=0.8)
ax.axvline(np.median(util_valid), color='#3498db', linestyle='--', linewidth=2.5, 
          label=f'Median = {np.median(util_valid):.2f}', alpha=0.8)

# Add percentage statistics
pct_under_50 = (utilization < 0.5).sum() / len(utilization) * 100
pct_over_100 = (utilization > 1.0).sum() / len(utilization) * 100
ax.text(0.98, 0.97, f'Under 50%: {pct_under_50:.1f}%\nOver 100%: {pct_over_100:.1f}%', 
       transform=ax.transAxes, fontsize=9, fontweight='bold',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
       verticalalignment='top', horizontalalignment='right')

ax.set_xlabel('Utilization Ratio (Volume / Capacity)\n[0.0 = empty | 1.0 = at capacity | >1.0 = over-capacity]', fontsize=10, fontweight='bold')
ax.set_ylabel('Number of Roads\n[Frequency count]', fontsize=10, fontweight='bold')
ax.set_title('B. Network Capacity Utilization Distribution\n[Shows how efficiently the network capacity is being used]\n[Red dashed line = theoretical maximum (100% utilization)]', 
            fontsize=10, fontweight='bold', pad=10)
ax.legend(loc='best', framealpha=0.9, fontsize=9)
ax.grid(True, alpha=0.3)

# 10.3 Capacity vs Volume with utilization zones
ax = axes[1, 0]
valid_mask = (capacity > 0) & (vol_base_case != 0)
cap_subset = capacity[valid_mask]
vol_subset = np.abs(vol_base_case[valid_mask])
util_subset = utilization[valid_mask]

# Color by utilization level
colors_scatter = np.where(util_subset < 0.5, '#27ae60',
                 np.where(util_subset < 0.75, '#3498db',
                 np.where(util_subset < 1.0, '#f39c12', '#e74c3c')))

ax.scatter(cap_subset, vol_subset, c=colors_scatter, alpha=0.4, s=3, edgecolors='none')

# Add reference lines
max_val = min(cap_subset.max(), vol_subset.max())
ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='100% utilization', alpha=0.7)
ax.plot([0, max_val], [0, max_val*0.5], 'b--', linewidth=1.5, label='50% utilization', alpha=0.6)

# Add legend for colors
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#27ae60', alpha=0.7, label='0-50% (Under-utilized)'),
    Patch(facecolor='#3498db', alpha=0.7, label='50-75% (Moderate)'),
    Patch(facecolor='#f39c12', alpha=0.7, label='75-100% (Heavy)'),
    Patch(facecolor='#e74c3c', alpha=0.7, label='>100% (Over-capacity)')
]
ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9, fontsize=8)

ax.set_xlabel('Road Capacity (vehicles/hour)\n[Maximum design capacity]', fontsize=10, fontweight='bold')
ax.set_ylabel('Baseline Traffic Volume (vehicles/hour)\n[Actual current traffic]', fontsize=10, fontweight='bold')
ax.set_title('C. Capacity-Volume Relationship with Utilization Zones\n[Points colored by utilization level: Green (light) to Red (over-capacity)]\n[Most points below red line = network has spare capacity]', 
            fontsize=10, fontweight='bold', pad=10)
ax.grid(True, alpha=0.3)

# 10.4 Efficiency metrics summary table
ax = axes[1, 1]
ax.axis('off')

# Calculate various efficiency metrics
total_capacity = capacity.sum()
total_volume = np.abs(vol_base_case).sum()
avg_utilization = np.mean(utilization[np.isfinite(utilization) & (utilization < 10)])
roads_over_capacity = (utilization > 1.0).sum()
roads_under_50 = (utilization < 0.5).sum()
spare_capacity = total_capacity - total_volume
efficiency_ratio = total_volume / total_capacity if total_capacity > 0 else 0

# Capacity concentration
sorted_cap = np.sort(capacity[capacity > 0])[::-1]
cum_cap = np.cumsum(sorted_cap)
idx_50 = np.where(cum_cap >= total_capacity * 0.5)[0][0]
pct_roads_for_50 = (idx_50 / len(capacity)) * 100

efficiency_data = [
    ['NETWORK CAPACITY EFFICIENCY', '', ''],
    ['', '', ''],
    ['Total Network Capacity', f'{total_capacity:,.0f}', 'veh/h'],
    ['Total Baseline Volume', f'{total_volume:,.0f}', 'veh/h'],
    ['Spare Capacity', f'{spare_capacity:,.0f}', 'veh/h'],
    ['', '', ''],
    ['Network Efficiency Ratio', f'{efficiency_ratio:.3f}', f'({efficiency_ratio*100:.1f}%)'],
    ['Average Utilization', f'{avg_utilization:.3f}', f'({avg_utilization*100:.1f}%)'],
    ['', '', ''],
    ['Under-utilized (<50%)', f'{roads_under_50:,}', f'({roads_under_50/n_edges*100:.1f}%)'],
    ['Over-capacity (>100%)', f'{roads_over_capacity:,}', f'({roads_over_capacity/n_edges*100:.1f}%)'],
    ['', '', ''],
    ['Roads for 50% Capacity', f'{pct_roads_for_50:.1f}%', 'of network'],
    ['Mean Capacity per Road', f'{capacity.mean():.0f}', 'veh/h'],
    ['Mean Volume per Road', f'{np.abs(vol_base_case).mean():.0f}', 'veh/h'],
    ['', '', ''],
    ['INTERPRETATION', '', ''],
    ['Low network efficiency', '<50%', 'Under-utilized'],
    ['Optimal efficiency', '70-90%', 'Balanced'],
    ['High congestion risk', '>95%', 'Over-utilized'],
]

table = ax.table(cellText=efficiency_data, cellLoc='left', loc='center',
                colWidths=[0.5, 0.25, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.0)

# Highlight header rows
for i in [0, 1, 5, 8, 11, 15]:
    for j in range(3):
        table[(i, j)].set_facecolor('#3498db')
        table[(i, j)].set_text_props(weight='bold', color='white')

# Highlight interpretation rows
for i in [16, 17, 18, 19]:
    for j in range(3):
        table[(i, j)].set_facecolor('#f0f0f0')
        table[(i, j)].set_text_props(fontsize=8)

ax.set_title('D. Network Efficiency Metrics Summary\n[Complete assessment of capacity utilization and efficiency]\n[Green highlight = good performance | Red = potential issues]', 
            fontsize=10, fontweight='bold', pad=10)

plt.tight_layout()
plt.savefig('feature1_chart10_efficiency.png', dpi=300, bbox_inches='tight')
print("✓ Saved: feature1_chart10_efficiency.png")
plt.show()
plt.close()

print(f"\nEfficiency Insights:")
print(f"  • Network efficiency ratio: {efficiency_ratio:.1%} (volume/capacity)")
print(f"  • Average utilization: {avg_utilization:.1%}")
print(f"  • {roads_under_50:,} roads ({roads_under_50/n_edges*100:.1f}%) are under-utilized (<50%)")
print(f"  • {roads_over_capacity:,} roads ({roads_over_capacity/n_edges*100:.1f}%) are over-capacity (>100%)")

################################################################################
# CHART 11: CAPACITY BY ROAD CHARACTERISTICS
################################################################################
print("\n" + "=" * 80)
print("CHART 11: Capacity by Road Characteristics")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('FEATURE 1: Capacity Analysis by Road Physical Characteristics\nExamining How Road Properties Influence Capacity\nSpeed Zones, Length Categories, and Geometric Factors', 
             fontsize=16, fontweight='bold', y=0.995)
plt.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.06, hspace=0.40, wspace=0.28)

# 11.1 Capacity by speed categories
ax = axes[0, 0]
speed_bins = [0, 30, 50, 70, 90, free_speed.max()+1]
speed_labels = ['0-30\nkm/h', '30-50\nkm/h', '50-70\nkm/h', '70-90\nkm/h', '>90\nkm/h']
cap_by_speed = []

for i in range(len(speed_bins)-1):
    mask = (free_speed >= speed_bins[i]) & (free_speed < speed_bins[i+1]) & (capacity > 0)
    cap_by_speed.append(capacity[mask])

# Filter empty categories
cap_filtered = [c for c in cap_by_speed if len(c) > 0]
labels_filtered = [speed_labels[i] for i in range(len(cap_by_speed)) if len(cap_by_speed[i]) > 0]

if len(cap_filtered) > 0:
    bp = ax.boxplot(cap_filtered, tick_labels=labels_filtered,
                   patch_artist=True, showfliers=False, widths=0.6)
    colors = ['#e74c3c', '#f39c12', '#27ae60', '#3498db', '#9b59b6']
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.2)
    for median in bp['medians']:
        median.set_color('white')
        median.set_linewidth(3)
    
    # Add count labels
    for idx, (c, label) in enumerate(zip(cap_filtered, labels_filtered)):
        ax.text(idx+1, ax.get_ylim()[1]*0.95, f'n={len(c):,}', 
               ha='center', fontsize=8, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

ax.set_xlabel('Speed Category\n[Roads grouped by free flow speed limit]', fontsize=10, fontweight='bold')
ax.set_ylabel('Road Capacity (vehicles/hour)\n[Box = middle 50% | White line = median]', fontsize=10, fontweight='bold')
ax.set_title('A. Capacity Distribution by Speed Zone\n[Expectation: Higher speed roads should have higher capacity]\n[Shows if speed limit correlates with designed capacity]', 
            fontsize=10, fontweight='bold', pad=10)
ax.grid(True, alpha=0.3, axis='y')

# 11.2 Capacity by length categories
ax = axes[0, 1]
length_percentiles = [0, 25, 50, 75, 90, 100]
length_thresholds = [np.percentile(length[length > 0], p) for p in length_percentiles]
length_labels = []
cap_by_length = []

for i in range(len(length_thresholds)-1):
    mask = (length >= length_thresholds[i]) & (length < length_thresholds[i+1]) & (capacity > 0)
    if mask.sum() > 0:
        cap_by_length.append(capacity[mask])
        length_labels.append(f'P{length_percentiles[i]}-P{length_percentiles[i+1]}\n({length_thresholds[i]:.0f}-{length_thresholds[i+1]:.0f}m)')

if len(cap_by_length) > 0:
    bp = ax.boxplot(cap_by_length, tick_labels=length_labels,
                   patch_artist=True, showfliers=False, widths=0.6)
    colors = ['#3498db', '#27ae60', '#f39c12', '#e67e22', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.2)
    for median in bp['medians']:
        median.set_color('white')
        median.set_linewidth(3)

ax.set_xlabel('Length Percentile Category\n[Roads grouped by length percentiles]', fontsize=10, fontweight='bold')
ax.set_ylabel('Road Capacity (vehicles/hour)\n[Box = middle 50% | White line = median]', fontsize=10, fontweight='bold')
ax.set_title('B. Capacity vs Road Length Categories\n[Do shorter or longer road segments have different capacity?]\n[Length categories: shortest 25% to longest 10%]', 
            fontsize=10, fontweight='bold', pad=10)
ax.grid(True, alpha=0.3, axis='y')

# 11.3 Mean capacity by highway type (bar chart with error bars)
ax = axes[1, 0]
mean_caps = []
std_caps = []
type_names_clean = []
counts = []

for ht in unique_types:
    mask = (highway == ht) & (capacity > 0)
    if mask.sum() > 0:
        mean_caps.append(capacity[mask].mean())
        std_caps.append(capacity[mask].std())
        type_names_clean.append(highway_type_names.get(int(ht), f'Type {int(ht)}'))
        counts.append(mask.sum())

colors_bar = ['#e74c3c', '#3498db', '#27ae60', '#f39c12', '#9b59b6', 
             '#e67e22', '#1abc9c', '#34495e', '#95a5a6', '#2c3e50']
bars = ax.bar(range(len(mean_caps)), mean_caps, yerr=std_caps, 
             alpha=0.8, color=colors_bar[:len(mean_caps)], 
             edgecolor='black', linewidth=1.2, capsize=5, error_kw={'linewidth': 2})

ax.set_xlabel('Highway Type\n[OpenStreetMap road classification]', fontsize=10, fontweight='bold')
ax.set_ylabel('Mean Capacity (vehicles/hour)\n[Average ± standard deviation]', fontsize=10, fontweight='bold')
ax.set_title('C. Average Capacity by Highway Type with Variability\n[Error bars show standard deviation = spread within each type]\n[Different colors help distinguish road types]', 
            fontsize=10, fontweight='bold', pad=10)
ax.set_xticks(range(len(mean_caps)))
ax.set_xticklabels(type_names_clean, fontsize=8, rotation=45, ha='right')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for idx, (bar, val, count) in enumerate(zip(bars, mean_caps, counts)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(std_caps)*1.1, 
           f'{val:.0f}\n(n={count:,})', ha='center', va='bottom', fontsize=7, fontweight='bold')

# 11.4 Capacity vs Speed scatter with density
ax = axes[1, 1]
valid_mask = (capacity > 0) & (free_speed > 0)
cap_valid = capacity[valid_mask]
speed_valid = free_speed[valid_mask]

# Create 2D histogram for density
from matplotlib.colors import LogNorm
h = ax.hist2d(speed_valid, cap_valid, bins=50, cmap='YlOrRd', 
             norm=LogNorm(), alpha=0.8)
plt.colorbar(h[3], ax=ax, label='Number of Roads\n(log scale)')

# Add trend line
if len(speed_valid) > 100:
    z = np.polyfit(speed_valid, cap_valid, 1)
    p = np.poly1d(z)
    speed_range = np.linspace(speed_valid.min(), speed_valid.max(), 100)
    ax.plot(speed_range, p(speed_range), "b--", linewidth=3, alpha=0.9, label='Linear trend')
    corr = np.corrcoef(speed_valid, cap_valid)[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes, 
           fontsize=10, fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
           verticalalignment='top')
    ax.legend(loc='lower right', framealpha=0.9, fontsize=9)

ax.set_xlabel('Free Flow Speed (km/h)\n[Design speed limit]', fontsize=10, fontweight='bold')
ax.set_ylabel('Road Capacity (vehicles/hour)\n[Maximum vehicle capacity]', fontsize=10, fontweight='bold')
ax.set_title(f'D. Capacity-Speed Density Plot (n={len(speed_valid):,} roads)\n[Heat map shows concentration of roads: Yellow (few) to Red (many)]\n[Blue dashed line = overall trend]', 
            fontsize=10, fontweight='bold', pad=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('feature1_chart11_characteristics.png', dpi=300, bbox_inches='tight')
print("✓ Saved: feature1_chart11_characteristics.png")
plt.show()
plt.close()

################################################################################
# CHART 12: COMPREHENSIVE SUMMARY DASHBOARD
################################################################################
print("\n" + "=" * 80)
print("CHART 12: Comprehensive Summary Dashboard")
print("=" * 80)

fig = plt.figure(figsize=(20, 16))
fig.suptitle('FEATURE 1: Comprehensive Road Capacity Analysis Summary\nComplete Overview of Network Capacity Distribution, Utilization, and Patterns\nIntegrated Dashboard Combining Key Insights from All Analysis Components', 
             fontsize=16, fontweight='bold', y=0.995)

# Create grid for dashboard layout
gs = fig.add_gridspec(3, 3, left=0.08, right=0.95, top=0.93, bottom=0.06, 
                     hspace=0.35, wspace=0.30)

# 12.1 Main histogram (large, top left)
ax1 = fig.add_subplot(gs[0:2, 0])
cap_nonzero = capacity[capacity > 0]
ax1.hist(cap_nonzero, bins=60, alpha=0.7, color='#3498db', edgecolor='black', linewidth=0.5)
ax1.axvline(np.median(cap_nonzero), color='#e74c3c', linestyle='--', linewidth=3, 
           label=f'Median: {np.median(cap_nonzero):.0f}', alpha=0.8)
ax1.axvline(np.mean(cap_nonzero), color='#27ae60', linestyle='--', linewidth=3,
           label=f'Mean: {np.mean(cap_nonzero):.0f}', alpha=0.8)
Q1, Q3 = np.percentile(cap_nonzero, [25, 75])
ax1.axvspan(Q1, Q3, alpha=0.2, color='yellow', label=f'IQR: {Q1:.0f}-{Q3:.0f}')
ax1.set_xlabel('Road Capacity (veh/h)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Number of Roads', fontsize=11, fontweight='bold')
ax1.set_title('Overall Capacity Distribution\n[Primary summary of network capacity]', 
             fontsize=11, fontweight='bold', pad=8)
ax1.legend(loc='best', framealpha=0.9, fontsize=9)
ax1.grid(True, alpha=0.3)

# 12.2 Capacity by type (top middle)
ax2 = fig.add_subplot(gs[0, 1])
type_means = [capacity[highway == ht].mean() for ht in unique_types if (highway == ht).sum() > 100]
type_labels_short = [highway_type_names.get(int(ht), '?')[:6] for ht in unique_types if (highway == ht).sum() > 100]
colors_top = plt.cm.Set3(np.linspace(0, 1, len(type_means)))
bars = ax2.bar(range(len(type_means)), type_means, alpha=0.8, color=colors_top, edgecolor='black', linewidth=0.8)
ax2.set_xticks(range(len(type_means)))
ax2.set_xticklabels(type_labels_short, fontsize=8, rotation=45, ha='right')
ax2.set_ylabel('Mean Capacity (veh/h)', fontsize=10, fontweight='bold')
ax2.set_title('Capacity by Type\n[Average per road type]', fontsize=10, fontweight='bold', pad=8)
ax2.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, type_means):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.05, 
            f'{val:.0f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

# 12.3 Utilization distribution (top right)
ax3 = fig.add_subplot(gs[0, 2])
util_display = utilization[(utilization > 0) & (utilization < 2)]
ax3.hist(util_display, bins=40, alpha=0.7, color='#27ae60', edgecolor='black', linewidth=0.5)
ax3.axvline(1.0, color='#e74c3c', linestyle='--', linewidth=2.5, label='100%', alpha=0.8)
ax3.set_xlabel('Utilization Ratio', fontsize=10, fontweight='bold')
ax3.set_ylabel('Number of Roads', fontsize=10, fontweight='bold')
ax3.set_title('Utilization Distribution\n[Volume/Capacity ratio]', fontsize=10, fontweight='bold', pad=8)
ax3.legend(loc='best', framealpha=0.9, fontsize=8)
ax3.grid(True, alpha=0.3)

# 12.4 Statistics table (middle left)
ax4 = fig.add_subplot(gs[1, 1:])
ax4.axis('off')

# Compile comprehensive statistics
stats_data = [
    ['METRIC', 'VALUE', 'INTERPRETATION'],
    ['', '', ''],
    ['Total Roads', f'{n_edges:,}', 'Network size'],
    ['Total Capacity', f'{capacity.sum():,.0f} veh/h', 'Maximum throughput'],
    ['Mean Capacity', f'{capacity.mean():.0f} veh/h', 'Average per road'],
    ['Median Capacity', f'{np.median(capacity):.0f} veh/h', 'Typical road'],
    ['', '', ''],
    ['Std Deviation', f'{capacity.std():.0f} veh/h', 'Variability'],
    ['Coefficient of Variation', f'{capacity.std()/capacity.mean():.3f}', 'Relative spread'],
    ['Gini Coefficient', f'{1 - 2 * np.trapezoid(np.cumsum(np.sort(capacity[capacity>0]))/capacity.sum(), np.arange(len(capacity[capacity>0]))/len(capacity[capacity>0])):.3f}', 'Inequality measure'],
    ['', '', ''],
    ['Q1 (25th percentile)', f'{np.percentile(capacity, 25):.0f} veh/h', 'Lower quartile'],
    ['Q3 (75th percentile)', f'{np.percentile(capacity, 75):.0f} veh/h', 'Upper quartile'],
    ['P90 (90th percentile)', f'{np.percentile(capacity, 90):.0f} veh/h', 'High capacity'],
    ['', '', ''],
    ['Average Utilization', f'{avg_utilization:.1%}', 'Network efficiency'],
    ['Over-capacity Roads', f'{(utilization>1.0).sum():,} ({(utilization>1.0).sum()/n_edges*100:.1f}%)', 'Congested'],
    ['Under-utilized (<50%)', f'{(utilization<0.5).sum():,} ({(utilization<0.5).sum()/n_edges*100:.1f}%)', 'Spare capacity'],
]

table = ax4.table(cellText=stats_data, cellLoc='left', loc='center',
                 colWidths=[0.35, 0.35, 0.30])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.3)

# Style header rows
for i in [0, 1, 6, 10, 14]:
    for j in range(3):
        table[(i, j)].set_facecolor('#3498db')
        table[(i, j)].set_text_props(weight='bold', color='white')

# Style header
for j in range(3):
    table[(0, j)].set_text_props(weight='bold', color='white', fontsize=10)

ax4.set_title('Comprehensive Statistics Summary\n[Complete statistical overview]', 
             fontsize=11, fontweight='bold', pad=8)

# 12.5 Box plot comparison (bottom left)
ax5 = fig.add_subplot(gs[2, 0])
util_cats = ['0-25%', '25-50%', '50-75%', '75-100%', '>100%']
util_bins = [0, 0.25, 0.5, 0.75, 1.0, 100]
cap_by_util = []
for i in range(len(util_bins)-1):
    mask = (utilization >= util_bins[i]) & (utilization < util_bins[i+1]) & (capacity > 0)
    if mask.sum() > 10:
        cap_by_util.append(capacity[mask])

if len(cap_by_util) > 0:
    bp = ax5.boxplot(cap_by_util, tick_labels=util_cats[:len(cap_by_util)],
                    patch_artist=True, showfliers=False, widths=0.6)
    colors_box = ['#27ae60', '#3498db', '#f39c12', '#e67e22', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors_box[:len(bp['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for median in bp['medians']:
        median.set_color('white')
        median.set_linewidth(2.5)

ax5.set_xlabel('Utilization Category', fontsize=10, fontweight='bold')
ax5.set_ylabel('Capacity (veh/h)', fontsize=10, fontweight='bold')
ax5.set_title('Capacity by Utilization\n[Green=light | Red=heavy]', fontsize=10, fontweight='bold', pad=8)
ax5.grid(True, alpha=0.3, axis='y')

# 12.6 Lorenz curve (bottom middle)
ax6 = fig.add_subplot(gs[2, 1])
sorted_cap_lorenz = np.sort(capacity[capacity > 0])
cum_cap_lorenz = np.cumsum(sorted_cap_lorenz)
cum_cap_pct = cum_cap_lorenz / cum_cap_lorenz[-1] * 100
cum_roads_pct = np.arange(1, len(sorted_cap_lorenz) + 1) / len(sorted_cap_lorenz) * 100
ax6.plot(cum_roads_pct, cum_cap_pct, linewidth=2.5, color='#3498db', label='Actual')
ax6.plot([0, 100], [0, 100], 'r--', linewidth=2, label='Perfect Equality', alpha=0.7)
ax6.fill_between(cum_roads_pct, cum_cap_pct, cum_roads_pct, alpha=0.3, color='lightcoral')
gini_final = 1 - 2 * np.trapezoid(cum_cap_pct / 100, cum_roads_pct / 100)
ax6.text(0.05, 0.95, f'Gini: {gini_final:.3f}', transform=ax6.transAxes, 
        fontsize=9, fontweight='bold', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        verticalalignment='top')
ax6.set_xlabel('Cumulative % of Roads', fontsize=10, fontweight='bold')
ax6.set_ylabel('Cumulative % of Capacity', fontsize=10, fontweight='bold')
ax6.set_title('Capacity Inequality\n[Lorenz Curve]', fontsize=10, fontweight='bold', pad=8)
ax6.legend(loc='lower right', framealpha=0.9, fontsize=8)
ax6.grid(True, alpha=0.3)

# 12.7 Scatter summary (bottom right)
ax7 = fig.add_subplot(gs[2, 2])
valid_scatter = (capacity > 0) & (vol_base_case != 0)
cap_scatter = capacity[valid_scatter]
vol_scatter = np.abs(vol_base_case[valid_scatter])
# Subsample for performance
if len(cap_scatter) > 5000:
    sample_idx = np.random.choice(len(cap_scatter), 5000, replace=False)
    cap_scatter = cap_scatter[sample_idx]
    vol_scatter = vol_scatter[sample_idx]
ax7.scatter(cap_scatter, vol_scatter, alpha=0.3, s=2, c='#9b59b6', edgecolors='none')
max_val = min(cap_scatter.max(), vol_scatter.max())
ax7.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='100% util', alpha=0.7)
if len(cap_scatter) > 10:
    corr_scatter = np.corrcoef(cap_scatter, vol_scatter)[0, 1]
    ax7.text(0.05, 0.95, f'r={corr_scatter:.3f}', transform=ax7.transAxes, 
            fontsize=9, fontweight='bold', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            verticalalignment='top')
ax7.set_xlabel('Capacity (veh/h)', fontsize=10, fontweight='bold')
ax7.set_ylabel('Volume (veh/h)', fontsize=10, fontweight='bold')
ax7.set_title('Capacity-Volume\n[Correlation check]', fontsize=10, fontweight='bold', pad=8)
ax7.legend(loc='best', framealpha=0.9, fontsize=8)
ax7.grid(True, alpha=0.3)

plt.savefig('feature1_chart12_summary.png', dpi=300, bbox_inches='tight')
print("✓ Saved: feature1_chart12_summary.png")
plt.show()
plt.close()

################################################################################
# FINAL SUMMARY
################################################################################
print("\n" + "=" * 80)
print("✓✓✓ PART 3 COMPLETE - CHARTS 9-12 GENERATED ✓✓✓")
print("=" * 80)
print("\nGenerated files:")
print("   9. feature1_chart9_correlations.png")
print("  10. feature1_chart10_efficiency.png")
print("  11. feature1_chart11_characteristics.png")
print("  12. feature1_chart12_summary.png")
print("\n" + "=" * 80)
print("✓✓✓ FEATURE 1 ANALYSIS COMPLETE - ALL 12 CHARTS GENERATED ✓✓✓")
print("=" * 80)
print("\nComplete set of Feature 1 (Road Capacity) visualizations:")
print("\nPART 1 (Charts 1-4):")
print("  1. Basic capacity distribution & statistics")
print("  2. Capacity patterns & temporal analysis")
print("  3. Highway type analysis")
print("  4. Advanced distribution analysis")
print("\nPART 2 (Charts 5-8):")
print("  5. Outliers & extreme values")
print("  6. Network capacity statistics")
print("  7. Capacity & policy interaction")
print("  8. Capacity reduction targeting")
print("\nPART 3 (Charts 9-12):")
print("  9. Feature correlations & relationships")
print(" 10. Capacity efficiency metrics")
print(" 11. Capacity by road characteristics")
print(" 12. Comprehensive summary dashboard")
print("\n" + "=" * 80)
print("Next: Proceed to Feature 2, 3, 4, or 5 analysis")
print("=" * 80)
