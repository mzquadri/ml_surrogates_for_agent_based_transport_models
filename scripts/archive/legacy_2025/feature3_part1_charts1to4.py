"""
FEATURE 3 ANALYSIS - PART 1: FREE SPEED (Charts 1-4)
=====================================================
Charts 1-4: Distribution, Highway Type & Relationships

Feature 3 (F3) represents free flow speed - the design speed limit of roads
Analyzing speed characteristics, patterns, and relationships with other features
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
print("#" + "  FEATURE 3 - PART 1: FREE SPEED (Charts 1-4)".center(78) + "#")
print("#" + "  Distribution, Highway Type & Relationships".center(78) + "#")
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

# Load first batch
batch_0 = torch.load(batch_files[0], weights_only=False)
first_scenario = batch_0[0]

# Extract features
vol_base_case = first_scenario.x[:, 0].numpy()
capacity = first_scenario.x[:, 1].numpy()
cap_reduction = first_scenario.x[:, 2].numpy()
free_speed = first_scenario.x[:, 3].numpy()
highway = first_scenario.x[:, 4].numpy()
length = first_scenario.x[:, 5].numpy()

n_edges = len(free_speed)
unique_types = np.unique(highway)
print(f"✓ Loaded {n_edges:,} edges")

# Highway type decoder
highway_type_names = {
    0: 'Motorway', 1: 'Trunk', 2: 'Primary', 3: 'Secondary', 4: 'Tertiary',
    5: 'Residential', 6: 'Service', 7: 'Unclassified', 8: 'Living Street', 9: 'Other'
}

# Basic statistics
print(f"\nFree Speed Statistics:")
print(f"  Mean: {free_speed.mean():.1f} km/h")
print(f"  Median: {np.median(free_speed):.1f} km/h")
print(f"  Range: {free_speed.min():.1f} - {free_speed.max():.1f} km/h")
print(f"  Std Dev: {free_speed.std():.1f} km/h")

################################################################################
# CHART 1: FREE SPEED DISTRIBUTION
################################################################################
print("\n" + "=" * 80)
print("CHART 1: Free Speed Distribution")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('FEATURE 3: Free Speed Distribution Analysis\nUnderstanding Road Speed Limits Across the Network\nAnalyzing Design Speed Characteristics and Patterns', 
             fontsize=16, fontweight='bold', y=0.995)
plt.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.06, hspace=0.40, wspace=0.28)

# 1.1 Main histogram
ax = axes[0, 0]
speed_nonzero = free_speed[free_speed > 0]
ax.hist(speed_nonzero, bins=60, alpha=0.7, color='#3498db', edgecolor='black', linewidth=0.5)
ax.axvline(np.median(speed_nonzero), color='#e74c3c', linestyle='--', linewidth=2.5,
          label=f'Median = {np.median(speed_nonzero):.1f} km/h', alpha=0.8)
ax.axvline(np.mean(speed_nonzero), color='#27ae60', linestyle='--', linewidth=2.5,
          label=f'Mean = {np.mean(speed_nonzero):.1f} km/h', alpha=0.8)
Q1, Q3 = np.percentile(speed_nonzero, [25, 75])
ax.axvspan(Q1, Q3, alpha=0.2, color='yellow', label=f'IQR: {Q1:.1f}-{Q3:.1f}')

ax.set_xlabel('Free Flow Speed (km/h)\n[Design speed limit of road]', fontsize=10, fontweight='bold')
ax.set_ylabel('Number of Roads\n[Frequency count]', fontsize=10, fontweight='bold')
ax.set_title(f'A. Overall Free Speed Distribution (n={len(speed_nonzero):,} roads)\n[Shows how speed limits are distributed across network]\n[Red=median | Green=mean | Yellow=middle 50% (IQR)]', 
            fontsize=10, fontweight='bold', pad=10)
ax.legend(loc='best', framealpha=0.9, fontsize=9)
ax.grid(True, alpha=0.3)

# 1.2 Speed categories
ax = axes[0, 1]
speed_bins = [0, 30, 50, 70, 90, 110, free_speed.max()+1]
bin_labels = ['0-30\nkm/h', '30-50\nkm/h', '50-70\nkm/h', '70-90\nkm/h', '90-110\nkm/h', '>110\nkm/h']
roads_per_bin = []
for i in range(len(speed_bins)-1):
    mask = (free_speed >= speed_bins[i]) & (free_speed < speed_bins[i+1])
    roads_per_bin.append(mask.sum())

colors = ['#e74c3c', '#e67e22', '#f39c12', '#27ae60', '#3498db', '#9b59b6']
bars = ax.bar(range(len(bin_labels)), roads_per_bin, alpha=0.8, 
             color=colors[:len(bin_labels)], edgecolor='black', linewidth=1.2)

# Add percentage labels
for bar, count in zip(bars, roads_per_bin):
    pct = (count / n_edges) * 100
    if count > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(roads_per_bin)*0.02,
               f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_xlabel('Speed Category\n[Roads grouped by speed limit ranges]', fontsize=10, fontweight='bold')
ax.set_ylabel('Number of Roads\n[Count in each speed category]', fontsize=10, fontweight='bold')
ax.set_title('B. Speed Distribution by Category\n[Common speed zones: urban (30-50) vs suburban (50-70) vs highway (>70)]\n[Color coding: Red (slow) to Purple (very fast)]', 
            fontsize=10, fontweight='bold', pad=10)
ax.set_xticks(range(len(bin_labels)))
ax.set_xticklabels(bin_labels, fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# 1.3 Cumulative distribution
ax = axes[1, 0]
sorted_speed = np.sort(speed_nonzero)
cum_count = np.arange(1, len(sorted_speed) + 1)
cum_pct = cum_count / len(sorted_speed) * 100

ax.plot(sorted_speed, cum_pct, linewidth=2.5, color='#3498db', label='Cumulative distribution')

# Mark key percentiles
percentiles = [25, 50, 75, 90]
for p in percentiles:
    val = np.percentile(speed_nonzero, p)
    ax.plot(val, p, 'o', markersize=10, color='#e74c3c' if p == 50 else '#f39c12')
    ax.axhline(p, color='gray', linestyle=':', alpha=0.3)
    ax.axvline(val, color='gray', linestyle=':', alpha=0.3)
    ax.text(val+2, p-3, f'P{p}: {val:.1f} km/h', fontsize=8, fontweight='bold')

ax.set_xlabel('Free Flow Speed (km/h)\n[X-axis: speed values]', fontsize=10, fontweight='bold')
ax.set_ylabel('Cumulative Percentage\n[% of roads with speed ≤ X]', fontsize=10, fontweight='bold')
ax.set_title('C. Cumulative Distribution Function (CDF)\n[Shows what % of roads have speed below any given value]\n[Key percentiles marked: P25, P50 (median), P75, P90]', 
            fontsize=10, fontweight='bold', pad=10)
ax.legend(loc='best', framealpha=0.9, fontsize=9)
ax.grid(True, alpha=0.3)

# 1.4 Statistics table
ax = axes[1, 1]
ax.axis('off')

stats_data = [
    ['FREE SPEED STATISTICS', '', ''],
    ['', '', ''],
    ['Total Roads', f'{n_edges:,}', 'roads'],
    ['Non-zero Speed', f'{len(speed_nonzero):,}', f'({len(speed_nonzero)/n_edges*100:.1f}%)'],
    ['Zero Speed', f'{(free_speed==0).sum():,}', f'({(free_speed==0).sum()/n_edges*100:.1f}%)'],
    ['', '', ''],
    ['CENTRAL TENDENCY', '', ''],
    ['', '', ''],
    ['Mean Speed', f'{speed_nonzero.mean():.1f}', 'km/h'],
    ['Median Speed', f'{np.median(speed_nonzero):.1f}', 'km/h'],
    ['Mode (approx)', f'{speed_nonzero[np.argmax(np.bincount(speed_nonzero.astype(int)))]:0.1f}', 'km/h'],
    ['', '', ''],
    ['VARIABILITY', '', ''],
    ['', '', ''],
    ['Std Deviation', f'{speed_nonzero.std():.1f}', 'km/h'],
    ['Coefficient of Variation', f'{speed_nonzero.std()/speed_nonzero.mean():.3f}', 'relative'],
    ['Range', f'{speed_nonzero.min():.1f} - {speed_nonzero.max():.1f}', 'km/h'],
    ['', '', ''],
    ['PERCENTILES', '', ''],
    ['', '', ''],
    ['P25 (Q1)', f'{np.percentile(speed_nonzero, 25):.1f}', 'km/h'],
    ['P50 (Median)', f'{np.percentile(speed_nonzero, 50):.1f}', 'km/h'],
    ['P75 (Q3)', f'{np.percentile(speed_nonzero, 75):.1f}', 'km/h'],
    ['P90', f'{np.percentile(speed_nonzero, 90):.1f}', 'km/h'],
]

table = ax.table(cellText=stats_data, cellLoc='left', loc='center',
                colWidths=[0.50, 0.30, 0.20])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.0)

# Highlight headers
for i in [0, 1, 6, 7, 12, 13, 18, 19]:
    for j in range(3):
        table[(i, j)].set_facecolor('#3498db')
        table[(i, j)].set_text_props(weight='bold', color='white')

ax.set_title('D. Comprehensive Statistics Summary\n[Complete statistical overview of speed distribution]\n[Key metrics: central tendency, spread, percentiles]', 
            fontsize=10, fontweight='bold', pad=10)

plt.tight_layout()
plt.savefig('feature3_chart1_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: feature3_chart1_distribution.png")
plt.show()
plt.close()

################################################################################
# CHART 2: SPEED BY HIGHWAY TYPE
################################################################################
print("\n" + "=" * 80)
print("CHART 2: Speed by Highway Type")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('FEATURE 3: Free Speed by Highway Type Analysis\nHow Speed Limits Vary Across Different Road Categories\nAnalyzing Speed Characteristics by OpenStreetMap Classification', 
             fontsize=16, fontweight='bold', y=0.995)
plt.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.06, hspace=0.40, wspace=0.28)

# 2.1 Mean speed by type (bar chart)
ax = axes[0, 0]
mean_speed_by_type = []
std_speed_by_type = []
type_labels = []
type_counts = []

for ht in unique_types:
    mask = (highway == ht) & (free_speed > 0)
    if mask.sum() > 0:
        mean_speed_by_type.append(free_speed[mask].mean())
        std_speed_by_type.append(free_speed[mask].std())
        type_labels.append(f'{int(ht)}\n{highway_type_names.get(int(ht), "?")[:6]}')
        type_counts.append(mask.sum())

colors_bar = ['#e74c3c', '#3498db', '#27ae60', '#f39c12', '#9b59b6', 
             '#e67e22', '#1abc9c', '#34495e', '#95a5a6', '#2c3e50']
bars = ax.bar(range(len(mean_speed_by_type)), mean_speed_by_type, yerr=std_speed_by_type,
             alpha=0.8, color=colors_bar[:len(mean_speed_by_type)], 
             edgecolor='black', linewidth=1.2, capsize=5, error_kw={'linewidth': 2})

ax.set_xlabel('Highway Type\n[OpenStreetMap road classification]', fontsize=10, fontweight='bold')
ax.set_ylabel('Mean Free Speed (km/h)\n[Average ± standard deviation]', fontsize=10, fontweight='bold')
ax.set_title('A. Average Speed by Road Type\n[Motorways typically fastest, residential slowest]\n[Error bars show variability within each type]', 
            fontsize=10, fontweight='bold', pad=10)
ax.set_xticks(range(len(mean_speed_by_type)))
ax.set_xticklabels(type_labels, fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, val, count in zip(bars, mean_speed_by_type, type_counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(std_speed_by_type)*1.15,
           f'{val:.1f}\n({count:,})', ha='center', va='bottom', fontsize=7, fontweight='bold')

# 2.2 Box plot by type
ax = axes[0, 1]
speed_by_type = []
type_labels_box = []

for ht in unique_types:
    mask = (highway == ht) & (free_speed > 0)
    if mask.sum() > 10:  # At least 10 roads
        speed_by_type.append(free_speed[mask])
        type_labels_box.append(f'{int(ht)}\n{highway_type_names.get(int(ht), "?")[:5]}')

if len(speed_by_type) > 0:
    bp = ax.boxplot(speed_by_type, tick_labels=type_labels_box,
                   patch_artist=True, showfliers=False, widths=0.6)
    
    colors_box = plt.cm.Set3(np.linspace(0, 1, len(speed_by_type)))
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.2)
    for median in bp['medians']:
        median.set_color('#e74c3c')
        median.set_linewidth(3)

ax.set_xlabel('Highway Type\n[Different road categories]', fontsize=10, fontweight='bold')
ax.set_ylabel('Free Speed Distribution (km/h)\n[Box = middle 50% | Red line = median]', fontsize=10, fontweight='bold')
ax.set_title('B. Speed Distribution by Highway Type\n[Box plot shows full distribution within each type]\n[Wider boxes = more speed variability]', 
            fontsize=10, fontweight='bold', pad=10)
ax.grid(True, alpha=0.3, axis='y')

# 2.3 Speed range by type
ax = axes[1, 0]
min_speeds = []
max_speeds = []
type_labels_range = []

for ht in unique_types:
    mask = (highway == ht) & (free_speed > 0)
    if mask.sum() > 0:
        min_speeds.append(free_speed[mask].min())
        max_speeds.append(free_speed[mask].max())
        type_labels_range.append(highway_type_names.get(int(ht), f'Type {int(ht)}'))

x_pos = np.arange(len(type_labels_range))
ranges = [max_s - min_s for min_s, max_s in zip(min_speeds, max_speeds)]

bars = ax.bar(x_pos, ranges, bottom=min_speeds, alpha=0.8,
             color=colors_bar[:len(type_labels_range)], edgecolor='black', linewidth=1.2)

ax.set_xlabel('Highway Type\n[OpenStreetMap classification]', fontsize=10, fontweight='bold')
ax.set_ylabel('Speed Range (km/h)\n[Bar shows min to max speed]', fontsize=10, fontweight='bold')
ax.set_title('C. Speed Range by Road Type\n[Bottom = minimum speed | Top = maximum speed]\n[Bar height = range of speeds within that type]', 
            fontsize=10, fontweight='bold', pad=10)
ax.set_xticks(x_pos)
ax.set_xticklabels(type_labels_range, fontsize=8, rotation=45, ha='right')
ax.grid(True, alpha=0.3, axis='y')

# Add range labels
for bar, min_s, max_s, rang in zip(bars, min_speeds, max_speeds, ranges):
    if rang > 5:  # Only label if range is significant
        ax.text(bar.get_x() + bar.get_width()/2, min_s + rang/2,
               f'{rang:.0f}', ha='center', va='center', fontsize=7, fontweight='bold', 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

# 2.4 Road count by type (sorted by speed)
ax = axes[1, 1]
# Sort by mean speed
sorted_indices = np.argsort(mean_speed_by_type)[::-1]
sorted_means = [mean_speed_by_type[i] for i in sorted_indices]
sorted_counts = [type_counts[i] for i in sorted_indices]
sorted_labels = [type_labels[i].split('\n')[1] for i in sorted_indices]  # Just the name

# Color by speed (gradient from slow to fast)
colors_sorted = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sorted_means)))

bars = ax.barh(range(len(sorted_counts)), sorted_counts, alpha=0.8,
              color=colors_sorted, edgecolor='black', linewidth=1.2)

ax.set_ylabel('Highway Type (sorted by speed)\n[Fastest at top, slowest at bottom]', fontsize=10, fontweight='bold')
ax.set_xlabel('Number of Roads\n[Count of roads in each type]', fontsize=10, fontweight='bold')
ax.set_title('D. Road Count by Type (Speed-Ordered)\n[Shows which road types are most common]\n[Color: Green (fast) to Red (slow)]', 
            fontsize=10, fontweight='bold', pad=10)
ax.set_yticks(range(len(sorted_counts)))
ax.set_yticklabels(sorted_labels, fontsize=9)
ax.grid(True, alpha=0.3, axis='x')

# Add percentage labels
for bar, count, speed in zip(bars, sorted_counts, sorted_means):
    pct = (count / n_edges) * 100
    ax.text(count + max(sorted_counts)*0.02, bar.get_y() + bar.get_height()/2,
           f'{count:,} ({pct:.1f}%)\n{speed:.0f} km/h', 
           ha='left', va='center', fontsize=7, fontweight='bold')

plt.tight_layout()
plt.savefig('feature3_chart2_by_type.png', dpi=300, bbox_inches='tight')
print("✓ Saved: feature3_chart2_by_type.png")
plt.show()
plt.close()

print("\n" + "=" * 80)
print("✓✓✓ PART 1 (Charts 1-2) COMPLETE ✓✓✓")
print("=" * 80)
print("\nGenerated files:")
print("  1. feature3_chart1_distribution.png")
print("  2. feature3_chart2_by_type.png")
print("\nNext: Run feature3_part2_charts3to4.py for Charts 3-4")
print("=" * 80)
