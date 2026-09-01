"""
FEATURE 4 ANALYSIS - PART 1
Highway Type (F4) - Charts 1 to 2

Dataset: Paris MATSim Transport Network
Feature: Highway Type (categorical: 0-12)
Scenarios: 250 (5 batches × 50 scenarios each)

CHART 1: Highway Type Distribution
  - 1A: Pie chart showing type proportions
  - 1B: Bar chart with counts and percentages
  - 1C: Type hierarchy visualization
  - 1D: Statistics table

CHART 2: Characteristics by Highway Type
  - 2A: Mean capacity by type
  - 2B: Mean free speed by type
  - 2C: Mean length by type
  - 2D: Traffic distribution by type
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from IPython.display import Image, display

# Paths
data_dir = Path('/content/drive/MyDrive/Zamin_thesis/ml_surrogates_for_agent_based_transport_models/data/train_data/dist_not_connected_10k_1pct')
batch_files = [
    'datalist_batch_1.pt', 'datalist_batch_2.pt', 'datalist_batch_3.pt', 
    'datalist_batch_4.pt', 'datalist_batch_5.pt'
]

# Highway type mapping
HW_MAPPING = {
    -1: 'Unknown',
    0: 'Motorway',
    1: 'Trunk',
    2: 'Primary',
    3: 'Secondary',
    4: 'Tertiary',
    5: 'Residential',
    6: 'PT',
    7: 'Service',
    8: 'Living Street',
    9: 'Motorway Link',
    10: 'Trunk Link',
    11: 'Primary Link',
    12: 'Secondary Link'
}

# Type hierarchy (for visualization)
HW_HIERARCHY = {
    'High Speed': ['Motorway', 'Motorway Link'],
    'Major Roads': ['Trunk', 'Trunk Link', 'Primary', 'Primary Link'],
    'Collector Roads': ['Secondary', 'Secondary Link', 'Tertiary'],
    'Local Roads': ['Residential', 'Living Street'],
    'Other': ['Service', 'PT']
}

# Color palette for 13 types
COLORS_13 = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
             '#ffff33', '#a65628', '#f781bf', '#999999', '#66c2a5',
             '#fc8d62', '#8da0cb', '#e78ac3']

print("\n" + "="*70)
print("FEATURE 4 (HIGHWAY TYPE) ANALYSIS - PART 1")
print("="*70)

# ============================================================================
# STEP 1: LOAD DATA FROM BATCH 1 (BASELINE)
# ============================================================================
print("\n[1/3] Loading Batch 1 (baseline)...")

batch_path = data_dir / batch_files[0]
if not batch_path.exists():
    raise FileNotFoundError(f"Batch file not found: {batch_path}")

graphs_list = torch.load(batch_path, weights_only=False)
print(f"   Loaded {len(graphs_list)} scenarios from batch")

# Use first scenario (baseline)
graph = graphs_list[0]

# Extract features
n_active = 31635  # Active road segments
highway_type = graph.x[:n_active, 4].numpy().astype(int)  # F4 = highway type
capacity = graph.x[:n_active, 1].numpy()  # F1 = capacity
free_speed = graph.x[:n_active, 3].numpy()  # F3 = free speed
road_length = graph.x[:n_active, 5].numpy()  # F5 = road length
baseline_volume = graph.x[:n_active, 2].numpy()  # F2 = baseline volume

print(f"   Total road segments: {n_active:,}")
print(f"   Highway types range: {highway_type.min()} to {highway_type.max()}")
print(f"   Number of unique types: {len(np.unique(highway_type))}")

# Count distribution
type_counts = Counter(highway_type)
print("\n   Highway Type Distribution:")
for hw_code in sorted(type_counts.keys()):
    hw_name = HW_MAPPING.get(hw_code, f'Unknown_{hw_code}')
    count = type_counts[hw_code]
    pct = (count / n_active) * 100
    print(f"   {hw_code:2d} {hw_name:20s}: {count:6,} ({pct:5.2f}%)")

# ============================================================================
# STEP 2: CHART 1 - HIGHWAY TYPE DISTRIBUTION (4 PANELS)
# ============================================================================
print("\n[2/3] Creating Chart 1: Highway Type Distribution...")

fig1 = plt.figure(figsize=(20, 16))

# Prepare data
type_codes = sorted(type_counts.keys())
type_names = [HW_MAPPING[code] for code in type_codes]
type_vals = [type_counts[code] for code in type_codes]
type_pcts = [(val / n_active) * 100 for val in type_vals]

# Sort by count for better visualization
sorted_indices = np.argsort(type_vals)[::-1]
type_codes_sorted = [type_codes[i] for i in sorted_indices]
type_names_sorted = [type_names[i] for i in sorted_indices]
type_vals_sorted = [type_vals[i] for i in sorted_indices]
type_pcts_sorted = [type_pcts[i] for i in sorted_indices]
colors_sorted = [COLORS_13[type_codes[i]] for i in sorted_indices]

# --- Panel 1A: Pie Chart ---
ax1a = plt.subplot(2, 2, 1)
wedges, texts, autotexts = ax1a.pie(
    type_vals_sorted,
    labels=type_names_sorted,
    autopct='%1.1f%%',
    startangle=90,
    colors=colors_sorted,
    textprops={'fontsize': 9}
)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
ax1a.set_title('Highway Type Distribution (Proportions)', fontsize=11, fontweight='bold', pad=15)

# --- Panel 1B: Bar Chart with Counts ---
ax1b = plt.subplot(2, 2, 2)
bars = ax1b.bar(range(len(type_names_sorted)), type_vals_sorted, color=colors_sorted, alpha=0.8, edgecolor='black')
ax1b.set_xticks(range(len(type_names_sorted)))
ax1b.set_xticklabels(type_names_sorted, rotation=45, ha='right', fontsize=9)
ax1b.set_ylabel('Number of Road Segments', fontsize=10, fontweight='bold')
ax1b.set_title('Highway Type Counts', fontsize=11, fontweight='bold', pad=15)
ax1b.grid(axis='y', alpha=0.3, linestyle='--')

# Add count labels on bars
for i, (bar, val) in enumerate(zip(bars, type_vals_sorted)):
    height = bar.get_height()
    ax1b.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:,}\n({type_pcts_sorted[i]:.1f}%)',
             ha='center', va='bottom', fontsize=8, fontweight='bold')

# --- Panel 1C: Type Hierarchy ---
ax1c = plt.subplot(2, 2, 3)
ax1c.axis('off')

y_pos = 0.95
hierarchy_data = []
for category, types in HW_HIERARCHY.items():
    category_count = sum(type_counts.get(code, 0) for code, name in HW_MAPPING.items() if name in types)
    category_pct = (category_count / n_active) * 100
    hierarchy_data.append((category, types, category_count, category_pct))

# Sort by count
hierarchy_data.sort(key=lambda x: x[2], reverse=True)

for category, types, count, pct in hierarchy_data:
    # Category header
    ax1c.text(0.05, y_pos, f'• {category}:', fontsize=10, fontweight='bold', color='#2c3e50')
    ax1c.text(0.95, y_pos, f'{count:,} ({pct:.1f}%)', fontsize=10, fontweight='bold',
             ha='right', color='#e74c3c')
    y_pos -= 0.05
    
    # Individual types
    for type_name in types:
        type_code = [k for k, v in HW_MAPPING.items() if v == type_name][0]
        if type_code in type_counts:
            type_count = type_counts[type_code]
            type_pct = (type_count / n_active) * 100
            ax1c.text(0.10, y_pos, f'  - {type_name}', fontsize=9, color='#34495e')
            ax1c.text(0.95, y_pos, f'{type_count:,} ({type_pct:.1f}%)', fontsize=9,
                     ha='right', color='#7f8c8d')
            y_pos -= 0.04
    
    y_pos -= 0.02  # Extra space between categories

ax1c.set_title('Highway Type Hierarchy', fontsize=11, fontweight='bold', pad=15)
ax1c.set_xlim(0, 1)
ax1c.set_ylim(0, 1)

# --- Panel 1D: Statistics Table ---
ax1d = plt.subplot(2, 2, 4)
ax1d.axis('off')

stats_data = [
    ['Total Road Segments', f'{n_active:,}'],
    ['Number of Types', f'{len(type_counts)}'],
    ['', ''],
    ['Most Common Type', f'{type_names_sorted[0]}'],
    ['  Count', f'{type_vals_sorted[0]:,} ({type_pcts_sorted[0]:.2f}%)'],
    ['', ''],
    ['Least Common Type', f'{type_names_sorted[-1]}'],
    ['  Count', f'{type_vals_sorted[-1]:,} ({type_pcts_sorted[-1]:.2f}%)'],
    ['', ''],
    ['Top 3 Types Coverage', f'{sum(type_vals_sorted[:3]):,} ({sum(type_pcts_sorted[:3]):.1f}%)'],
    ['Bottom 3 Types Coverage', f'{sum(type_vals_sorted[-3:]):,} ({sum(type_pcts_sorted[-3:]):.1f}%)'],
]

y_pos = 0.95
for row in stats_data:
    if row[0] == '':
        y_pos -= 0.03
        continue
    
    if row[0].startswith('  '):
        # Indented row
        ax1d.text(0.1, y_pos, row[0], fontsize=9, color='#34495e')
        ax1d.text(0.95, y_pos, row[1], fontsize=9, ha='right', color='#7f8c8d')
    else:
        # Main row
        ax1d.text(0.05, y_pos, row[0], fontsize=10, fontweight='bold', color='#2c3e50')
        ax1d.text(0.95, y_pos, row[1], fontsize=10, fontweight='bold', ha='right', color='#e74c3c')
    
    y_pos -= 0.05

ax1d.set_title('Distribution Statistics', fontsize=11, fontweight='bold', pad=15)
ax1d.set_xlim(0, 1)
ax1d.set_ylim(0, 1)

plt.tight_layout()
chart1_path = 'feature4_chart1_distribution.png'
plt.savefig(chart1_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"   ✓ Saved: {chart1_path}")
display(Image(chart1_path))

# ============================================================================
# STEP 3: CHART 2 - CHARACTERISTICS BY HIGHWAY TYPE (4 PANELS)
# ============================================================================
print("\n[3/3] Creating Chart 2: Characteristics by Highway Type...")

fig2 = plt.figure(figsize=(20, 16))

# Calculate means for each type
type_means = {}
for code in type_codes:
    mask = highway_type == code
    type_means[code] = {
        'name': HW_MAPPING[code],
        'count': np.sum(mask),
        'capacity_mean': np.mean(capacity[mask]),
        'capacity_std': np.std(capacity[mask]),
        'speed_mean': np.mean(free_speed[mask]),
        'speed_std': np.std(free_speed[mask]),
        'length_mean': np.mean(road_length[mask]),
        'length_std': np.std(road_length[mask]),
        'baseline_mean': np.mean(baseline_volume[mask]),
        'baseline_std': np.std(baseline_volume[mask]),
        'traffic_pct': (np.sum(baseline_volume[mask] != 0) / np.sum(mask)) * 100 if np.sum(mask) > 0 else 0
    }

# Sort by count for consistent ordering
sorted_codes = sorted(type_codes, key=lambda c: type_means[c]['count'], reverse=True)
sorted_names = [type_means[c]['name'] for c in sorted_codes]

# --- Panel 2A: Mean Capacity by Type ---
ax2a = plt.subplot(2, 2, 1)
cap_means = [type_means[c]['capacity_mean'] for c in sorted_codes]
cap_stds = [type_means[c]['capacity_std'] for c in sorted_codes]
colors_cap = [COLORS_13[c] for c in sorted_codes]

bars = ax2a.barh(range(len(sorted_names)), cap_means, xerr=cap_stds, 
                  color=colors_cap, alpha=0.8, edgecolor='black', capsize=3)
ax2a.set_yticks(range(len(sorted_names)))
ax2a.set_yticklabels(sorted_names, fontsize=9)
ax2a.set_xlabel('Mean Capacity (veh/h)', fontsize=10, fontweight='bold')
ax2a.set_title('Mean Road Capacity by Highway Type', fontsize=11, fontweight='bold', pad=15)
ax2a.grid(axis='x', alpha=0.3, linestyle='--')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, cap_means)):
    ax2a.text(val, bar.get_y() + bar.get_height()/2,
             f' {val:.0f}', va='center', fontsize=8, fontweight='bold')

# --- Panel 2B: Mean Free Speed by Type ---
ax2b = plt.subplot(2, 2, 2)
speed_means = [type_means[c]['speed_mean'] for c in sorted_codes]
speed_stds = [type_means[c]['speed_std'] for c in sorted_codes]

bars = ax2b.barh(range(len(sorted_names)), speed_means, xerr=speed_stds,
                  color=colors_cap, alpha=0.8, edgecolor='black', capsize=3)
ax2b.set_yticks(range(len(sorted_names)))
ax2b.set_yticklabels(sorted_names, fontsize=9)
ax2b.set_xlabel('Mean Free Speed (km/h)', fontsize=10, fontweight='bold')
ax2b.set_title('Mean Free Speed by Highway Type', fontsize=11, fontweight='bold', pad=15)
ax2b.grid(axis='x', alpha=0.3, linestyle='--')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, speed_means)):
    ax2b.text(val, bar.get_y() + bar.get_height()/2,
             f' {val:.1f}', va='center', fontsize=8, fontweight='bold')

# --- Panel 2C: Mean Road Length by Type ---
ax2c = plt.subplot(2, 2, 3)
length_means = [type_means[c]['length_mean'] for c in sorted_codes]
length_stds = [type_means[c]['length_std'] for c in sorted_codes]

bars = ax2c.barh(range(len(sorted_names)), length_means, xerr=length_stds,
                  color=colors_cap, alpha=0.8, edgecolor='black', capsize=3)
ax2c.set_yticks(range(len(sorted_names)))
ax2c.set_yticklabels(sorted_names, fontsize=9)
ax2c.set_xlabel('Mean Road Length (m)', fontsize=10, fontweight='bold')
ax2c.set_title('Mean Road Length by Highway Type', fontsize=11, fontweight='bold', pad=15)
ax2c.grid(axis='x', alpha=0.3, linestyle='--')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, length_means)):
    ax2c.text(val, bar.get_y() + bar.get_height()/2,
             f' {val:.1f}', va='center', fontsize=8, fontweight='bold')

# --- Panel 2D: Traffic Distribution by Type ---
ax2d = plt.subplot(2, 2, 4)
traffic_pcts = [type_means[c]['traffic_pct'] for c in sorted_codes]

bars = ax2d.barh(range(len(sorted_names)), traffic_pcts,
                  color=colors_cap, alpha=0.8, edgecolor='black')
ax2d.set_yticks(range(len(sorted_names)))
ax2d.set_yticklabels(sorted_names, fontsize=9)
ax2d.set_xlabel('Percentage with Traffic (%)', fontsize=10, fontweight='bold')
ax2d.set_title('Traffic Coverage by Highway Type (Baseline)', fontsize=11, fontweight='bold', pad=15)
ax2d.grid(axis='x', alpha=0.3, linestyle='--')
ax2d.set_xlim(0, 100)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, traffic_pcts)):
    ax2d.text(val, bar.get_y() + bar.get_height()/2,
             f' {val:.1f}%', va='center', fontsize=8, fontweight='bold')

plt.tight_layout()
chart2_path = 'feature4_chart2_characteristics_by_type.png'
plt.savefig(chart2_path, dpi=300, bbox_inches='tight')
plt.close()

display(Image(chart2_path))
print(f"   ✓ Saved: {chart2_path}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("SUMMARY - FEATURE 4 PART 1 COMPLETE")
print("="*70)
print(f"\nChart 1: Highway Type Distribution (4 panels)")
print(f"  - Pie chart, bar chart, hierarchy, statistics")
print(f"Chart 2: Characteristics by Type (4 panels)")
print(f"  - Capacity, speed, length, traffic coverage")
print(f"\nMost common type: {type_names_sorted[0]} ({type_vals_sorted[0]:,} roads, {type_pcts_sorted[0]:.1f}%)")
print(f"Highest capacity: {sorted_names[np.argmax(cap_means)]} ({max(cap_means):.0f} veh/h)")
print(f"Highest speed: {sorted_names[np.argmax(speed_means)]} ({max(speed_means):.1f} km/h)")
print(f"Longest roads: {sorted_names[np.argmax(length_means)]} ({max(length_means):.1f} m)")
print(f"Most traffic: {sorted_names[np.argmax(traffic_pcts)]} ({max(traffic_pcts):.1f}% coverage)")
print("\n" + "="*70)
