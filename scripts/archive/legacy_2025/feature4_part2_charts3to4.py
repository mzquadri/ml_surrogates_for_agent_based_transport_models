"""
FEATURE 4 ANALYSIS - PART 2
Highway Type (F4) - Charts 3 to 4

Dataset: Paris MATSim Transport Network
Feature: Highway Type (categorical: -1 to 9)
Scenarios: 250 (5 batches × 50 scenarios each)

CHART 3: Highway Type Relationships (4 panels)
  - 3A: Type vs Capacity (box plots)
  - 3B: Type vs Free Speed (box plots)
  - 3C: Type vs Road Length (box plots)
  - 3D: Type vs Traffic Volume (box plots)

CHART 4: Comprehensive Dashboard (12 panels)
  - Distribution, statistics, correlations, patterns
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from scipy import stats
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
    9: 'Motorway Link'
}

# Color palette
COLORS_11 = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
             '#ffff33', '#a65628', '#f781bf', '#999999', '#66c2a5', '#fc8d62']

print("\n" + "="*70)
print("FEATURE 4 (HIGHWAY TYPE) ANALYSIS - PART 2")
print("="*70)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n[1/3] Loading data...")

batch_path = data_dir / batch_files[0]
graphs_list = torch.load(batch_path, weights_only=False)
graph = graphs_list[0]

# Extract features
n_active = 31635
highway_type = graph.x[:n_active, 4].numpy().astype(int)  # F4
capacity = graph.x[:n_active, 1].numpy()  # F1
free_speed = graph.x[:n_active, 3].numpy()  # F3
road_length = graph.x[:n_active, 5].numpy()  # F5
baseline_volume = graph.x[:n_active, 2].numpy()  # F2
target_volume = graph.y[:n_active].numpy()  # Target

print(f"   Loaded {n_active:,} road segments")
print(f"   Highway types: {len(np.unique(highway_type))} unique")

# ============================================================================
# STEP 2: CHART 3 - HIGHWAY TYPE RELATIONSHIPS (4 PANELS)
# ============================================================================
print("\n[2/3] Creating Chart 3: Highway Type Relationships...")

fig3 = plt.figure(figsize=(20, 16))

# Get unique types sorted by frequency
type_counts = Counter(highway_type)
type_codes_sorted = sorted(type_counts.keys(), key=lambda x: type_counts[x], reverse=True)
type_names_sorted = [HW_MAPPING[code] for code in type_codes_sorted]

# Prepare data for box plots
capacity_by_type = [capacity[highway_type == code] for code in type_codes_sorted]
speed_by_type = [free_speed[highway_type == code] for code in type_codes_sorted]
length_by_type = [road_length[highway_type == code] for code in type_codes_sorted]
volume_by_type = [baseline_volume[highway_type == code] for code in type_codes_sorted]

# --- Panel 3A: Type vs Capacity ---
ax3a = plt.subplot(2, 2, 1)
bp1 = ax3a.boxplot(capacity_by_type, tick_labels=type_names_sorted, patch_artist=True)
for patch, color in zip(bp1['boxes'], COLORS_11):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax3a.set_xticklabels(type_names_sorted, rotation=45, ha='right', fontsize=9)
ax3a.set_ylabel('Capacity (veh/h)', fontsize=10, fontweight='bold')
ax3a.set_title('Road Capacity by Highway Type', fontsize=11, fontweight='bold', pad=15)
ax3a.grid(axis='y', alpha=0.3, linestyle='--')

# Add median labels
medians = [np.median(data) for data in capacity_by_type]
for i, median in enumerate(medians):
    ax3a.text(i+1, median, f'{median:.0f}', ha='center', va='bottom', 
             fontsize=8, fontweight='bold', color='darkred')

# --- Panel 3B: Type vs Free Speed ---
ax3b = plt.subplot(2, 2, 2)
bp2 = ax3b.boxplot(speed_by_type, tick_labels=type_names_sorted, patch_artist=True)
for patch, color in zip(bp2['boxes'], COLORS_11):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax3b.set_xticklabels(type_names_sorted, rotation=45, ha='right', fontsize=9)
ax3b.set_ylabel('Free Speed (km/h)', fontsize=10, fontweight='bold')
ax3b.set_title('Free Speed by Highway Type', fontsize=11, fontweight='bold', pad=15)
ax3b.grid(axis='y', alpha=0.3, linestyle='--')

# Add median labels
medians = [np.median(data) for data in speed_by_type]
for i, median in enumerate(medians):
    ax3b.text(i+1, median, f'{median:.1f}', ha='center', va='bottom', 
             fontsize=8, fontweight='bold', color='darkred')

# --- Panel 3C: Type vs Road Length ---
ax3c = plt.subplot(2, 2, 3)
bp3 = ax3c.boxplot(length_by_type, tick_labels=type_names_sorted, patch_artist=True)
for patch, color in zip(bp3['boxes'], COLORS_11):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax3c.set_xticklabels(type_names_sorted, rotation=45, ha='right', fontsize=9)
ax3c.set_ylabel('Road Length (m)', fontsize=10, fontweight='bold')
ax3c.set_title('Road Length by Highway Type', fontsize=11, fontweight='bold', pad=15)
ax3c.grid(axis='y', alpha=0.3, linestyle='--')
ax3c.set_yscale('log')  # Log scale for better visualization

# Add median labels
medians = [np.median(data) for data in length_by_type]
for i, median in enumerate(medians):
    ax3c.text(i+1, median, f'{median:.0f}', ha='center', va='bottom', 
             fontsize=8, fontweight='bold', color='darkred')

# --- Panel 3D: Type vs Traffic Volume ---
ax3d = plt.subplot(2, 2, 4)
# Only show non-zero volumes for clarity
volume_by_type_nonzero = [data[data != 0] for data in volume_by_type]
bp4 = ax3d.boxplot(volume_by_type_nonzero, tick_labels=type_names_sorted, patch_artist=True)
for patch, color in zip(bp4['boxes'], COLORS_11):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax3d.set_xticklabels(type_names_sorted, rotation=45, ha='right', fontsize=9)
ax3d.set_ylabel('Baseline Volume (veh/h, non-zero)', fontsize=10, fontweight='bold')
ax3d.set_title('Traffic Volume by Highway Type (Non-Zero Only)', fontsize=11, fontweight='bold', pad=15)
ax3d.grid(axis='y', alpha=0.3, linestyle='--')

# Add traffic coverage percentage
for i, (code, data) in enumerate(zip(type_codes_sorted, volume_by_type)):
    pct = (np.sum(data != 0) / len(data)) * 100
    ax3d.text(i+1, ax3d.get_ylim()[0], f'{pct:.1f}%', ha='center', va='top', 
             fontsize=7, color='blue', fontweight='bold')

plt.tight_layout()
chart3_path = 'feature4_chart3_relationships.png'
plt.savefig(chart3_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"   ✓ Saved: {chart3_path}")
display(Image(chart3_path))

# ============================================================================
# STEP 3: CHART 4 - COMPREHENSIVE DASHBOARD (12 PANELS)
# ============================================================================
print("\n[3/3] Creating Chart 4: Comprehensive Dashboard...")

fig4 = plt.figure(figsize=(24, 20))

# --- Panel 4A: Type Distribution (Pie) ---
ax4a = plt.subplot(4, 3, 1)
type_counts_sorted = [type_counts[code] for code in type_codes_sorted]
colors_sorted = COLORS_11[:len(type_codes_sorted)]
wedges, texts, autotexts = ax4a.pie(type_counts_sorted, labels=type_names_sorted, 
                                      autopct='%1.1f%%', startangle=90, 
                                      colors=colors_sorted, textprops={'fontsize': 8})
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(7)
ax4a.set_title('Type Distribution', fontsize=10, fontweight='bold')

# --- Panel 4B: Type Counts (Bar) ---
ax4b = plt.subplot(4, 3, 2)
bars = ax4b.barh(range(len(type_names_sorted)), type_counts_sorted, color=colors_sorted, alpha=0.8)
ax4b.set_yticks(range(len(type_names_sorted)))
ax4b.set_yticklabels(type_names_sorted, fontsize=8)
ax4b.set_xlabel('Count', fontsize=9, fontweight='bold')
ax4b.set_title('Type Counts', fontsize=10, fontweight='bold')
ax4b.grid(axis='x', alpha=0.3)
for i, val in enumerate(type_counts_sorted):
    ax4b.text(val, i, f' {val:,}', va='center', fontsize=7)

# --- Panel 4C: Statistics Table ---
ax4c = plt.subplot(4, 3, 3)
ax4c.axis('off')
stats_text = f"""HIGHWAY TYPE STATISTICS

Total Segments: {n_active:,}
Unique Types: {len(np.unique(highway_type))}

Most Common:
  {type_names_sorted[0]}: {type_counts_sorted[0]:,} ({type_counts_sorted[0]/n_active*100:.1f}%)

Least Common:
  {type_names_sorted[-1]}: {type_counts_sorted[-1]:,} ({type_counts_sorted[-1]/n_active*100:.1f}%)

Top 3 Coverage:
  {sum(type_counts_sorted[:3]):,} ({sum(type_counts_sorted[:3])/n_active*100:.1f}%)

Data Quality:
  Missing/Unknown: {type_counts.get(-1, 0):,} ({type_counts.get(-1, 0)/n_active*100:.1f}%)
"""
ax4c.text(0.1, 0.95, stats_text, fontsize=9, verticalalignment='top', 
         family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
ax4c.set_title('Key Statistics', fontsize=10, fontweight='bold')

# --- Panel 4D: Capacity by Type (Mean) ---
ax4d = plt.subplot(4, 3, 4)
cap_means = [np.mean(data) for data in capacity_by_type]
bars = ax4d.barh(range(len(type_names_sorted)), cap_means, color=colors_sorted, alpha=0.8)
ax4d.set_yticks(range(len(type_names_sorted)))
ax4d.set_yticklabels(type_names_sorted, fontsize=8)
ax4d.set_xlabel('Mean Capacity (veh/h)', fontsize=9, fontweight='bold')
ax4d.set_title('Mean Capacity by Type', fontsize=10, fontweight='bold')
ax4d.grid(axis='x', alpha=0.3)
for i, val in enumerate(cap_means):
    ax4d.text(val, i, f' {val:.0f}', va='center', fontsize=7)

# --- Panel 4E: Speed by Type (Mean) ---
ax4e = plt.subplot(4, 3, 5)
speed_means = [np.mean(data) for data in speed_by_type]
bars = ax4e.barh(range(len(type_names_sorted)), speed_means, color=colors_sorted, alpha=0.8)
ax4e.set_yticks(range(len(type_names_sorted)))
ax4e.set_yticklabels(type_names_sorted, fontsize=8)
ax4e.set_xlabel('Mean Free Speed (km/h)', fontsize=9, fontweight='bold')
ax4e.set_title('Mean Speed by Type', fontsize=10, fontweight='bold')
ax4e.grid(axis='x', alpha=0.3)
for i, val in enumerate(speed_means):
    ax4e.text(val, i, f' {val:.1f}', va='center', fontsize=7)

# --- Panel 4F: Length by Type (Mean) ---
ax4f = plt.subplot(4, 3, 6)
length_means = [np.mean(data) for data in length_by_type]
bars = ax4f.barh(range(len(type_names_sorted)), length_means, color=colors_sorted, alpha=0.8)
ax4f.set_yticks(range(len(type_names_sorted)))
ax4f.set_yticklabels(type_names_sorted, fontsize=8)
ax4f.set_xlabel('Mean Length (m)', fontsize=9, fontweight='bold')
ax4f.set_title('Mean Road Length by Type', fontsize=10, fontweight='bold')
ax4f.grid(axis='x', alpha=0.3)
for i, val in enumerate(length_means):
    ax4f.text(val, i, f' {val:.0f}', va='center', fontsize=7)

# --- Panel 4G: Traffic Coverage by Type ---
ax4g = plt.subplot(4, 3, 7)
traffic_pcts = [(np.sum(data != 0) / len(data)) * 100 for data in volume_by_type]
bars = ax4g.barh(range(len(type_names_sorted)), traffic_pcts, color=colors_sorted, alpha=0.8)
ax4g.set_yticks(range(len(type_names_sorted)))
ax4g.set_yticklabels(type_names_sorted, fontsize=8)
ax4g.set_xlabel('Traffic Coverage (%)', fontsize=9, fontweight='bold')
ax4g.set_title('Roads with Traffic by Type', fontsize=10, fontweight='bold')
ax4g.grid(axis='x', alpha=0.3)
ax4g.set_xlim(0, 100)
for i, val in enumerate(traffic_pcts):
    ax4g.text(val, i, f' {val:.1f}%', va='center', fontsize=7)

# --- Panel 4H: Capacity Distribution by Type (Violin) ---
ax4h = plt.subplot(4, 3, 8)
parts = ax4h.violinplot(capacity_by_type, positions=range(len(type_names_sorted)), 
                         showmedians=True, widths=0.7)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors_sorted[i])
    pc.set_alpha(0.7)
ax4h.set_xticks(range(len(type_names_sorted)))
ax4h.set_xticklabels(type_names_sorted, rotation=45, ha='right', fontsize=8)
ax4h.set_ylabel('Capacity (veh/h)', fontsize=9, fontweight='bold')
ax4h.set_title('Capacity Distribution by Type', fontsize=10, fontweight='bold')
ax4h.grid(axis='y', alpha=0.3)

# --- Panel 4I: Speed Distribution by Type (Violin) ---
ax4i = plt.subplot(4, 3, 9)
parts = ax4i.violinplot(speed_by_type, positions=range(len(type_names_sorted)), 
                         showmedians=True, widths=0.7)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors_sorted[i])
    pc.set_alpha(0.7)
ax4i.set_xticks(range(len(type_names_sorted)))
ax4i.set_xticklabels(type_names_sorted, rotation=45, ha='right', fontsize=8)
ax4i.set_ylabel('Free Speed (km/h)', fontsize=9, fontweight='bold')
ax4i.set_title('Speed Distribution by Type', fontsize=10, fontweight='bold')
ax4i.grid(axis='y', alpha=0.3)

# --- Panel 4J: Type Correlation with Features ---
ax4j = plt.subplot(4, 3, 10)
# Calculate point-biserial correlation for each type
correlations = {}
for code in type_codes_sorted:
    type_mask = (highway_type == code).astype(int)
    correlations[HW_MAPPING[code]] = {
        'capacity': stats.pointbiserialr(type_mask, capacity)[0],
        'speed': stats.pointbiserialr(type_mask, free_speed)[0],
        'length': stats.pointbiserialr(type_mask, road_length)[0],
        'baseline': stats.pointbiserialr(type_mask, baseline_volume)[0]
    }

corr_matrix = np.array([[correlations[name]['capacity'], correlations[name]['speed'], 
                         correlations[name]['length'], correlations[name]['baseline']] 
                        for name in type_names_sorted])

im = ax4j.imshow(corr_matrix.T, cmap='RdBu_r', aspect='auto', vmin=-0.5, vmax=0.5)
ax4j.set_xticks(range(len(type_names_sorted)))
ax4j.set_xticklabels(type_names_sorted, rotation=45, ha='right', fontsize=8)
ax4j.set_yticks(range(4))
ax4j.set_yticklabels(['Capacity', 'Speed', 'Length', 'Baseline'], fontsize=8)
ax4j.set_title('Type Correlation with Features', fontsize=10, fontweight='bold')
plt.colorbar(im, ax=ax4j, label='Correlation')

# --- Panel 4K: Capacity vs Speed by Type (Scatter) ---
ax4k = plt.subplot(4, 3, 11)
for i, code in enumerate(type_codes_sorted[:5]):  # Top 5 types only
    mask = highway_type == code
    ax4k.scatter(capacity[mask], free_speed[mask], alpha=0.3, s=10, 
                color=colors_sorted[i], label=HW_MAPPING[code])
ax4k.set_xlabel('Capacity (veh/h)', fontsize=9, fontweight='bold')
ax4k.set_ylabel('Free Speed (km/h)', fontsize=9, fontweight='bold')
ax4k.set_title('Capacity vs Speed (Top 5 Types)', fontsize=10, fontweight='bold')
ax4k.legend(fontsize=7, loc='best')
ax4k.grid(alpha=0.3)

# --- Panel 4L: Key Insights ---
ax4l = plt.subplot(4, 3, 12)
ax4l.axis('off')

# Calculate insights
dominant_type = type_names_sorted[0]
dominant_pct = type_counts_sorted[0] / n_active * 100
highest_cap_type = type_names_sorted[np.argmax(cap_means)]
highest_speed_type = type_names_sorted[np.argmax(speed_means)]
most_traffic_type = type_names_sorted[np.argmax(traffic_pcts)]

insights_text = f"""KEY INSIGHTS

* {dominant_type} dominates the network 
  ({dominant_pct:.1f}% of all roads)

* {highest_cap_type} has highest capacity
  ({max(cap_means):.0f} veh/h average)

* {highest_speed_type} has highest speed
  ({max(speed_means):.1f} km/h average)

* {most_traffic_type} has most traffic
  ({max(traffic_pcts):.1f}% coverage)

* Highway type is STATIC
  (does not change across scenarios)

* Type strongly correlates with
  capacity and speed limits

* Unknown type present in {type_counts.get(-1, 0):,} roads
  ({type_counts.get(-1, 0)/n_active*100:.1f}% of network)
"""

ax4l.text(0.05, 0.95, insights_text, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
ax4l.set_title('Key Insights', fontsize=10, fontweight='bold')

plt.tight_layout()
chart4_path = 'feature4_chart4_comprehensive_dashboard.png'
plt.savefig(chart4_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"   ✓ Saved: {chart4_path}")
display(Image(chart4_path))

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("SUMMARY - FEATURE 4 PART 2 COMPLETE")
print("="*70)
print(f"\nChart 3: Highway Type Relationships (4 panels)")
print(f"  - Type vs Capacity, Speed, Length, Traffic")
print(f"Chart 4: Comprehensive Dashboard (12 panels)")
print(f"  - Distribution, statistics, means, violin plots, correlations, insights")
print(f"\nKey findings:")
print(f"  - {dominant_type} is dominant type ({dominant_pct:.1f}%)")
print(f"  - {highest_cap_type} has highest capacity")
print(f"  - {highest_speed_type} has highest speed")
print(f"  - {most_traffic_type} has most traffic coverage")
print(f"  - Highway type is STATIC (design parameter)")
print("\n" + "="*70)
