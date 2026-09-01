"""
FEATURE 5 - CHART 5
Length-Capacity Deep Dive Analysis

Detailed analysis of the relationship between road length and capacity.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from scipy import stats
from IPython.display import Image, display

print("\n" + "="*80)
print("FEATURE 5 - CHART 5: Length-Capacity Deep Dive")
print("="*80)

# Setup
data_dir = Path('/content/drive/MyDrive/Zamin_thesis/ml_surrogates_for_agent_based_transport_models/data/train_data/dist_not_connected_10k_1pct')
batch_path = data_dir / 'datalist_batch_1.pt'

HW_MAPPING = {
    -1: 'Unknown', 0: 'Motorway', 1: 'Trunk', 2: 'Primary', 3: 'Secondary',
    4: 'Tertiary', 5: 'Residential', 6: 'PT', 7: 'Service', 
    8: 'Living Street', 9: 'Motorway Link'
}

# Load data
graphs_list = torch.load(batch_path, weights_only=False)
graph = graphs_list[0]

n_active = 31635
road_length = graph.x[:n_active, 5].numpy()
capacity = graph.x[:n_active, 1].numpy()
highway_type = graph.x[:n_active, 4].numpy().astype(int)

print(f"\nActive road segments: {n_active:,}")

# Create figure
fig, axes = plt.subplots(3, 3, figsize=(24, 20))

# Panel 1: Main scatter plot with density
ax1 = axes[0, 0]
ax1.hexbin(road_length, capacity, gridsize=50, cmap='Blues', mincnt=1)
ax1.set_xlabel('Road Length (m)', fontsize=10, fontweight='bold')
ax1.set_ylabel('Capacity (veh/h)', fontsize=10, fontweight='bold')
ax1.set_title('Length vs Capacity (Density Plot)', fontsize=11, fontweight='bold', pad=10)
ax1.grid(alpha=0.3)

corr = np.corrcoef(road_length, capacity)[0, 1]
ax1.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax1.transAxes,
         fontsize=10, fontweight='bold', verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

# Panel 2: By highway type
ax2 = axes[0, 1]
type_counts = Counter(highway_type)
top_types = sorted(type_counts.keys(), key=lambda x: type_counts[x], reverse=True)[:6]
colors = plt.cm.Set3(np.linspace(0, 1, 6))

for idx, type_code in enumerate(top_types):
    mask = highway_type == type_code
    ax2.scatter(road_length[mask], capacity[mask], alpha=0.5, s=10, 
               label=HW_MAPPING[type_code], color=colors[idx])

ax2.set_xlabel('Road Length (m)', fontsize=10, fontweight='bold')
ax2.set_ylabel('Capacity (veh/h)', fontsize=10, fontweight='bold')
ax2.set_title('Length vs Capacity by Type (Top 6)', fontsize=11, fontweight='bold', pad=10)
ax2.legend(fontsize=8, loc='upper right')
ax2.grid(alpha=0.3)

# Panel 3: Correlation by highway type
ax3 = axes[0, 2]
correlations = []
type_names = []
for type_code in top_types:
    mask = highway_type == type_code
    if np.sum(mask) > 10:
        corr = np.corrcoef(road_length[mask], capacity[mask])[0, 1]
        correlations.append(corr)
        type_names.append(HW_MAPPING[type_code])

bars = ax3.barh(range(len(type_names)), correlations, 
                color=['green' if c > 0.3 else 'orange' if c > 0.1 else 'red' for c in correlations],
                alpha=0.7, edgecolor='black')
ax3.set_yticks(range(len(type_names)))
ax3.set_yticklabels(type_names, fontsize=9)
ax3.set_xlabel('Correlation', fontsize=10, fontweight='bold')
ax3.set_title('Correlation by Highway Type', fontsize=11, fontweight='bold', pad=10)
ax3.axvline(0, color='black', linewidth=1)
ax3.grid(axis='x', alpha=0.3)

for i, (bar, corr) in enumerate(zip(bars, correlations)):
    ax3.text(corr, i, f' {corr:.3f}', va='center', fontsize=8, fontweight='bold')

# Panel 4: Length categories vs mean capacity
ax4 = axes[1, 0]
categories = ['<50m', '50-100m', '100-200m', '200-500m', '500-1000m', '>1000m']
ranges = [(0, 50), (50, 100), (100, 200), (200, 500), (500, 1000), (1000, np.inf)]
mean_capacities = []
std_capacities = []

for low, high in ranges:
    mask = (road_length >= low) & (road_length < high)
    if np.sum(mask) > 0:
        mean_capacities.append(np.mean(capacity[mask]))
        std_capacities.append(np.std(capacity[mask]))
    else:
        mean_capacities.append(0)
        std_capacities.append(0)

bars = ax4.bar(range(len(categories)), mean_capacities, 
               yerr=std_capacities, capsize=5,
               color=plt.cm.viridis(np.linspace(0, 1, len(categories))),
               alpha=0.8, edgecolor='black')
ax4.set_xticks(range(len(categories)))
ax4.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
ax4.set_ylabel('Mean Capacity (veh/h)', fontsize=10, fontweight='bold')
ax4.set_title('Mean Capacity by Length Category', fontsize=11, fontweight='bold', pad=10)
ax4.grid(axis='y', alpha=0.3)

for i, (bar, cap) in enumerate(zip(bars, mean_capacities)):
    ax4.text(i, cap, f'{cap:.0f}', ha='center', va='bottom', fontsize=8)

# Panel 5: Box plots by length category
ax5 = axes[1, 1]
capacity_by_cat = []
for low, high in ranges:
    mask = (road_length >= low) & (road_length < high)
    if np.sum(mask) > 0:
        capacity_by_cat.append(capacity[mask])
    else:
        capacity_by_cat.append([0])

bp = ax5.boxplot(capacity_by_cat, tick_labels=categories, patch_artist=True)
colors = plt.cm.viridis(np.linspace(0, 1, len(categories)))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax5.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
ax5.set_ylabel('Capacity (veh/h)', fontsize=10, fontweight='bold')
ax5.set_title('Capacity Distribution by Length Category', fontsize=11, fontweight='bold', pad=10)
ax5.grid(axis='y', alpha=0.3)

# Panel 6: Length bins vs capacity bins heatmap
ax6 = axes[1, 2]
length_bins = [0, 50, 100, 200, 500, 1000, 3000]
capacity_bins = [0, 500, 1000, 1500, 2000, 2500, 3000]

heatmap_data = np.zeros((len(capacity_bins)-1, len(length_bins)-1))
for i in range(len(length_bins)-1):
    for j in range(len(capacity_bins)-1):
        mask = ((road_length >= length_bins[i]) & (road_length < length_bins[i+1]) &
                (capacity >= capacity_bins[j]) & (capacity < capacity_bins[j+1]))
        heatmap_data[j, i] = np.sum(mask)

im = ax6.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
ax6.set_xticks(range(len(length_bins)-1))
ax6.set_yticks(range(len(capacity_bins)-1))
ax6.set_xticklabels([f'{length_bins[i]}-{length_bins[i+1]}' for i in range(len(length_bins)-1)], 
                     rotation=45, ha='right', fontsize=8)
ax6.set_yticklabels([f'{capacity_bins[i]}-{capacity_bins[i+1]}' for i in range(len(capacity_bins)-1)], 
                     fontsize=8)
ax6.set_xlabel('Length (m)', fontsize=10, fontweight='bold')
ax6.set_ylabel('Capacity (veh/h)', fontsize=10, fontweight='bold')
ax6.set_title('Joint Distribution Heatmap', fontsize=11, fontweight='bold', pad=10)
plt.colorbar(im, ax=ax6, label='Count')

# Panel 7: Residuals plot
ax7 = axes[2, 0]
z = np.polyfit(road_length, capacity, 1)
p = np.poly1d(z)
predicted = p(road_length)
residuals = capacity - predicted

ax7.scatter(road_length, residuals, alpha=0.3, s=3, c='steelblue')
ax7.axhline(0, color='red', linestyle='--', linewidth=2)
ax7.set_xlabel('Road Length (m)', fontsize=10, fontweight='bold')
ax7.set_ylabel('Residuals', fontsize=10, fontweight='bold')
ax7.set_title('Residual Plot (Linear Fit)', fontsize=11, fontweight='bold', pad=10)
ax7.grid(alpha=0.3)

# Panel 8: Statistics by length quartiles
ax8 = axes[2, 1]
ax8.axis('off')
quartiles = [np.percentile(road_length, q) for q in [0, 25, 50, 75, 100]]
stats_text = "CAPACITY STATISTICS BY LENGTH QUARTILE\n\n"

for i in range(len(quartiles)-1):
    mask = (road_length >= quartiles[i]) & (road_length < quartiles[i+1])
    if i == len(quartiles)-2:
        mask = road_length >= quartiles[i]
    
    cap_subset = capacity[mask]
    stats_text += f"Q{i+1} ({quartiles[i]:.1f}-{quartiles[i+1]:.1f}m):\n"
    stats_text += f"  Roads: {np.sum(mask):,}\n"
    stats_text += f"  Mean Cap: {np.mean(cap_subset):.1f} veh/h\n"
    stats_text += f"  Median Cap: {np.median(cap_subset):.1f} veh/h\n"
    stats_text += f"  Std Cap: {np.std(cap_subset):.1f}\n\n"

ax8.text(0.1, 0.9, stats_text, fontsize=8, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
ax8.set_title('Quartile Analysis', fontsize=11, fontweight='bold', pad=10)

# Panel 9: Key insights
ax9 = axes[2, 2]
ax9.axis('off')

# Calculate insights
overall_corr = np.corrcoef(road_length, capacity)[0, 1]
strongest_corr_idx = np.argmax(np.abs(correlations))
strongest_type = type_names[strongest_corr_idx]
strongest_corr = correlations[strongest_corr_idx]

insights_text = f"""KEY INSIGHTS: LENGTH-CAPACITY

OVERALL RELATIONSHIP:
• Correlation: {overall_corr:.3f}
• Very weak relationship
• Length does NOT predict capacity

BY HIGHWAY TYPE:
• Strongest: {strongest_type}
  (r = {strongest_corr:.3f})
• Type matters more than length
• Each type has typical capacity

BY LENGTH CATEGORY:
• <50m: {mean_capacities[0]:.0f} veh/h avg
• >1000m: {mean_capacities[-1]:.0f} veh/h avg
• No clear length-capacity pattern

CONCLUSION:
• Road capacity is design parameter
• Independent of physical length
• Determined by road type & lanes
• STATIC feature
"""

ax9.text(0.1, 0.9, insights_text, fontsize=8, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.4))
ax9.set_title('Summary & Insights', fontsize=11, fontweight='bold', pad=10)

plt.tight_layout()
output_path = 'feature5_chart5_length_capacity.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✓ Saved: {output_path}")
display(Image(output_path))

print("\n" + "="*80)
print("CHART 5 COMPLETE")
print("="*80)
print(f"\nOverall correlation: {overall_corr:.3f} (very weak)")
print(f"Strongest type correlation: {strongest_type} ({strongest_corr:.3f})")
print(f"Conclusion: Length does NOT determine capacity")
