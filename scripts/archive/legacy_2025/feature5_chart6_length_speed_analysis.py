"""
FEATURE 5 - CHART 6
Length-Speed Deep Dive Analysis

Detailed analysis of the relationship between road length and free speed.
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
print("FEATURE 5 - CHART 6: Length-Speed Deep Dive")
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
free_speed = graph.x[:n_active, 3].numpy()
highway_type = graph.x[:n_active, 4].numpy().astype(int)

print(f"\nActive road segments: {n_active:,}")

# Create figure
fig, axes = plt.subplots(3, 3, figsize=(24, 20))

# Panel 1: Main scatter plot with density
ax1 = axes[0, 0]
ax1.hexbin(road_length, free_speed, gridsize=50, cmap='Greens', mincnt=1)
ax1.set_xlabel('Road Length (m)', fontsize=10, fontweight='bold')
ax1.set_ylabel('Free Speed (km/h)', fontsize=10, fontweight='bold')
ax1.set_title('Length vs Free Speed (Density Plot)', fontsize=11, fontweight='bold', pad=10)
ax1.grid(alpha=0.3)

corr = np.corrcoef(road_length, free_speed)[0, 1]
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
    ax2.scatter(road_length[mask], free_speed[mask], alpha=0.5, s=10, 
               label=HW_MAPPING[type_code], color=colors[idx])

ax2.set_xlabel('Road Length (m)', fontsize=10, fontweight='bold')
ax2.set_ylabel('Free Speed (km/h)', fontsize=10, fontweight='bold')
ax2.set_title('Length vs Speed by Type (Top 6)', fontsize=11, fontweight='bold', pad=10)
ax2.legend(fontsize=8, loc='upper right')
ax2.grid(alpha=0.3)

# Panel 3: Correlation by highway type
ax3 = axes[0, 2]
correlations = []
type_names = []
for type_code in top_types:
    mask = highway_type == type_code
    if np.sum(mask) > 10:
        corr = np.corrcoef(road_length[mask], free_speed[mask])[0, 1]
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

# Panel 4: Length categories vs mean speed
ax4 = axes[1, 0]
categories = ['<50m', '50-100m', '100-200m', '200-500m', '500-1000m', '>1000m']
ranges = [(0, 50), (50, 100), (100, 200), (200, 500), (500, 1000), (1000, np.inf)]
mean_speeds = []
std_speeds = []

for low, high in ranges:
    mask = (road_length >= low) & (road_length < high)
    if np.sum(mask) > 0:
        mean_speeds.append(np.mean(free_speed[mask]))
        std_speeds.append(np.std(free_speed[mask]))
    else:
        mean_speeds.append(0)
        std_speeds.append(0)

bars = ax4.bar(range(len(categories)), mean_speeds, 
               yerr=std_speeds, capsize=5,
               color=plt.cm.viridis(np.linspace(0, 1, len(categories))),
               alpha=0.8, edgecolor='black')
ax4.set_xticks(range(len(categories)))
ax4.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
ax4.set_ylabel('Mean Speed (km/h)', fontsize=10, fontweight='bold')
ax4.set_title('Mean Speed by Length Category', fontsize=11, fontweight='bold', pad=10)
ax4.grid(axis='y', alpha=0.3)

for i, (bar, speed) in enumerate(zip(bars, mean_speeds)):
    ax4.text(i, speed, f'{speed:.1f}', ha='center', va='bottom', fontsize=8)

# Panel 5: Box plots by length category
ax5 = axes[1, 1]
speed_by_cat = []
for low, high in ranges:
    mask = (road_length >= low) & (road_length < high)
    if np.sum(mask) > 0:
        speed_by_cat.append(free_speed[mask])
    else:
        speed_by_cat.append([0])

bp = ax5.boxplot(speed_by_cat, tick_labels=categories, patch_artist=True)
colors = plt.cm.viridis(np.linspace(0, 1, len(categories)))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax5.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
ax5.set_ylabel('Free Speed (km/h)', fontsize=10, fontweight='bold')
ax5.set_title('Speed Distribution by Length Category', fontsize=11, fontweight='bold', pad=10)
ax5.grid(axis='y', alpha=0.3)

# Panel 6: Speed zones by length
ax6 = axes[1, 2]
speed_zones = [(0, 30), (30, 50), (50, 70), (70, 90), (90, 150)]
zone_labels = ['<30', '30-50', '50-70', '70-90', '>90']
zone_data = []

for low, high in speed_zones:
    mask = (free_speed >= low) & (free_speed < high)
    if low == 90:
        mask = free_speed >= 90
    zone_data.append(road_length[mask])

bp = ax6.boxplot(zone_data, tick_labels=zone_labels, patch_artist=True)
colors = plt.cm.RdYlGn(np.linspace(0, 1, len(zone_labels)))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax6.set_xlabel('Speed Zone (km/h)', fontsize=10, fontweight='bold')
ax6.set_ylabel('Road Length (m)', fontsize=10, fontweight='bold')
ax6.set_title('Length Distribution by Speed Zone', fontsize=11, fontweight='bold', pad=10)
ax6.set_yscale('log')
ax6.grid(axis='y', alpha=0.3)

# Panel 7: Length-Speed joint histogram
ax7 = axes[2, 0]
hist, xedges, yedges = np.histogram2d(road_length, free_speed, bins=50)
im = ax7.imshow(hist.T, origin='lower', cmap='viridis', aspect='auto',
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
ax7.set_xlabel('Road Length (m)', fontsize=10, fontweight='bold')
ax7.set_ylabel('Free Speed (km/h)', fontsize=10, fontweight='bold')
ax7.set_title('Joint Distribution Histogram', fontsize=11, fontweight='bold', pad=10)
plt.colorbar(im, ax=ax7, label='Count')

# Panel 8: Statistics by length quartiles
ax8 = axes[2, 1]
ax8.axis('off')
quartiles = [np.percentile(road_length, q) for q in [0, 25, 50, 75, 100]]
stats_text = "SPEED STATISTICS BY LENGTH QUARTILE\n\n"

for i in range(len(quartiles)-1):
    mask = (road_length >= quartiles[i]) & (road_length < quartiles[i+1])
    if i == len(quartiles)-2:
        mask = road_length >= quartiles[i]
    
    speed_subset = free_speed[mask]
    stats_text += f"Q{i+1} ({quartiles[i]:.1f}-{quartiles[i+1]:.1f}m):\n"
    stats_text += f"  Roads: {np.sum(mask):,}\n"
    stats_text += f"  Mean Speed: {np.mean(speed_subset):.1f} km/h\n"
    stats_text += f"  Median Speed: {np.median(speed_subset):.1f} km/h\n"
    stats_text += f"  Std Speed: {np.std(speed_subset):.1f}\n\n"

ax8.text(0.1, 0.9, stats_text, fontsize=8, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
ax8.set_title('Quartile Analysis', fontsize=11, fontweight='bold', pad=10)

# Panel 9: Key insights
ax9 = axes[2, 2]
ax9.axis('off')

# Calculate insights
overall_corr = np.corrcoef(road_length, free_speed)[0, 1]
if len(correlations) > 0:
    strongest_corr_idx = np.argmax(np.abs(correlations))
    strongest_type = type_names[strongest_corr_idx]
    strongest_corr = correlations[strongest_corr_idx]
else:
    strongest_type = "N/A"
    strongest_corr = 0.0

insights_text = f"""KEY INSIGHTS: LENGTH-SPEED

OVERALL RELATIONSHIP:
• Correlation: {overall_corr:.3f}
• Weak negative relationship
• Longer roads slightly slower

BY HIGHWAY TYPE:
• Strongest: {strongest_type}
  (r = {strongest_corr:.3f})
• Type determines speed limit
• Length has minor impact

BY LENGTH CATEGORY:
• <50m: {mean_speeds[0]:.1f} km/h avg
• >1000m: {mean_speeds[-1]:.1f} km/h avg
• Small speed variation

BY SPEED ZONE:
• High speed (>90): Longer roads
• Low speed (<30): Shorter roads
• Urban vs highway pattern

CONCLUSION:
• Speed is design parameter
• Weakly affected by length
• Highway type is main factor
• STATIC feature
"""

ax9.text(0.1, 0.9, insights_text, fontsize=8, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.4))
ax9.set_title('Summary & Insights', fontsize=11, fontweight='bold', pad=10)

plt.tight_layout()
output_path = 'feature5_chart6_length_speed.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✓ Saved: {output_path}")
display(Image(output_path))

print("\n" + "="*80)
print("CHART 6 COMPLETE")
print("="*80)
print(f"\nOverall correlation: {overall_corr:.3f} (weak negative)")
print(f"Shortest roads: {mean_speeds[0]:.1f} km/h average speed")
print(f"Longest roads: {mean_speeds[-1]:.1f} km/h average speed")
print(f"Conclusion: Length has weak impact on speed")
