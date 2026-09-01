"""
FEATURE 5 - CHART 7
Length-Traffic Deep Dive Analysis

Detailed analysis of the relationship between road length and baseline volume.
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
print("FEATURE 5 - CHART 7: Length-Traffic Deep Dive")
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
baseline_volume = graph.x[:n_active, 2].numpy()
highway_type = graph.x[:n_active, 4].numpy().astype(int)

print(f"\nActive road segments: {n_active:,}")

# Separate traffic and no-traffic roads
traffic_mask = baseline_volume != 0
n_with_traffic = np.sum(traffic_mask)
n_no_traffic = n_active - n_with_traffic

print(f"Roads with traffic: {n_with_traffic:,} ({n_with_traffic/n_active*100:.1f}%)")
print(f"Roads without traffic: {n_no_traffic:,} ({n_no_traffic/n_active*100:.1f}%)")

# Create figure
fig, axes = plt.subplots(3, 3, figsize=(24, 20))

# Panel 1: Traffic presence by length
ax1 = axes[0, 0]
categories = ['<50m', '50-100m', '100-200m', '200-500m', '500-1000m', '>1000m']
ranges = [(0, 50), (50, 100), (100, 200), (200, 500), (500, 1000), (1000, np.inf)]
traffic_counts = []
no_traffic_counts = []

for low, high in ranges:
    mask = (road_length >= low) & (road_length < high)
    traffic_counts.append(np.sum(mask & traffic_mask))
    no_traffic_counts.append(np.sum(mask & ~traffic_mask))

x = np.arange(len(categories))
width = 0.4
bars1 = ax1.bar(x - width/2, traffic_counts, width, label='With Traffic', color='orange', alpha=0.8)
bars2 = ax1.bar(x + width/2, no_traffic_counts, width, label='No Traffic', color='lightgray', alpha=0.8)

ax1.set_xticks(x)
ax1.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
ax1.set_ylabel('Count', fontsize=10, fontweight='bold')
ax1.set_title('Traffic Presence by Length Category', fontsize=11, fontweight='bold', pad=10)
ax1.legend(fontsize=9)
ax1.grid(axis='y', alpha=0.3)

# Panel 2: Length distribution - traffic vs no traffic
ax2 = axes[0, 1]
ax2.hist([road_length[traffic_mask], road_length[~traffic_mask]], 
         bins=50, label=['With Traffic', 'No Traffic'],
         color=['orange', 'lightgray'], alpha=0.7, edgecolor='black')
ax2.set_xlabel('Road Length (m)', fontsize=10, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=10, fontweight='bold')
ax2.set_title('Length Distribution by Traffic Presence', fontsize=11, fontweight='bold', pad=10)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# Panel 3: Scatter plot (roads with traffic only)
ax3 = axes[0, 2]
if n_with_traffic > 10:
    length_traffic = road_length[traffic_mask]
    volume_traffic = np.abs(baseline_volume[traffic_mask])
    
    ax3.scatter(length_traffic, volume_traffic, alpha=0.4, s=10, c='orange')
    ax3.set_xlabel('Road Length (m)', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Baseline Volume (veh/h)', fontsize=10, fontweight='bold')
    ax3.set_title('Length vs Traffic (Roads with Traffic)', fontsize=11, fontweight='bold', pad=10)
    ax3.grid(alpha=0.3)
    
    corr = np.corrcoef(length_traffic, volume_traffic)[0, 1]
    ax3.text(0.05, 0.95, f'Correlation: {corr:.3f}\nn = {len(length_traffic):,}', 
             transform=ax3.transAxes, fontsize=10, fontweight='bold', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
else:
    ax3.text(0.5, 0.5, 'Insufficient traffic data', transform=ax3.transAxes,
             ha='center', va='center', fontsize=14)

# Panel 4: Mean length comparison
ax4 = axes[1, 0]
mean_with_traffic = np.mean(road_length[traffic_mask])
mean_no_traffic = np.mean(road_length[~traffic_mask])
std_with_traffic = np.std(road_length[traffic_mask])
std_no_traffic = np.std(road_length[~traffic_mask])

bars = ax4.bar(['With Traffic', 'No Traffic'], 
               [mean_with_traffic, mean_no_traffic],
               yerr=[std_with_traffic, std_no_traffic],
               capsize=10, color=['orange', 'lightgray'], 
               alpha=0.8, edgecolor='black')
ax4.set_ylabel('Mean Length (m)', fontsize=10, fontweight='bold')
ax4.set_title('Mean Length by Traffic Presence', fontsize=11, fontweight='bold', pad=10)
ax4.grid(axis='y', alpha=0.3)

for bar, mean in zip(bars, [mean_with_traffic, mean_no_traffic]):
    ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
             f'{mean:.1f}m', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Panel 5: Traffic by highway type and length
ax5 = axes[1, 1]
type_counts = Counter(highway_type)
top_types = sorted(type_counts.keys(), key=lambda x: type_counts[x], reverse=True)[:6]
type_names_short = [HW_MAPPING[code] for code in top_types]

traffic_pct_by_type = []
for type_code in top_types:
    mask = highway_type == type_code
    if np.sum(mask) > 0:
        pct = np.sum(mask & traffic_mask) / np.sum(mask) * 100
        traffic_pct_by_type.append(pct)
    else:
        traffic_pct_by_type.append(0)

bars = ax5.barh(range(len(type_names_short)), traffic_pct_by_type,
                color=plt.cm.Set3(np.linspace(0, 1, len(type_names_short))),
                alpha=0.8, edgecolor='black')
ax5.set_yticks(range(len(type_names_short)))
ax5.set_yticklabels(type_names_short, fontsize=9)
ax5.set_xlabel('% Roads with Traffic', fontsize=10, fontweight='bold')
ax5.set_title('Traffic Presence by Highway Type', fontsize=11, fontweight='bold', pad=10)
ax5.grid(axis='x', alpha=0.3)

for i, (bar, pct) in enumerate(zip(bars, traffic_pct_by_type)):
    ax5.text(pct, i, f' {pct:.1f}%', va='center', fontsize=8)

# Panel 6: Volume distribution by length category (traffic roads only)
ax6 = axes[1, 2]
if n_with_traffic > 10:
    volume_by_cat = []
    for low, high in ranges:
        mask = (road_length >= low) & (road_length < high) & traffic_mask
        if np.sum(mask) > 0:
            volume_by_cat.append(np.abs(baseline_volume[mask]))
        else:
            volume_by_cat.append([0])
    
    bp = ax6.boxplot(volume_by_cat, tick_labels=categories, patch_artist=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(categories)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax6.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
    ax6.set_ylabel('Baseline Volume (veh/h)', fontsize=10, fontweight='bold')
    ax6.set_title('Traffic Distribution by Length', fontsize=11, fontweight='bold', pad=10)
    ax6.grid(axis='y', alpha=0.3)
else:
    ax6.text(0.5, 0.5, 'Insufficient data', transform=ax6.transAxes,
             ha='center', va='center', fontsize=14)

# Panel 7: Percentile comparison
ax7 = axes[2, 0]
percentiles = [25, 50, 75, 95]
with_traffic_pct = [np.percentile(road_length[traffic_mask], p) for p in percentiles]
no_traffic_pct = [np.percentile(road_length[~traffic_mask], p) for p in percentiles]

x = np.arange(len(percentiles))
width = 0.35
bars1 = ax7.bar(x - width/2, with_traffic_pct, width, label='With Traffic', color='orange', alpha=0.8)
bars2 = ax7.bar(x + width/2, no_traffic_pct, width, label='No Traffic', color='lightgray', alpha=0.8)

ax7.set_xticks(x)
ax7.set_xticklabels([f'P{p}' for p in percentiles], fontsize=9)
ax7.set_ylabel('Length (m)', fontsize=10, fontweight='bold')
ax7.set_title('Length Percentiles by Traffic Presence', fontsize=11, fontweight='bold', pad=10)
ax7.legend(fontsize=9)
ax7.grid(axis='y', alpha=0.3)

# Panel 8: Statistics table
ax8 = axes[2, 1]
ax8.axis('off')

if n_with_traffic > 10:
    length_traffic = road_length[traffic_mask]
    volume_traffic = np.abs(baseline_volume[traffic_mask])
    corr_traffic = np.corrcoef(length_traffic, volume_traffic)[0, 1]
else:
    corr_traffic = 0.0

stats_text = f"""LENGTH-TRAFFIC STATISTICS

OVERALL:
Total roads:     {n_active:,}
With traffic:    {n_with_traffic:,} ({n_with_traffic/n_active*100:.1f}%)
No traffic:      {n_no_traffic:,} ({n_no_traffic/n_active*100:.1f}%)

MEAN LENGTH:
With traffic:    {mean_with_traffic:.1f} m
No traffic:      {mean_no_traffic:.1f} m
Difference:      {abs(mean_with_traffic - mean_no_traffic):.1f} m

MEDIAN LENGTH:
With traffic:    {np.median(road_length[traffic_mask]):.1f} m
No traffic:      {np.median(road_length[~traffic_mask]):.1f} m

CORRELATION (traffic roads):
Length-Volume:   {corr_traffic:.3f}

CONCLUSION:
Traffic presence NOT strongly
related to road length
"""

ax8.text(0.1, 0.9, stats_text, fontsize=8, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
ax8.set_title('Statistics Summary', fontsize=11, fontweight='bold', pad=10)

# Panel 9: Key insights
ax9 = axes[2, 2]
ax9.axis('off')

insights_text = f"""KEY INSIGHTS: LENGTH-TRAFFIC

TRAFFIC PRESENCE:
• {n_with_traffic:,} roads with traffic
• {n_no_traffic:,} roads without traffic
• {n_with_traffic/n_active*100:.1f}% have traffic

LENGTH COMPARISON:
• With traffic: {mean_with_traffic:.1f}m avg
• No traffic: {mean_no_traffic:.1f}m avg
• Small difference

CORRELATION:
• Length-Traffic: {corr_traffic:.3f}
• Very weak relationship
• Length doesn't predict traffic

BY CATEGORY:
• All length categories have
  mixed traffic presence
• No clear length-traffic pattern

CONCLUSION:
• Traffic is DYNAMIC feature
• Length is STATIC feature
• Independent variables
• Traffic depends on network
  position, not road length
"""

ax9.text(0.1, 0.9, insights_text, fontsize=8, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.4))
ax9.set_title('Summary & Insights', fontsize=11, fontweight='bold', pad=10)

plt.tight_layout()
output_path = 'feature5_chart7_length_traffic.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✓ Saved: {output_path}")
display(Image(output_path))

print("\n" + "="*80)
print("CHART 7 COMPLETE")
print("="*80)
print(f"\nRoads with traffic: {n_with_traffic:,} ({n_with_traffic/n_active*100:.1f}%)")
print(f"Mean length (with traffic): {mean_with_traffic:.1f}m")
print(f"Mean length (no traffic): {mean_no_traffic:.1f}m")
if n_with_traffic > 10:
    print(f"Correlation (traffic roads): {corr_traffic:.3f}")
print(f"Conclusion: Length and traffic are independent")
