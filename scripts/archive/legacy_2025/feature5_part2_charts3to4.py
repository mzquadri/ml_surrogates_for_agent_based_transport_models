"""
FEATURE 5 - PART 2 (CHARTS 3-4)
Road Length Relationships and Dashboard

This script creates Charts 3-4 for Feature 5 (Road Length):
- Chart 3: Relationships with Other Features (scatter plots, correlations)
- Chart 4: Comprehensive Dashboard (12-panel overview)
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
print("FEATURE 5: ROAD LENGTH - PART 2")
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
highway_type = graph.x[:n_active, 4].numpy().astype(int)
capacity = graph.x[:n_active, 1].numpy()
baseline_volume = graph.x[:n_active, 2].numpy()
free_speed = graph.x[:n_active, 3].numpy()

print(f"\nLoaded scenario 1 from batch")
print(f"Active road segments: {n_active:,}")

# ============================================================================
# CHART 3: RELATIONSHIPS WITH OTHER FEATURES
# ============================================================================
print("\n" + "-"*80)
print("CHART 3: Road Length Relationships")
print("-"*80)

fig, axes = plt.subplots(2, 2, figsize=(20, 16))

# Panel 3A: Length vs Capacity scatter
ax1 = axes[0, 0]
ax1.scatter(road_length, capacity, alpha=0.3, s=5, c='steelblue')
ax1.set_xlabel('Road Length (m)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Capacity (veh/h)', fontsize=11, fontweight='bold')
ax1.set_title('Road Length vs Capacity', fontsize=12, fontweight='bold', pad=15)
ax1.grid(alpha=0.3)

# Add correlation
corr = np.corrcoef(road_length, capacity)[0, 1]
ax1.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax1.transAxes,
         fontsize=11, fontweight='bold', verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Add trend line
z = np.polyfit(road_length, capacity, 1)
p = np.poly1d(z)
x_trend = np.linspace(road_length.min(), road_length.max(), 100)
ax1.plot(x_trend, p(x_trend), "r--", linewidth=2, label='Trend line')
ax1.legend(fontsize=9)

# Panel 3B: Length vs Free Speed scatter
ax2 = axes[0, 1]
ax2.scatter(road_length, free_speed, alpha=0.3, s=5, c='darkgreen')
ax2.set_xlabel('Road Length (m)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Free Speed (km/h)', fontsize=11, fontweight='bold')
ax2.set_title('Road Length vs Free Speed', fontsize=12, fontweight='bold', pad=15)
ax2.grid(alpha=0.3)

# Add correlation
corr = np.corrcoef(road_length, free_speed)[0, 1]
ax2.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax2.transAxes,
         fontsize=11, fontweight='bold', verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Add trend line
z = np.polyfit(road_length, free_speed, 1)
p = np.poly1d(z)
ax2.plot(x_trend, p(x_trend), "r--", linewidth=2, label='Trend line')
ax2.legend(fontsize=9)

# Panel 3C: Length vs Traffic (for roads with traffic)
ax3 = axes[1, 0]
traffic_mask = baseline_volume != 0
if np.sum(traffic_mask) > 10:
    length_traffic = road_length[traffic_mask]
    volume_traffic = baseline_volume[traffic_mask]
    
    ax3.scatter(length_traffic, np.abs(volume_traffic), alpha=0.4, s=10, c='orange')
    ax3.set_xlabel('Road Length (m)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Baseline Volume (veh/h)', fontsize=11, fontweight='bold')
    ax3.set_title('Road Length vs Traffic (Roads with Traffic Only)', fontsize=12, fontweight='bold', pad=15)
    ax3.grid(alpha=0.3)
    
    # Add correlation
    corr = np.corrcoef(length_traffic, np.abs(volume_traffic))[0, 1]
    ax3.text(0.05, 0.95, f'Correlation: {corr:.3f}\nSample: {len(length_traffic):,} roads', 
             transform=ax3.transAxes, fontsize=11, fontweight='bold', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
else:
    ax3.text(0.5, 0.5, 'Insufficient traffic data', transform=ax3.transAxes,
             ha='center', va='center', fontsize=14)

# Panel 3D: Correlation heatmap
ax4 = axes[1, 1]
features = ['Road Length', 'Capacity', 'Free Speed', 'Baseline Vol']
feature_data = np.column_stack([road_length, capacity, free_speed, baseline_volume])
corr_matrix = np.corrcoef(feature_data.T)

im = ax4.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
ax4.set_xticks(range(len(features)))
ax4.set_yticks(range(len(features)))
ax4.set_xticklabels(features, rotation=45, ha='right', fontsize=10)
ax4.set_yticklabels(features, fontsize=10)
ax4.set_title('Feature Correlation Matrix', fontsize=12, fontweight='bold', pad=15)

# Add correlation values
for i in range(len(features)):
    for j in range(len(features)):
        text = ax4.text(j, i, f'{corr_matrix[i, j]:.3f}',
                       ha="center", va="center", color="black" if abs(corr_matrix[i, j]) < 0.5 else "white",
                       fontsize=10, fontweight='bold')

plt.colorbar(im, ax=ax4, label='Correlation')

plt.tight_layout()
chart3_path = 'feature5_chart3_relationships.png'
plt.savefig(chart3_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Saved: {chart3_path}")
display(Image(chart3_path))

# ============================================================================
# CHART 4: COMPREHENSIVE DASHBOARD
# ============================================================================
print("\n" + "-"*80)
print("CHART 4: Comprehensive Dashboard")
print("-"*80)

fig = plt.figure(figsize=(24, 20))

# Calculate statistics
type_counts = Counter(highway_type)
type_codes_sorted = sorted(type_counts.keys(), key=lambda x: type_counts[x], reverse=True)
type_names_sorted = [HW_MAPPING[code] for code in type_codes_sorted]
length_by_type = [road_length[highway_type == code] for code in type_codes_sorted]

# Panel 1: Distribution histogram
ax1 = plt.subplot(3, 4, 1)
ax1.hist(road_length, bins=80, alpha=0.7, color='steelblue', edgecolor='black')
ax1.axvline(np.mean(road_length), color='red', linestyle='--', linewidth=2, label='Mean')
ax1.axvline(np.median(road_length), color='orange', linestyle='--', linewidth=2, label='Median')
ax1.set_xlabel('Road Length (m)', fontsize=9, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=9, fontweight='bold')
ax1.set_title('Length Distribution', fontsize=10, fontweight='bold')
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

# Panel 2: CDF
ax2 = plt.subplot(3, 4, 2)
sorted_lengths = np.sort(road_length)
cdf = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)
ax2.plot(sorted_lengths, cdf, linewidth=2, color='darkblue')
ax2.set_xlabel('Road Length (m)', fontsize=9, fontweight='bold')
ax2.set_ylabel('CDF', fontsize=9, fontweight='bold')
ax2.set_title('Cumulative Distribution', fontsize=10, fontweight='bold')
ax2.grid(alpha=0.3)

# Panel 3: Box plot by type (top 5)
ax3 = plt.subplot(3, 4, 3)
top5_data = length_by_type[:5]
top5_names = type_names_sorted[:5]
bp = ax3.boxplot(top5_data, tick_labels=top5_names, patch_artist=True)
colors = plt.cm.Set3(np.linspace(0, 1, 5))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax3.set_xticklabels(top5_names, rotation=45, ha='right', fontsize=8)
ax3.set_ylabel('Length (m)', fontsize=9, fontweight='bold')
ax3.set_title('Length by Type (Top 5)', fontsize=10, fontweight='bold')
ax3.set_yscale('log')
ax3.grid(axis='y', alpha=0.3)

# Panel 4: Statistics table
ax4 = plt.subplot(3, 4, 4)
ax4.axis('off')
stats_text = f"""ROAD LENGTH STATISTICS

Count:     {len(road_length):,} roads
Mean:      {np.mean(road_length):.1f} m
Median:    {np.median(road_length):.1f} m
Std Dev:   {np.std(road_length):.1f} m

Min:       {road_length.min():.1f} m
Max:       {road_length.max():.1f} m
Range:     {road_length.max() - road_length.min():.1f} m

P25:       {np.percentile(road_length, 25):.1f} m
P75:       {np.percentile(road_length, 75):.1f} m
P95:       {np.percentile(road_length, 95):.1f} m

Skewness:  {stats.skew(road_length):.3f}
Kurtosis:  {stats.kurtosis(road_length):.3f}
"""
ax4.text(0.1, 0.9, stats_text, fontsize=9, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
ax4.set_title('Key Statistics', fontsize=10, fontweight='bold', pad=15)

# Panel 5: Mean length by type
ax5 = plt.subplot(3, 4, 5)
mean_lengths = [np.mean(data) for data in length_by_type]
bars = ax5.barh(range(len(type_names_sorted)), mean_lengths, 
                color=plt.cm.Set3(np.linspace(0, 1, len(type_names_sorted))), alpha=0.8)
ax5.set_yticks(range(len(type_names_sorted)))
ax5.set_yticklabels(type_names_sorted, fontsize=8)
ax5.set_xlabel('Mean Length (m)', fontsize=9, fontweight='bold')
ax5.set_title('Mean Length by Type', fontsize=10, fontweight='bold')
ax5.grid(axis='x', alpha=0.3)

# Panel 6: Length categories
ax6 = plt.subplot(3, 4, 6)
categories = ['<50m', '50-100m', '100-200m', '200-500m', '500-1000m', '>1000m']
ranges = [(0, 50), (50, 100), (100, 200), (200, 500), (500, 1000), (1000, np.inf)]
counts = [np.sum((road_length >= low) & (road_length < high)) for low, high in ranges]
colors_cat = plt.cm.viridis(np.linspace(0, 1, len(categories)))
bars = ax6.bar(range(len(categories)), counts, color=colors_cat, alpha=0.8, edgecolor='black')
ax6.set_xticks(range(len(categories)))
ax6.set_xticklabels(categories, rotation=45, ha='right', fontsize=8)
ax6.set_ylabel('Count', fontsize=9, fontweight='bold')
ax6.set_title('Length Categories', fontsize=10, fontweight='bold')
ax6.grid(axis='y', alpha=0.3)

# Panel 7: Length vs Capacity scatter
ax7 = plt.subplot(3, 4, 7)
ax7.scatter(road_length, capacity, alpha=0.2, s=3, c='steelblue')
ax7.set_xlabel('Length (m)', fontsize=9, fontweight='bold')
ax7.set_ylabel('Capacity (veh/h)', fontsize=9, fontweight='bold')
ax7.set_title('Length vs Capacity', fontsize=10, fontweight='bold')
ax7.grid(alpha=0.3)
corr = np.corrcoef(road_length, capacity)[0, 1]
ax7.text(0.05, 0.95, f'r={corr:.3f}', transform=ax7.transAxes, fontsize=9,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Panel 8: Length vs Speed scatter
ax8 = plt.subplot(3, 4, 8)
ax8.scatter(road_length, free_speed, alpha=0.2, s=3, c='darkgreen')
ax8.set_xlabel('Length (m)', fontsize=9, fontweight='bold')
ax8.set_ylabel('Speed (km/h)', fontsize=9, fontweight='bold')
ax8.set_title('Length vs Speed', fontsize=10, fontweight='bold')
ax8.grid(alpha=0.3)
corr = np.corrcoef(road_length, free_speed)[0, 1]
ax8.text(0.05, 0.95, f'r={corr:.3f}', transform=ax8.transAxes, fontsize=9,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Panel 9: Correlation matrix
ax9 = plt.subplot(3, 4, 9)
features = ['Length', 'Capacity', 'Speed', 'Volume']
feature_data = np.column_stack([road_length, capacity, free_speed, baseline_volume])
corr_matrix = np.corrcoef(feature_data.T)
im = ax9.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
ax9.set_xticks(range(len(features)))
ax9.set_yticks(range(len(features)))
ax9.set_xticklabels(features, rotation=45, ha='right', fontsize=8)
ax9.set_yticklabels(features, fontsize=8)
ax9.set_title('Correlation Matrix', fontsize=10, fontweight='bold')
for i in range(len(features)):
    for j in range(len(features)):
        text = ax9.text(j, i, f'{corr_matrix[i, j]:.2f}',
                       ha="center", va="center", color="black" if abs(corr_matrix[i, j]) < 0.5 else "white",
                       fontsize=8)

# Panel 10: Outlier analysis
ax10 = plt.subplot(3, 4, 10)
q1 = np.percentile(road_length, 25)
q3 = np.percentile(road_length, 75)
iqr = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr
outliers_mask = (road_length < lower_fence) | (road_length > upper_fence)
outlier_counts = np.sum(outliers_mask)

categories_out = ['Normal', 'Outliers']
counts_out = [len(road_length) - outlier_counts, outlier_counts]
colors_out = ['green', 'red']
bars = ax10.bar(categories_out, counts_out, color=colors_out, alpha=0.7, edgecolor='black')
ax10.set_ylabel('Count', fontsize=9, fontweight='bold')
ax10.set_title('Outlier Analysis', fontsize=10, fontweight='bold')
ax10.grid(axis='y', alpha=0.3)
for bar, count in zip(bars, counts_out):
    pct = (count / len(road_length)) * 100
    ax10.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
             f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=8)

# Panel 11: Type comparison table
ax11 = plt.subplot(3, 4, 11)
ax11.axis('off')
table_data = [['Type', 'Count', 'Mean (m)', 'Median (m)']]
for i, (code, name) in enumerate(zip(type_codes_sorted[:6], type_names_sorted[:6])):
    data = length_by_type[i]
    table_data.append([name[:12], f'{len(data):,}', f'{np.mean(data):.1f}', f'{np.median(data):.1f}'])

table = ax11.table(cellText=table_data, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 2)
for i in range(4):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')
ax11.set_title('Top 6 Types Comparison', fontsize=10, fontweight='bold', pad=15)

# Panel 12: Key insights
ax12 = plt.subplot(3, 4, 12)
ax12.axis('off')

longest_type = type_names_sorted[np.argmax(mean_lengths)]
shortest_type = type_names_sorted[np.argmin(mean_lengths)]
most_common_cat_idx = np.argmax(counts)

insights_text = f"""KEY INSIGHTS

DISTRIBUTION:
• Mean: {np.mean(road_length):.1f}m
• Median: {np.median(road_length):.1f}m
• Right-skewed distribution
• 6 length categories

BY HIGHWAY TYPE:
• Longest: {longest_type}
  ({max(mean_lengths):.1f}m mean)
• Shortest: {shortest_type}
  ({min(mean_lengths):.1f}m mean)

CORRELATIONS:
• Capacity: {np.corrcoef(road_length, capacity)[0, 1]:.3f}
• Speed: {np.corrcoef(road_length, free_speed)[0, 1]:.3f}
• Very weak correlations

FEATURE STATUS:
• STATIC (design parameter)
• Physical dimension
• Does NOT vary with traffic
"""

ax12.text(0.1, 0.9, insights_text, fontsize=8, verticalalignment='top', family='monospace',
          bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.4))
ax12.set_title('Summary & Insights', fontsize=10, fontweight='bold', pad=15)

plt.tight_layout()
chart4_path = 'feature5_chart4_dashboard.png'
plt.savefig(chart4_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Saved: {chart4_path}")
display(Image(chart4_path))

print("\n" + "="*80)
print("FEATURE 5 PART 2 COMPLETE")
print("="*80)
print("\nKey Findings:")
print(f"  Road length is STATIC (physical dimension)")
print(f"  Very weak correlation with other features")
print(f"  Wide range: {road_length.min():.1f}m to {road_length.max():.1f}m")
print(f"  Right-skewed: Mean ({np.mean(road_length):.1f}m) > Median ({np.median(road_length):.1f}m)")
print(f"  Most roads are short: {counts[0]:,} roads <50m ({counts[0]/len(road_length)*100:.1f}%)")
