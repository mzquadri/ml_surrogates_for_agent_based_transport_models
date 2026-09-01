"""
FEATURE 5 - PART 1 (CHARTS 1-2)
Road Length Analysis

This script creates the first 2 comprehensive charts for Feature 5 (Road Length):
- Chart 1: Distribution Analysis (histogram, CDF, box plot, statistics)
- Chart 2: Characteristics Analysis (by highway type, correlations)
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
print("FEATURE 5: ROAD LENGTH ANALYSIS")
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
# CHART 1: DISTRIBUTION ANALYSIS
# ============================================================================
print("\n" + "-"*80)
print("CHART 1: Road Length Distribution Analysis")
print("-"*80)

fig = plt.figure(figsize=(20, 16))

# Panel 1A: Histogram with KDE
ax1 = plt.subplot(2, 2, 1)
ax1.hist(road_length, bins=100, alpha=0.7, color='steelblue', edgecolor='black', density=True)
# Add KDE
from scipy.stats import gaussian_kde
kde = gaussian_kde(road_length)
x_range = np.linspace(road_length.min(), road_length.max(), 1000)
ax1.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
ax1.set_xlabel('Road Length (m)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Density', fontsize=11, fontweight='bold')
ax1.set_title('Road Length Distribution with KDE', fontsize=12, fontweight='bold', pad=15)
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)

# Add statistics text
mean_len = np.mean(road_length)
median_len = np.median(road_length)
std_len = np.std(road_length)
ax1.axvline(mean_len, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_len:.1f}m')
ax1.axvline(median_len, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_len:.1f}m')
ax1.legend(fontsize=9)

# Panel 1B: Cumulative Distribution Function (CDF)
ax2 = plt.subplot(2, 2, 2)
sorted_lengths = np.sort(road_length)
cdf = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)
ax2.plot(sorted_lengths, cdf, linewidth=2, color='darkblue')
ax2.set_xlabel('Road Length (m)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Cumulative Probability', fontsize=11, fontweight='bold')
ax2.set_title('Cumulative Distribution Function', fontsize=12, fontweight='bold', pad=15)
ax2.grid(alpha=0.3)

# Add percentile markers
percentiles = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
for p in percentiles:
    val = np.percentile(road_length, p * 100)
    ax2.axhline(y=p, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax2.axvline(x=val, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax2.text(val, p, f' P{int(p*100)}: {val:.0f}m', fontsize=7, verticalalignment='bottom')

# Panel 1C: Box Plot with outlier analysis
ax3 = plt.subplot(2, 2, 3)
bp = ax3.boxplot([road_length], vert=True, patch_artist=True, widths=0.5,
                  boxprops=dict(facecolor='lightblue', edgecolor='black', linewidth=2),
                  medianprops=dict(color='red', linewidth=2),
                  whiskerprops=dict(color='black', linewidth=1.5),
                  capprops=dict(color='black', linewidth=1.5),
                  flierprops=dict(marker='o', markerfacecolor='red', markersize=3, alpha=0.3))

ax3.set_ylabel('Road Length (m)', fontsize=11, fontweight='bold')
ax3.set_title('Box Plot with Outliers', fontsize=12, fontweight='bold', pad=15)
ax3.set_xticklabels(['All Roads'])
ax3.grid(axis='y', alpha=0.3)

# Add statistics labels
q1 = np.percentile(road_length, 25)
q3 = np.percentile(road_length, 75)
iqr = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr
outliers = np.sum((road_length < lower_fence) | (road_length > upper_fence))
outlier_pct = (outliers / len(road_length)) * 100

stats_text = f'Q1: {q1:.1f}m\nMedian: {median_len:.1f}m\nQ3: {q3:.1f}m\nIQR: {iqr:.1f}m\nOutliers: {outliers:,} ({outlier_pct:.1f}%)'
ax3.text(1.3, np.median(road_length), stats_text, fontsize=9, 
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Panel 1D: Statistics Table
ax4 = plt.subplot(2, 2, 4)
ax4.axis('off')

stats_data = [
    ['Statistic', 'Value'],
    ['Count', f'{len(road_length):,} roads'],
    ['Mean', f'{mean_len:.2f} m'],
    ['Median', f'{median_len:.2f} m'],
    ['Std Dev', f'{std_len:.2f} m'],
    ['Min', f'{road_length.min():.2f} m'],
    ['Max', f'{road_length.max():.2f} m'],
    ['Range', f'{road_length.max() - road_length.min():.2f} m'],
    ['', ''],
    ['P25', f'{q1:.2f} m'],
    ['P50', f'{median_len:.2f} m'],
    ['P75', f'{q3:.2f} m'],
    ['P90', f'{np.percentile(road_length, 90):.2f} m'],
    ['P95', f'{np.percentile(road_length, 95):.2f} m'],
    ['P99', f'{np.percentile(road_length, 99):.2f} m'],
    ['', ''],
    ['Skewness', f'{stats.skew(road_length):.3f}'],
    ['Kurtosis', f'{stats.kurtosis(road_length):.3f}'],
    ['CV', f'{std_len/mean_len:.3f}'],
]

table = ax4.table(cellText=stats_data, cellLoc='left', loc='center',
                  colWidths=[0.4, 0.6])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header
for i in range(2):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(stats_data)):
    for j in range(2):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#F0F0F0')

ax4.set_title('Road Length Statistics', fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
chart1_path = 'feature5_chart1_distribution.png'
plt.savefig(chart1_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Saved: {chart1_path}")
display(Image(chart1_path))

# Summary
print(f"\nKey Statistics:")
print(f"  Mean length: {mean_len:.1f} m")
print(f"  Median length: {median_len:.1f} m")
print(f"  Std deviation: {std_len:.1f} m")
print(f"  Range: {road_length.min():.1f} - {road_length.max():.1f} m")
print(f"  Outliers: {outliers:,} ({outlier_pct:.1f}%)")

# ============================================================================
# CHART 2: CHARACTERISTICS ANALYSIS
# ============================================================================
print("\n" + "-"*80)
print("CHART 2: Road Length Characteristics Analysis")
print("-"*80)

fig = plt.figure(figsize=(20, 16))

# Panel 2A: Length by Highway Type
ax1 = plt.subplot(2, 2, 1)
type_counts = Counter(highway_type)
type_codes_sorted = sorted(type_counts.keys(), key=lambda x: type_counts[x], reverse=True)
type_names_sorted = [HW_MAPPING[code] for code in type_codes_sorted]
length_by_type = [road_length[highway_type == code] for code in type_codes_sorted]

bp = ax1.boxplot(length_by_type, tick_labels=type_names_sorted, patch_artist=True)
colors = plt.cm.Set3(np.linspace(0, 1, len(type_names_sorted)))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax1.set_xticklabels(type_names_sorted, rotation=45, ha='right', fontsize=9)
ax1.set_ylabel('Road Length (m)', fontsize=11, fontweight='bold')
ax1.set_title('Road Length by Highway Type', fontsize=12, fontweight='bold', pad=15)
ax1.set_yscale('log')  # Log scale for better visualization
ax1.grid(axis='y', alpha=0.3)

# Panel 2B: Mean Length by Type
ax2 = plt.subplot(2, 2, 2)
mean_lengths = [np.mean(data) for data in length_by_type]
std_lengths = [np.std(data) for data in length_by_type]
y_pos = np.arange(len(type_names_sorted))

bars = ax2.barh(y_pos, mean_lengths, xerr=std_lengths, color=colors, 
                alpha=0.8, capsize=5, edgecolor='black')
ax2.set_yticks(y_pos)
ax2.set_yticklabels(type_names_sorted, fontsize=9)
ax2.set_xlabel('Mean Road Length (m)', fontsize=11, fontweight='bold')
ax2.set_title('Mean Road Length by Highway Type', fontsize=12, fontweight='bold', pad=15)
ax2.grid(axis='x', alpha=0.3)

# Add value labels
for i, (mean, std) in enumerate(zip(mean_lengths, std_lengths)):
    ax2.text(mean + std, i, f' {mean:.1f}m', va='center', fontsize=8)

# Panel 2C: Correlation with other features
ax3 = plt.subplot(2, 2, 3)

# Calculate correlations
corr_capacity = np.corrcoef(road_length, capacity)[0, 1]
corr_speed = np.corrcoef(road_length, free_speed)[0, 1]
# Traffic correlation (for roads with traffic)
traffic_mask = baseline_volume != 0
if np.sum(traffic_mask) > 1:
    corr_traffic = np.corrcoef(road_length[traffic_mask], baseline_volume[traffic_mask])[0, 1]
else:
    corr_traffic = 0

features = ['Capacity', 'Free Speed', 'Baseline\nVolume']
correlations = [corr_capacity, corr_speed, corr_traffic]
colors_corr = ['green' if abs(c) > 0.3 else 'orange' if abs(c) > 0.1 else 'red' for c in correlations]

bars = ax3.bar(features, correlations, color=colors_corr, alpha=0.7, edgecolor='black', linewidth=2)
ax3.axhline(y=0, color='black', linewidth=1)
ax3.set_ylabel('Pearson Correlation', fontsize=11, fontweight='bold')
ax3.set_title('Road Length Correlation with Other Features', fontsize=12, fontweight='bold', pad=15)
ax3.set_ylim(-1, 1)
ax3.grid(axis='y', alpha=0.3)

# Add value labels
for bar, corr in zip(bars, correlations):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{corr:.3f}', ha='center', va='bottom' if height > 0 else 'top', 
            fontsize=10, fontweight='bold')

# Panel 2D: Length categories distribution
ax4 = plt.subplot(2, 2, 4)

# Define length categories
categories = {
    'Very Short\n(<50m)': (0, 50),
    'Short\n(50-100m)': (50, 100),
    'Medium\n(100-200m)': (100, 200),
    'Long\n(200-500m)': (200, 500),
    'Very Long\n(500-1000m)': (500, 1000),
    'Extra Long\n(>1000m)': (1000, np.inf)
}

category_counts = []
category_names = list(categories.keys())
for cat_name, (low, high) in categories.items():
    count = np.sum((road_length >= low) & (road_length < high))
    category_counts.append(count)

colors_cat = plt.cm.viridis(np.linspace(0, 1, len(category_names)))
bars = ax4.bar(range(len(category_names)), category_counts, color=colors_cat, 
               alpha=0.8, edgecolor='black', linewidth=1.5)
ax4.set_xticks(range(len(category_names)))
ax4.set_xticklabels(category_names, rotation=0, ha='center', fontsize=9)
ax4.set_ylabel('Number of Roads', fontsize=11, fontweight='bold')
ax4.set_title('Road Length Categories Distribution', fontsize=12, fontweight='bold', pad=15)
ax4.grid(axis='y', alpha=0.3)

# Add percentage labels
for i, (bar, count) in enumerate(zip(bars, category_counts)):
    pct = (count / len(road_length)) * 100
    ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom', 
            fontsize=8, fontweight='bold')

plt.tight_layout()
chart2_path = 'feature5_chart2_characteristics.png'
plt.savefig(chart2_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Saved: {chart2_path}")
display(Image(chart2_path))

# Summary
print(f"\nKey Findings:")
print(f"  Longest mean type: {type_names_sorted[np.argmax(mean_lengths)]} ({max(mean_lengths):.1f}m)")
print(f"  Shortest mean type: {type_names_sorted[np.argmin(mean_lengths)]} ({min(mean_lengths):.1f}m)")
print(f"  Correlation with capacity: {corr_capacity:.3f}")
print(f"  Correlation with speed: {corr_speed:.3f}")
print(f"  Most common category: {category_names[np.argmax(category_counts)]} ({max(category_counts):,} roads)")

print("\n" + "="*80)
print("FEATURE 5 PART 1 COMPLETE")
print("="*80)
