"""
################################################################################
#                                                                              #
#                  FEATURE 3 - PART 2: FREE SPEED (Charts 3-4)                #
#                    Relationships & Comprehensive Summary                     #
#                                                                              #
################################################################################

Feature 3 (Free Speed) represents the design speed limit of roads - the maximum
speed vehicles can travel on a road segment under ideal conditions.

Part 2 includes:
  - Chart 3: Speed relationships with capacity, volume, length (scatter plots + correlation)
  - Chart 4: Comprehensive summary dashboard (7-panel integrated visualization)

"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Configure plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams.update({
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

print("="*80)
print("LOADING DATA...")
print("="*80)

# Path detection
local_path = Path(r'D:\Python Projects\Zamin_Thesis\ml_surrogates_for_agent_based_transport_models\data\train_data\dist_not_connected_10k_1pct')
colab_path = Path('/content/drive/MyDrive/Zamin_thesis/ml_surrogates_for_agent_based_transport_models/data/train_data/dist_not_connected_10k_1pct')

if local_path.exists():
    data_path = local_path
elif colab_path.exists():
    data_path = colab_path
else:
    raise FileNotFoundError("Data path not found!")

print(f"✓ Found data path: {data_path}")

# Load first batch
batch_file = data_path / 'datalist_batch_1.pt'
data_list = torch.load(batch_file, map_location='cpu', weights_only=False)

# Extract features from first scenario
graph = data_list[0]
n_edges = graph.x.shape[0]

length = graph.x[:, 0].numpy()
capacity = graph.x[:, 1].numpy()
baseline_volume = graph.x[:, 2].numpy()
capacity_reduction = graph.x[:, 3].numpy()
highway_type = graph.x[:, 4].numpy()
free_speed = graph.x[:, 5].numpy()
target = graph.y.numpy().flatten()

print(f"✓ Loaded {n_edges:,} edges")
print()

# Highway type mapping
hw_mapping = {
    0: 'Motorway', 1: 'Trunk', 2: 'Primary', 3: 'Secondary',
    4: 'Tertiary', 5: 'Residential', 6: 'PT', 7: 'Service',
    8: 'Living Street', 9: 'Motorway Link', 10: 'Trunk Link',
    11: 'Primary Link', 12: 'Secondary Link'
}

################################################################################
#                            CHART 3: RELATIONSHIPS                            #
################################################################################

print("="*80)
print("CHART 3: Speed Relationships with Other Features")
print("="*80)

fig = plt.figure(figsize=(20, 16))
fig.suptitle('Feature 3: Free Speed Relationships with Other Features', 
             fontsize=16, fontweight='bold', y=0.995)

# 3A: Speed vs Capacity
ax1 = plt.subplot(2, 2, 1)
scatter1 = ax1.scatter(capacity, free_speed, alpha=0.5, s=20, c=free_speed,
                      cmap='RdYlGn', edgecolors='black', linewidth=0.3)
corr_cap = np.corrcoef(capacity, free_speed)[0, 1]
ax1.set_xlabel('Road Capacity (veh/h)', fontweight='bold')
ax1.set_ylabel('Free Speed (km/h)', fontweight='bold')
ax1.set_title(f'A. Speed vs Capacity\nCorrelation: {corr_cap:.3f}', 
             fontweight='bold', pad=10)
ax1.grid(True, alpha=0.3)
plt.colorbar(scatter1, ax=ax1, label='Speed (km/h)')

# Add trend line
z = np.polyfit(capacity, free_speed, 1)
p = np.poly1d(z)
ax1.plot(capacity, p(capacity), "r--", linewidth=2, alpha=0.7, label='Trend line')
ax1.legend()

# 3B: Speed vs Baseline Volume (roads with traffic only)
ax2 = plt.subplot(2, 2, 2)
has_traffic = baseline_volume < 0
if has_traffic.sum() > 0:
    vol_with_traffic = baseline_volume[has_traffic]
    speed_with_traffic = free_speed[has_traffic]
    
    scatter2 = ax2.scatter(vol_with_traffic, speed_with_traffic, alpha=0.5, s=20,
                          c=speed_with_traffic, cmap='RdYlGn',
                          edgecolors='black', linewidth=0.3)
    corr_vol = np.corrcoef(vol_with_traffic, speed_with_traffic)[0, 1]
    ax2.set_xlabel('Baseline Volume (veh/h)', fontweight='bold')
    ax2.set_ylabel('Free Speed (km/h)', fontweight='bold')
    ax2.set_title(f'B. Speed vs Baseline Volume (Roads with Traffic)\nCorrelation: {corr_vol:.3f}', 
                 fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='Speed (km/h)')
    
    # Add trend line
    z = np.polyfit(vol_with_traffic, speed_with_traffic, 1)
    p = np.poly1d(z)
    ax2.plot(vol_with_traffic, p(vol_with_traffic), "r--", linewidth=2, alpha=0.7, label='Trend line')
    ax2.legend()
else:
    ax2.text(0.5, 0.5, 'No traffic data available', 
            ha='center', va='center', transform=ax2.transAxes, fontsize=14)

# 3C: Speed vs Length
ax3 = plt.subplot(2, 2, 3)
scatter3 = ax3.scatter(length, free_speed, alpha=0.5, s=20, c=free_speed,
                      cmap='RdYlGn', edgecolors='black', linewidth=0.3)
corr_len = np.corrcoef(length, free_speed)[0, 1]
ax3.set_xlabel('Road Length (m)', fontweight='bold')
ax3.set_ylabel('Free Speed (km/h)', fontweight='bold')
ax3.set_title(f'C. Speed vs Road Length\nCorrelation: {corr_len:.3f}', 
             fontweight='bold', pad=10)
ax3.grid(True, alpha=0.3)
plt.colorbar(scatter3, ax=ax3, label='Speed (km/h)')

# Add trend line
z = np.polyfit(length, free_speed, 1)
p = np.poly1d(z)
ax3.plot(length, p(length), "r--", linewidth=2, alpha=0.7, label='Trend line')
ax3.legend()

# 3D: Correlation summary table
ax4 = plt.subplot(2, 2, 4)
ax4.axis('off')

# Calculate all correlations
correlations = [
    ('Capacity', corr_cap),
    ('Baseline Volume (traffic)', corr_vol if has_traffic.sum() > 0 else 0),
    ('Road Length', corr_len),
    ('Highway Type', np.corrcoef(highway_type, free_speed)[0, 1]),
    ('Target Volume', np.corrcoef(target, free_speed)[0, 1]),
    ('Capacity Reduction', np.corrcoef(capacity_reduction, free_speed)[0, 1])
]

# Sort by absolute correlation
correlations_sorted = sorted(correlations, key=lambda x: abs(x[1]), reverse=True)

# Create table
table_data = [['Feature', 'Correlation', 'Strength']]
for feat, corr in correlations_sorted:
    if abs(corr) > 0.7:
        strength = 'Strong'
        color = '#2ecc71'
    elif abs(corr) > 0.4:
        strength = 'Moderate'
        color = '#f39c12'
    elif abs(corr) > 0.2:
        strength = 'Weak'
        color = '#e67e22'
    else:
        strength = 'Very Weak'
        color = '#95a5a6'
    
    table_data.append([feat, f'{corr:+.3f}', strength])

# Plot table
table = ax4.table(cellText=table_data, cellLoc='left', loc='center',
                 colWidths=[0.5, 0.2, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Style header
for i in range(3):
    cell = table[(0, i)]
    cell.set_facecolor('#3498db')
    cell.set_text_props(weight='bold', color='white')

# Color code rows by strength
for i in range(1, len(table_data)):
    strength = table_data[i][2]
    if strength == 'Strong':
        color = '#d5f4e6'
    elif strength == 'Moderate':
        color = '#fef5e7'
    elif strength == 'Weak':
        color = '#fdebd0'
    else:
        color = '#ecf0f1'
    
    for j in range(3):
        table[(i, j)].set_facecolor(color)

ax4.set_title('D. Correlation Summary\n(Sorted by Strength)', 
             fontweight='bold', pad=20, fontsize=13)

plt.tight_layout()
plt.savefig('feature3_chart3_relationships.png', dpi=300, bbox_inches='tight')
print("✓ Saved: feature3_chart3_relationships.png")
print()

################################################################################
#                    CHART 4: COMPREHENSIVE SUMMARY DASHBOARD                  #
################################################################################

print("="*80)
print("CHART 4: Comprehensive Summary Dashboard")
print("="*80)

fig = plt.figure(figsize=(20, 24))
fig.suptitle('Feature 3: Free Speed - Comprehensive Summary Dashboard', 
             fontsize=16, fontweight='bold', y=0.995)

# 4A: Speed distribution histogram (top left)
ax1 = plt.subplot(4, 3, 1)
n, bins, patches = ax1.hist(free_speed, bins=60, edgecolor='black', linewidth=1.2, alpha=0.7)
# Color code bins
for i, patch in enumerate(patches):
    speed_val = (bins[i] + bins[i+1]) / 2
    if speed_val < 5:
        patch.set_facecolor('#e74c3c')
    elif speed_val < 10:
        patch.set_facecolor('#f39c12')
    elif speed_val < 15:
        patch.set_facecolor('#f1c40f')
    else:
        patch.set_facecolor('#2ecc71')

ax1.axvline(np.median(free_speed), color='red', linestyle='--', linewidth=2.5, label=f'Median: {np.median(free_speed):.1f}')
ax1.axvline(np.mean(free_speed), color='green', linestyle='--', linewidth=2.5, label=f'Mean: {np.mean(free_speed):.1f}')
ax1.set_xlabel('Free Speed (km/h)', fontweight='bold')
ax1.set_ylabel('Number of Roads', fontweight='bold')
ax1.set_title('A. Overall Distribution', fontweight='bold', pad=10)
ax1.legend()
ax1.grid(True, alpha=0.3)

# 4B: Speed categories (top middle)
ax2 = plt.subplot(4, 3, 2)
speed_bins = [0, 5, 10, 15, 20, 25, 100]
speed_labels = ['0-5\n(Very Slow)', '5-10\n(Slow)', '10-15\n(Moderate)', 
                '15-20\n(Fast)', '20-25\n(Very Fast)', '>25\n(Highway)']
speed_counts = [np.sum((free_speed >= speed_bins[i]) & (free_speed < speed_bins[i+1])) 
                for i in range(len(speed_bins)-1)]
colors = ['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71', '#3498db', '#9b59b6']

bars = ax2.bar(range(len(speed_labels)), speed_counts, color=colors, 
              edgecolor='black', linewidth=1.5, alpha=0.8)
for bar, count in zip(bars, speed_counts):
    height = bar.get_height()
    pct = (count / len(free_speed)) * 100
    ax2.text(bar.get_x() + bar.get_width()/2, height + 50,
            f'{count:,}\n({pct:.1f}%)', ha='center', fontsize=9, fontweight='bold')

ax2.set_xticks(range(len(speed_labels)))
ax2.set_xticklabels(speed_labels, fontsize=10)
ax2.set_ylabel('Number of Roads', fontweight='bold')
ax2.set_title('B. Speed Categories', fontweight='bold', pad=10)
ax2.grid(axis='y', alpha=0.3)

# 4C: Statistics table (top right)
ax3 = plt.subplot(4, 3, 3)
ax3.axis('off')

stats_data = [
    ['Metric', 'Value'],
    ['Total Roads', f'{len(free_speed):,}'],
    ['', ''],
    ['Mean Speed', f'{np.mean(free_speed):.2f} km/h'],
    ['Median Speed', f'{np.median(free_speed):.2f} km/h'],
    ['Std Deviation', f'{np.std(free_speed):.2f} km/h'],
    ['', ''],
    ['Minimum', f'{np.min(free_speed):.2f} km/h'],
    ['Maximum', f'{np.max(free_speed):.2f} km/h'],
    ['Range', f'{np.max(free_speed) - np.min(free_speed):.2f} km/h'],
    ['', ''],
    ['25th Percentile', f'{np.percentile(free_speed, 25):.2f} km/h'],
    ['50th Percentile', f'{np.percentile(free_speed, 50):.2f} km/h'],
    ['75th Percentile', f'{np.percentile(free_speed, 75):.2f} km/h'],
    ['90th Percentile', f'{np.percentile(free_speed, 90):.2f} km/h'],
    ['', ''],
    ['Unique Values', f'{len(np.unique(free_speed))}'],
    ['Coefficient of Var', f'{(np.std(free_speed)/np.mean(free_speed)):.3f}']
]

table = ax3.table(cellText=stats_data, cellLoc='left', loc='center',
                 colWidths=[0.6, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.0)

# Style header
for i in range(2):
    cell = table[(0, i)]
    cell.set_facecolor('#3498db')
    cell.set_text_props(weight='bold', color='white')

# Color alternating rows
for i in range(1, len(stats_data)):
    color = '#ecf0f1' if i % 2 == 0 else 'white'
    for j in range(2):
        if stats_data[i][0] == '':
            table[(i, j)].set_facecolor('white')
        else:
            table[(i, j)].set_facecolor(color)

ax3.set_title('C. Summary Statistics', fontweight='bold', pad=20, fontsize=13)

# 4D: Mean speed by highway type (middle left)
ax4 = plt.subplot(4, 3, 4)
hw_types_present = np.unique(highway_type)
hw_means = []
hw_labels = []
hw_counts = []

for hw_id in hw_types_present:
    if hw_id in hw_mapping:
        mask = highway_type == hw_id
        hw_means.append(np.mean(free_speed[mask]))
        hw_labels.append(hw_mapping[hw_id])
        hw_counts.append(np.sum(mask))

# Sort by mean speed
sorted_indices = np.argsort(hw_means)[::-1]
hw_means = [hw_means[i] for i in sorted_indices]
hw_labels = [hw_labels[i] for i in sorted_indices]
hw_counts = [hw_counts[i] for i in sorted_indices]

colors_hw = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(hw_labels)))
bars = ax4.barh(range(len(hw_labels)), hw_means, color=colors_hw, 
               edgecolor='black', linewidth=1.2, alpha=0.8)

for i, (bar, mean, count) in enumerate(zip(bars, hw_means, hw_counts)):
    ax4.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
            f'{mean:.1f} km/h (n={count:,})', va='center', fontsize=9, fontweight='bold')

ax4.set_yticks(range(len(hw_labels)))
ax4.set_yticklabels(hw_labels, fontsize=10)
ax4.set_xlabel('Mean Free Speed (km/h)', fontweight='bold')
ax4.set_title('D. Mean Speed by Highway Type', fontweight='bold', pad=10)
ax4.grid(axis='x', alpha=0.3)

# 4E: CDF (middle middle)
ax5 = plt.subplot(4, 3, 5)
sorted_speed = np.sort(free_speed)
cumulative = np.arange(1, len(sorted_speed) + 1) / len(sorted_speed) * 100

ax5.plot(sorted_speed, cumulative, linewidth=2.5, color='#2c3e50')
ax5.set_xlabel('Free Speed (km/h)', fontweight='bold')
ax5.set_ylabel('Cumulative Percentage (%)', fontweight='bold')
ax5.set_title('E. Cumulative Distribution Function', fontweight='bold', pad=10)
ax5.grid(True, alpha=0.3)

# Add percentile markers
percentiles = [25, 50, 75, 90]
colors_pct = ['#e74c3c', '#f39c12', '#2ecc71', '#9b59b6']
for pct, color in zip(percentiles, colors_pct):
    value = np.percentile(free_speed, pct)
    ax5.axvline(value, color=color, linestyle='--', linewidth=2, alpha=0.7, 
               label=f'P{pct}: {value:.1f}')
    ax5.axhline(pct, color=color, linestyle='--', linewidth=1.5, alpha=0.5)

ax5.legend(loc='lower right', fontsize=9)

# 4F: Box plot by highway type (middle right)
ax6 = plt.subplot(4, 3, 6)
box_data = []
box_labels = []

for hw_id in hw_types_present[:10]:  # Top 10 types
    if hw_id in hw_mapping:
        mask = highway_type == hw_id
        if np.sum(mask) > 10:  # At least 10 roads
            box_data.append(free_speed[mask])
            box_labels.append(f'{hw_mapping[hw_id]}\n(n={np.sum(mask)})')

bp = ax6.boxplot(box_data, tick_labels=box_labels, patch_artist=True,
                showmeans=True, meanline=True,
                boxprops=dict(linewidth=1.5),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5),
                medianprops=dict(linewidth=2.5, color='red'),
                meanprops=dict(linewidth=2.5, color='blue', linestyle='--'))

colors_box = plt.cm.Set3(np.linspace(0, 1, len(box_data)))
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax6.set_ylabel('Free Speed (km/h)', fontweight='bold')
ax6.set_title('F. Distribution by Highway Type', fontweight='bold', pad=10)
ax6.tick_params(axis='x', rotation=45)
ax6.grid(axis='y', alpha=0.3)

# 4G: Speed-Capacity relationship (bottom left)
ax7 = plt.subplot(4, 3, 7)
scatter = ax7.scatter(capacity, free_speed, alpha=0.4, s=15, c=free_speed,
                     cmap='RdYlGn', edgecolors='black', linewidth=0.2)
ax7.set_xlabel('Capacity (veh/h)', fontweight='bold')
ax7.set_ylabel('Free Speed (km/h)', fontweight='bold')
ax7.set_title(f'G. Speed vs Capacity (r={corr_cap:.3f})', fontweight='bold', pad=10)
ax7.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax7, label='Speed')

# 4H: Speed-Length relationship (bottom middle)
ax8 = plt.subplot(4, 3, 8)
scatter = ax8.scatter(length, free_speed, alpha=0.4, s=15, c=free_speed,
                     cmap='RdYlGn', edgecolors='black', linewidth=0.2)
ax8.set_xlabel('Length (m)', fontweight='bold')
ax8.set_ylabel('Free Speed (km/h)', fontweight='bold')
ax8.set_title(f'H. Speed vs Length (r={corr_len:.3f})', fontweight='bold', pad=10)
ax8.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax8, label='Speed')

# 4I: Key insights text box (bottom right)
ax9 = plt.subplot(4, 3, 9)
ax9.axis('off')

# Calculate key insights
unique_speeds = len(np.unique(free_speed))
most_common_speed = stats.mode(free_speed, keepdims=True).mode[0]
most_common_count = np.sum(free_speed == most_common_speed)
low_speed_pct = np.sum(free_speed < 10) / len(free_speed) * 100
high_speed_pct = np.sum(free_speed > 15) / len(free_speed) * 100

insights_text = f"""
KEY INSIGHTS:

DISTRIBUTION:
  * {unique_speeds} unique speed values
  * Mean: {np.mean(free_speed):.1f} km/h
  * Most common: {most_common_speed:.1f} km/h 
    ({most_common_count:,} roads, {most_common_count/len(free_speed)*100:.1f}%)

SPEED PROFILE:
  * Low speed (<10 km/h): {low_speed_pct:.1f}%
  * High speed (>15 km/h): {high_speed_pct:.1f}%
  * Moderate speed (10-15): {100-low_speed_pct-high_speed_pct:.1f}%

CORRELATIONS:
  * Strongest: {correlations_sorted[0][0]}
    (r={correlations_sorted[0][1]:.3f})
  * Weakest: {correlations_sorted[-1][0]}
    (r={correlations_sorted[-1][1]:.3f})

BY HIGHWAY TYPE:
  * Fastest: {hw_labels[0]} ({hw_means[0]:.1f} km/h)
  * Slowest: {hw_labels[-1]} ({hw_means[-1]:.1f} km/h)
  * Range: {hw_means[0] - hw_means[-1]:.1f} km/h difference
"""

ax9.text(0.05, 0.95, insights_text, transform=ax9.transAxes,
        fontsize=10, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# 4J-L: Additional panels for comprehensive view
# 4J: Speed range by highway type (bottom row)
ax10 = plt.subplot(4, 3, 10)
hw_mins = []
hw_maxs = []
hw_ranges = []

for hw_id in hw_types_present[:8]:
    if hw_id in hw_mapping:
        mask = highway_type == hw_id
        if np.sum(mask) > 10:
            hw_mins.append(np.min(free_speed[mask]))
            hw_maxs.append(np.max(free_speed[mask]))
            hw_ranges.append(np.max(free_speed[mask]) - np.min(free_speed[mask]))

hw_labels_plot = [hw_labels[i] for i in range(len(hw_ranges))]

bars = ax10.bar(range(len(hw_ranges)), hw_ranges, 
               color=plt.cm.viridis(np.linspace(0.2, 0.8, len(hw_ranges))),
               edgecolor='black', linewidth=1.2, alpha=0.8)

for bar, rng in zip(bars, hw_ranges):
    height = bar.get_height()
    ax10.text(bar.get_x() + bar.get_width()/2, height + 0.2,
             f'{rng:.1f}', ha='center', fontsize=9, fontweight='bold')

ax10.set_xticks(range(len(hw_ranges)))
ax10.set_xticklabels(hw_labels_plot, rotation=45, ha='right', fontsize=9)
ax10.set_ylabel('Speed Range (km/h)', fontweight='bold')
ax10.set_title('J. Speed Range by Highway Type', fontweight='bold', pad=10)
ax10.grid(axis='y', alpha=0.3)

# 4K: Unique speed values distribution
ax11 = plt.subplot(4, 3, 11)
unique_vals, unique_counts = np.unique(free_speed, return_counts=True)
top_n = 15
sorted_indices = np.argsort(-unique_counts)[:top_n]

bars = ax11.bar(range(top_n), unique_counts[sorted_indices],
               color=plt.cm.plasma(np.linspace(0.2, 0.8, top_n)),
               edgecolor='black', linewidth=1.2, alpha=0.8)

for i, (bar, count) in enumerate(zip(bars, unique_counts[sorted_indices])):
    pct = count / len(free_speed) * 100
    ax11.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
             f'{count}\n({pct:.1f}%)', ha='center', fontsize=8, fontweight='bold')

ax11.set_xticks(range(top_n))
ax11.set_xticklabels([f'{unique_vals[i]:.1f}' for i in sorted_indices], 
                     rotation=45, ha='right', fontsize=9)
ax11.set_xlabel('Speed Value (km/h)', fontweight='bold')
ax11.set_ylabel('Number of Roads', fontweight='bold')
ax11.set_title('K. Top 15 Most Common Speed Values', fontweight='bold', pad=10)
ax11.grid(axis='y', alpha=0.3)

# 4L: Speed variability summary
ax12 = plt.subplot(4, 3, 12)
ax12.axis('off')

# Calculate variability metrics
cv = np.std(free_speed) / np.mean(free_speed)
iqr = np.percentile(free_speed, 75) - np.percentile(free_speed, 25)
q1 = np.percentile(free_speed, 25)
q3 = np.percentile(free_speed, 75)

variability_data = [
    ['Metric', 'Value', 'Interpretation'],
    ['Coeff. of Variation', f'{cv:.3f}', 'Moderate' if cv > 0.3 else 'Low'],
    ['IQR (Q3-Q1)', f'{iqr:.2f} km/h', 'Central 50% spread'],
    ['Q1 (25th)', f'{q1:.2f} km/h', 'Lower quartile'],
    ['Q3 (75th)', f'{q3:.2f} km/h', 'Upper quartile'],
    ['Std Deviation', f'{np.std(free_speed):.2f} km/h', 'Average deviation'],
    ['Range', f'{np.max(free_speed) - np.min(free_speed):.2f} km/h', 'Full spread']
]

table = ax12.table(cellText=variability_data, cellLoc='left', loc='center',
                  colWidths=[0.4, 0.3, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.2)

# Style header
for i in range(3):
    cell = table[(0, i)]
    cell.set_facecolor('#9b59b6')
    cell.set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(variability_data)):
    color = '#f8f9fa' if i % 2 == 0 else 'white'
    for j in range(3):
        table[(i, j)].set_facecolor(color)

ax12.set_title('L. Variability Summary', fontweight='bold', pad=20, fontsize=13)

plt.tight_layout()
plt.savefig('feature3_chart4_summary_dashboard.png', dpi=300, bbox_inches='tight')
print("✓ Saved: feature3_chart4_summary_dashboard.png")
print()

################################################################################
#                                  COMPLETION                                   #
################################################################################

print("="*80)
print("✓✓✓ PART 2 (Charts 3-4) COMPLETE ✓✓✓")
print("="*80)
print()
print("Generated files:")
print("  3. feature3_chart3_relationships.png")
print("  4. feature3_chart4_summary_dashboard.png")
print()
print("Feature 3 (Free Speed) analysis complete!")
print("Total charts: 4 (2 from Part 1 + 2 from Part 2)")
print("="*80)
