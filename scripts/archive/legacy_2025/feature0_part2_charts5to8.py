"""
FEATURE 0 ANALYSIS - PART 2: NETWORK ANALYSIS (Charts 5-8)
============================================================
- Chart 5: Volume-Capacity Relationship
- Chart 6: Traffic by Highway Type
- Chart 7: Spatial Distribution
- Chart 8: Outliers & Anomalies

Repository Code: process_simulations_for_gnn.py Line 104
Source: pop_1pct_basecase_average_output_links.geojson
F0 = Baseline traffic volume WITHOUT policy intervention
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from scipy import stats
import matplotlib.ticker as ticker

# Set professional plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

print("\n" + "#" * 80)
print("#" + " " * 78 + "#")
print("#" + "  FEATURE 0 - PART 2: NETWORK ANALYSIS (Charts 5-8)".center(78) + "#")
print("#" + "  Paris MATSim Network Analysis".center(78) + "#")
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
vol_base_case = first_scenario.x[:, 0].numpy()
capacity = first_scenario.x[:, 1].numpy()
highway = first_scenario.x[:, 4].numpy()
length = first_scenario.x[:, 5].numpy()

n_edges = len(vol_base_case)
print(f"✓ Loaded {n_edges:,} edges")

# Highway type decoder (OpenStreetMap classification)
highway_type_names = {
    0: 'Motorway',        # High-speed divided highways (autoroute)
    1: 'Trunk',           # Important non-motorway roads
    2: 'Primary',         # Major roads connecting cities
    3: 'Secondary',       # Regional connector roads
    4: 'Tertiary',        # Local connector roads
    5: 'Residential',     # Roads in residential areas
    6: 'Service',         # Service/access roads (parking lots)
    7: 'Unclassified',    # Minor public roads
    8: 'Living Street',   # Low-speed residential streets
    9: 'Other'            # Other road types
}

################################################################################
# CHART 5: VOLUME-CAPACITY RELATIONSHIP
################################################################################
print("\n" + "=" * 80)
print("CHART 5: Volume-Capacity Relationship")
print("=" * 80)

# Calculate utilization with safe division (avoid division by zero warning)
with np.errstate(divide='ignore', invalid='ignore'):
    utilization = np.abs(vol_base_case) / capacity
    utilization = np.nan_to_num(utilization, nan=0.0, posinf=0.0, neginf=0.0)

fig, axes = plt.subplots(2, 2, figsize=(16, 13))
fig.suptitle('FEATURE 0: Volume-Capacity Relationship - Utilization Analysis', 
             fontsize=15, fontweight='bold', y=0.995)
plt.subplots_adjust(left=0.08, right=0.95, top=0.94, bottom=0.06, hspace=0.35, wspace=0.25)

# 5.1 Scatter: Volume vs Capacity
traffic_mask = vol_base_case != 0
axes[0, 0].scatter(capacity[traffic_mask], vol_base_case[traffic_mask], 
                  alpha=0.4, s=2, c='#3498db', edgecolors='none')
axes[0, 0].plot([0, capacity.max()], [0, capacity.max()], 'r--', linewidth=3, 
               label='100% utilization (red line = fully used)', alpha=0.8)
axes[0, 0].set_xlabel('Road Capacity (vehicles/hour)\n[Example: 2000 veh/h = road designed to handle 2000 cars/hour maximum]', fontsize=10)
axes[0, 0].set_ylabel('Baseline Traffic Volume (vehicles/hour)\n[Example: 500 veh/h = currently 500 cars/hour using this road]', fontsize=10)
axes[0, 0].set_title(f'A. Actual Traffic vs Maximum Capacity (n={traffic_mask.sum():,} active roads)\n[Points below red line = under-utilized | On line = fully utilized | Above = over capacity]', 
                     fontsize=11, fontweight='bold', pad=10)
axes[0, 0].legend(loc='upper left', framealpha=0.9, fontsize=9)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].xaxis.set_major_locator(ticker.MultipleLocator(1000))
axes[0, 0].yaxis.set_major_locator(ticker.MultipleLocator(200))

# 5.2 Utilization distribution
util_nonzero = utilization[utilization > 0]
axes[0, 1].hist(util_nonzero, bins=80, alpha=0.75, color='#e67e22', edgecolor='black', linewidth=0.5)
axes[0, 1].set_xlabel('Utilization Ratio (Current Traffic / Maximum Capacity)\n[Example: 0.25 = 25% utilized | 0.50 = 50% | 1.0 = 100% full capacity]', fontsize=10)
axes[0, 1].set_ylabel('Number of Roads\n[How many roads have each utilization level]', fontsize=10)
axes[0, 1].set_title(f'B. Road Utilization Distribution\n[Mean={util_nonzero.mean():.3f} = Average road uses {util_nonzero.mean()*100:.1f}% of its capacity]', 
                     fontsize=11, fontweight='bold', pad=10)
axes[0, 1].axvline(1.0, color='#e74c3c', linestyle='--', linewidth=2.5, label='100% = Full capacity (congested)', alpha=0.8)
axes[0, 1].axvline(util_nonzero.mean(), color='#27ae60', linestyle='--', linewidth=2.5, 
                  label=f'Mean={util_nonzero.mean():.3f} ({util_nonzero.mean()*100:.1f}%)', alpha=0.8)
axes[0, 1].legend(loc='best', framealpha=0.9, fontsize=9)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].xaxis.set_major_locator(ticker.MultipleLocator(0.1))

# 5.3 Volume by capacity bins
cap_bins = [0, 500, 1000, 2000, 5000, capacity.max()+1]
cap_labels = ['0-500', '500-1k', '1k-2k', '2k-5k', '5k+']
mean_vols = []
for i in range(len(cap_bins)-1):
    mask = (capacity >= cap_bins[i]) & (capacity < cap_bins[i+1])
    mean_vols.append(vol_base_case[mask].mean() if mask.sum() > 0 else 0)

x = np.arange(len(cap_labels))
bars = axes[1, 0].bar(x, mean_vols, alpha=0.8, color='#27ae60', edgecolor='black', linewidth=0.7)
axes[1, 0].set_xlabel('Road Capacity Category (vehicles/hour)\n[Example: "500-1k" = roads that can handle 500 to 1000 cars/hour]', fontsize=10)
axes[1, 0].set_ylabel('Average Traffic Volume (vehicles/hour)\n[Mean traffic on roads in each capacity category]', fontsize=10)
axes[1, 0].set_title('C. Do Higher-Capacity Roads Carry More Traffic?\n[Shows relationship between road size and actual traffic usage]', fontsize=11, fontweight='bold', pad=10)
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(cap_labels)
axes[1, 0].grid(True, alpha=0.3, axis='y')
axes[1, 0].yaxis.set_major_locator(ticker.MultipleLocator(50))
for bar, val in zip(bars, mean_vols):
    axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                   f'{val:.0f}\nveh/h', ha='center', va='bottom', fontsize=8, fontweight='bold')

# 5.4 Utilization by capacity bins
mean_utils = []
for i in range(len(cap_bins)-1):
    mask = (capacity >= cap_bins[i]) & (capacity < cap_bins[i+1])
    mean_utils.append(utilization[mask].mean() if mask.sum() > 0 else 0)

bars = axes[1, 1].bar(x, mean_utils, alpha=0.8, color='#c0392b', edgecolor='black', linewidth=0.7)
axes[1, 1].set_xlabel('Road Capacity Category (vehicles/hour)\n[Small roads (0-500) vs Large highways (5k+)]', fontsize=10)
axes[1, 1].set_ylabel('Average Utilization Ratio (Traffic/Capacity)\n[Example: 0.30 = roads using 30% of their capacity on average]', fontsize=10)
axes[1, 1].set_title('D. Are Smaller or Larger Roads More Congested?\n[Higher bar = more utilized/congested | Lower bar = under-utilized]', fontsize=11, fontweight='bold', pad=10)
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(cap_labels)
axes[1, 1].axhline(1.0, color='#e74c3c', linestyle='--', linewidth=2.5, label='100% = Full capacity (maximum congestion)', alpha=0.8)
axes[1, 1].legend(loc='best', framealpha=0.9, fontsize=9)
axes[1, 1].grid(True, alpha=0.3, axis='y')
axes[1, 1].yaxis.set_major_locator(ticker.MultipleLocator(0.1))
for bar, val in zip(bars, mean_utils):
    axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                   f'{val*100:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig('feature0_chart5_capacity_relationship.png', dpi=300, bbox_inches='tight')
print("✓ Saved: feature0_chart5_capacity_relationship.png")
plt.show()  # Display in Colab
plt.close()

# Calculate correlation
valid_mask = (capacity > 0) & (vol_base_case != 0)
corr_vol_cap = np.corrcoef(vol_base_case[valid_mask], capacity[valid_mask])[0, 1]
print(f"Correlation (Volume vs Capacity): {corr_vol_cap:.4f}")

################################################################################
# CHART 6: TRAFFIC BY HIGHWAY TYPE
################################################################################
print("\n" + "=" * 80)
print("CHART 6: Traffic Patterns by Highway Type")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(17, 14))
fig.suptitle('FEATURE 0: Traffic Patterns by Highway Type (OpenStreetMap Classification)', 
             fontsize=14, fontweight='bold', y=0.995)
plt.subplots_adjust(left=0.07, right=0.96, top=0.94, bottom=0.08, hspace=0.42, wspace=0.26)

unique_types = np.unique(highway)

# 6.1 Box plot by type with full names
ax = axes[0, 0]
data_for_boxplot = [vol_base_case[highway == ht] for ht in unique_types]
bp = ax.boxplot(data_for_boxplot, positions=unique_types, widths=0.6,
                patch_artist=True, showfliers=False)
for patch in bp['boxes']:
    patch.set_facecolor('#3498db')
    patch.set_alpha(0.7)
for median in bp['medians']:
    median.set_color('#e74c3c')
    median.set_linewidth(2.5)
ax.set_xlabel('Road Type\n[0=Motorway, 1=Trunk, 2=Primary, 3=Secondary, 4=Tertiary,\n5=Residential, 6=Service, 7=Unclassified, 8=Living St., 9=Other]', fontsize=9)
ax.set_ylabel('Baseline Traffic Volume (vehicles/hour)\n[Red line = median | Blue box = middle 50% of roads (IQR)]', fontsize=10)
ax.set_title('A. Traffic Distribution by Road Type\n[Compare major highways (0-2) vs local streets (5-8) traffic patterns]', fontsize=11, fontweight='bold', pad=10)
ax.grid(True, alpha=0.3, axis='y')
ax.set_xticks(unique_types)
ax.set_xticklabels([f'{int(ht)}\n{highway_type_names.get(int(ht), "?")[:5]}' for ht in unique_types], fontsize=8)
ax.yaxis.set_major_locator(ticker.MultipleLocator(200))

# 6.2 Mean volume by type with full names
ax = axes[0, 1]
means = [vol_base_case[highway == ht].mean() for ht in unique_types]
colors = ['#e74c3c' if m < 20 else '#f39c12' if m < 50 else '#27ae60' for m in means]
bars = ax.bar(unique_types, means, width=0.6, alpha=0.8, color=colors, edgecolor='black', linewidth=0.7)
ax.set_xlabel('Road Type (OpenStreetMap Classification)\n[0=Motorway | 1=Trunk | 2=Primary | 3=Secondary | 4=Tertiary\n5=Residential | 6=Service | 7=Unclassified | 8=Living Street | 9=Other]', fontsize=8.5)
ax.set_ylabel('Mean Traffic Volume (vehicles/hour)\n[Average traffic across all roads of each type]', fontsize=10)
ax.set_title('B. Which Road Types Carry Most Traffic?\n[Red (<20)=barely used | Orange (20-50)=light use | Green (>50)=moderate+ traffic]', 
             fontsize=11, fontweight='bold', pad=10)
ax.grid(True, alpha=0.3, axis='y')
ax.set_xticks(unique_types)
ax.set_xticklabels([f'{int(ht)}\n{highway_type_names.get(int(ht), "?")[:4]}' for ht in unique_types], fontsize=8)
ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
for bar, val, ht in zip(bars, means, unique_types):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
           f'{val:.1f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

# 6.3 Zero traffic percentage by type with full names
ax = axes[1, 0]
zero_pcts = [(highway == ht).sum() and 100 * (vol_base_case[highway == ht] == 0).sum() / (highway == ht).sum() for ht in unique_types]
colors = ['#e74c3c' if pct > 50 else '#f39c12' if pct > 20 else '#27ae60' for pct in zero_pcts]
bars = ax.bar(unique_types, zero_pcts, color=colors, alpha=0.7, edgecolor='black', linewidth=0.7)
ax.set_xlabel('Road Type\n[Full Classification: 0=Motorway, 1=Trunk, 2=Primary, 3=Secondary, 4=Tertiary,\n5=Residential, 6=Service, 7=Unclassified, 8=Living Street, 9=Other]', fontsize=8.5)
ax.set_ylabel('Zero Traffic Percentage (%)\n[What fraction of each road type is empty?]', fontsize=10)
ax.set_title('C. Road Utilization by Type\n[Red (>50% empty)=poor utilization | Green (<20% empty)=well-utilized network]', fontsize=11, fontweight='bold', pad=10)
ax.grid(True, alpha=0.3, axis='y')
ax.set_xticks(unique_types)
ax.set_xticklabels([f'{int(ht)}\n{highway_type_names.get(int(ht), "?")[:4]}' for ht in unique_types], fontsize=8)
ax.axhline(50, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='50% critical')
ax.axhline(20, color='orange', linestyle='--', alpha=0.5, linewidth=1.5, label='20% warning')
ax.legend(loc='upper right', framealpha=0.9, fontsize=8)
ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
# Add percentage labels
for bar, pct in zip(bars, zero_pcts):
    if pct > 5:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
               f'{pct:.0f}%', ha='center', va='bottom', fontsize=7, fontweight='bold')

# 6.4 Road count by type with full names and percentages
ax = axes[1, 1]
counts = [(highway == ht).sum() for ht in unique_types]
total_roads = sum(counts)
bars = ax.bar(unique_types, counts, width=0.6, alpha=0.8, color='#16a085', edgecolor='black', linewidth=0.7)
ax.set_xlabel('Road Type (Full Classification)\n[0=Motorway | 1=Trunk | 2=Primary | 3=Secondary | 4=Tertiary\n5=Residential | 6=Service | 7=Unclassified | 8=Living Street | 9=Other]', fontsize=8.5)
ax.set_ylabel('Number of Road Segments\n[Total count in Paris MATSim network]', fontsize=10)
ax.set_title('D. Network Composition by Road Type\n[Which road types dominate the network? Highway-heavy or local-street-heavy?]', fontsize=11, fontweight='bold', pad=10)
ax.grid(True, alpha=0.3, axis='y')
ax.set_xticks(unique_types)
ax.set_xticklabels([f'{int(ht)}\n{highway_type_names.get(int(ht), "?")[:4]}' for ht in unique_types], fontsize=8)
ax.yaxis.set_major_locator(ticker.MultipleLocator(2000))
for bar, val, ht in zip(bars, counts, unique_types):
    pct = 100 * val / total_roads
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, 
           f'{val:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

plt.tight_layout()
plt.savefig('feature0_chart6_highway_types.png', dpi=300, bbox_inches='tight')
print("✓ Saved: feature0_chart6_highway_types.png")
plt.show()  # Display in Colab
plt.close()

################################################################################
# CHART 7: SPATIAL DISTRIBUTION
################################################################################
print("\n" + "=" * 80)
print("CHART 7: Spatial Distribution")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(16, 13))
fig.suptitle('FEATURE 0: Spatial Distribution of Traffic', 
             fontsize=14, fontweight='bold', y=0.995)
plt.subplots_adjust(left=0.08, right=0.95, top=0.94, bottom=0.06, hspace=0.30, wspace=0.22)

if hasattr(first_scenario, 'pos') and first_scenario.pos is not None:
    pos_raw = first_scenario.pos.numpy()
    edge_index = first_scenario.edge_index.numpy()
    
    # Handle 3D pos array
    if len(pos_raw.shape) == 3:
        pos = pos_raw[:, 0, :]
        print(f"Extracted 2D coordinates from {pos_raw.shape} -> {pos.shape}")
    else:
        pos = pos_raw
    
    # Calculate edge midpoints
    n_edges_to_plot = min(vol_base_case.shape[0], edge_index.shape[1])
    src_indices = edge_index[0, :n_edges_to_plot]
    dst_indices = edge_index[1, :n_edges_to_plot]
    src_pos = pos[src_indices]
    dst_pos = pos[dst_indices]
    edge_midpoints = (src_pos + dst_pos) / 2
    
    # 7.1 All traffic
    ax = axes[0, 0]
    scatter = ax.scatter(edge_midpoints[:, 0], edge_midpoints[:, 1], 
                        c=vol_base_case, cmap='YlOrRd', s=1, alpha=0.6, 
                        vmin=0, vmax=np.percentile(vol_base_case, 95))
    plt.colorbar(scatter, ax=ax, label='Traffic Volume (veh/h)')
    ax.set_xlabel('X Coordinate (meters from origin)\n[Geographic position: West to East across Paris]', fontsize=10)
    ax.set_ylabel('Y Coordinate (meters from origin)\n[Geographic position: South to North across Paris]', fontsize=10)
    ax.set_title('A. Spatial Traffic Map - Where Is Traffic Concentrated?\n[Yellow=low traffic | Orange=moderate | Red=high traffic]', fontsize=11, fontweight='bold', pad=10)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    
    # 7.2 High-traffic corridors
    ax = axes[0, 1]
    high_threshold = np.percentile(vol_base_case[vol_base_case > 0], 90)
    high_mask = vol_base_case > high_threshold
    ax.scatter(edge_midpoints[~high_mask, 0], edge_midpoints[~high_mask, 1],
              c='lightgray', s=0.5, alpha=0.3, label=f'Normal traffic ({(~high_mask).sum():,} roads)')
    scatter = ax.scatter(edge_midpoints[high_mask, 0], edge_midpoints[high_mask, 1],
                        c=vol_base_case[high_mask], cmap='hot', s=5, alpha=0.8, label=f'High traffic ({high_mask.sum():,} roads)')
    plt.colorbar(scatter, ax=ax, label='Traffic Volume (veh/h)')
    ax.set_xlabel('X Coordinate (meters from origin)\n[Shows location of busiest roads in city]', fontsize=10)
    ax.set_ylabel('Y Coordinate (meters from origin)\n[North-South position in network]', fontsize=10)
    ax.set_title(f'B. Major Traffic Corridors - Busiest 10% of Roads\n[Threshold: >{high_threshold:.0f} veh/h | Yellow=moderate | Red=extremely busy]', fontsize=11, fontweight='bold', pad=10)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8, framealpha=0.9)
    
    # 7.3 Zero-traffic roads
    ax = axes[1, 0]
    zero_mask = vol_base_case == 0
    ax.scatter(edge_midpoints[~zero_mask, 0], edge_midpoints[~zero_mask, 1],
              c='#27ae60', s=0.5, alpha=0.3, label=f'Active roads ({(~zero_mask).sum():,})')
    ax.scatter(edge_midpoints[zero_mask, 0], edge_midpoints[zero_mask, 1],
              c='#e74c3c', s=2, alpha=0.6, label=f'Unused roads ({zero_mask.sum():,} = {zero_mask.sum()/len(vol_base_case)*100:.1f}%)')
    ax.set_xlabel('X Coordinate (meters from origin)\n[Geographic location across city]', fontsize=10)
    ax.set_ylabel('Y Coordinate (meters from origin)\n[Are empty roads in city center or outskirts?]', fontsize=10)
    ax.set_title('C. Where Are Empty Roads Located?\n[Green=roads with traffic | Red=unused roads in simulation]', fontsize=11, fontweight='bold', pad=10)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8, framealpha=0.9)
    
    # 7.4 Density heatmap
    ax = axes[1, 1]
    h, xedges, yedges, im = ax.hist2d(edge_midpoints[:, 0], edge_midpoints[:, 1],
                                      bins=50, weights=vol_base_case, cmap='viridis')
    plt.colorbar(im, ax=ax, label='Cumulative Traffic (veh/h)')
    ax.set_xlabel('X Coordinate (meters from origin)\n[West to East across Paris network]', fontsize=10)
    ax.set_ylabel('Y Coordinate (meters from origin)\n[South to North across Paris network]', fontsize=10)
    ax.set_title('D. Traffic Density Heatmap - Where Is Traffic Most Concentrated?\n[Dark blue=low density area | Yellow/Green=high density traffic zones]', fontsize=11, fontweight='bold', pad=10)
    ax.set_aspect('equal', adjustable='box')
else:
    for ax in axes.flat:
        ax.text(0.5, 0.5, 'No spatial data available', 
               ha='center', va='center', transform=ax.transAxes)

plt.tight_layout()
plt.savefig('feature0_chart7_spatial_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: feature0_chart7_spatial_distribution.png")
plt.show()  # Display in Colab
plt.close()

################################################################################
# CHART 8: OUTLIERS & ANOMALIES
################################################################################
print("\n" + "=" * 80)
print("CHART 8: Outliers & Anomalies")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(16, 13))
fig.suptitle('FEATURE 0: Outlier and Anomaly Detection', 
             fontsize=14, fontweight='bold', y=0.995)
plt.subplots_adjust(left=0.08, right=0.95, top=0.94, bottom=0.06, hspace=0.30, wspace=0.22)

# IQR method
vol_nonzero = vol_base_case[vol_base_case > 0]
Q1 = np.percentile(vol_nonzero, 25)
Q3 = np.percentile(vol_nonzero, 75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers_iqr = (vol_nonzero < lower_bound) | (vol_nonzero > upper_bound)

# Z-score method
z_scores = np.abs(stats.zscore(vol_nonzero))
outliers_z = z_scores > 3

# 8.1 IQR visualization
ax = axes[0, 0]
ax.hist(vol_nonzero, bins=100, alpha=0.7, color='#3498db', edgecolor='black', linewidth=0.5)
ax.axvline(Q1, color='green', linestyle='--', linewidth=2, label=f'Q1 (25th percentile) = {Q1:.1f} veh/h', alpha=0.8)
ax.axvline(Q3, color='orange', linestyle='--', linewidth=2, label=f'Q3 (75th percentile) = {Q3:.1f} veh/h', alpha=0.8)
ax.axvline(lower_bound, color='red', linestyle='--', linewidth=2, label=f'Lower bound = {lower_bound:.1f}', alpha=0.8)
ax.axvline(upper_bound, color='red', linestyle='--', linewidth=2, label=f'Upper bound = {upper_bound:.1f}', alpha=0.8)
ax.set_xlabel('Traffic Volume (vehicles/hour)\n[Values outside red lines are considered outliers]', fontsize=10)
ax.set_ylabel('Frequency (Number of Roads)\n[How many roads have each traffic volume]', fontsize=10)
ax.set_title(f'A. IQR Outlier Detection Method\n[Found {outliers_iqr.sum():,} outliers = {outliers_iqr.sum()/len(vol_nonzero)*100:.1f}% of active roads | IQR = {IQR:.1f}]', 
             fontsize=11, fontweight='bold', pad=10)
ax.legend(loc='best', fontsize=8, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(ticker.MultipleLocator(200))

# 8.2 Z-score distribution
ax = axes[0, 1]
ax.hist(z_scores, bins=100, alpha=0.7, color='#e67e22', edgecolor='black', linewidth=0.5)
ax.axvline(3, color='red', linestyle='--', linewidth=2.5, label='Z=3 threshold (3 std deviations)', alpha=0.8)
ax.set_xlabel('Z-Score (Absolute Value)\n[Measures how many standard deviations from mean | Example: Z=3 means 3× away]', fontsize=10)
ax.set_ylabel('Frequency (Number of Roads)\n[Most roads near Z=0 (average) | Few at high Z (extreme)]', fontsize=10)
ax.set_title(f'B. Z-Score Statistical Outlier Detection\n[Found {outliers_z.sum():,} extreme outliers = {outliers_z.sum()/len(vol_nonzero)*100:.2f}% | Z>3 is unusual]', 
             fontsize=11, fontweight='bold', pad=10)
ax.legend(loc='best', fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))

# 8.3 Outlier characteristics
ax = axes[1, 0]
outlier_mask = np.zeros(n_edges, dtype=bool)
outlier_mask[vol_base_case > 0] = outliers_iqr
normal_mask = ~outlier_mask & (vol_base_case > 0)

ax.scatter(capacity[normal_mask], vol_base_case[normal_mask], 
          alpha=0.3, s=5, c='gray', label=f'Normal roads ({normal_mask.sum():,})', edgecolors='none')
ax.scatter(capacity[outlier_mask], vol_base_case[outlier_mask], 
          alpha=0.7, s=20, c='red', label=f'Outlier roads ({outlier_mask.sum():,})', edgecolors='black', linewidth=0.5)
ax.set_xlabel('Road Capacity (vehicles/hour)\n[Maximum traffic the road can handle]', fontsize=10)
ax.set_ylabel('Baseline Traffic Volume (vehicles/hour)\n[Current actual traffic - outliers shown in red]', fontsize=10)
ax.set_title('C. What Makes Outliers Different?\n[Do outliers have unusually high capacity or volume?]', fontsize=11, fontweight='bold', pad=10)
ax.legend(loc='best', fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1000))
ax.yaxis.set_major_locator(ticker.MultipleLocator(200))

# 8.4 Top extreme values
ax = axes[1, 1]
top_n = min(20, len(vol_nonzero))
top_indices = np.argsort(vol_nonzero)[-top_n:]
top_values = vol_nonzero[top_indices]
bars = ax.barh(range(top_n), top_values, alpha=0.8, color='#c0392b', edgecolor='black', linewidth=0.7)
ax.set_xlabel('Traffic Volume (vehicles/hour)\n[Example: 1500 veh/h = 1500 cars pass through that road per hour]', fontsize=10)
ax.set_ylabel('Ranking (1 = Busiest Road in Network)\n[Top 20 most congested road segments]', fontsize=10)
ax.set_title(f'D. The 20 Busiest Roads in Paris Network\n[Maximum traffic: {top_values[-1]:.0f} veh/h | Minimum in top 20: {top_values[0]:.0f} veh/h]', fontsize=11, fontweight='bold', pad=10)
ax.grid(True, alpha=0.3, axis='x')
ax.xaxis.set_major_locator(ticker.MultipleLocator(200))
ax.invert_yaxis()  # Highest traffic at top
ax.set_yticks([0, 5, 10, 15, 19])
ax.set_yticklabels(['#1\n(Busiest)', '#5', '#10', '#15', '#20'])
# Add value labels
for i, (bar, val) in enumerate(zip(bars, top_values)):
    ax.text(val + 20, bar.get_y() + bar.get_height()/2, f'{val:.0f}', 
           va='center', ha='left', fontsize=7, fontweight='bold')

plt.tight_layout()
plt.savefig('feature0_chart8_outliers.png', dpi=300, bbox_inches='tight')
print("✓ Saved: feature0_chart8_outliers.png")
plt.show()  # Display in Colab
plt.close()

print(f"\nOutlier Detection:")
print(f"  IQR method: {outliers_iqr.sum()} outliers ({outliers_iqr.sum()/len(vol_nonzero)*100:.1f}%)")
print(f"  Z-score method: {outliers_z.sum()} outliers ({outliers_z.sum()/len(vol_nonzero)*100:.1f}%)")
print(f"  Top 5 values: {np.sort(vol_base_case)[-5:]}")

print("\n" + "=" * 80)
print("✓✓✓ PART 2 COMPLETE - Charts 5-8 Generated Successfully ✓✓✓")
print("=" * 80)
print("\nGenerated Charts:")
print("  5. feature0_chart5_capacity_relationship.png - Volume-capacity analysis")
print("  6. feature0_chart6_highway_types.png - Traffic by road type")
print("  7. feature0_chart7_spatial_distribution.png - Geographic traffic patterns")
print("  8. feature0_chart8_outliers.png - Outlier detection analysis")
print("\n✓ Charts displayed inline above (Colab)")
print("✓ PNG files saved in current directory")
print(f"\nKey Findings:")
print(f"  • Volume-Capacity Correlation: {corr_vol_cap:.4f} (moderate positive)")
print(f"  • Average Utilization: {util_nonzero.mean()*100:.1f}% of road capacity")
print(f"  • Outliers Detected: {outliers_iqr.sum():,} roads (IQR method)")
print(f"  • Top Traffic Volume: {np.sort(vol_base_case)[-1]:.0f} veh/h")
print("\nNext: Run feature0_part3_charts9to12.py for Advanced Analysis")
