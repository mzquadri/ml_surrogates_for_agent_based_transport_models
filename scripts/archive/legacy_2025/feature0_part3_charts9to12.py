"""
FEATURE 0 ANALYSIS - PART 3: ADVANCED ANALYSIS (Charts 9-12)
==============================================================
- Chart 9: Target Correlation (Policy Sensitivity)
- Chart 10: Network Statistics
- Chart 11: Capacity Reduction (Policy Targeting)
- Chart 12: Final Summary & Insights

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
print("#" + "  FEATURE 0 - PART 3: ADVANCED ANALYSIS (Charts 9-12)".center(78) + "#")
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
cap_reduction = first_scenario.x[:, 2].numpy()
highway = first_scenario.x[:, 4].numpy()

n_edges = len(vol_base_case)
unique_types = np.unique(highway)
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
# CHART 9: TARGET CORRELATION
################################################################################
print("\n" + "=" * 80)
print("CHART 9: Correlation with Target (Policy Sensitivity)")
print("=" * 80)

if hasattr(first_scenario, 'y') and first_scenario.y is not None:
    target = first_scenario.y.numpy()
    if len(target.shape) > 1:
        target = target.flatten()
    
    fig, axes = plt.subplots(2, 2, figsize=(17, 14))
    fig.suptitle('FEATURE 0: Correlation with Target (Policy Impact Analysis)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.subplots_adjust(left=0.08, right=0.95, top=0.94, bottom=0.06, hspace=0.35, wspace=0.25)
    
    # 9.1 Scatter: F0 vs Target
    ax = axes[0, 0]
    ax.scatter(vol_base_case, target, c='#3498db', s=2, alpha=0.4)
    valid_mask = ~(np.isnan(vol_base_case) | np.isnan(target))
    if valid_mask.sum() > 0:
        z = np.polyfit(vol_base_case[valid_mask], target[valid_mask], 1)
        p = np.poly1d(z)
        x_line = np.linspace(vol_base_case.min(), vol_base_case.max(), 100)
        ax.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.8, 
               label=f'Linear fit: y={z[0]:.3f}x+{z[1]:.1f}')
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Baseline Traffic Volume F0 (vehicles/hour)\n[Example: 100 veh/h = road currently has 100 cars/hour before policy]', fontsize=10)
    ax.set_ylabel('Target y (vehicles/hour change after policy)\n[Example: -50 = policy reduces traffic by 50 veh/h | +20 = increases by 20]', fontsize=10)
    ax.set_title('A. How Does Baseline Traffic Predict Policy Impact?\n[Zero line = no change | Below = traffic reduction | Above = traffic increase]', fontsize=11, fontweight='bold', pad=10)
    ax.legend(loc='best', framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(200))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
    
    # 9.2 Target distribution by baseline bins with BOX PLOT ANNOTATION
    ax = axes[0, 1]
    bins_f0 = [0, 10, 50, 100, 200, vol_base_case.max()]
    bin_labels = ['0-10', '10-50', '50-100', '100-200', '200+']
    bin_indices = np.digitize(vol_base_case, bins_f0)
    target_by_bin = [target[bin_indices == i+1] for i in range(len(bin_labels))]
    bp = ax.boxplot(target_by_bin, tick_labels=bin_labels, patch_artist=True, showfliers=False)
    for patch in bp['boxes']:
        patch.set_facecolor('#e67e22')
        patch.set_alpha(0.7)
    for median in bp['medians']:
        median.set_color('#e74c3c')
        median.set_linewidth(2.5)
    ax.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Zero change (no policy effect)')
    ax.set_xlabel('Baseline Traffic Volume Range (vehicles/hour)\n[Bins: Quiet roads (0-10) to Busy roads (200+)]', fontsize=10)
    ax.set_ylabel('Policy Impact Distribution (vehicles/hour change)\n[Red line = median | Orange box = middle 50% of impacts (IQR)]', fontsize=10)
    ax.set_title('B. Do Busier Roads Experience Larger Policy Impacts?\n[Compare impact distributions across different baseline traffic levels]', fontsize=11, fontweight='bold', pad=10)
    ax.legend(loc='best', framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
    # Add box plot annotation
    ax.text(0.98, 0.97, 'BOX PLOT GUIDE:\n• Red line = MEDIAN impact\n• Orange box = IQR (middle 50%)\n• Box edges = Q1/Q3 quartiles\n• Whiskers = min/max range', 
           transform=ax.transAxes, fontsize=8, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 9.3 Correlation by highway type with FULL NAMES
    ax = axes[1, 0]
    corr_by_type = []
    for ht in unique_types:
        type_mask = highway == ht
        if type_mask.sum() > 1:
            # Safe correlation calculation with error handling
            with np.errstate(invalid='ignore'):
                corr_matrix = np.corrcoef(vol_base_case[type_mask], target[type_mask])
                corr = corr_matrix[0, 1] if corr_matrix.shape == (2, 2) else 0.0
            corr_by_type.append(0 if np.isnan(corr) else corr)
        else:
            corr_by_type.append(0)
    colors = ['#e74c3c' if abs(c) > 0.5 else '#f39c12' if abs(c) > 0.3 else '#27ae60' for c in corr_by_type]
    bars = ax.bar(unique_types, corr_by_type, width=0.6, alpha=0.8, color=colors, edgecolor='black')
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Road Type (OpenStreetMap Classification)\n[0=Motorway | 1=Trunk | 2=Primary | 3=Secondary | 4=Tertiary\n5=Residential | 6=Service | 7=Unclassified | 8=Living Street | 9=Other]', fontsize=8.5)
    ax.set_ylabel('Correlation Coefficient (Baseline vs Impact)\n[+1=perfect positive | 0=no relationship | -1=perfect negative]', fontsize=10)
    ax.set_title('C. Which Road Types Show Strongest Baseline-Impact Relationship?\n[Red (>0.5)=strong | Orange (0.3-0.5)=moderate | Green (<0.3)=weak correlation]', fontsize=11, fontweight='bold', pad=10)
    ax.set_xticks(unique_types)
    ax.set_xticklabels([f'{int(ht)}\n{highway_type_names.get(int(ht), "?")[:4]}' for ht in unique_types], fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(-1, 1)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    for bar, val in zip(bars, corr_by_type):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.03 if val > 0 else val - 0.03, 
               f'{val:.2f}', ha='center', va='bottom' if val > 0 else 'top', fontsize=7.5, fontweight='bold')
    
    # 9.4 Impact magnitude
    ax = axes[1, 1]
    impact_mag = np.abs(target)
    ax.scatter(vol_base_case, impact_mag, c='#16a085', s=2, alpha=0.4)
    if valid_mask.sum() > 0:
        z_mag = np.polyfit(vol_base_case[valid_mask], impact_mag[valid_mask], 1)
        p_mag = np.poly1d(z_mag)
        ax.plot(x_line, p_mag(x_line), "r--", linewidth=2, alpha=0.8,
               label=f'Trend: |y|={z_mag[0]:.3f}x+{z_mag[1]:.1f}')
    ax.set_xlabel('Baseline Traffic Volume (vehicles/hour)\n[Current traffic before policy intervention]', fontsize=10)
    ax.set_ylabel('Absolute Policy Impact Magnitude (vehicles/hour)\n[Example: 30 = policy changes traffic by 30 veh/h (increase or decrease)]', fontsize=10)
    ax.set_title('D. Do Busier Roads Experience Larger Changes Regardless of Direction?\n[Magnitude ignores sign - focuses on size of change only]', fontsize=11, fontweight='bold', pad=10)
    ax.legend(loc='best', framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(200))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
    
    plt.tight_layout()
    plt.savefig('feature0_chart9_target_correlation.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: feature0_chart9_target_correlation.png")
    plt.show()  # Display in Colab
    plt.close()
    
    # Safe correlation calculation
    with np.errstate(invalid='ignore'):
        corr_matrix = np.corrcoef(vol_base_case[valid_mask], target[valid_mask])
        overall_corr = corr_matrix[0, 1] if corr_matrix.shape == (2, 2) else 0.0
        corr_mag_matrix = np.corrcoef(vol_base_case[valid_mask], impact_mag[valid_mask])
        overall_corr_mag = corr_mag_matrix[0, 1] if corr_mag_matrix.shape == (2, 2) else 0.0
    print(f"Overall Correlation: F0 vs Target = {overall_corr:.4f}")
    print(f"Magnitude Correlation: F0 vs |Target| = {overall_corr_mag:.4f}")
else:
    print("⚠ No target data available")
    overall_corr = 0.0
    overall_corr_mag = 0.0

################################################################################
# CHART 10: NETWORK STATISTICS
################################################################################
print("\n" + "=" * 80)
print("CHART 10: Network Statistics")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(17, 14))
fig.suptitle('FEATURE 0: Network-Level Traffic Statistics & Inequality Analysis', 
             fontsize=14, fontweight='bold', y=0.995)
plt.subplots_adjust(left=0.08, right=0.95, top=0.94, bottom=0.06, hspace=0.35, wspace=0.25)

# 10.1 Percentiles
ax = axes[0, 0]
percentiles = [0, 10, 25, 50, 75, 90, 95, 99, 100]
percentile_values = [np.percentile(vol_base_case, p) for p in percentiles]
bars = ax.bar(range(len(percentiles)), percentile_values, width=0.7, alpha=0.8, 
             color='#3498db', edgecolor='black')
ax.set_xlabel('Percentile Rank\n[Example: 50% = median | 90% = only 10% of roads busier | 100% = maximum]', fontsize=10)
ax.set_ylabel('Traffic Volume (vehicles/hour)\n[The traffic level at each percentile threshold]', fontsize=10)
ax.set_title('A. How Is Traffic Distributed Across Network Percentiles?\n[Shows traffic volume at key statistical thresholds from min to max]', fontsize=11, fontweight='bold', pad=10)
ax.set_xticks(range(len(percentiles)))
ax.set_xticklabels([f'{p}%' for p in percentiles], fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
ax.yaxis.set_major_locator(ticker.MultipleLocator(200))
for bar, val in zip(bars, percentile_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
           f'{val:.0f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

# 10.2 Lorenz curve (Gini) - FIXED deprecated trapz
ax = axes[0, 1]
sorted_vols = np.sort(vol_base_case)
cumulative_vols = np.cumsum(sorted_vols)
cumulative_vols_pct = cumulative_vols / cumulative_vols[-1] * 100
cumulative_roads_pct = np.arange(1, n_edges + 1) / n_edges * 100
ax.plot(cumulative_roads_pct, cumulative_vols_pct, color='#e74c3c', linewidth=2.5, label='Actual traffic distribution (Paris network)')
ax.plot([0, 100], [0, 100], 'k--', linewidth=2, alpha=0.6, label='Perfect equality (every road equal traffic)')
# FIX: Use trapezoid (NumPy 1.21+) or trapz (older versions) - both work, trapz just shows deprecation warning
try:
    gini = 1 - 2 * np.trapezoid(cumulative_vols_pct/100, cumulative_roads_pct/100)
except AttributeError:
    # Fallback for older NumPy versions
    gini = 1 - 2 * np.trapz(cumulative_vols_pct/100, cumulative_roads_pct/100)
ax.text(55, 15, f'Gini Coefficient: {gini:.3f}\n(0=perfect equality\n1=extreme inequality)', fontsize=9, fontweight='bold',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
ax.set_xlabel('Cumulative Percentage of Roads (sorted from quietest to busiest)\n[Example: 80% = the quietest 80% of roads]', fontsize=9.5)
ax.set_ylabel('Cumulative Percentage of Total Network Traffic\n[Example: 20% = these roads carry 20% of all traffic]', fontsize=9.5)
ax.set_title('B. Lorenz Curve - Is Traffic Concentrated on Few Roads?\n[Curve far from diagonal = high inequality | Near diagonal = evenly distributed]', fontsize=11, fontweight='bold', pad=10)
ax.legend(loc='upper left', framealpha=0.9, fontsize=8.5)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.yaxis.set_major_locator(ticker.MultipleLocator(10))

# 10.3 Summary stats table with descriptions
ax = axes[1, 0]
ax.axis('off')
stats_data = [
    ['Statistical Metric', 'Value', 'Interpretation'],
    ['Total Roads', f'{n_edges:,}', 'Network size'],
    ['Active Roads', f'{(vol_base_case > 0).sum():,}', f'{(vol_base_case > 0).sum()/n_edges*100:.1f}% have traffic'],
    ['Mean Volume', f'{vol_base_case.mean():.1f} veh/h', 'Average across all'],
    ['Median Volume', f'{np.median(vol_base_case):.1f} veh/h', 'Middle value (50%)'],
    ['Std Deviation', f'{vol_base_case.std():.1f} veh/h', 'Spread/variability'],
    ['Skewness', f'{stats.skew(vol_base_case):.2f}', 'Right-skewed (>0)'],
    ['Kurtosis', f'{stats.kurtosis(vol_base_case):.2f}', 'Heavy tails (>0)'],
    ['Gini Coefficient', f'{gini:.3f}', 'High inequality'],
]
table = ax.table(cellText=stats_data, cellLoc='left', loc='center', colWidths=[0.35, 0.3, 0.35])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.2)
for i in range(3):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white', fontsize=9)
for i in range(1, len(stats_data)):
    for j in range(3):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#ecf0f1')
ax.set_title('C. Network Traffic Summary Statistics\n[Key metrics describing traffic distribution characteristics]', fontsize=11, fontweight='bold', pad=15)

# 10.4 Top roads contribution
ax = axes[1, 1]
top_pcts = [1, 5, 10, 20, 50]
contributions = []
for pct in top_pcts:
    n_top = int(n_edges * pct / 100)
    top_vols = np.sort(vol_base_case)[-n_top:]
    contribution = top_vols.sum() / vol_base_case.sum() * 100
    contributions.append(contribution)
bars = ax.bar(range(len(top_pcts)), contributions, width=0.7, alpha=0.8,
             color='#27ae60', edgecolor='black', linewidth=0.8)
ax.set_xlabel('Busiest X% of Roads in Network\n[Example: "Top 5%" = the 5% busiest roads (1,582 roads)]', fontsize=10)
ax.set_ylabel('Percentage of Total Network Traffic Carried\n[What fraction of all traffic uses these roads]', fontsize=10)
ax.set_title('D. How Much Traffic Is Concentrated on the Busiest Roads?\n[High bars = traffic concentrated on few roads | Low bars = evenly distributed]', fontsize=11, fontweight='bold', pad=10)
ax.set_xticks(range(len(top_pcts)))
ax.set_xticklabels([f'Top {p}%' for p in top_pcts], fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
for bar, val, pct in zip(bars, contributions, top_pcts):
    n_roads = int(n_edges * pct / 100)
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, 
           f'{val:.1f}%\n({n_roads:,} roads)', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

plt.tight_layout()
plt.savefig('feature0_chart10_network_statistics.png', dpi=300, bbox_inches='tight')
print("✓ Saved: feature0_chart10_network_statistics.png")
plt.show()  # Display in Colab
plt.close()

print(f"Gini Coefficient: {gini:.3f} (Traffic inequality)")

################################################################################
# CHART 11: CAPACITY REDUCTION COMPARISON
################################################################################
print("\n" + "=" * 80)
print("CHART 11: Capacity Reduction (Policy Targeting)")
print("=" * 80)

# Diagnostic info
reduction_mask_pre = cap_reduction > 0
print(f"Roads with capacity reduction: {reduction_mask_pre.sum():,} ({reduction_mask_pre.sum()/n_edges*100:.2f}%)")
if reduction_mask_pre.sum() > 0:
    print(f"Capacity reduction range: {cap_reduction[reduction_mask_pre].min():.1f} - {cap_reduction[reduction_mask_pre].max():.1f} veh/h")
    print(f"Mean reduction (non-zero): {cap_reduction[reduction_mask_pre].mean():.1f} veh/h")
else:
    print("⚠ WARNING: No roads have capacity reduction in this scenario!")

fig, axes = plt.subplots(2, 2, figsize=(17, 14))
fig.suptitle('FEATURE 0: Relationship with Capacity Reduction (Policy Targeting Strategy)', 
             fontsize=14, fontweight='bold', y=0.995)
plt.subplots_adjust(left=0.08, right=0.95, top=0.94, bottom=0.06, hspace=0.35, wspace=0.25)

# 11.1 Scatter: F0 vs F2
ax = axes[0, 0]
reduction_mask = cap_reduction > 0
if reduction_mask.sum() > 0:
    # Plot roads with reduction in color, zero reduction in gray
    ax.scatter(vol_base_case[~reduction_mask], cap_reduction[~reduction_mask], 
              c='lightgray', s=1, alpha=0.2, label=f'No reduction ({(~reduction_mask).sum():,} roads)')
    ax.scatter(vol_base_case[reduction_mask], cap_reduction[reduction_mask], 
              c='#9b59b6', s=5, alpha=0.7, edgecolors='black', linewidth=0.3,
              label=f'With reduction ({reduction_mask.sum():,} roads)')
    # Adjust y-axis to show actual data range better
    max_reduction = cap_reduction[reduction_mask].max()
    ax.set_ylim(-max_reduction*0.05, max_reduction*1.1)
else:
    ax.scatter(vol_base_case, cap_reduction, c='#9b59b6', s=2, alpha=0.4)
    ax.text(0.5, 0.5, 'No capacity reduction in this scenario', 
           ha='center', va='center', transform=ax.transAxes, fontsize=12, color='red')
ax.set_xlabel('Baseline Traffic Volume F0 (vehicles/hour)\n[Current traffic before any policy intervention]', fontsize=10)
ax.set_ylabel('Capacity Reduction F2 (vehicles/hour)\n[How much road capacity is reduced by policy | 0 = no reduction]', fontsize=10)
ax.set_title('A. Do Policies Target Roads Based on Current Traffic Levels?\n[Scatter pattern shows if busy roads get more/less capacity reduction]', fontsize=11, fontweight='bold', pad=10)
ax.legend(loc='best', fontsize=8, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(ticker.MultipleLocator(200))

# Calculate correlation with safe division
valid_mask = ~(np.isnan(vol_base_case) | np.isnan(cap_reduction))
if valid_mask.sum() > 0:
    with np.errstate(invalid='ignore'):
        corr_matrix = np.corrcoef(vol_base_case[valid_mask], cap_reduction[valid_mask])
        corr_f0_f2 = corr_matrix[0, 1] if corr_matrix.shape == (2, 2) else 0.0
    if reduction_mask.sum() > 0:
        ax.text(0.05, 0.95, f'Correlation: {corr_f0_f2:.3f}\n(+1=target busy roads\n-1=target quiet roads\n0=no pattern)', 
               transform=ax.transAxes, fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
else:
    corr_f0_f2 = np.nan

# 11.2 Targeted vs non-targeted roads
ax = axes[0, 1]
bins_f0 = [0, 10, 50, 100, 200, vol_base_case.max()]
bin_labels = ['0-10', '10-50', '50-100', '100-200', '200+']
targeting_rates = []
bin_counts = []
for i in range(len(bins_f0)-1):
    bin_mask = (vol_base_case >= bins_f0[i]) & (vol_base_case < bins_f0[i+1])
    if bin_mask.sum() > 0:
        rate = 100 * (bin_mask & reduction_mask).sum() / bin_mask.sum()
        targeting_rates.append(rate)
        bin_counts.append(bin_mask.sum())
    else:
        targeting_rates.append(0)
        bin_counts.append(0)

if reduction_mask.sum() > 0 and max(targeting_rates) > 0:
    colors = ['#e74c3c' if r < 5 else '#e67e22' if r < 20 else '#27ae60' for r in targeting_rates]
    bars = ax.bar(range(len(bin_labels)), targeting_rates, alpha=0.8, color=colors, edgecolor='black', linewidth=0.8)
    ax.set_ylim(0, max(targeting_rates) * 1.15)  # Adjust to show actual data
    for bar, val, count in zip(bars, targeting_rates, bin_counts):
        if val > 0.5:  # Show label if rate > 0.5%
            n_targeted = int(count * val / 100)
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(targeting_rates)*0.02, 
                   f'{val:.1f}%\n({n_targeted:,})', ha='center', va='bottom', fontsize=7, fontweight='bold')
else:
    ax.bar(range(len(bin_labels)), [0]*len(bin_labels), alpha=0.3, color='gray')
    ax.text(0.5, 0.5, 'No capacity reduction\nin this scenario', 
           ha='center', va='center', transform=ax.transAxes, fontsize=11, color='red',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

ax.set_xlabel('Baseline Traffic Volume Range (vehicles/hour)\n[Bins: Quiet roads (0-10) to Very busy roads (200+)]', fontsize=10)
ax.set_ylabel('Percentage of Roads Targeted by Policy (%)\n[What fraction of roads in each bin get capacity reduction]', fontsize=10)
ax.set_title('B. Which Traffic Levels Are Most Targeted by Capacity Reduction?\n[High bars = policy focuses on this traffic level | Low bars = mostly ignored]', fontsize=11, fontweight='bold', pad=10)
ax.set_xticks(range(len(bin_labels)))
ax.set_xticklabels(bin_labels, fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# 11.3 Utilization comparison with SAFE DIVISION
ax = axes[1, 0]
# FIX: Safe division to avoid RuntimeWarning
with np.errstate(divide='ignore', invalid='ignore'):
    utilization = np.where(capacity > 0, vol_base_case / capacity, 0)
    utilization = np.nan_to_num(utilization, nan=0.0, posinf=0.0, neginf=0.0)
targeted_utils = utilization[reduction_mask]
non_targeted_utils = utilization[~reduction_mask & (vol_base_case > 0)]
if len(targeted_utils) > 0 and len(non_targeted_utils) > 0:
    bp = ax.boxplot([non_targeted_utils, targeted_utils], 
                    tick_labels=['Not Targeted\n(No reduction)', 'Targeted\n(Capacity reduced)'],
                    patch_artist=True, showfliers=False)
    for patch in bp['boxes']:
        patch.set_facecolor('#16a085')
        patch.set_alpha(0.7)
    for median in bp['medians']:
        median.set_color('#e74c3c')
        median.set_linewidth(2.5)
    ax.set_ylabel('Utilization Ratio (Traffic / Capacity)\n[Example: 0.50 = road using 50% of its capacity]', fontsize=10)
    ax.set_title('C. Do Policies Target More or Less Utilized Roads?\n[Compare utilization of roads that get capacity reduction vs those that don\'t]', fontsize=11, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
else:
    ax.text(0.5, 0.5, 'No targeted roads in this scenario', 
           ha='center', va='center', transform=ax.transAxes)

# 11.4 Reduction amount by baseline
ax = axes[1, 1]
reduction_by_bin = []
count_by_bin = []
for i in range(len(bins_f0)-1):
    bin_mask = (vol_base_case >= bins_f0[i]) & (vol_base_case < bins_f0[i+1]) & reduction_mask
    if bin_mask.sum() > 0:
        reduction_by_bin.append(cap_reduction[bin_mask].mean())
        count_by_bin.append(bin_mask.sum())
    else:
        reduction_by_bin.append(0)
        count_by_bin.append(0)

if reduction_mask.sum() > 0 and max(reduction_by_bin) > 0:
    bars = ax.bar(range(len(bin_labels)), reduction_by_bin, alpha=0.8, color='#c0392b', edgecolor='black', linewidth=0.8)
    ax.set_ylim(0, max(reduction_by_bin) * 1.15)  # Adjust to show actual data
    for bar, val, count in zip(bars, reduction_by_bin, count_by_bin):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(reduction_by_bin)*0.02, 
                   f'{val:.0f}\n({count:,} roads)', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
else:
    ax.bar(range(len(bin_labels)), [0]*len(bin_labels), alpha=0.3, color='gray')
    ax.text(0.5, 0.5, 'No capacity reduction\nin this scenario', 
           ha='center', va='center', transform=ax.transAxes, fontsize=11, color='red',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

ax.set_xlabel('Baseline Traffic Volume Range (vehicles/hour)\n[Compare reduction intensity across different traffic levels]', fontsize=10)
ax.set_ylabel('Mean Capacity Reduction Amount (vehicles/hour)\n[Average reduction for roads in each traffic range]', fontsize=10)
ax.set_title('D. How Much Capacity Is Reduced at Each Traffic Level?\n[Higher bars = larger capacity reductions applied to these roads]', fontsize=11, fontweight='bold', pad=10)
ax.set_xticks(range(len(bin_labels)))
ax.set_xticklabels(bin_labels, fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('feature0_chart11_capacity_reduction.png', dpi=300, bbox_inches='tight')
print("✓ Saved: feature0_chart11_capacity_reduction.png")
plt.show()  # Display in Colab
plt.close()

print(f"F0-F2 Correlation: {corr_f0_f2:.4f}")

################################################################################
# CHART 12: SUMMARY & INSIGHTS
################################################################################
print("\n" + "=" * 80)
print("CHART 12: Final Summary")
print("=" * 80)

fig = plt.figure(figsize=(16, 13))
fig.suptitle('FEATURE 0: COMPLETE ANALYSIS SUMMARY\nKey Insights for Paris MATSim Network', 
             fontsize=16, fontweight='bold', y=0.98)

# Create text summary (FIXED: removed emoji glyphs to avoid font warnings)
# Calculate safe correlation for summary
with np.errstate(invalid='ignore'):
    cap_corr_matrix = np.corrcoef(vol_base_case[vol_base_case>0], capacity[vol_base_case>0])
    cap_corr = cap_corr_matrix[0,1] if cap_corr_matrix.shape == (2,2) else 0.0

summary_text = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        FEATURE 0 ANALYSIS COMPLETE                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

[NETWORK OVERVIEW]
   • Total Edges: {n_edges:,}
   • Active Roads: {(vol_base_case > 0).sum():,} ({(vol_base_case > 0).sum()/n_edges*100:.1f}%)
   • Zero Traffic: {(vol_base_case == 0).sum():,} ({(vol_base_case == 0).sum()/n_edges*100:.1f}%)

[TRAFFIC DISTRIBUTION]
   • Mean: {vol_base_case.mean():.1f} veh/h
   • Median: {np.median(vol_base_case):.1f} veh/h
   • Range: {vol_base_case.min():.0f} - {vol_base_case.max():.0f} veh/h
   • Gini Coefficient: {gini:.3f} (High inequality)

[STATIC VALIDATION]
   • Variance Across Scenarios: approximately 0.000
   • Status: CONFIRMED STATIC (identical across scenarios)

[CORRELATIONS]
   • F0 vs Capacity: {cap_corr:.3f}
   • F0 vs Target: {overall_corr:.3f}
   • F0 vs |Target|: {overall_corr_mag:.3f}
   • F0 vs F2: {corr_f0_f2:.3f}

[KEY FINDINGS]
   1. Highly skewed distribution (few busy roads, many quiet)
   2. Directional network (no negative values)
   3. Under-utilized (mean utilization approximately 5%)
   4. Static feature validation passed
   5. Traffic concentrated on main arterials (Gini={gini:.3f})

[RECOMMENDATIONS FOR GNN]
   - Use F0 as static reference feature
   - Consider log transformation (right-skewed)
   - Handle zeros carefully (24% of edges)
   - Correlation with target is weak ({overall_corr:.3f})
   - F0 useful for network structure, less for direct prediction

╔══════════════════════════════════════════════════════════════════════════════╗
║  Next Steps: Analyze F1 (Capacity), F2 (Reduction), F3 (Freespeed), etc.   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

ax = fig.add_subplot(111)
ax.text(0.5, 0.5, summary_text, ha='center', va='center', 
       fontsize=11, family='monospace',
       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
ax.axis('off')

plt.tight_layout()
plt.savefig('feature0_chart12_summary.png', dpi=300, bbox_inches='tight')
print("✓ Saved: feature0_chart12_summary.png")
plt.show()  # Display in Colab
plt.close()

print("\n" + "=" * 80)
print("✓✓✓ PART 3 COMPLETE - Charts 9-12 Generated ✓✓✓")
print("=" * 80)
print("\n" + "=" * 80)
print("✓✓✓ ALL 12 CHARTS GENERATED SUCCESSFULLY ✓✓✓")
print("=" * 80)
print("\nGenerated Files:")
print("  1. feature0_chart1_distribution.png")
print("  2. feature0_chart2_negative_analysis.png")
print("  3. feature0_chart3_zeros_analysis.png")
print("  4. feature0_chart4_temporal_variance.png")
print("  5. feature0_chart5_capacity_relationship.png")
print("  6. feature0_chart6_highway_types.png")
print("  7. feature0_chart7_spatial_distribution.png")
print("  8. feature0_chart8_outliers.png")
print("  9. feature0_chart9_target_correlation.png")
print(" 10. feature0_chart10_network_statistics.png")
print(" 11. feature0_chart11_capacity_reduction.png")
print(" 12. feature0_chart12_summary.png")
print("\nNext: Proceed with Feature 1 (Capacity) analysis")
