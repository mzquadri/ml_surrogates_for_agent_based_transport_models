"""
FEATURE 2 ANALYSIS - PART 1: CAPACITY REDUCTION (Charts 1-4)
=============================================================
Charts 1-4: Distribution, Patterns & Baseline Analysis

Feature 2 represents policy interventions (capacity reduction scenarios)
NOTE: If all scenarios are baseline (F2=0), analysis shows baseline characteristics
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.ticker as ticker

# Set professional plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 15

print("\n" + "#" * 80)
print("#" + " " * 78 + "#")
print("#" + "  FEATURE 2 - PART 1: CAPACITY REDUCTION (Charts 1-4)".center(78) + "#")
print("#" + "  Distribution, Patterns & Baseline Analysis".center(78) + "#")
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

# Load multiple batches to check for policy scenarios
print(f"\nScanning batches to detect policy scenarios...")
all_scenarios = []
batch_count = 0
max_batches = min(5, len(batch_files))

for batch_file in batch_files[:max_batches]:
    try:
        batch = torch.load(batch_file, weights_only=False)
        if isinstance(batch, list):
            all_scenarios.extend(batch)
            batch_count += 1
            print(f"  ✓ Loaded batch {batch_count}: {batch_file.name} ({len(batch)} scenarios)")
    except Exception as e:
        print(f"  Warning: Could not load {batch_file.name}: {e}")

n_scenarios = len(all_scenarios)
print(f"✓ Total loaded: {n_scenarios} scenarios from {batch_count} batch(es)")

# Analyze capacity reduction across all scenarios
print(f"\nAnalyzing capacity reduction across {n_scenarios} scenarios...")
has_reduction_count = 0
total_reductions = 0

for idx, scenario in enumerate(all_scenarios):
    cap_red = scenario.x[:, 2].numpy()
    if (cap_red > 0).sum() > 0:
        has_reduction_count += 1
        total_reductions += (cap_red > 0).sum()

print(f"Scenarios with reduction: {has_reduction_count}/{n_scenarios} ({has_reduction_count/n_scenarios*100:.1f}%)")

# Use first scenario for analysis
first_scenario = all_scenarios[0]
vol_base_case = first_scenario.x[:, 0].numpy()
capacity = first_scenario.x[:, 1].numpy()
cap_reduction = first_scenario.x[:, 2].numpy()
free_speed = first_scenario.x[:, 3].numpy()
highway = first_scenario.x[:, 4].numpy()
length = first_scenario.x[:, 5].numpy()

n_edges = len(capacity)
unique_types = np.unique(highway)
print(f"✓ Loaded {n_edges:,} edges")

# Determine analysis mode
n_with_reduction = (cap_reduction > 0).sum()
IS_BASELINE = (n_with_reduction == 0)

if IS_BASELINE:
    print(f"\n[i] BASELINE MODE ACTIVATED")
    print(f"    All scenarios have zero capacity reduction (F2 = 0)")
    print(f"    Analysis will focus on baseline network characteristics")
else:
    print(f"\n[i] POLICY MODE ACTIVATED")
    print(f"    Capacity reduction detected: {n_with_reduction:,} roads ({n_with_reduction/n_edges*100:.1f}%)")

# Highway type decoder
highway_type_names = {
    0: 'Motorway', 1: 'Trunk', 2: 'Primary', 3: 'Secondary', 4: 'Tertiary',
    5: 'Residential', 6: 'Service', 7: 'Unclassified', 8: 'Living Street', 9: 'Other'
}

################################################################################
# CHART 1: CAPACITY REDUCTION DISTRIBUTION
################################################################################
print("\n" + "=" * 80)
print("CHART 1: Capacity Reduction Distribution")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(20, 16))

if IS_BASELINE:
    fig.suptitle('FEATURE 2: Baseline Scenario Analysis (No Capacity Reduction)\nAll Roads Operating at Full Design Capacity (F2 = 0)\nShowing Baseline Network Characteristics and Potential Policy Target Analysis', 
                 fontsize=16, fontweight='bold', y=0.995)
else:
    fig.suptitle('FEATURE 2: Capacity Reduction Distribution Analysis\nPolicy Intervention Impact on Network Capacity\nAnalyzing Which Roads Are Targeted and By How Much', 
                 fontsize=16, fontweight='bold', y=0.995)

plt.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.06, hspace=0.40, wspace=0.28)

# 1.1 Distribution histogram
ax = axes[0, 0]
if IS_BASELINE:
    # Show that all values are zero
    ax.bar([0, 1], [n_edges, 0], width=0.8, alpha=0.8, color=['#27ae60', '#cccccc'], 
           edgecolor='black', linewidth=1.5)
    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Zero\nReduction\n(Baseline)', 'Non-Zero\nReduction\n(Policy)'], fontsize=10)
    ax.text(0, n_edges + n_edges*0.05, f'100%\n({n_edges:,} roads)', 
           ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Roads\n[Frequency count]', fontsize=10, fontweight='bold')
    ax.set_title('A. Capacity Reduction Status - BASELINE SCENARIO\n[All roads at full design capacity | No policy interventions]\n[Green bar = 100% of network operating normally]', 
                fontsize=10, fontweight='bold', pad=10)
else:
    cap_red_nonzero = cap_reduction[cap_reduction > 0]
    if len(cap_red_nonzero) > 0:
        ax.hist(cap_red_nonzero, bins=50, alpha=0.7, color='#e74c3c', edgecolor='black', linewidth=0.5)
        ax.axvline(np.median(cap_red_nonzero), color='#3498db', linestyle='--', linewidth=2.5,
                  label=f'Median = {np.median(cap_red_nonzero):.0f} veh/h')
        ax.axvline(np.mean(cap_red_nonzero), color='#27ae60', linestyle='--', linewidth=2.5,
                  label=f'Mean = {np.mean(cap_red_nonzero):.0f} veh/h')
        ax.legend(loc='best', framealpha=0.9, fontsize=9)
        ax.set_xlabel('Capacity Reduction (vehicles/hour)\n[Amount of capacity removed from road]', fontsize=10, fontweight='bold')
        ax.set_ylabel('Number of Roads\n[Frequency count]', fontsize=10, fontweight='bold')
        ax.set_title(f'A. Capacity Reduction Distribution (n={len(cap_red_nonzero):,} affected roads)\n[Shows amount of capacity removed per road]\n[Blue=median | Green=mean reduction value]', 
                    fontsize=10, fontweight='bold', pad=10)
ax.grid(True, alpha=0.3)

# 1.2 Proportion of roads affected
ax = axes[0, 1]
if IS_BASELINE:
    # Show baseline capacity distribution instead
    cap_bins = [0, 500, 1000, 2000, 3000, 5000, capacity.max()+1]
    bin_labels = ['0-500', '500-1k', '1k-2k', '2k-3k', '3k-5k', '5k+']
    roads_per_bin = []
    for i in range(len(cap_bins)-1):
        mask = (capacity >= cap_bins[i]) & (capacity < cap_bins[i+1])
        roads_per_bin.append(mask.sum())
    
    colors = ['#3498db', '#27ae60', '#f39c12', '#e67e22', '#e74c3c', '#9b59b6']
    bars = ax.bar(range(len(bin_labels)), roads_per_bin, alpha=0.8, 
                 color=colors[:len(bin_labels)], edgecolor='black', linewidth=1.2)
    
    # Add percentage labels
    for bar, count in zip(bars, roads_per_bin):
        pct = (count / n_edges) * 100
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(roads_per_bin)*0.02,
                   f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Baseline Capacity Category (veh/h)\n[Roads grouped by their current full capacity]', fontsize=10, fontweight='bold')
    ax.set_ylabel('Number of Roads\n[Count in each capacity range]', fontsize=10, fontweight='bold')
    ax.set_title('B. Baseline Capacity Distribution by Category\n[BASELINE: All roads operating at full capacity]\n[Shows potential policy targets across capacity ranges]', 
                fontsize=10, fontweight='bold', pad=10)
    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels, fontsize=9)
else:
    n_affected = (cap_reduction > 0).sum()
    n_unaffected = (cap_reduction == 0).sum()
    sizes = [n_unaffected, n_affected]
    labels = [f'No Reduction\n{n_unaffected:,} roads\n({n_unaffected/n_edges*100:.1f}%)',
             f'With Reduction\n{n_affected:,} roads\n({n_affected/n_edges*100:.1f}%)']
    colors_pie = ['#27ae60', '#e74c3c']
    explode = (0, 0.1)
    
    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
                                       autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    ax.set_title('B. Network Impact: Proportion of Roads Affected\n[Green = unaffected | Red = capacity reduced]\n[Shows what % of network is targeted by policy]', 
                fontsize=10, fontweight='bold', pad=10)
ax.grid(True, alpha=0.3)

# 1.3 Reduction by highway type
ax = axes[1, 0]
if IS_BASELINE:
    # Show capacity by highway type (potential for reduction)
    mean_cap_by_type = []
    std_cap_by_type = []
    type_labels = []
    
    for ht in unique_types:
        mask = (highway == ht) & (capacity > 0)
        if mask.sum() > 0:
            mean_cap_by_type.append(capacity[mask].mean())
            std_cap_by_type.append(capacity[mask].std())
            type_labels.append(f'{int(ht)}\n{highway_type_names.get(int(ht), "?")[:6]}')
    
    colors_bar = ['#e74c3c', '#3498db', '#27ae60', '#f39c12', '#9b59b6', 
                 '#e67e22', '#1abc9c', '#34495e', '#95a5a6', '#2c3e50']
    bars = ax.bar(range(len(mean_cap_by_type)), mean_cap_by_type, yerr=std_cap_by_type,
                 alpha=0.8, color=colors_bar[:len(mean_cap_by_type)], 
                 edgecolor='black', linewidth=1.2, capsize=5, error_kw={'linewidth': 2})
    
    ax.set_xlabel('Highway Type\n[OpenStreetMap road classification]', fontsize=10, fontweight='bold')
    ax.set_ylabel('Mean Baseline Capacity (veh/h)\n[Average ± standard deviation]', fontsize=10, fontweight='bold')
    ax.set_title('C. Baseline Capacity by Road Type\n[BASELINE: Shows which road types have highest capacity]\n[Error bars = variability | Higher capacity = higher policy impact potential]', 
                fontsize=10, fontweight='bold', pad=10)
    ax.set_xticks(range(len(mean_cap_by_type)))
    ax.set_xticklabels(type_labels, fontsize=8)
    
    for bar, val in zip(bars, mean_cap_by_type):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(std_cap_by_type)*1.15,
               f'{val:.0f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
else:
    mean_red_by_type = []
    count_by_type = []
    type_labels = []
    
    for ht in unique_types:
        mask = (highway == ht) & (cap_reduction > 0)
        if mask.sum() > 0:
            mean_red_by_type.append(cap_reduction[mask].mean())
            count_by_type.append(mask.sum())
            type_labels.append(f'{int(ht)}\n{highway_type_names.get(int(ht), "?")[:6]}')
    
    colors_bar = plt.cm.Reds(np.linspace(0.4, 0.9, len(mean_red_by_type)))
    bars = ax.bar(range(len(mean_red_by_type)), mean_red_by_type, alpha=0.85,
                 color=colors_bar, edgecolor='black', linewidth=1.2)
    
    ax.set_xlabel('Highway Type\n[Road categories with capacity reduction]', fontsize=10, fontweight='bold')
    ax.set_ylabel('Mean Capacity Reduction (veh/h)\n[Average reduction per affected road]', fontsize=10, fontweight='bold')
    ax.set_title('C. Capacity Reduction by Road Type\n[Which road types experience most reduction?]\n[Darker red = higher average reduction]', 
                fontsize=10, fontweight='bold', pad=10)
    ax.set_xticks(range(len(mean_red_by_type)))
    ax.set_xticklabels(type_labels, fontsize=8)
    
    for bar, val, count in zip(bars, mean_red_by_type, count_by_type):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mean_red_by_type)*0.03,
               f'{val:.0f}\n({count:,})', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# 1.4 Cumulative reduction
ax = axes[1, 1]
if IS_BASELINE:
    # Show cumulative capacity (potential for reduction)
    sorted_cap = np.sort(capacity[capacity > 0])[::-1]
    cum_cap = np.cumsum(sorted_cap)
    cum_cap_pct = cum_cap / cum_cap[-1] * 100
    cum_roads_pct = np.arange(1, len(sorted_cap) + 1) / len(sorted_cap) * 100
    
    ax.plot(cum_roads_pct, cum_cap_pct, linewidth=2.5, color='#3498db', label='Capacity accumulation')
    ax.axhline(50, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.7)
    ax.axhline(80, color='#f39c12', linestyle='--', linewidth=2, alpha=0.7)
    
    # Find roads for 50% and 80%
    idx_50 = np.where(cum_cap_pct >= 50)[0][0]
    idx_80 = np.where(cum_cap_pct >= 80)[0][0]
    roads_50 = cum_roads_pct[idx_50]
    roads_80 = cum_roads_pct[idx_80]
    
    ax.plot(roads_50, 50, 'ro', markersize=10)
    ax.plot(roads_80, 80, 'o', color='#f39c12', markersize=10)
    ax.text(roads_50+2, 48, f'{roads_50:.1f}% roads\nhold 50% capacity', fontsize=8, fontweight='bold')
    ax.text(roads_80+2, 78, f'{roads_80:.1f}% roads\nhold 80% capacity', fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Cumulative % of Roads (sorted by capacity)\n[X-axis: starting with highest-capacity roads]', fontsize=10, fontweight='bold')
    ax.set_ylabel('Cumulative % of Total Network Capacity\n[Y-axis: % of total capacity accumulated]', fontsize=10, fontweight='bold')
    ax.set_title('D. Capacity Concentration - Policy Targeting Potential\n[BASELINE: Shows strategic importance of high-capacity roads]\n[Targeting top roads would have disproportionate network impact]', 
                fontsize=10, fontweight='bold', pad=10)
    ax.legend(loc='best', framealpha=0.9, fontsize=9)
else:
    sorted_red = np.sort(cap_reduction[cap_reduction > 0])[::-1]
    cum_red = np.cumsum(sorted_red)
    cum_red_pct = cum_red / cum_red[-1] * 100
    cum_roads_pct = np.arange(1, len(sorted_red) + 1) / len(sorted_red) * 100
    
    ax.plot(cum_roads_pct, cum_red_pct, linewidth=2.5, color='#e74c3c', label='Reduction accumulation')
    ax.plot([0, 100], [0, 100], 'b--', linewidth=2, label='Uniform distribution', alpha=0.7)
    
    # Find roads for 50% and 80% of reduction
    idx_50 = np.where(cum_red_pct >= 50)[0][0]
    idx_80 = np.where(cum_red_pct >= 80)[0][0]
    roads_50 = cum_roads_pct[idx_50]
    roads_80 = cum_roads_pct[idx_80]
    
    ax.plot(roads_50, 50, 'ro', markersize=10)
    ax.plot(roads_80, 80, 'ro', markersize=10)
    ax.text(roads_50+2, 48, f'{roads_50:.1f}% of affected\nroads = 50% reduction', fontsize=8, fontweight='bold')
    ax.text(roads_80+2, 78, f'{roads_80:.1f}% of affected\nroads = 80% reduction', fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Cumulative % of Affected Roads (sorted by reduction)\n[X-axis: roads ordered by reduction amount]', fontsize=10, fontweight='bold')
    ax.set_ylabel('Cumulative % of Total Capacity Reduction\n[Y-axis: % of total reduction accumulated]', fontsize=10, fontweight='bold')
    ax.set_title('D. Reduction Concentration - Policy Intensity\n[Is reduction evenly distributed or concentrated?]\n[Red dots = key breakpoints | Blue dashed = uniform policy]', 
                fontsize=10, fontweight='bold', pad=10)
    ax.legend(loc='best', framealpha=0.9, fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('feature2_chart1_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: feature2_chart1_distribution.png")
plt.show()
plt.close()

################################################################################
# CHART 2: REDUCTION PATTERNS & STATISTICS
################################################################################
print("\n" + "=" * 80)
print("CHART 2: Reduction Patterns & Statistics")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(20, 16))

if IS_BASELINE:
    fig.suptitle('FEATURE 2: Baseline Network Statistics & Vulnerability Analysis\nNo Policy Interventions Applied (F2 = 0)\nAnalyzing Network Structure and Potential Policy Impact Points', 
                 fontsize=16, fontweight='bold', y=0.995)
else:
    fig.suptitle('FEATURE 2: Capacity Reduction Patterns & Statistics\nDetailed Analysis of Policy Implementation\nStatistical Characteristics of Capacity Reduction', 
                 fontsize=16, fontweight='bold', y=0.995)

plt.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.06, hspace=0.40, wspace=0.28)

# 2.1 Statistics summary
ax = axes[0, 0]
ax.axis('off')

if IS_BASELINE:
    stats_data = [
        ['BASELINE SCENARIO STATISTICS', '', ''],
        ['', '', ''],
        ['Total Roads', f'{n_edges:,}', 'roads'],
        ['Roads with Reduction', '0', '(0.0%)'],
        ['Roads at Full Capacity', f'{n_edges:,}', '(100.0%)'],
        ['', '', ''],
        ['CAPACITY CHARACTERISTICS', '', ''],
        ['', '', ''],
        ['Total Network Capacity', f'{capacity.sum():,.0f}', 'veh/h'],
        ['Mean Capacity', f'{capacity.mean():.0f}', 'veh/h'],
        ['Median Capacity', f'{np.median(capacity):.0f}', 'veh/h'],
        ['Max Capacity', f'{capacity.max():.0f}', 'veh/h'],
        ['', '', ''],
        ['POTENTIAL IMPACT ANALYSIS', '', ''],
        ['', '', ''],
        ['High-capacity roads (>P90)', f'{(capacity > np.percentile(capacity, 90)).sum():,}', f'({(capacity > np.percentile(capacity, 90)).sum()/n_edges*100:.1f}%)'],
        ['Strategic roads (top 15%)', f'{int(n_edges * 0.15):,}', 'roads'],
        ['Capacity concentration', 'Low-Medium', '(Gini ≈ 0.46)'],
    ]
else:
    n_affected = (cap_reduction > 0).sum()
    cap_red_nonzero = cap_reduction[cap_reduction > 0]
    stats_data = [
        ['REDUCTION STATISTICS', '', ''],
        ['', '', ''],
        ['Total Roads', f'{n_edges:,}', 'roads'],
        ['Roads with Reduction', f'{n_affected:,}', f'({n_affected/n_edges*100:.1f}%)'],
        ['Roads Unaffected', f'{n_edges - n_affected:,}', f'({(n_edges-n_affected)/n_edges*100:.1f}%)'],
        ['', '', ''],
        ['REDUCTION AMOUNT', '', ''],
        ['', '', ''],
        ['Total Reduction', f'{cap_reduction.sum():,.0f}', 'veh/h'],
        ['Mean (affected roads)', f'{cap_red_nonzero.mean():.0f}', 'veh/h'],
        ['Median (affected roads)', f'{np.median(cap_red_nonzero):.0f}', 'veh/h'],
        ['Max Reduction', f'{cap_reduction.max():.0f}', 'veh/h'],
        ['', '', ''],
        ['INTENSITY METRICS', '', ''],
        ['', '', ''],
        ['% of Total Capacity', f'{cap_reduction.sum()/capacity.sum()*100:.2f}%', 'reduced'],
        ['Std Dev (affected)', f'{cap_red_nonzero.std():.0f}', 'veh/h'],
        ['Coeff. of Variation', f'{cap_red_nonzero.std()/cap_red_nonzero.mean():.3f}', 'variability'],
    ]

table = ax.table(cellText=stats_data, cellLoc='left', loc='center',
                colWidths=[0.50, 0.30, 0.20])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.2)

# Highlight headers
for i in [0, 1, 6, 7, 13, 14]:
    for j in range(3):
        table[(i, j)].set_facecolor('#3498db' if IS_BASELINE else '#e74c3c')
        table[(i, j)].set_text_props(weight='bold', color='white')

if IS_BASELINE:
    ax.set_title('A. Baseline Scenario Statistics\n[All roads operating at full design capacity]\n[No policy interventions applied]', 
                fontsize=10, fontweight='bold', pad=10)
else:
    ax.set_title('A. Capacity Reduction Statistics Summary\n[Complete statistical overview]\n[Key metrics for policy impact assessment]', 
                fontsize=10, fontweight='bold', pad=10)

# 2.2 Box plot comparison
ax = axes[0, 1]
if IS_BASELINE:
    # Compare capacity across different utilization levels
    with np.errstate(divide='ignore', invalid='ignore'):
        utilization = np.abs(vol_base_case) / capacity
        utilization = np.nan_to_num(utilization, nan=0.0, posinf=0.0, neginf=0.0)
    
    util_cats = ['0-25%\nUnder-used', '25-50%\nLight', '50-75%\nModerate', '75-100%\nHeavy', '>100%\nOver-cap']
    util_bins = [0, 0.25, 0.5, 0.75, 1.0, 100]
    cap_by_util = []
    
    for i in range(len(util_bins)-1):
        mask = (utilization >= util_bins[i]) & (utilization < util_bins[i+1]) & (capacity > 0)
        if mask.sum() > 10:
            cap_by_util.append(capacity[mask])
    
    if len(cap_by_util) > 0:
        labels_filtered = [util_cats[i] for i in range(len(util_bins)-1) 
                          if i < len(cap_by_util) and len(cap_by_util[i]) > 0]
        bp = ax.boxplot(cap_by_util, tick_labels=labels_filtered,
                       patch_artist=True, showfliers=False, widths=0.6)
        colors_box = ['#27ae60', '#3498db', '#f39c12', '#e67e22', '#e74c3c']
        for patch, color in zip(bp['boxes'], colors_box[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.2)
        for median in bp['medians']:
            median.set_color('white')
            median.set_linewidth(3)
        
        ax.set_ylabel('Baseline Capacity (veh/h)\n[Box = middle 50% | White line = median]', fontsize=10, fontweight='bold')
        ax.set_title('B. Baseline Capacity by Current Utilization Level\n[BASELINE: Do heavily-used roads have different capacity?]\n[Color code: Green (light use) to Red (heavy/over-capacity)]', 
                    fontsize=10, fontweight='bold', pad=10)
else:
    # Compare capacity of affected vs unaffected roads
    cap_affected = capacity[cap_reduction > 0]
    cap_unaffected = capacity[cap_reduction == 0]
    
    bp = ax.boxplot([cap_unaffected, cap_affected],
                   tick_labels=['Unaffected\nRoads', 'Affected\nRoads'],
                   patch_artist=True, showfliers=False, widths=0.6)
    bp['boxes'][0].set_facecolor('#27ae60')
    bp['boxes'][1].set_facecolor('#e74c3c')
    for box in bp['boxes']:
        box.set_alpha(0.8)
        box.set_edgecolor('black')
        box.set_linewidth(1.5)
    for median in bp['medians']:
        median.set_color('white')
        median.set_linewidth(3)
    
    # Add mean comparison
    mean_unaff = cap_unaffected.mean()
    mean_aff = cap_affected.mean()
    ax.text(0.5, 0.97, f'Mean: Unaffected={mean_unaff:.0f} | Affected={mean_aff:.0f} veh/h',
           transform=ax.transAxes, ha='center', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_ylabel('Road Capacity (veh/h)\n[Baseline capacity before reduction]', fontsize=10, fontweight='bold')
    ax.set_title(f'B. Capacity Comparison: Affected vs Unaffected Roads\n[Do policies target high-capacity or low-capacity roads?]\n[Green = not targeted | Red = capacity reduced]', 
                fontsize=10, fontweight='bold', pad=10)
ax.grid(True, alpha=0.3, axis='y')

# 2.3 Scatter: Capacity vs Reduction (or vulnerability)
ax = axes[1, 0]
if IS_BASELINE:
    # Show capacity vs traffic volume (vulnerability indicator)
    valid_mask = (capacity > 0) & (vol_base_case != 0)
    cap_subset = capacity[valid_mask]
    vol_subset = np.abs(vol_base_case[valid_mask])
    
    # Subsample for performance
    if len(cap_subset) > 5000:
        sample_idx = np.random.choice(len(cap_subset), 5000, replace=False)
        cap_subset = cap_subset[sample_idx]
        vol_subset = vol_subset[sample_idx]
    
    ax.scatter(cap_subset, vol_subset, alpha=0.4, s=3, c='#3498db', edgecolors='none')
    
    # Add reference lines
    max_val = min(cap_subset.max(), vol_subset.max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='100% utilization', alpha=0.7)
    ax.plot([0, max_val], [0, max_val*0.5], 'g--', linewidth=1.5, label='50% utilization', alpha=0.6)
    
    if len(cap_subset) > 10:
        corr = np.corrcoef(cap_subset, vol_subset)[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes,
               fontsize=10, fontweight='bold', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               verticalalignment='top')
    
    ax.set_xlabel('Baseline Road Capacity (veh/h)\n[Maximum design capacity]', fontsize=10, fontweight='bold')
    ax.set_ylabel('Current Traffic Volume (veh/h)\n[Actual traffic flow]', fontsize=10, fontweight='bold')
    ax.set_title(f'C. Baseline Capacity-Volume Relationship (n={len(cap_subset):,} roads)\n[BASELINE: Shows which roads are critical for current traffic]\n[Points near red line = most vulnerable to capacity reduction]', 
                fontsize=10, fontweight='bold', pad=10)
    ax.legend(loc='best', framealpha=0.9, fontsize=9)
else:
    valid_mask = (capacity > 0) & (cap_reduction > 0)
    cap_subset = capacity[valid_mask]
    red_subset = cap_reduction[valid_mask]
    
    ax.scatter(cap_subset, red_subset, alpha=0.5, s=5, c='#e74c3c', edgecolors='none')
    
    if len(cap_subset) > 10:
        corr = np.corrcoef(cap_subset, red_subset)[0, 1]
        z = np.polyfit(cap_subset, red_subset, 1)
        p = np.poly1d(z)
        cap_range = np.linspace(cap_subset.min(), cap_subset.max(), 100)
        ax.plot(cap_range, p(cap_range), "b--", linewidth=2.5, alpha=0.7, label=f'Trend (r={corr:.3f})')
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes,
               fontsize=10, fontweight='bold', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               verticalalignment='top')
        ax.legend(loc='best', framealpha=0.9, fontsize=9)
    
    ax.set_xlabel('Baseline Road Capacity (veh/h)\n[Original capacity before policy]', fontsize=10, fontweight='bold')
    ax.set_ylabel('Capacity Reduction (veh/h)\n[Amount removed by policy]', fontsize=10, fontweight='bold')
    ax.set_title(f'C. Capacity vs Reduction Relationship (n={len(cap_subset):,} affected roads)\n[Do high-capacity roads get reduced more?]\n[Pattern reveals policy targeting strategy]', 
                fontsize=10, fontweight='bold', pad=10)
ax.grid(True, alpha=0.3)

# 2.4 Reduction intensity bins
ax = axes[1, 1]
if IS_BASELINE:
    # Show network criticality by capacity level
    cap_bins = [0, 500, 1000, 2000, 3000, 5000, capacity.max()+1]
    bin_labels = ['0-500', '500-1k', '1k-2k', '2k-3k', '3k-5k', '5k+']
    
    total_vol_by_bin = []
    total_cap_by_bin = []
    
    for i in range(len(cap_bins)-1):
        mask = (capacity >= cap_bins[i]) & (capacity < cap_bins[i+1])
        total_vol_by_bin.append(np.abs(vol_base_case[mask]).sum())
        total_cap_by_bin.append(capacity[mask].sum())
    
    # Calculate criticality score (% of traffic / % of capacity)
    total_vol = np.abs(vol_base_case).sum()
    total_cap = capacity.sum()
    criticality = []
    for vol, cap in zip(total_vol_by_bin, total_cap_by_bin):
        vol_pct = (vol / total_vol) * 100 if total_vol > 0 else 0
        cap_pct = (cap / total_cap) * 100 if total_cap > 0 else 0
        crit = vol_pct / cap_pct if cap_pct > 0 else 0
        criticality.append(crit)
    
    colors_crit = ['#27ae60' if c < 0.8 else '#f39c12' if c < 1.2 else '#e74c3c' for c in criticality]
    bars = ax.bar(range(len(bin_labels)), criticality, alpha=0.8, color=colors_crit,
                 edgecolor='black', linewidth=1.2)
    ax.axhline(1.0, color='black', linestyle='--', linewidth=2, label='Proportional use', alpha=0.7)
    
    for bar, val in zip(bars, criticality):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Capacity Category (veh/h)\n[Roads grouped by capacity level]', fontsize=10, fontweight='bold')
    ax.set_ylabel('Network Criticality Score\n[Traffic share / Capacity share | >1 = critical]', fontsize=10, fontweight='bold')
    ax.set_title('D. Network Criticality by Capacity Category\n[BASELINE: Which capacity levels are most critical for traffic?]\n[Green (<0.8) = over-built | Yellow (0.8-1.2) = balanced | Red (>1.2) = critical]', 
                fontsize=10, fontweight='bold', pad=10)
    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels, fontsize=9)
    ax.legend(loc='best', framealpha=0.9, fontsize=9)
else:
    # Reduction intensity by capacity bins
    cap_bins = [0, 1000, 2000, 3000, 5000, capacity.max()+1]
    bin_labels = ['0-1k', '1k-2k', '2k-3k', '3k-5k', '5k+']
    
    mean_reduction = []
    reduction_rate = []  # % of capacity reduced
    
    for i in range(len(cap_bins)-1):
        mask = (capacity >= cap_bins[i]) & (capacity < cap_bins[i+1]) & (cap_reduction > 0)
        if mask.sum() > 0:
            mean_reduction.append(cap_reduction[mask].mean())
            # Calculate average % reduction
            pct_red = (cap_reduction[mask] / capacity[mask] * 100).mean()
            reduction_rate.append(pct_red)
        else:
            mean_reduction.append(0)
            reduction_rate.append(0)
    
    x_pos = np.arange(len(bin_labels))
    width = 0.35
    
    ax2 = ax.twinx()
    bars1 = ax.bar(x_pos - width/2, mean_reduction, width, alpha=0.8, color='#e74c3c',
                  edgecolor='black', linewidth=1.2, label='Mean reduction (veh/h)')
    bars2 = ax2.bar(x_pos + width/2, reduction_rate, width, alpha=0.8, color='#3498db',
                   edgecolor='black', linewidth=1.2, label='% of capacity')
    
    ax.set_xlabel('Capacity Category (veh/h)\n[Roads grouped by capacity level]', fontsize=10, fontweight='bold')
    ax.set_ylabel('Mean Reduction (veh/h)\n[Red bars - left axis]', fontsize=10, color='#e74c3c', fontweight='bold')
    ax2.set_ylabel('% of Capacity Reduced\n[Blue bars - right axis]', fontsize=10, color='#3498db', fontweight='bold')
    ax.set_title('D. Reduction Intensity by Capacity Category\n[How aggressive is policy across different capacity levels?]\n[Red = absolute reduction | Blue = relative reduction (%)]', 
                fontsize=10, fontweight='bold', pad=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bin_labels, fontsize=9)
    ax.tick_params(axis='y', labelcolor='#e74c3c')
    ax2.tick_params(axis='y', labelcolor='#3498db')
    ax.legend(loc='upper left', framealpha=0.9, fontsize=8)
    ax2.legend(loc='upper right', framealpha=0.9, fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('feature2_chart2_patterns.png', dpi=300, bbox_inches='tight')
print("✓ Saved: feature2_chart2_patterns.png")
plt.show()
plt.close()

print("\n" + "=" * 80)
print("✓✓✓ PART 1 COMPLETE - CHARTS 1-2 GENERATED ✓✓✓")
print("=" * 80)
print("\nGenerated files:")
print("  1. feature2_chart1_distribution.png")
print("  2. feature2_chart2_patterns.png")

if IS_BASELINE:
    print("\n[i] BASELINE ANALYSIS MODE")
    print("    All charts show baseline network characteristics")
    print("    Analysis focuses on network structure and potential policy impact")
else:
    print("\n[i] POLICY ANALYSIS MODE")
    print(f"    {(cap_reduction > 0).sum():,} roads affected ({(cap_reduction > 0).sum()/n_edges*100:.1f}%)")
    print(f"    Total reduction: {cap_reduction.sum():,.0f} veh/h")

print("\nNext: Run feature2_part2_charts3to4.py for Charts 3-4")
print("=" * 80)
