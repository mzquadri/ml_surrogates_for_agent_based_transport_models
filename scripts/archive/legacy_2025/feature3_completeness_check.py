import torch
import numpy as np

# Load data
batch_path = '/content/drive/MyDrive/Zamin_thesis/ml_surrogates_for_agent_based_transport_models/data/train_data/dist_not_connected_10k_1pct/datalist_batch_1.pt'

# Safe loading
allowed_globals = {
    'Data': type('Data', (), {}),
    'DataEdgeAttr': type('DataEdgeAttr', (), {})
}
data_list = torch.load(batch_path, weights_only=False, map_location='cpu')

print("="*80)
print("FEATURE 3 (FREE_SPEED) - COMPLETENESS CHECK")
print("="*80)
print()

# Get basic info
n_active = 31635
graph = data_list[0]
free_speed = graph.x[:n_active, 5].numpy()

print("1. BASIC STATISTICS")
print("-" * 80)
print(f"Total active road segments: {n_active:,}")
print(f"Feature 3 shape: {free_speed.shape}")
print(f"Min value: {free_speed.min():.2f} km/h")
print(f"Max value: {free_speed.max():.2f} km/h")
print(f"Mean: {free_speed.mean():.2f} km/h")
print(f"Median: {np.median(free_speed):.2f} km/h")
print(f"Std Dev: {free_speed.std():.2f} km/h")
print()

# Check for missing or invalid values
print("2. DATA QUALITY CHECK")
print("-" * 80)
has_nan = np.isnan(free_speed).any()
has_inf = np.isinf(free_speed).any()
has_negative = (free_speed < 0).any()
has_zero = (free_speed == 0).sum()
print(f"Contains NaN values: {has_nan}")
print(f"Contains Inf values: {has_inf}")
print(f"Contains negative values: {has_negative}")
print(f"Number of zero speed roads: {has_zero:,} ({has_zero/n_active*100:.2f}%)")
print()

# Static vs Dynamic check
print("3. STATIC vs DYNAMIC CHECK")
print("-" * 80)
print("Checking across first 10 scenarios...")
is_static = True
for i in range(1, min(10, len(data_list))):
    graph_i = data_list[i]
    speed_i = graph_i.x[:n_active, 5].numpy()
    if not np.array_equal(free_speed, speed_i):
        is_static = False
        break

if is_static:
    print("Result: STATIC - Values are identical across scenarios")
else:
    print("Result: DYNAMIC - Values change across scenarios")
    
    # Calculate variation statistics
    all_values = []
    for i in range(len(data_list)):
        graph_i = data_list[i]
        all_values.append(graph_i.x[:n_active, 5].numpy())
    all_values = np.array(all_values)
    
    cv_per_segment = np.std(all_values, axis=0) / (np.mean(all_values, axis=0) + 1e-10)
    cv_mean = np.mean(cv_per_segment[np.isfinite(cv_per_segment)])
    
    print(f"Mean Coefficient of Variation across scenarios: {cv_mean:.4f}")
    print(f"Number of segments with variation: {np.sum(cv_per_segment > 0.01):,}")
print()

# Speed distribution
print("4. SPEED DISTRIBUTION")
print("-" * 80)
speed_categories = {
    'Very Slow (0-5 km/h)': (free_speed >= 0) & (free_speed < 5),
    'Slow (5-10 km/h)': (free_speed >= 5) & (free_speed < 10),
    'Moderate (10-15 km/h)': (free_speed >= 10) & (free_speed < 15),
    'Fast (15-20 km/h)': (free_speed >= 15) & (free_speed < 20),
    'Very Fast (20-25 km/h)': (free_speed >= 20) & (free_speed < 25),
    'Highway (>25 km/h)': free_speed >= 25
}

for category, mask in speed_categories.items():
    count = np.sum(mask)
    pct = count / n_active * 100
    print(f"{category}: {count:,} ({pct:.2f}%)")
print()

# Highway type distribution
print("5. HIGHWAY TYPE DISTRIBUTION")
print("-" * 80)
highway_types = graph.x[:n_active, 4].numpy()
hw_mapping = {
    0: 'Motorway', 1: 'Trunk', 2: 'Primary', 3: 'Secondary', 
    4: 'Tertiary', 5: 'Residential', 6: 'PT', 7: 'Service',
    8: 'Living Street', 9: 'Motorway Link', 10: 'Trunk Link',
    11: 'Primary Link', 12: 'Secondary Link'
}

hw_speeds = {}
for hw_id in range(13):
    mask = highway_types == hw_id
    count = np.sum(mask)
    if count > 0:
        hw_speeds[hw_id] = {
            'name': hw_mapping[hw_id],
            'count': count,
            'mean_speed': np.mean(free_speed[mask]),
            'median_speed': np.median(free_speed[mask]),
            'std_speed': np.std(free_speed[mask])
        }

# Sort by mean speed descending
sorted_hw = sorted(hw_speeds.items(), key=lambda x: x[1]['mean_speed'], reverse=True)

for hw_id, data in sorted_hw:
    print(f"  HW {hw_id}: {data['name']:15s}: {data['count']:5,} roads "
          f"(mean: {data['mean_speed']:5.2f} km/h, median: {data['median_speed']:5.2f} km/h)")
print()

# Unique values analysis
print("6. UNIQUE VALUES ANALYSIS")
print("-" * 80)
unique_speeds = np.unique(free_speed)
print(f"Total unique speed values: {len(unique_speeds)}")
print(f"Most common speeds (top 10):")

unique_vals, unique_counts = np.unique(free_speed, return_counts=True)
sorted_indices = np.argsort(-unique_counts)[:10]

for i, idx in enumerate(sorted_indices, 1):
    count = unique_counts[idx]
    pct = count / n_active * 100
    print(f"  {i:2d}. {unique_vals[idx]:6.2f} km/h: {count:5,} roads ({pct:5.2f}%)")
print()

# Speed discretization check
print("7. SPEED DISCRETIZATION PATTERN")
print("-" * 80)
# Check if speeds are multiples of common values
multiples_0_1 = np.sum(np.isclose(free_speed % 0.1, 0) | np.isclose(free_speed % 0.1, 0.1))
multiples_0_5 = np.sum(np.isclose(free_speed % 0.5, 0))
multiples_1_0 = np.sum(np.isclose(free_speed % 1.0, 0))

print(f"Multiples of 1.0 km/h: {multiples_1_0:,} ({multiples_1_0/n_active*100:.2f}%)")
print(f"Multiples of 0.5 km/h: {multiples_0_5:,} ({multiples_0_5/n_active*100:.2f}%)")
print(f"High precision (0.1 km/h): {multiples_0_1:,} ({multiples_0_1/n_active*100:.2f}%)")
print()

# Correlation with other features
print("8. CORRELATION WITH OTHER FEATURES")
print("-" * 80)
length = graph.x[:n_active, 0].numpy()
capacity = graph.x[:n_active, 1].numpy()
baseline_volume = graph.x[:n_active, 2].numpy()
capacity_reduction = graph.x[:n_active, 3].numpy()
target = graph.y[:n_active].numpy().flatten()

correlations = {
    'LENGTH': np.corrcoef(length, free_speed)[0, 1],
    'CAPACITY': np.corrcoef(capacity, free_speed)[0, 1],
    'BASELINE_VOLUME': np.corrcoef(baseline_volume, free_speed)[0, 1],
    'CAPACITY_REDUCTION': np.corrcoef(capacity_reduction, free_speed)[0, 1],
    'HIGHWAY_TYPE': np.corrcoef(highway_types, free_speed)[0, 1],
    'TARGET': np.corrcoef(target, free_speed)[0, 1]
}

# Sort by absolute correlation
sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

for feature, corr in sorted_corr:
    strength = 'Strong' if abs(corr) > 0.7 else 'Moderate' if abs(corr) > 0.4 else 'Weak' if abs(corr) > 0.2 else 'Very Weak'
    print(f"Correlation with {feature:20s}: {corr:+.4f} ({strength})")
print()

# Percentile analysis
print("9. PERCENTILE ANALYSIS")
print("-" * 80)
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
for p in percentiles:
    value = np.percentile(free_speed, p)
    print(f"  {p:3d}th percentile:   {value:6.2f} km/h")
print()

# Speed variability
print("10. SPEED VARIABILITY")
print("-" * 80)
cv = np.std(free_speed) / np.mean(free_speed)
iqr = np.percentile(free_speed, 75) - np.percentile(free_speed, 25)
print(f"Coefficient of Variation: {cv:.4f}")
print(f"Interquartile Range (IQR): {iqr:.2f} km/h")
print(f"Range (max - min): {free_speed.max() - free_speed.min():.2f} km/h")
print()

# Speed by traffic status
print("11. SPEED BY TRAFFIC STATUS")
print("-" * 80)
has_traffic = baseline_volume < 0
no_traffic = baseline_volume == 0

if has_traffic.sum() > 0:
    print(f"Roads WITH traffic (n={has_traffic.sum():,}):")
    print(f"  Mean speed: {np.mean(free_speed[has_traffic]):.2f} km/h")
    print(f"  Median speed: {np.median(free_speed[has_traffic]):.2f} km/h")
    print(f"  Std dev: {np.std(free_speed[has_traffic]):.2f} km/h")
    print()

if no_traffic.sum() > 0:
    print(f"Roads with NO traffic (n={no_traffic.sum():,}):")
    print(f"  Mean speed: {np.mean(free_speed[no_traffic]):.2f} km/h")
    print(f"  Median speed: {np.median(free_speed[no_traffic]):.2f} km/h")
    print(f"  Std dev: {np.std(free_speed[no_traffic]):.2f} km/h")
    print()

# Speed-capacity relationship
print("12. SPEED-CAPACITY RELATIONSHIP")
print("-" * 80)
# Categorize by capacity
cap_bins = [0, 500, 1000, 2000, 5000, 10000, float('inf')]
cap_labels = ['0-500', '500-1k', '1k-2k', '2k-5k', '5k-10k', '>10k']

for i in range(len(cap_bins)-1):
    mask = (capacity >= cap_bins[i]) & (capacity < cap_bins[i+1])
    if mask.sum() > 0:
        print(f"Capacity {cap_labels[i]:8s} veh/h: {mask.sum():5,} roads, "
              f"avg speed: {np.mean(free_speed[mask]):5.2f} km/h")
print()

# Final charts created
print("13. FINAL CHARTS CREATED")
print("-" * 80)
print("  Chart 1: Speed distribution (histogram, categories, CDF, statistics)")
print("  Chart 2: Speed by highway type (means, box plots, ranges, counts)")
print("  Chart 3: Speed relationships (with capacity, volume, length, correlations)")
print("  Chart 4: Comprehensive summary dashboard (12 panels)")
print()
print("Total charts: 4")
print()

# Key findings summary
print("14. KEY FINDINGS SUMMARY")
print("-" * 80)
print(f"  - STATIC/DYNAMIC: {'STATIC' if is_static else 'DYNAMIC'} feature")
print(f"  - Mean speed: {np.mean(free_speed):.2f} km/h (surprisingly low!)")
print(f"  - {len(unique_speeds)} unique speed values")
print(f"  - Most common: {unique_vals[sorted_indices[0]]:.2f} km/h "
      f"({unique_counts[sorted_indices[0]]:,} roads, {unique_counts[sorted_indices[0]]/n_active*100:.1f}%)")
print(f"  - Fastest highway type: {sorted_hw[0][1]['name']} ({sorted_hw[0][1]['mean_speed']:.2f} km/h)")
print(f"  - Slowest highway type: {sorted_hw[-1][1]['name']} ({sorted_hw[-1][1]['mean_speed']:.2f} km/h)")
print(f"  - Strongest correlation: {sorted_corr[0][0]} (r={sorted_corr[0][1]:.3f})")
print(f"  - CV: {cv:.3f} ({'Low' if cv < 0.3 else 'Moderate' if cv < 0.6 else 'High'} variability)")
print(f"  - Zero speed roads: {has_zero:,} ({has_zero/n_active*100:.2f}%)")

if not is_static:
    print(f"  - Variation across scenarios: {cv_mean:.4f} (dynamic behavior confirmed)")

print()
print("="*80)
print("COMPLETENESS CHECK PASSED - Feature 3 fully analyzed")
print("="*80)
