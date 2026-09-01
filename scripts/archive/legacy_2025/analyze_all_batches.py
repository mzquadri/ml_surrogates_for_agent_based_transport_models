"""
Comprehensive analysis of all 20 batch files
"""
import torch
from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr
import numpy as np
import pandas as pd
from pathlib import Path

# Add safe globals for PyTorch loading
torch.serialization.add_safe_globals([Data, DataEdgeAttr])

# Define data directory
data_dir = Path(r"d:\Python Projects\Zamin_Thesis\ml_surrogates_for_agent_based_transport_models\data\train_data\dist_not_connected_10k_1pct")

print("="*80)
print("COMPREHENSIVE ANALYSIS OF ALL 20 BATCH FILES")
print("="*80)

# Part 1: Load all batches and check structure
print("\n[PART 1: BATCH FILE STRUCTURE ANALYSIS]")
print("-"*80)

batch_info = []
all_graphs = []

for batch_num in range(1, 21):
    batch_file = data_dir / f"datalist_batch_{batch_num}.pt"
    
    if not batch_file.exists():
        print(f"Warning: {batch_file.name} not found!")
        continue
    
    # Load batch
    try:
        data_list = torch.load(batch_file, weights_only=False)
        
        batch_data = {
            'batch_num': batch_num,
            'num_graphs': len(data_list),
            'file_size_mb': batch_file.stat().st_size / (1024*1024)
        }
        
        # Check first and last graph in batch
        if len(data_list) > 0:
            first_graph = data_list[0]
            last_graph = data_list[-1]
            
            batch_data.update({
                'num_nodes': first_graph.num_nodes,
                'num_edges': first_graph.num_edges,
                'num_features': first_graph.x.shape[1],
                'has_y': hasattr(first_graph, 'y'),
                'y_shape': first_graph.y.shape if hasattr(first_graph, 'y') else None,
                'last_graph_matches': (
                    last_graph.num_nodes == first_graph.num_nodes and
                    last_graph.num_edges == first_graph.num_edges and
                    last_graph.x.shape == first_graph.x.shape
                )
            })
        
        batch_info.append(batch_data)
        all_graphs.extend(data_list)
        
        print(f"Batch {batch_num:2d}: {len(data_list):2d} graphs | "
              f"Nodes: {batch_data['num_nodes']:,} | "
              f"Edges: {batch_data['num_edges']:,} | "
              f"Features: {batch_data['num_features']} | "
              f"Size: {batch_data['file_size_mb']:.1f} MB")
        
    except Exception as e:
        print(f"Error loading batch {batch_num}: {str(e)}")
        continue

# Summary statistics
df_batches = pd.DataFrame(batch_info)

print("\n" + "="*80)
print("BATCH FILE SUMMARY")
print("="*80)
print(f"Total batches loaded: {len(batch_info)}")
print(f"Total graphs: {len(all_graphs)}")
print(f"Total data size: {df_batches['file_size_mb'].sum():.1f} MB")

print("\nConsistency Check:")
print(f"  Graphs per batch: min={df_batches['num_graphs'].min()}, "
      f"max={df_batches['num_graphs'].max()}, "
      f"mean={df_batches['num_graphs'].mean():.1f}")
print(f"  Nodes per graph: unique values = {df_batches['num_nodes'].nunique()}")
print(f"  Edges per graph: unique values = {df_batches['num_edges'].nunique()}")
print(f"  Features per node: unique values = {df_batches['num_features'].nunique()}")
print(f"  All last graphs match first: {df_batches['last_graph_matches'].all()}")

# Part 2: Feature Analysis Across All Graphs
print("\n" + "="*80)
print("[PART 2: FEATURE ANALYSIS ACROSS ALL 1,000 GRAPHS]")
print("="*80)

# Collect feature statistics from all graphs
feature_stats = {i: [] for i in range(6)}
target_stats = []

print("\nCollecting features from all 1,000 graphs...")
for idx, graph in enumerate(all_graphs):
    if (idx + 1) % 200 == 0:
        print(f"  Processed {idx+1}/{len(all_graphs)} graphs...")
    
    # Feature statistics
    for feat_idx in range(6):
        feature_vals = graph.x[:, feat_idx].numpy()
        feature_stats[feat_idx].append({
            'graph_idx': idx,
            'min': feature_vals.min(),
            'max': feature_vals.max(),
            'mean': feature_vals.mean(),
            'median': np.median(feature_vals),
            'std': feature_vals.std(),
            'num_zeros': (feature_vals == 0).sum(),
            'num_negatives': (feature_vals < 0).sum(),
            'num_unique': len(np.unique(feature_vals))
        })
    
    # Target statistics
    if hasattr(graph, 'y'):
        target_vals = graph.y.numpy()
        target_stats.append({
            'graph_idx': idx,
            'min': target_vals.min(),
            'max': target_vals.max(),
            'mean': target_vals.mean(),
            'median': np.median(target_vals),
            'std': target_vals.std(),
            'num_zeros': (target_vals == 0).sum(),
            'num_negatives': (target_vals < 0).sum()
        })

print(f"  Completed: {len(all_graphs)} graphs processed.")

# Aggregate statistics for each feature
print("\n" + "-"*80)
print("FEATURE STATISTICS ACROSS ALL GRAPHS")
print("-"*80)

for feat_idx in range(6):
    df_feat = pd.DataFrame(feature_stats[feat_idx])
    
    print(f"\nFeature {feat_idx}:")
    print(f"  Global Range: [{df_feat['min'].min():.2f}, {df_feat['max'].max():.2f}]")
    print(f"  Mean across graphs: {df_feat['mean'].mean():.2f} (std: {df_feat['mean'].std():.2f})")
    print(f"  Median across graphs: {df_feat['median'].mean():.2f}")
    print(f"  Zeros: {df_feat['num_zeros'].mean():.0f} nodes/graph ({df_feat['num_zeros'].mean()/31635*100:.1f}%)")
    print(f"  Negatives: {df_feat['num_negatives'].mean():.0f} nodes/graph ({df_feat['num_negatives'].mean()/31635*100:.1f}%)")
    print(f"  Unique values per graph: {df_feat['num_unique'].mean():.0f}")
    print(f"  Variation between graphs:")
    print(f"    - Min value range: [{df_feat['min'].min():.2f}, {df_feat['min'].max():.2f}]")
    print(f"    - Max value range: [{df_feat['max'].min():.2f}, {df_feat['max'].max():.2f}]")
    print(f"    - Mean value range: [{df_feat['mean'].min():.2f}, {df_feat['mean'].max():.2f}]")

# Target statistics
if target_stats:
    df_target = pd.DataFrame(target_stats)
    
    print(f"\nTarget Variable (Traffic Volume Change):")
    print(f"  Global Range: [{df_target['min'].min():.2f}, {df_target['max'].max():.2f}]")
    print(f"  Mean across graphs: {df_target['mean'].mean():.2f} (std: {df_target['mean'].std():.2f})")
    print(f"  Median across graphs: {df_target['median'].mean():.2f}")
    print(f"  Zeros: {df_target['num_zeros'].mean():.0f} nodes/graph ({df_target['num_zeros'].mean()/31635*100:.1f}%)")
    print(f"  Negatives: {df_target['num_negatives'].mean():.0f} nodes/graph ({df_target['num_negatives'].mean()/31635*100:.1f}%)")
    print(f"  Variation between graphs:")
    print(f"    - Min value range: [{df_target['min'].min():.2f}, {df_target['min'].max():.2f}]")
    print(f"    - Max value range: [{df_target['max'].min():.2f}, {df_target['max'].max():.2f}]")
    print(f"    - Mean value range: [{df_target['mean'].min():.2f}, {df_target['mean'].max():.2f}]")

# Part 3: Check which features vary between scenarios
print("\n" + "="*80)
print("[PART 3: SCENARIO VARIATION ANALYSIS]")
print("="*80)

print("\nChecking which features are scenario-dependent (vary across graphs)...")

# For efficiency, sample 10 random graphs
sample_indices = np.random.choice(len(all_graphs), size=min(10, len(all_graphs)), replace=False)
sample_graphs = [all_graphs[i] for i in sample_indices]

for feat_idx in range(6):
    # Get features from all sampled graphs
    feat_arrays = [g.x[:, feat_idx].numpy() for g in sample_graphs]
    
    # Check if all are identical
    all_identical = all(np.array_equal(feat_arrays[0], arr) for arr in feat_arrays[1:])
    
    # Calculate variation coefficient
    means_across_graphs = [arr.mean() for arr in feat_arrays]
    cv = np.std(means_across_graphs) / np.mean(means_across_graphs) if np.mean(means_across_graphs) != 0 else 0
    
    print(f"\nFeature {feat_idx}:")
    print(f"  Identical across sampled graphs: {all_identical}")
    print(f"  Coefficient of variation (CV) of means: {cv:.4f}")
    if cv < 0.01:
        print(f"  -> STATIC (network property)")
    else:
        print(f"  -> DYNAMIC (scenario-dependent)")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
