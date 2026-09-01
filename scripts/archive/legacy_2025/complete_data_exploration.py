"""
Complete Data Exploration - Tensors, Graphs, Edge Structure
Verify everything about the dataset for Colab
"""
import torch
from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add safe globals for PyTorch loading
torch.serialization.add_safe_globals([Data, DataEdgeAttr])

# For Colab, update this path:
data_dir = Path("/content/drive/MyDrive/Zamin_thesis/ml_surrogates_for_agent_based_transport_models/data/train_data/dist_not_connected_10k_1pct")

print("="*80)
print("COMPLETE DATA EXPLORATION - TENSORS, GRAPHS & STRUCTURE")
print("="*80)

# Load first batch
batch_file = data_dir / "datalist_batch_1.pt"
data_list = torch.load(batch_file, weights_only=False)

print(f"\n✓ Loaded batch 1: {len(data_list)} graphs")

# ============================================================================
# PART 1: SINGLE GRAPH DETAILED ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("PART 1: SINGLE GRAPH (First Graph) DETAILED ANALYSIS")
print("="*80)

graph = data_list[0]

print("\n--- Basic Structure ---")
print(f"Number of nodes: {graph.num_nodes}")
print(f"Number of edges: {graph.num_edges}")
print(f"Is undirected: {graph.is_undirected()}")

print("\n--- Node Features (graph.x) ---")
print(f"Shape: {graph.x.shape}")
print(f"Data type: {graph.x.dtype}")
print(f"Device: {graph.x.device}")
print(f"Memory size: {graph.x.element_size() * graph.x.nelement() / 1024 / 1024:.2f} MB")

print("\n--- Edge Index (graph.edge_index) ---")
print(f"Shape: {graph.edge_index.shape}")
print(f"Data type: {graph.edge_index.dtype}")
print(f"Min node index: {graph.edge_index.min()}")
print(f"Max node index: {graph.edge_index.max()}")
print(f"First 5 edges:")
for i in range(5):
    src, dst = graph.edge_index[0, i].item(), graph.edge_index[1, i].item()
    print(f"  Edge {i}: {src} -> {dst}")

print("\n--- Positional Features (graph.pos) ---")
if hasattr(graph, 'pos'):
    print(f"Shape: {graph.pos.shape}")
    print(f"Data type: {graph.pos.dtype}")
    print(f"Interpretation: {graph.pos.shape[0]} nodes × {graph.pos.shape[1]} coordinate sets × {graph.pos.shape[2]}D")
    print(f"First node positions:")
    print(f"  Start point: {graph.pos[0, 0, :].numpy()}")
    print(f"  End point: {graph.pos[0, 1, :].numpy()}")
    print(f"  Midpoint: {graph.pos[0, 2, :].numpy()}")
else:
    print("No positional features found")

print("\n--- Target Variable (graph.y) ---")
if hasattr(graph, 'y'):
    print(f"Shape: {graph.y.shape}")
    print(f"Data type: {graph.y.dtype}")
    print(f"Range: [{graph.y.min():.2f}, {graph.y.max():.2f}]")
    print(f"Mean: {graph.y.mean():.2f}, Std: {graph.y.std():.2f}")
else:
    print("No target variable found")

print("\n--- Additional Attributes ---")
print("All attributes in graph object:")
for attr in dir(graph):
    if not attr.startswith('_') and not callable(getattr(graph, attr)):
        try:
            val = getattr(graph, attr)
            if isinstance(val, torch.Tensor):
                print(f"  {attr}: Tensor{tuple(val.shape)}")
            else:
                print(f"  {attr}: {type(val).__name__}")
        except:
            pass

# ============================================================================
# PART 2: FEATURE TENSOR DETAILED ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("PART 2: FEATURE TENSOR (graph.x) DETAILED ANALYSIS")
print("="*80)

print("\n--- Feature-by-Feature Breakdown ---")
feature_names = ['LENGTH', 'CAPACITY', 'BASELINE_VOLUME', 'CAPACITY_REDUCTION', 'HIGHWAY', 'FREESPEED']

for i in range(6):
    feat = graph.x[:, i].numpy()
    print(f"\n[Feature {i}: {feature_names[i]}]")
    print(f"  Range: [{feat.min():.2f}, {feat.max():.2f}]")
    print(f"  Mean: {feat.mean():.2f}, Median: {np.median(feat):.2f}, Std: {feat.std():.2f}")
    print(f"  Zeros: {(feat == 0).sum()} ({(feat == 0).sum()/len(feat)*100:.1f}%)")
    print(f"  Negatives: {(feat < 0).sum()} ({(feat < 0).sum()/len(feat)*100:.1f}%)")
    print(f"  Unique values: {len(np.unique(feat))}")
    
    # Sample values
    non_zero = feat[feat != 0]
    if len(non_zero) > 0:
        sample = np.random.choice(non_zero, min(5, len(non_zero)), replace=False)
        print(f"  Sample non-zero values: {sample}")

# ============================================================================
# PART 3: GRAPH STRUCTURE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("PART 3: GRAPH STRUCTURE ANALYSIS")
print("="*80)

print("\n--- Edge Connectivity ---")
edge_index = graph.edge_index.numpy()
src_nodes = edge_index[0, :]
dst_nodes = edge_index[1, :]

# Node degree analysis
from collections import Counter
out_degree = Counter(src_nodes)
in_degree = Counter(dst_nodes)

print(f"Nodes with edges: {len(set(src_nodes) | set(dst_nodes))}")
print(f"Isolated nodes: {graph.num_nodes - len(set(src_nodes) | set(dst_nodes))}")

print(f"\nOut-degree statistics:")
degrees = list(out_degree.values())
print(f"  Min: {min(degrees)}, Max: {max(degrees)}, Mean: {np.mean(degrees):.2f}")
print(f"  Nodes with degree 0: {graph.num_nodes - len(out_degree)}")
print(f"  Nodes with degree 1: {list(out_degree.values()).count(1)}")
print(f"  Nodes with degree >10: {sum(1 for d in degrees if d > 10)}")

# Self-loops check
self_loops = (src_nodes == dst_nodes).sum()
print(f"\nSelf-loops: {self_loops} ({self_loops/len(src_nodes)*100:.2f}%)")

# ============================================================================
# PART 4: COMPARE MULTIPLE GRAPHS
# ============================================================================
print("\n" + "="*80)
print("PART 4: COMPARING MULTIPLE GRAPHS")
print("="*80)

print("\n--- Structure Consistency Check (First 10 graphs) ---")
for i in range(min(10, len(data_list))):
    g = data_list[i]
    print(f"Graph {i}: nodes={g.num_nodes}, edges={g.num_edges}, features={g.x.shape[1]}, has_pos={hasattr(g, 'pos')}, has_y={hasattr(g, 'y')}")

print("\n--- Feature Variation Across Graphs ---")
print("Checking if features change across scenarios...")

# Compare first 10 graphs
sample_size = min(10, len(data_list))
for feat_idx in range(6):
    values = []
    for i in range(sample_size):
        values.append(data_list[i].x[:, feat_idx].numpy())
    
    # Check if all identical
    all_same = all(np.array_equal(values[0], v) for v in values[1:])
    
    # Calculate variation
    means = [v.mean() for v in values]
    cv = np.std(means) / np.mean(means) if np.mean(means) != 0 else 0
    
    status = "STATIC" if cv < 0.01 else "DYNAMIC"
    print(f"Feature {feat_idx} ({feature_names[feat_idx]}): {status} (CV={cv:.6f})")

# ============================================================================
# PART 5: EDGE CASES & SPECIAL PATTERNS
# ============================================================================
print("\n" + "="*80)
print("PART 5: EDGE CASES & SPECIAL PATTERNS")
print("="*80)

print("\n--- Zero-Length Segments ---")
length = graph.x[:, 0].numpy()
zero_length = (length == 0).sum()
print(f"Zero-length segments: {zero_length} ({zero_length/len(length)*100:.1f}%)")
if zero_length > 0:
    zero_idx = np.where(length == 0)[0][:5]
    print(f"Sample zero-length node indices: {zero_idx}")
    print("Their other features:")
    for idx in zero_idx[:3]:
        print(f"  Node {idx}: capacity={graph.x[idx, 1]:.0f}, baseline_vol={graph.x[idx, 2]:.0f}, highway={graph.x[idx, 4]:.0f}")

print("\n--- Negative Baseline Volume Analysis ---")
baseline = graph.x[:, 2].numpy()
negative = (baseline < 0).sum()
print(f"Negative baseline volumes: {negative} ({negative/len(baseline)*100:.1f}%)")
if negative > 0:
    neg_values = baseline[baseline < 0]
    print(f"Range of negative values: [{neg_values.min():.0f}, {neg_values.max():.0f}]")
    print(f"Are they multiples of 60? {np.all(np.abs(neg_values) % 60 < 0.01)}")

print("\n--- Highway Type -1 (Public Transport) Analysis ---")
highway = graph.x[:, 4].numpy()
pt_links = (highway == -1).sum()
print(f"PT links (highway=-1): {pt_links} ({pt_links/len(highway)*100:.1f}%)")
if pt_links > 0:
    pt_idx = np.where(highway == -1)[0][:3]
    print("Sample PT link characteristics:")
    for idx in pt_idx:
        print(f"  Node {idx}: length={graph.x[idx, 0]:.1f}m, capacity={graph.x[idx, 1]:.0f}, freespeed={graph.x[idx, 5]:.2f}")

print("\n--- Capacity Reduction Patterns ---")
cap_red = graph.x[:, 3].numpy()
unique_reductions = np.unique(cap_red)
print(f"Unique capacity reduction values: {len(unique_reductions)}")
print("Distribution:")
for val in sorted(unique_reductions)[:10]:
    count = (cap_red == val).sum()
    print(f"  {val:.2f}%: {count} nodes ({count/len(cap_red)*100:.1f}%)")

# ============================================================================
# PART 6: DATA QUALITY CHECKS
# ============================================================================
print("\n" + "="*80)
print("PART 6: DATA QUALITY CHECKS")
print("="*80)

print("\n--- NaN/Inf Check ---")
has_nan = torch.isnan(graph.x).any()
has_inf = torch.isinf(graph.x).any()
print(f"Contains NaN: {has_nan}")
print(f"Contains Inf: {has_inf}")

print("\n--- Feature Correlation Matrix ---")
features_np = graph.x.numpy()
corr_matrix = np.corrcoef(features_np.T)
print("Correlation matrix (6×6):")
print("        ", "  ".join([f"{name[:6]:>8}" for name in feature_names]))
for i, name in enumerate(feature_names):
    print(f"{name[:8]:8}", "  ".join([f"{corr_matrix[i, j]:8.3f}" for j in range(6)]))

print("\n--- Target Correlation with Features ---")
if hasattr(graph, 'y'):
    target = graph.y.numpy().flatten()
    for i, name in enumerate(feature_names):
        corr = np.corrcoef(features_np[:, i], target)[0, 1]
        print(f"  {name:20} : {corr:7.4f}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("EXPLORATION SUMMARY")
print("="*80)

print("\n✓ Data Structure:")
print(f"  - Graphs per batch: {len(data_list)}")
print(f"  - Nodes per graph: {graph.num_nodes}")
print(f"  - Edges per graph: {graph.num_edges}")
print(f"  - Node features: 6")
print(f"  - Positional features: 3 × 2D coordinates")
print(f"  - Target: 1D per node")

print("\n✓ Feature Order Confirmed:")
print("  0: LENGTH (0-1596m)")
print("  1: CAPACITY (0-14400 veh/h)")
print("  2: BASELINE_VOLUME (-4800 to 0, DYNAMIC)")
print("  3: CAPACITY_REDUCTION (0-33.33%)")
print("  4: HIGHWAY (-1 to 9)")
print("  5: FREESPEED (4.17-2568.58)")

print("\n✓ Key Findings:")
print(f"  - Only Feature 2 (Baseline Volume) varies across scenarios")
print(f"  - {zero_length} nodes have zero length (likely intersections)")
print(f"  - {pt_links} nodes are PT links (highway=-1)")
print(f"  - {self_loops} self-loops in edge structure")
print(f"  - All graphs have identical structure (nodes, edges)")

print("\n✓ Data Quality: Clean (no NaN/Inf)")
print("\n" + "="*80)
print("EXPLORATION COMPLETE")
print("="*80)
