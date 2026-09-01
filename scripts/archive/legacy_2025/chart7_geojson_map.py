import torch
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import os
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Load Paris districts GeoJSON
geojson_path = '/content/drive/MyDrive/Zamin_thesis/ml_surrogates_for_agent_based_transport_models/data/visualisation/districts_paris.geojson'
gdf = gpd.read_file(geojson_path)

# Load first scenario to get network statistics
data = torch.load('/content/drive/MyDrive/Zamin_thesis/ml_surrogates_for_agent_based_transport_models/data/train_data/dist_not_connected_10k_1pct/datalist_batch_1.pt', weights_only=False)
scenario_0 = data[0]

print("\n" + "="*80)
print("GEOJSON DATA INSPECTION")
print("="*80)
print(f"Number of districts: {len(gdf)}")
print(f"\nGeoJSON Columns Explanation:")
print(f"  • 'c_ar'      = District Number (1-20 arrondissements)")
print(f"  • 'surface'   = Area of district (square meters)")
print(f"  • 'perimetre' = Perimeter/boundary length (meters)")
print(f"  • 'geometry'  = Geographic shape coordinates (polygon)")
print(f"\nCoordinate Reference System: {gdf.crs}")
print(f"  → EPSG:4326 = Standard latitude/longitude coordinates")
print(f"\nDistrict Numbers (c_ar): {sorted(gdf['c_ar'].unique())}")
print(f"All 20 Paris arrondissements numbered 1-20")
print(f"\nSample district data:")
print(gdf[['c_ar', 'surface', 'perimetre']].head(10))
print("="*80 + "\n")

# Create figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Plot 1: Paris Districts Map - Modern Blue Theme
ax = axes[0]
gdf.plot(ax=ax, edgecolor='#2C3E50', facecolor='#3498DB', linewidth=2, alpha=0.7)
ax.set_title('Paris Network Coverage\n20 Arrondissements (Districts)', fontsize=14, fontweight='bold', color='#2C3E50')
ax.set_xlabel('Longitude (East-West Position)', fontsize=11)
ax.set_ylabel('Latitude (North-South Position)', fontsize=11)
ax.grid(True, alpha=0.3, linestyle='--', color='gray')
ax.set_facecolor('#ECF0F1')

# Add district numbers to each polygon
for idx, row in gdf.iterrows():
    centroid = row.geometry.centroid
    district_num = int(row['c_ar'])
    ax.annotate(f"{district_num}", xy=(centroid.x, centroid.y), 
               ha='center', va='center', fontsize=11, 
               color='white', fontweight='bold',
               bbox=dict(boxstyle='circle', facecolor='#E74C3C', 
                        edgecolor='#C0392B', linewidth=2, alpha=0.95))

# Add detailed network statistics box
stats_text = "PARIS ROAD NETWORK:\n"
stats_text += "─" * 35 + "\n"
stats_text += f"📍 Road Segments: {scenario_0.x.shape[0]:,}\n"
stats_text += f"   (Total individual roads)\n\n"
stats_text += f"🔗 Connections: {scenario_0.edge_index.shape[1]:,}\n"
stats_text += f"   (Junctions between roads)\n\n"
stats_text += f"🏙️  Districts: {len(gdf)} arrondissements\n"
stats_text += f"   (Numbered 1-20)\n\n"
stats_text += f"📊 Average: {scenario_0.x.shape[0]//len(gdf):,} roads/district"
ax.text(0.02, 0.98, stats_text,
        transform=ax.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='#FFFFFF', 
                 edgecolor='#2C3E50', linewidth=2.5, alpha=0.97),
        fontsize=10, family='monospace', fontweight='bold', color='#2C3E50')

# Plot 2: Network Feature Summary Map - Modern Green Theme
ax = axes[1]
gdf.plot(ax=ax, edgecolor='#16A085', facecolor='#A8E6CF', linewidth=2, alpha=0.7)
ax.set_title('GNN Training Features Summary\n(5 Features × 1000 Scenarios)', fontsize=14, fontweight='bold', color='#16A085')
ax.set_xlabel('Longitude (East-West Position)', fontsize=11)
ax.set_ylabel('Latitude (North-South Position)', fontsize=11)
ax.grid(True, alpha=0.3, linestyle='--', color='gray')
ax.set_facecolor('#E8F8F5')

# Add district numbers to second panel as well
for idx, row in gdf.iterrows():
    centroid = row.geometry.centroid
    district_num = int(row['c_ar'])
    ax.annotate(f"{district_num}", xy=(centroid.x, centroid.y), 
               ha='center', va='center', fontsize=11, 
               color='white', fontweight='bold',
               bbox=dict(boxstyle='circle', facecolor='#16A085', 
                        edgecolor='#138D75', linewidth=2, alpha=0.95))

# Add detailed feature summary with full explanations
feature_summary = "GNN INPUT FEATURES:\n"
feature_summary += "─" * 40 + "\n"
feature_summary += "(Each road has these 5 properties)\n\n"
feature_summary += "✓ [0] VOL_BASE_CASE\n"
feature_summary += "    Traffic volume in normal scenario\n"
feature_summary += "    Range: 0-1,596 vehicles/day\n\n"
feature_summary += "✓ [1] CAPACITY_BASE_CASE\n"
feature_summary += "    Maximum traffic road can handle\n"
feature_summary += "    Range: 0-14,400 vehicles/hour\n\n"
feature_summary += "✓ [2] CAPACITY_REDUCTION ⭐\n"
feature_summary += "    Policy intervention value\n"
feature_summary += "    Negative = reduced capacity\n"
feature_summary += "    VARIES across 1000 scenarios!\n\n"
feature_summary += "✓ [3] FREESPEED\n"
feature_summary += "    Speed limit of road\n"
feature_summary += "    Encoded in meters/second\n\n"
feature_summary += "✓ [4] LENGTH\n"
feature_summary += "    Road segment length\n"
feature_summary += "    Encoded as category (0-9)\n\n"
feature_summary += "✗ [5] HIGHWAY - NOT USED\n"
feature_summary += "    Road type (redundant info)"

ax.text(0.02, 0.98, feature_summary,
        transform=ax.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='#FFFFFF', 
                 edgecolor='#16A085', linewidth=2.5, alpha=0.97),
        fontsize=9, family='monospace', fontweight='bold', color='#16A085')

plt.tight_layout()

# Save the figure
os.makedirs('data/visualisation', exist_ok=True)
plt.savefig('data/visualisation/chart7_geojson_network_map.png', dpi=150, bbox_inches='tight')
plt.show()

# Print detailed information
print("\n" + "="*80)
print("CHART 7: GEOJSON NETWORK VISUALIZATION")
print("="*80)
print()
print("📍 Geographic Context:")
print("  → Paris road network from OpenStreetMap (OSM)")
print("  → GeoJSON shows district boundaries in study area")
print("  → Network distributed across multiple districts")
print()
print("Network Coverage:")
print(f"  Total road segments:     {scenario_0.x.shape[0]:,}")
print(f"  Total connections:       {scenario_0.edge_index.shape[1]:,}")
print(f"  Districts covered:       {len(gdf)}")
print(f"  Avg roads per district:  {scenario_0.x.shape[0]//len(gdf) if len(gdf) > 0 else 0:,}")
print()
print("📄 REFERENCE FROM PAPER (Elena Boreale, 2024):")
print("  'The Paris road network is extracted from OpenStreetMap and covers")
print("  the central metropolitan area. The network is represented as a graph")
print("  with road segments as nodes and intersections defining connections.'")
print()
print("GNN Graph Structure:")
print("  NODE FEATURES (x):")
print(f"    • Shape: {scenario_0.x.shape} → ({scenario_0.x.shape[0]:,} roads × 6 features)")
print("    • 5 features used for training")
print("    • 1 feature (HIGHWAY) excluded")
print()
print("  EDGE CONNECTIVITY (edge_index):")
print(f"    • Shape: {scenario_0.edge_index.shape} → (2 × {scenario_0.edge_index.shape[1]:,} connections)")
print("    • Represents which roads are connected (adjacency)")
print("    • Bidirectional connections (if A→B exists, B→A also exists)")
print()
print("  EDGE FEATURES (edge_attr):")
if hasattr(scenario_0, 'edge_attr') and scenario_0.edge_attr is not None:
    print(f"    • Shape: {scenario_0.edge_attr.shape}")
    print("    • Additional edge properties (if any)")
else:
    print("    • No edge attributes (only connectivity matters)")
print()
print("Why Geographic Visualization Matters:")
print("  1. SPATIAL CONTEXT:")
print("     → Shows real-world layout of Paris road network")
print("     → Helps understand policy impact on specific districts")
print("     → Validates that network covers realistic area")
print()
print("  2. NETWORK STRUCTURE:")
print("     → Dense urban grid → many alternative routes")
print("     → High connectivity → traffic can redistribute easily")
print("     → Central Paris has denser network than periphery")
print()
print("  3. POLICY DESIGN:")
print("     → Capacity reductions applied to specific geographic roads")
print("     → District-level policies affect localized traffic")
print("     → GNN must learn spatial redistribution patterns")
print()
print("  4. MODEL VALIDATION:")
print("     → Can visualize predictions on actual map")
print("     → Compare before/after traffic flows geographically")
print("     → Identify which districts are most affected by policies")
print()
print("📊 How GNN Uses This Data:")
print()
print("  STEP 1 - Graph Construction:")
print("  ┌──────────────────────────────────────────────────────┐")
print("  │  Each road segment = NODE in graph                  │")
print("  │  Connections at junctions = EDGES in graph          │")
print("  │  Features (capacity, speed, etc.) = NODE ATTRIBUTES │")
print("  └──────────────────────────────────────────────────────┘")
print()
print("  STEP 2 - Message Passing:")
print("  ┌──────────────────────────────────────────────────────┐")
print("  │  GNN passes messages between connected roads         │")
print("  │  'If my road closes, where will traffic go?'        │")
print("  │  Learns to propagate traffic through network        │")
print("  └──────────────────────────────────────────────────────┘")
print()
print("  STEP 3 - Prediction:")
print("  ┌──────────────────────────────────────────────────────┐")
print("  │  Input: CAPACITY_REDUCTION (policy)                 │")
print("  │  Output: New traffic volumes on all roads           │")
print("  │  Compares to MATSim simulation ground truth         │")
print("  └──────────────────────────────────────────────────────┘")
print()
print("Training Process:")
print("  • 1000 scenarios with DIFFERENT capacity reductions")
print("  • Each scenario = different policy intervention")
print("  • GNN learns: 'Given policy X, traffic redistributes to Y'")
print("  • Goal: Replace slow MATSim simulation with fast GNN prediction")
print()
print("📌 Key Insight:")
print("  Geographic structure is IMPLICIT in the graph connections.")
print("  GNN doesn't need lat/lon coordinates - it learns from topology!")
print()
print("  Why this works:")
print("  • Connected roads in graph = connected roads geographically")
print("  • Traffic flows along connections (edges)")
print("  • GNN learns spatial patterns through message passing")
print()
print("Summary:")
print("  ✓ Paris network covers central metropolitan area")
print(f"  ✓ {scenario_0.x.shape[0]:,} road segments with 6 features each")
print(f"  ✓ {scenario_0.edge_index.shape[1]:,} connections defining network topology")
print("  ✓ GeoJSON shows geographic context (NOT used by GNN directly)")
print("  ✓ GNN learns from graph structure, not geographic coordinates")
print()
print("="*80)
print("Map saved: data/visualisation/chart7_geojson_network_map.png")
print("="*80 + "\n")
