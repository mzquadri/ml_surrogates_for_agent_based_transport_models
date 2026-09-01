# MONTE CARLO DROPOUT FOR TRIAL 8 (BEST MODEL)
# Based on original repo method - Adapted for Google Colab
# Trial 8: R² = 0.5957, Dropout = 0.2, Batch Size = 8

"""
CELL 1: Mount Drive and Setup
"""
from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir('/content/drive/MyDrive/Zamin-thesis/ml_surrogates_for_agent_based_transport_models-main/scripts/misc')
print(f"Current directory: {os.getcwd()}")

"""
CELL 2: Imports
"""
import sys
import json
import joblib
import numpy as np
from tqdm import tqdm
import geopandas as gpd
import torch

# Add the 'scripts' directory to Python Path
project_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
scripts_path = os.path.join(project_root, 'scripts')
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

import evaluation.help_functions as hf
import evaluation.plot_functions as pf
from gnn.help_functions import mc_dropout_predict
from gnn.models.point_net_transf_gat import PointNetTransfGAT
from data_preprocessing.help_functions import highway_mapping

print("Imports successful!")

"""
CELL 3: Configuration - Trial 8
"""
# Paths - TRIAL 8
run_path = os.path.join(project_root, "data", "TR-C_Benchmarks", "point_net_transf_gat_8th_trial_lower_dropout")
districts = gpd.read_file(os.path.join(project_root, "data", "visualisation", "districts_paris.geojson"))
base_case_path = os.path.join(project_root, "data", "links_and_stats", "pop_1pct_basecase_average_output_links.geojson")
result_path = 'results/'

# GNN Parameters - TRIAL 8 CONFIGURATION
point_net_conv_layer_structure_local_mlp = "256"
point_net_conv_layer_structure_global_mlp = "512"
gat_conv_layer_structure = "128,256,512"
dropout = 0.2  # Trial 8: Lower dropout
use_dropout = True  # Trial 8 uses dropout (MC Dropout needs this)
predict_mode_stats = False
in_channels = 5
out_channels = 1

links_base_case = gpd.read_file(base_case_path, crs="EPSG:4326")
data_created_during_training = os.path.join(run_path, 'data_created_during_training')

print("="*80)
print("TRIAL 8 CONFIGURATION")
print("="*80)
print(f"Model Path: {run_path}")
print(f"Dropout: {dropout}")
print(f"Use Dropout: {use_dropout}")
print(f"Expected R²: 0.5957")
print("="*80)

"""
CELL 4: Load Test Data
"""
# Load scalers
scaler_x = joblib.load(os.path.join(data_created_during_training, 'test_x_scaler.pkl'))
scaler_pos = joblib.load(os.path.join(data_created_during_training, 'test_pos_scaler.pkl'))

# Load the test dataset created during training - IMPORTANT: weights_only=False
test_set_dl = torch.load(os.path.join(data_created_during_training, 'test_dl.pt'), weights_only=False)

# Load the DataLoader parameters
with open(os.path.join(data_created_during_training, 'test_loader_params.json'), 'r') as f:
    test_set_dl_loader_params = json.load(f)
    
# Remove or correct collate_fn if it is incorrectly specified
if 'collate_fn' in test_set_dl_loader_params and isinstance(test_set_dl_loader_params['collate_fn'], str):
    del test_set_dl_loader_params['collate_fn']

test_set_loader = torch.utils.data.DataLoader(test_set_dl, **test_set_dl_loader_params)

print(f"Test data loaded: {len(test_set_loader.dataset)} scenarios")

"""
CELL 5: Load Model
"""
# Parse layer structures
point_net_conv_layer_structure_local_mlp = [int(x) for x in point_net_conv_layer_structure_local_mlp.split(',')]
point_net_conv_layer_structure_global_mlp = [int(x) for x in point_net_conv_layer_structure_global_mlp.split(',')]
gat_conv_layer_structure = [int(x) for x in gat_conv_layer_structure.split(',')]

# Initialize model
model = PointNetTransfGAT(
    in_channels=in_channels, 
    out_channels=out_channels,
    point_net_conv_layer_structure_local_mlp=point_net_conv_layer_structure_local_mlp, 
    point_net_conv_layer_structure_global_mlp=point_net_conv_layer_structure_global_mlp,
    gat_conv_layer_structure=gat_conv_layer_structure,
    dropout=dropout,
    use_dropout=use_dropout,
    predict_mode_stats=predict_mode_stats
)

# Load the model state dictionary - IMPORTANT: weights_only=False
model_path = os.path.join(run_path, 'trained_model/model.pth')
model.load_state_dict(torch.load(model_path, weights_only=False), strict=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

loss_fct = torch.nn.MSELoss().to(dtype=torch.float32).to(device)

print(f"Trial 8 Model loaded successfully!")
print(f"Device: {device}")

"""
CELL 6: Validate Model on Full Test Set
"""
test_loss, r_squared, actual_vals, predictions, baseline_loss = hf.validate_model_on_test_set(
    model, test_set_loader.dataset, loss_fct, device
)

print("="*80)
print("TRIAL 8 - FULL TEST SET VALIDATION")
print("="*80)
print(f"Test Loss: {test_loss:.4f}")
print(f"R-squared: {r_squared:.4f}")
print(f"Baseline Loss: {baseline_loss:.4f}")
print("="*80)

"""
CELL 7: MC Dropout on Single Sample (Spatial Visualization)
"""
i = 32  # Sample index to visualize

test_data = test_set_loader.dataset[i]
test_x = test_set_loader.dataset[i].x
test_x = test_x.to('cpu')

test_loss, r_squared, actual_vals, predictions, baseline_loss = hf.validate_model_on_test_set(
    model, test_data, loss_fct, device
)

print(f"Sample {i}")
print(f"Test Loss: {test_loss:.4f}")
print(f"R-squared: {r_squared:.4f}")
print(f"Baseline Loss: {baseline_loss:.4f}")

# Run MC Dropout (50 samples)
inversed_x = scaler_x.inverse_transform(test_x)
mean_predictions, uncertainties = mc_dropout_predict(model, test_data, num_samples=50, device=device)

# Create GeoDataFrame with uncertainty values
gdf_with_og_values = hf.data_to_geodataframe_with_og_values(
    data=test_data, 
    original_gdf=links_base_case, 
    predicted_values=predictions, 
    inversed_x=inversed_x, 
    use_all_features=False
)
gdf_with_og_values['capacity_reduction_rounded'] = gdf_with_og_values['capacity_reduction'].round(decimals=3)
gdf_with_og_values['highway'] = gdf_with_og_values['highway'].map(highway_mapping)
gdf_with_og_values['mc_uncertainty'] = uncertainties

# Plot uncertainty map
pf.plot_combined_output(
    gdf_input=gdf_with_og_values, 
    column_to_plot="mc_uncertainty", 
    plot_contour_lines=False,
    save_it=False, 
    number_to_plot=i, 
    zone_to_plot="this zone", 
    is_predicted=True, 
    use_fixed_norm=False,
    known_districts=False, 
    buffer=0.0005, 
    districts_of_interest=None, 
    cmap='Reds'
)

print(f"Sample {i} Uncertainty Statistics:")
print(f"  Mean Uncertainty: {uncertainties.mean():.4f}")
print(f"  Max Uncertainty: {uncertainties.max():.4f}")
print(f"  Min Uncertainty: {uncertainties.min():.4f}")

"""
CELL 8: MC Dropout on Entire Test Set (Network-wide Average Uncertainty)
"""
print("Running MC Dropout on entire test set...")
print("This may take 15-20 minutes...")

mean_uncertainties = []

for i in tqdm(range(len(test_set_loader.dataset)), desc="Processing scenarios"):
    test_data = test_set_loader.dataset[i]
    test_x = test_set_loader.dataset[i].x
    test_x = test_x.to('cpu')
    
    mean_predictions, uncertainties = mc_dropout_predict(model, test_data, num_samples=50, device=device)
    mean_uncertainties.append(uncertainties)

# Average uncertainties across all scenarios
mean_uncertainties = np.array(mean_uncertainties).mean(axis=0)

print(f"\nNetwork-wide Average Uncertainty Statistics:")
print(f"  Mean: {mean_uncertainties.mean():.4f}")
print(f"  Std: {mean_uncertainties.std():.4f}")
print(f"  Max: {mean_uncertainties.max():.4f}")
print(f"  Min: {mean_uncertainties.min():.4f}")

"""
CELL 9: Plot Network-wide Average Uncertainty Map
"""
# Use last sample's structure (geometry doesn't change)
inversed_x = scaler_x.inverse_transform(test_x)
gdf_with_og_values = hf.data_to_geodataframe_with_og_values(
    data=test_data, 
    original_gdf=links_base_case, 
    predicted_values=mean_predictions, 
    inversed_x=inversed_x, 
    use_all_features=False
)
gdf_with_og_values['capacity_reduction_rounded'] = gdf_with_og_values['capacity_reduction'].round(decimals=3)
gdf_with_og_values['highway'] = gdf_with_og_values['highway'].map(highway_mapping)
gdf_with_og_values['mc_uncertainty'] = mean_uncertainties

# Plot network-wide average uncertainty
pf.plot_combined_output(
    gdf_input=gdf_with_og_values, 
    column_to_plot="mc_uncertainty", 
    plot_contour_lines=False,
    save_it=True,  # Save this important plot
    number_to_plot=999,  # Special number for average
    zone_to_plot="Network-wide Average", 
    is_predicted=True, 
    use_fixed_norm=False,
    known_districts=False, 
    buffer=0.0005, 
    districts_of_interest=None, 
    cmap='Reds'
)

print("\nNetwork-wide average uncertainty map generated!")

"""
CELL 10: Save Results
"""
import pandas as pd
from datetime import datetime

# Save network-wide uncertainty data
uncertainty_results = pd.DataFrame({
    'Link_ID': range(len(mean_uncertainties)),
    'Average_Uncertainty': mean_uncertainties
})
uncertainty_results.to_csv('trial8_networkwide_mc_uncertainty.csv', index=False)

# Summary statistics
summary = {
    'Trial': 8,
    'Model': 'PointNetTransfGAT',
    'Test_R2': r_squared,
    'MC_Samples': 50,
    'Test_Scenarios': len(test_set_loader.dataset),
    'Mean_Uncertainty': mean_uncertainties.mean(),
    'Std_Uncertainty': mean_uncertainties.std(),
    'Max_Uncertainty': mean_uncertainties.max(),
    'Min_Uncertainty': mean_uncertainties.min(),
    'Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv('trial8_mc_dropout_summary.csv', index=False)

print("="*80)
print("RESULTS SAVED")
print("="*80)
print("Files created:")
print("  - trial8_networkwide_mc_uncertainty.csv")
print("  - trial8_mc_dropout_summary.csv")
print("="*80)
print("\nSummary:")
print(summary_df.T.to_string())
print("="*80)
