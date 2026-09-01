import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import sys
import os

# Setup paths
BASE_PATH = "/content/drive/MyDrive/Zamin-thesis/ml_surrogates_for_agent_based_transport_models-main"
sys.path.insert(0, os.path.join(BASE_PATH, "scripts"))

MODEL_DIR = f"{BASE_PATH}/data/TR-C_Benchmarks/point_net_transf_gat_7th_trial_80_10_10_split"

# ============================================================================
# LOAD MODEL
# ============================================================================
print("="*80)
print(" LOADING MODEL - 7TH TRIAL (HIGHER LR + 80-10-10 SPLIT)")
print("="*80)

from torch_geometric.loader import DataLoader
from gnn.models.point_net_transf_gat import PointNetTransfGAT
from gnn.help_functions import GNN_Loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# Load model
model = PointNetTransfGAT(
    in_channels=5,
    out_channels=1,
    point_net_conv_layer_structure_local_mlp=[256],
    point_net_conv_layer_structure_global_mlp=[512],
    gat_conv_layer_structure=[128, 256, 512],
    dropout=0.3,
    use_dropout=True,
    predict_mode_stats=False,
    dtype=torch.float32,
    log_to_wandb=False
).to(device)

model.load_state_dict(torch.load(f"{MODEL_DIR}/trained_model/model.pth", map_location=device))
model.eval()
print("✓ Model loaded successfully!\n")

# ============================================================================
# LOAD TEST DATA
# ============================================================================
print("="*80)
print(" LOADING TEST DATA")
print("="*80)

# Load dataset
test_dataset = torch.load(f"{MODEL_DIR}/data_created_during_training/test_dl.pt", weights_only=False)
print(f"✓ Test dataset loaded: {len(test_dataset)} samples")

# Load loader params
with open(f"{MODEL_DIR}/data_created_during_training/test_loader_params.json", 'r') as f:
    loader_params = json.load(f)

# Create DataLoader
test_loader = DataLoader(
    test_dataset,
    batch_size=loader_params.get('batch_size', 8),
    shuffle=False,
    num_workers=0,
    pin_memory=False
)
print(f"✓ Test loader created: batch_size={test_loader.batch_size}\n")

# ============================================================================
# EVALUATE
# ============================================================================
print("="*80)
print(" RUNNING EVALUATION")
print("="*80)

num_nodes = test_dataset[0].x.shape[0]
loss_fn = GNN_Loss("mse", num_nodes, device, False)

all_predictions = []
all_targets = []
batch_losses = []

print("Running inference...")
with torch.no_grad():
    for batch in tqdm(test_loader):
        batch = batch.to(device)
        predictions = model(batch)
        loss = loss_fn(predictions, batch.y)
        
        batch_losses.append(loss.item())
        all_predictions.append(predictions.cpu().numpy())
        all_targets.append(batch.y.cpu().numpy())

# Combine
all_predictions = np.concatenate(all_predictions, axis=0)
all_targets = np.concatenate(all_targets, axis=0)
avg_test_loss = np.mean(batch_losses)

# ============================================================================
# COMPUTE METRICS
# ============================================================================
print("\n" + "="*80)
print(" TEST PERFORMANCE METRICS")
print("="*80)

# MSE, RMSE, MAE
mse = np.mean((all_predictions - all_targets) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(all_predictions - all_targets))

# R² Score
ss_res = np.sum((all_targets - all_predictions) ** 2)
ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
r2 = 1 - (ss_res / ss_tot)

# Pearson Correlation
from scipy.stats import pearsonr, spearmanr
pearson_r, _ = pearsonr(all_targets.flatten(), all_predictions.flatten())
spearman_r, _ = spearmanr(all_targets.flatten(), all_predictions.flatten())

# MAPE
mape = np.mean(np.abs((all_targets - all_predictions) / (np.abs(all_targets) + 1e-8))) * 100

print(f"\n{'='*40}")
print(f"  COMPARISON: Validation vs Test")
print(f"{'='*40}")
print(f"{'Metric':<20} {'Validation':<15} {'Test':<15}")
print(f"{'-'*50}")
print(f"{'R² Score':<20} {0.5497:<15.4f} {r2:<15.4f}")
print(f"{'Pearson':<20} {0.7427:<15.4f} {pearson_r:<15.4f}")
print(f"{'Spearman':<20} {0.2255:<15.4f} {spearman_r:<15.4f}")
print(f"{'Loss (MSE)':<20} {54.94:<15.2f} {avg_test_loss:<15.2f}")

print(f"\n{'='*40}")
print(f"  DETAILED TEST METRICS")
print(f"{'='*40}")
print(f"Test Loss (MSE):        {avg_test_loss:.4f}")
print(f"MSE:                    {mse:.4f}")
print(f"RMSE:                   {rmse:.4f}")
print(f"MAE:                    {mae:.4f}")
print(f"R² Score:               {r2:.4f}")
print(f"Pearson Correlation:    {pearson_r:.4f}")
print(f"Spearman Correlation:   {spearman_r:.4f}")
print(f"MAPE:                   {mape:.2f}%")

print(f"\n{'='*40}")
print(f"  TARGET STATISTICS")
print(f"{'='*40}")
print(f"Mean:    {np.mean(all_targets):.4f}")
print(f"Std:     {np.std(all_targets):.4f}")
print(f"Min:     {np.min(all_targets):.4f}")
print(f"Max:     {np.max(all_targets):.4f}")

print(f"\n{'='*40}")
print(f"  PREDICTION STATISTICS")
print(f"{'='*40}")
print(f"Mean:    {np.mean(all_predictions):.4f}")
print(f"Std:     {np.std(all_predictions):.4f}")
print(f"Min:     {np.min(all_predictions):.4f}")
print(f"Max:     {np.max(all_predictions):.4f}")

# ============================================================================
# COMPARISON WITH TRIAL 5 (BEST MODEL SO FAR)
# ============================================================================
print(f"\n{'='*40}")
print(f"  COMPARISON WITH TRIAL 5 (BEST MODEL)")
print(f"{'='*40}")
print(f"{'Metric':<20} {'Trial 5':<15} {'Trial 7':<15} {'Difference':<15}")
print(f"{'-'*65}")
print(f"{'R² Score':<20} {0.5553:<15.4f} {r2:<15.4f} {r2-0.5553:<+15.4f}")
print(f"{'Pearson':<20} {0.7468:<15.4f} {pearson_r:<15.4f} {pearson_r-0.7468:<+15.4f}")
print(f"{'MAE':<20} {4.2421:<15.4f} {mae:<15.4f} {mae-4.2421:<+15.4f}")
print(f"{'Learning Rate':<20} {'5e-4':<15} {'6e-4':<15} {'+20%':<15}")
print(f"{'Test Set Size':<20} {'50':<15} {'100':<15} {'2x':<15}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print(" GENERATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 13))

# 1. Predictions vs Actual
ax = axes[0, 0]
ax.scatter(all_targets, all_predictions, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
ax.plot([all_targets.min(), all_targets.max()], 
        [all_targets.min(), all_targets.max()], 
        'r--', linewidth=2.5, label='Perfect Prediction')
ax.set_xlabel('Actual Values', fontsize=13, fontweight='bold')
ax.set_ylabel('Predicted Values', fontsize=13, fontweight='bold')
ax.set_title(f'Predictions vs Actual\n(R²={r2:.4f}, Pearson={pearson_r:.4f})', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# 2. Residuals
ax = axes[0, 1]
residuals = all_predictions - all_targets
ax.scatter(all_targets, residuals, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
ax.axhline(y=0, color='r', linestyle='--', linewidth=2.5)
ax.set_xlabel('Actual Values', fontsize=13, fontweight='bold')
ax.set_ylabel('Residuals (Predicted - Actual)', fontsize=13, fontweight='bold')
ax.set_title('Residual Plot', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# 3. Error Distribution
ax = axes[1, 0]
ax.hist(residuals, bins=40, edgecolor='black', alpha=0.8, color='steelblue')
ax.axvline(x=0, color='r', linestyle='--', linewidth=2.5, label='Zero Error')
ax.axvline(x=np.mean(residuals), color='green', linestyle='--', linewidth=2, 
           label=f'Mean: {np.mean(residuals):.2f}')
ax.set_xlabel('Residuals', fontsize=13, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=13, fontweight='bold')
ax.set_title(f'Error Distribution (MAE={mae:.4f})', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

# 4. Trial Comparison
ax = axes[1, 1]
trials = ['Trial 5\n(LR=5e-4)', 'Trial 7\n(LR=6e-4)']
r2_scores = [0.5553, r2]
colors = ['#51cf66', '#ff6b6b' if r2 < 0.5553 else '#51cf66']

bars = ax.bar(trials, r2_scores, color=colors, edgecolor='black', linewidth=2, alpha=0.8)
ax.axhline(y=0.76, color='gold', linestyle='--', linewidth=2.5, label='Elena Benchmark (0.76)')
ax.set_ylabel('R² Score', fontsize=13, fontweight='bold')
ax.set_title('Higher LR Experiment Result', fontsize=14, fontweight='bold')
ax.set_ylim(0, 0.85)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, score in zip(bars, r2_scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{score:.4f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.suptitle('7th Trial (Higher LR + 80-10-10 Split) - Test Set Evaluation', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()

# Save
plt.savefig(f"{MODEL_DIR}/test_evaluation_results.png", dpi=300, bbox_inches='tight')
print(f"✓ Visualization saved: {MODEL_DIR}/test_evaluation_results.png")
plt.show()

# ============================================================================
# SAVE RESULTS
# ============================================================================
results = {
    "trial_name": "7th_trial_higher_lr_80_10_10",
    "hyperparameters": {
        "batch_size": 8,
        "gradient_accumulation": 3,
        "use_dropout": True,
        "dropout": 0.3,
        "use_weighted_loss": False,
        "lr": 0.0006,
        "split": "80-10-10"
    },
    "validation_metrics": {
        "r2": 0.5497,
        "pearson": 0.7427,
        "spearman": 0.2255,
        "loss": 54.94
    },
    "test_metrics": {
        "r2": float(r2),
        "pearson": float(pearson_r),
        "spearman": float(spearman_r),
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "mape": float(mape),
        "loss": float(avg_test_loss)
    },
    "num_test_samples": len(all_targets),
    "comparison_with_trial5": {
        "trial5_r2": 0.5553,
        "trial7_r2": float(r2),
        "r2_difference": float(r2 - 0.5553),
        "percentage_change": float((r2 - 0.5553) / 0.5553 * 100)
    }
}

with open(f"{MODEL_DIR}/test_evaluation_complete.json", 'w') as f:
    json.dump(results, f, indent=2)

print(f"✓ Results saved: {MODEL_DIR}/test_evaluation_complete.json")

print("\n" + "="*80)
print("  EVALUATION COMPLETE!")
print("="*80)
print("\nNEXT STEP: Compare Trial 7 with Trial 5 to decide on Trial 8 strategy")
