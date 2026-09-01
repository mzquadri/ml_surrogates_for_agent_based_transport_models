"""
Comprehensive Analysis of All 6 WandB CSV Files
Analyzes complete training metrics and compares with Elena's benchmarks
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Define folder path
folder = Path(r"D:\Python Projects\Zamin_Thesis\ml_surrogates_for_agent_based_transport_models\New folder")

# Load all 6 CSV files
print("=" * 80)
print("LOADING ALL 6 CSV FILES FROM WANDB")
print("=" * 80)

files = sorted(folder.glob("*.csv"))
print(f"\nFound {len(files)} CSV files:")
for f in files:
    print(f"  - {f.name}")

# Load each file
dfs = {}
for file in files:
    df = pd.read_csv(file)
    # Extract metric name from column names
    cols = [col for col in df.columns if 'olive-shadow-2' in col and '__MIN' not in col and '__MAX' not in col and '_step' not in col]
    if cols:
        metric_name = cols[0].replace('olive-shadow-2 - ', '')
        dfs[metric_name] = df
        print(f"\n{metric_name:15s}: {len(df)} epochs, columns: {list(df.columns[:3])}")

# Extract data from each metric
r2_data = dfs.get('r^2', dfs.get('r2'))
pearson_data = dfs.get('pearson')
spearman_data = dfs.get('spearman')
train_loss_data = dfs.get('train_loss')
val_loss_data = dfs.get('val_loss')
lr_data = dfs.get('lr')

print("\n" + "=" * 80)
print("EXTRACTING BEST RESULTS")
print("=" * 80)

# Get column names (handle both 'r^2' and 'r2')
r2_col = [c for c in r2_data.columns if 'r^2' in c.lower() and '__MIN' not in c and '__MAX' not in c and '_step' not in c][0]
pearson_col = [c for c in pearson_data.columns if 'pearson' in c and '__MIN' not in c and '__MAX' not in c and '_step' not in c][0]
spearman_col = [c for c in spearman_data.columns if 'spearman' in c and '__MIN' not in c and '__MAX' not in c and '_step' not in c][0]
train_loss_col = [c for c in train_loss_data.columns if 'train_loss' in c and '__MIN' not in c and '__MAX' not in c and '_step' not in c][0]
val_loss_col = [c for c in val_loss_data.columns if 'val_loss' in c and '__MIN' not in c and '__MAX' not in c and '_step' not in c][0]
lr_col = [c for c in lr_data.columns if 'lr' in c and '__MIN' not in c and '__MAX' not in c and '_step' not in c][0]

# Extract best metrics
best_r2 = r2_data[r2_col].max()
best_r2_epoch = r2_data[r2_col].idxmax()

best_val_loss = val_loss_data[val_loss_col].min()
best_val_loss_epoch = val_loss_data[val_loss_col].idxmin()

final_pearson = pearson_data[pearson_col].iloc[-1]
final_spearman = spearman_data[spearman_col].iloc[-1]

final_train_loss = train_loss_data[train_loss_col].iloc[-1]

# Elena's benchmarks
elena_r2 = 0.76
elena_pearson = 0.87
elena_spearman = 0.85
elena_mse = 24.95  # Mean Squared Error
elena_mae = 2.74   # Mean Absolute Error

# Display results
print(f"\n{'METRIC':<20s} {'YOUR MODEL':<15s} {'ELENA MODEL':<15s} {'DIFFERENCE':<15s}")
print("-" * 70)
print(f"{'R² Score':<20s} {best_r2:<15.4f} {elena_r2:<15.4f} {(best_r2-elena_r2):+.4f}")
print(f"{'  (at epoch)':<20s} {best_r2_epoch:<15d}")
print(f"{'Pearson Corr':<20s} {final_pearson:<15.4f} {elena_pearson:<15.4f} {(final_pearson-elena_pearson):+.4f}")
print(f"{'Spearman Corr':<20s} {final_spearman:<15.4f} {elena_spearman:<15.4f} {(final_spearman-elena_spearman):+.4f}")
print(f"{'Val Loss (best)':<20s} {best_val_loss:<15.2f} {elena_mse:<15.2f} {(best_val_loss-elena_mse):+.2f}")
print(f"{'  (at epoch)':<20s} {best_val_loss_epoch:<15d}")
print(f"{'Train Loss (final)':<20s} {final_train_loss:<15.2f}")

# Performance summary
print("\n" + "=" * 80)
print("PERFORMANCE SUMMARY")
print("=" * 80)

if best_r2 >= elena_r2:
    print(f"✓ R² Score: BETTER than Elena ({best_r2:.4f} vs {elena_r2:.4f})")
else:
    diff = elena_r2 - best_r2
    print(f"✗ R² Score: {diff:.4f} points below Elena ({best_r2:.4f} vs {elena_r2:.4f})")

if final_pearson >= elena_pearson:
    print(f"✓ Pearson Correlation: BETTER than Elena ({final_pearson:.4f} vs {elena_pearson:.4f})")
else:
    diff = elena_pearson - final_pearson
    print(f"✗ Pearson Correlation: {diff:.4f} points below Elena ({final_pearson:.4f} vs {elena_pearson:.4f})")

if final_spearman >= elena_spearman:
    print(f"✓ Spearman Correlation: BETTER than Elena ({final_spearman:.4f} vs {elena_spearman:.4f})")
else:
    diff = elena_spearman - final_spearman
    print(f"✗ Spearman Correlation: {diff:.4f} points below Elena ({final_spearman:.4f} vs {elena_spearman:.4f})")

# Training statistics
print("\n" + "=" * 80)
print("TRAINING STATISTICS")
print("=" * 80)
print(f"Total Epochs: {len(r2_data)}")
print(f"Initial Learning Rate: {lr_data[lr_col].iloc[0]:.6f}")
print(f"Final Learning Rate: {lr_data[lr_col].iloc[-1]:.6f}")
print(f"Initial R²: {r2_data[r2_col].iloc[0]:.6f}")
print(f"Final R²: {r2_data[r2_col].iloc[-1]:.6f}")
print(f"R² Improvement: {r2_data[r2_col].iloc[-1] - r2_data[r2_col].iloc[0]:.6f}")

# Create comprehensive visualization
print("\n" + "=" * 80)
print("CREATING VISUALIZATION")
print("=" * 80)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Complete Training Analysis: Your Model vs Elena Benchmark', fontsize=16, fontweight='bold')

# Plot 1: R² over time
ax = axes[0, 0]
ax.plot(r2_data['epoch'], r2_data[r2_col], 'b-', linewidth=2, label='Your Model')
ax.axhline(y=elena_r2, color='r', linestyle='--', linewidth=2, label=f'Elena Benchmark ({elena_r2})')
ax.axhline(y=best_r2, color='g', linestyle=':', linewidth=1.5, label=f'Your Best ({best_r2:.4f})')
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('R² Score', fontsize=11)
ax.set_title('R² Score Progress', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Loss curves
ax = axes[0, 1]
ax.plot(train_loss_data['epoch'], train_loss_data[train_loss_col], 'b-', linewidth=2, label='Training Loss')
ax.plot(val_loss_data['epoch'], val_loss_data[val_loss_col], 'orange', linewidth=2, label='Validation Loss')
ax.axhline(y=elena_mse, color='r', linestyle='--', linewidth=2, label=f'Elena MSE ({elena_mse})')
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Loss (MSE)', fontsize=11)
ax.set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# Plot 3: Learning Rate Schedule
ax = axes[0, 2]
ax.plot(lr_data['epoch'], lr_data[lr_col], 'purple', linewidth=2)
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Learning Rate', fontsize=11)
ax.set_title('Learning Rate Schedule (Cosine Annealing)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# Plot 4: Pearson Correlation
ax = axes[1, 0]
ax.plot(pearson_data['epoch'], pearson_data[pearson_col], 'green', linewidth=2, label='Your Model')
ax.axhline(y=elena_pearson, color='r', linestyle='--', linewidth=2, label=f'Elena Benchmark ({elena_pearson})')
ax.axhline(y=final_pearson, color='g', linestyle=':', linewidth=1.5, label=f'Your Final ({final_pearson:.4f})')
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Pearson Correlation', fontsize=11)
ax.set_title('Pearson Correlation Progress', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 5: Spearman Correlation
ax = axes[1, 1]
ax.plot(spearman_data['epoch'], spearman_data[spearman_col], 'teal', linewidth=2, label='Your Model')
ax.axhline(y=elena_spearman, color='r', linestyle='--', linewidth=2, label=f'Elena Benchmark ({elena_spearman})')
ax.axhline(y=final_spearman, color='g', linestyle=':', linewidth=1.5, label=f'Your Final ({final_spearman:.4f})')
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Spearman Correlation', fontsize=11)
ax.set_title('Spearman Correlation Progress', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 6: Combined Metrics Comparison
ax = axes[1, 2]
metrics = ['R²', 'Pearson', 'Spearman']
your_values = [best_r2, final_pearson, final_spearman]
elena_values = [elena_r2, elena_pearson, elena_spearman]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, your_values, width, label='Your Model', color='steelblue')
bars2 = ax.bar(x + width/2, elena_values, width, label='Elena Benchmark', color='coral')

ax.set_ylabel('Score', fontsize=11)
ax.set_title('Final Metrics Comparison', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('complete_training_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved: complete_training_analysis.png")

# Save detailed CSV summary
summary_data = {
    'Metric': ['R² Score', 'Best R² Epoch', 'Pearson Correlation', 'Spearman Correlation', 
               'Validation Loss', 'Best Val Loss Epoch', 'Training Loss (Final)',
               'Total Epochs', 'Initial LR', 'Final LR'],
    'Your Model': [best_r2, best_r2_epoch, final_pearson, final_spearman,
                   best_val_loss, best_val_loss_epoch, final_train_loss,
                   len(r2_data), lr_data[lr_col].iloc[0], lr_data[lr_col].iloc[-1]],
    'Elena Benchmark': [elena_r2, '-', elena_pearson, elena_spearman,
                        elena_mse, '-', '-', 
                        750, 5e-4, 5e-6],
    'Difference': [best_r2-elena_r2, '-', final_pearson-elena_pearson, final_spearman-elena_spearman,
                   best_val_loss-elena_mse, '-', '-',
                   '-', '-', '-']
}

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('complete_training_summary.csv', index=False)
print("✓ Summary saved: complete_training_summary.csv")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print("\nGenerated files:")
print("  1. complete_training_analysis.png (6-panel visualization)")
print("  2. complete_training_summary.csv (detailed metrics table)")
print("\nCheck the visualization for complete training progress and comparison!")
