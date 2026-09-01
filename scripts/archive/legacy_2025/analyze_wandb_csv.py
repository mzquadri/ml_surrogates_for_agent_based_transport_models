"""
ANALYZE WANDB CSV FILES - COMPLETE TRAINING RESULTS
Load downloaded CSV files and extract all metrics
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("="*80)
print("WANDB CSV ANALYSIS - TRAINING RESULTS")
print("="*80)

# ============================================================================
# STEP 1: LOAD CSV FILE
# ============================================================================

print("\n📂 Loading CSV file...")

# Update this path to your CSV file location
csv_file = "olive-shadow-2.csv"  # Or full path if needed

try:
    df = pd.read_csv(csv_file)
    print(f"✓ Loaded: {csv_file}")
    print(f"  Total rows: {len(df)}")
    print(f"  Columns: {len(df.columns)}")
except FileNotFoundError:
    print(f"❌ File not found: {csv_file}")
    print("\nPlease update 'csv_file' variable with correct path")
    print("Example paths:")
    print("  - Colab: '/content/drive/MyDrive/olive-shadow-2.csv'")
    print("  - Local: 'downloads/olive-shadow-2.csv'")
    exit()

# ============================================================================
# STEP 2: EXPLORE COLUMNS
# ============================================================================

print("\n📊 Available Metrics:")
print("="*80)

print("\nAll columns in CSV:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col}")

# ============================================================================
# STEP 3: KEY METRICS SUMMARY
# ============================================================================

print("\n🎯 KEY TRAINING METRICS")
print("="*80)

# Find relevant columns (WandB uses different naming)
metric_cols = {
    'epoch': ['epoch', 'Step', '_step'],
    'train_loss': ['train_loss', 'train/loss'],
    'val_loss': ['val_loss', 'validation_loss', 'val/loss'],
    'r2': ['r^2', 'r2', 'r_squared', 'val_r2'],
    'pearson': ['pearson', 'val_pearson'],
    'spearman': ['spearman', 'val_spearman'],
    'lr': ['lr', 'learning_rate']
}

# Find actual column names
actual_cols = {}
for metric, possible_names in metric_cols.items():
    for name in possible_names:
        if name in df.columns:
            actual_cols[metric] = name
            break

print("\nFound metrics:")
for metric, col in actual_cols.items():
    if col in df.columns:
        values = df[col].dropna()
        if len(values) > 0:
            print(f"  ✓ {metric:12s}: {col}")
            print(f"      Best: {values.max():.4f}, Latest: {values.iloc[-1]:.4f}")

# ============================================================================
# STEP 4: BEST RESULTS
# ============================================================================

print("\n🏆 BEST TRAINING RESULTS")
print("="*80)

if 'r2' in actual_cols:
    r2_col = actual_cols['r2']
    best_r2_idx = df[r2_col].idxmax()
    best_r2 = df.loc[best_r2_idx, r2_col]
    best_r2_epoch = df.loc[best_r2_idx, actual_cols.get('epoch', 0)] if 'epoch' in actual_cols else best_r2_idx
    
    print(f"\n✅ Best R² Score: {best_r2:.4f}")
    print(f"   At epoch: {int(best_r2_epoch)}")

if 'val_loss' in actual_cols:
    loss_col = actual_cols['val_loss']
    best_loss_idx = df[loss_col].idxmin()
    best_loss = df.loc[best_loss_idx, loss_col]
    best_loss_epoch = df.loc[best_loss_idx, actual_cols.get('epoch', 0)] if 'epoch' in actual_cols else best_loss_idx
    
    print(f"\n✅ Best Validation Loss: {best_loss:.4f}")
    print(f"   At epoch: {int(best_loss_epoch)}")

if 'pearson' in actual_cols:
    pearson = df[actual_cols['pearson']].dropna().iloc[-1]
    print(f"\n✅ Final Pearson Correlation: {pearson:.4f}")

if 'spearman' in actual_cols:
    spearman = df[actual_cols['spearman']].dropna().iloc[-1]
    print(f"\n✅ Final Spearman Correlation: {spearman:.4f}")

# ============================================================================
# STEP 5: ELENA COMPARISON
# ============================================================================

print("\n📈 COMPARISON WITH ELENA'S PAPER")
print("="*80)

elena_results = {
    'R² Score': 0.76,
    'MSE': 24.95,
    'MAE': 2.74,
    'Pearson': 0.87,
    'Spearman': 0.85
}

print("\nElena's Results vs Your Results:")
print("-" * 60)

if 'r2' in actual_cols:
    your_r2 = df[actual_cols['r2']].max()
    diff = your_r2 - elena_results['R² Score']
    status = "✅ Better" if diff > 0 else "⚠ Lower" if diff < -0.05 else "✓ Similar"
    print(f"  R² Score:     Elena: {elena_results['R² Score']:.4f}  |  You: {your_r2:.4f}  |  {status}")

if 'pearson' in actual_cols:
    your_pearson = df[actual_cols['pearson']].dropna().iloc[-1]
    diff = your_pearson - elena_results['Pearson']
    status = "✅ Better" if diff > 0 else "⚠ Lower" if diff < -0.05 else "✓ Similar"
    print(f"  Pearson:      Elena: {elena_results['Pearson']:.4f}  |  You: {your_pearson:.4f}  |  {status}")

if 'spearman' in actual_cols:
    your_spearman = df[actual_cols['spearman']].dropna().iloc[-1]
    diff = your_spearman - elena_results['Spearman']
    status = "✅ Better" if diff > 0 else "⚠ Lower" if diff < -0.05 else "✓ Similar"
    print(f"  Spearman:     Elena: {elena_results['Spearman']:.4f}  |  You: {your_spearman:.4f}  |  {status}")

# ============================================================================
# STEP 6: VISUALIZATIONS
# ============================================================================

print("\n📊 GENERATING PLOTS...")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Training Progress - Elena Replica Model', fontsize=16, fontweight='bold')

# Plot 1: R² over epochs
if 'r2' in actual_cols and 'epoch' in actual_cols:
    ax = axes[0, 0]
    epochs = df[actual_cols['epoch']]
    r2_values = df[actual_cols['r2']]
    
    ax.plot(epochs, r2_values, 'b-', linewidth=2, label='Your Model')
    ax.axhline(y=0.76, color='r', linestyle='--', linewidth=2, label='Elena (0.76)')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('R² Score Progress', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim([0, 1])

# Plot 2: Loss curves
if 'train_loss' in actual_cols and 'val_loss' in actual_cols and 'epoch' in actual_cols:
    ax = axes[0, 1]
    epochs = df[actual_cols['epoch']]
    
    ax.plot(epochs, df[actual_cols['train_loss']], 'b-', linewidth=2, label='Training Loss')
    ax.plot(epochs, df[actual_cols['val_loss']], 'r-', linewidth=2, label='Validation Loss')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

# Plot 3: Correlation metrics
if 'pearson' in actual_cols and 'spearman' in actual_cols and 'epoch' in actual_cols:
    ax = axes[1, 0]
    epochs = df[actual_cols['epoch']]
    
    ax.plot(epochs, df[actual_cols['pearson']], 'g-', linewidth=2, label='Pearson')
    ax.plot(epochs, df[actual_cols['spearman']], 'orange', linewidth=2, label='Spearman')
    ax.axhline(y=0.87, color='g', linestyle='--', alpha=0.5, label='Elena Pearson')
    ax.axhline(y=0.85, color='orange', linestyle='--', alpha=0.5, label='Elena Spearman')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Correlation', fontsize=12)
    ax.set_title('Correlation Metrics', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

# Plot 4: Learning rate
if 'lr' in actual_cols and 'epoch' in actual_cols:
    ax = axes[1, 1]
    epochs = df[actual_cols['epoch']]
    
    ax.plot(epochs, df[actual_cols['lr']], 'purple', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

plt.tight_layout()
plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
print("\n✓ Plots saved: training_results.png")
plt.show()

# ============================================================================
# STEP 7: DETAILED STATISTICS
# ============================================================================

print("\n📋 DETAILED TRAINING STATISTICS")
print("="*80)

print(f"\nTotal epochs trained: {len(df)}")

if 'r2' in actual_cols:
    r2_values = df[actual_cols['r2']].dropna()
    print(f"\nR² Statistics:")
    print(f"  Mean:   {r2_values.mean():.4f}")
    print(f"  Median: {r2_values.median():.4f}")
    print(f"  Max:    {r2_values.max():.4f}")
    print(f"  Min:    {r2_values.min():.4f}")
    print(f"  Std:    {r2_values.std():.4f}")

if 'val_loss' in actual_cols:
    loss_values = df[actual_cols['val_loss']].dropna()
    print(f"\nValidation Loss Statistics:")
    print(f"  Mean:   {loss_values.mean():.4f}")
    print(f"  Median: {loss_values.median():.4f}")
    print(f"  Min:    {loss_values.min():.4f}")
    print(f"  Max:    {loss_values.max():.4f}")
    print(f"  Std:    {loss_values.std():.4f}")

# ============================================================================
# STEP 8: EXPORT SUMMARY
# ============================================================================

print("\n💾 EXPORTING SUMMARY...")

summary = {
    'Total Epochs': len(df),
    'Best R²': df[actual_cols['r2']].max() if 'r2' in actual_cols else None,
    'Best Val Loss': df[actual_cols['val_loss']].min() if 'val_loss' in actual_cols else None,
    'Final Pearson': df[actual_cols['pearson']].dropna().iloc[-1] if 'pearson' in actual_cols else None,
    'Final Spearman': df[actual_cols['spearman']].dropna().iloc[-1] if 'spearman' in actual_cols else None,
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv('training_summary.csv', index=False)
print("✓ Summary saved: training_summary.csv")

# ============================================================================
# FINAL REPORT
# ============================================================================

print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE")
print("="*80)

print(f"""
Results Summary:
  📊 CSV file analyzed: {csv_file}
  📈 Plots generated: training_results.png
  💾 Summary exported: training_summary.csv

Your Model Performance:
  R² Score: {df[actual_cols['r2']].max():.4f} (Elena: 0.76)
  Val Loss: {df[actual_cols['val_loss']].min():.4f} (Elena: ~24.95)

Next Steps:
  1. Check training_results.png for visual analysis
  2. Review training_summary.csv for key metrics
  3. Compare with Elena's benchmarks above
""")

print("="*80)
