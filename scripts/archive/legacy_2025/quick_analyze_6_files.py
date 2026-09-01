"""
Quick Analysis of All 6 WandB CSV Files (No Plotting Required)
Extracts key metrics and compares with Elena's benchmarks
"""

import pandas as pd
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
    size_mb = f.stat().st_size / (1024 * 1024)
    print(f"  - {f.name} ({size_mb:.2f} MB)")

# Load each file and identify metric
dfs = {}
for file in files:
    df = pd.read_csv(file)
    # Extract metric name from column names
    cols = [col for col in df.columns if 'olive-shadow-2' in col and '__MIN' not in col and '__MAX' not in col and '_step' not in col]
    if cols:
        metric_name = cols[0].replace('olive-shadow-2 - ', '')
        dfs[metric_name] = df
        print(f"\n{metric_name:15s}: {len(df)} epochs loaded")

print("\n" + "=" * 80)
print("EXTRACTING BEST RESULTS FROM ALL METRICS")
print("=" * 80)

# Get data for each metric
r2_data = dfs.get('r^2', dfs.get('r2'))
pearson_data = dfs.get('pearson')
spearman_data = dfs.get('spearman')
train_loss_data = dfs.get('train_loss')
val_loss_data = dfs.get('val_loss')
lr_data = dfs.get('lr')

# Get column names
r2_col = [c for c in r2_data.columns if 'r^2' in c.lower() and '__MIN' not in c and '__MAX' not in c and '_step' not in c][0]
pearson_col = [c for c in pearson_data.columns if 'pearson' in c and '__MIN' not in c and '__MAX' not in c and '_step' not in c][0]
spearman_col = [c for c in spearman_data.columns if 'spearman' in c and '__MIN' not in c and '__MAX' not in c and '_step' not in c][0]
train_loss_col = [c for c in train_loss_data.columns if 'train_loss' in c and '__MIN' not in c and '__MAX' not in c and '_step' not in c][0]
val_loss_col = [c for c in val_loss_data.columns if 'val_loss' in c and '__MIN' not in c and '__MAX' not in c and '_step' not in c][0]
lr_col = [c for c in lr_data.columns if 'lr' in c and '__MIN' not in c and '__MAX' not in c and '_step' not in c][0]

# Extract metrics
best_r2 = r2_data[r2_col].max()
best_r2_epoch = int(r2_data[r2_col].idxmax())
final_r2 = r2_data[r2_col].iloc[-1]
initial_r2 = r2_data[r2_col].iloc[0]

best_val_loss = val_loss_data[val_loss_col].min()
best_val_loss_epoch = int(val_loss_data[val_loss_col].idxmin())
final_val_loss = val_loss_data[val_loss_col].iloc[-1]

final_train_loss = train_loss_data[train_loss_col].iloc[-1]
initial_train_loss = train_loss_data[train_loss_col].iloc[0]

final_pearson = pearson_data[pearson_col].iloc[-1]
initial_pearson = pearson_data[pearson_col].iloc[0]

final_spearman = spearman_data[spearman_col].iloc[-1]
initial_spearman = spearman_data[spearman_col].iloc[0]

initial_lr = lr_data[lr_col].iloc[0]
final_lr = lr_data[lr_col].iloc[-1]

total_epochs = len(r2_data)

# Elena's benchmarks from paper
elena_r2 = 0.76
elena_pearson = 0.87
elena_spearman = 0.85
elena_mse = 24.95  # Mean Squared Error
elena_mae = 2.74   # Mean Absolute Error

# Display main results
print("\n" + "=" * 80)
print("YOUR MODEL RESULTS (olive-shadow-2)")
print("=" * 80)

print(f"\n{'METRIC':<25s} {'BEST/FINAL':<15s} {'EPOCH':<10s} {'INITIAL':<15s}")
print("-" * 65)
print(f"{'R² Score':<25s} {best_r2:<15.6f} {best_r2_epoch:<10d} {initial_r2:<15.6f}")
print(f"{'R² Score (final)':<25s} {final_r2:<15.6f} {total_epochs-1:<10d}")
print(f"{'Pearson Correlation':<25s} {final_pearson:<15.6f} {total_epochs-1:<10d} {initial_pearson:<15.6f}")
print(f"{'Spearman Correlation':<25s} {final_spearman:<15.6f} {total_epochs-1:<10d} {initial_spearman:<15.6f}")
print(f"{'Validation Loss (best)':<25s} {best_val_loss:<15.4f} {best_val_loss_epoch:<10d}")
print(f"{'Validation Loss (final)':<25s} {final_val_loss:<15.4f} {total_epochs-1:<10d}")
print(f"{'Training Loss (final)':<25s} {final_train_loss:<15.4f} {total_epochs-1:<10d} {initial_train_loss:<15.4f}")
print(f"{'Learning Rate':<25s} {final_lr:<15.8f} {total_epochs-1:<10d} {initial_lr:<15.8f}")

print("\n" + "=" * 80)
print("COMPARISON WITH ELENA BOREALE'S MODEL")
print("=" * 80)

print(f"\n{'METRIC':<25s} {'YOUR MODEL':<15s} {'ELENA MODEL':<15s} {'DIFFERENCE':<15s} {'STATUS':<10s}")
print("-" * 85)

# R² comparison
diff_r2 = best_r2 - elena_r2
status_r2 = "✓ BETTER" if diff_r2 >= 0 else "✗ LOWER"
print(f"{'R² Score':<25s} {best_r2:<15.6f} {elena_r2:<15.6f} {diff_r2:+15.6f} {status_r2:<10s}")

# Pearson comparison
diff_pearson = final_pearson - elena_pearson
status_pearson = "✓ BETTER" if diff_pearson >= 0 else "✗ LOWER"
print(f"{'Pearson Correlation':<25s} {final_pearson:<15.6f} {elena_pearson:<15.6f} {diff_pearson:+15.6f} {status_pearson:<10s}")

# Spearman comparison
diff_spearman = final_spearman - elena_spearman
status_spearman = "✓ BETTER" if diff_spearman >= 0 else "✗ LOWER"
print(f"{'Spearman Correlation':<25s} {final_spearman:<15.6f} {elena_spearman:<15.6f} {diff_spearman:+15.6f} {status_spearman:<10s}")

# Loss comparison
diff_loss = best_val_loss - elena_mse
status_loss = "✓ BETTER" if diff_loss <= 0 else "✗ HIGHER"
print(f"{'Validation Loss (MSE)':<25s} {best_val_loss:<15.4f} {elena_mse:<15.4f} {diff_loss:+15.4f} {status_loss:<10s}")

print("\n" + "=" * 80)
print("TRAINING PROGRESS SUMMARY")
print("=" * 80)

print(f"\nTotal Training Epochs: {total_epochs}")
print(f"Learning Rate Schedule: {initial_lr:.8f} → {final_lr:.8f} (Cosine Annealing)")
print(f"\nR² Improvement:")
print(f"  Epoch 0:   {initial_r2:.6f}")
print(f"  Best:      {best_r2:.6f} (epoch {best_r2_epoch})")
print(f"  Final:     {final_r2:.6f} (epoch {total_epochs-1})")
print(f"  Total Gain: {final_r2 - initial_r2:.6f}")

print(f"\nLoss Reduction:")
print(f"  Train Loss:  {initial_train_loss:.4f} → {final_train_loss:.4f} ({((final_train_loss-initial_train_loss)/initial_train_loss*100):.2f}%)")
print(f"  Val Loss:    {val_loss_data[val_loss_col].iloc[0]:.4f} → {best_val_loss:.4f} (best)")

print(f"\nCorrelation Evolution:")
print(f"  Pearson:   {initial_pearson:.6f} → {final_pearson:.6f}")
print(f"  Spearman:  {initial_spearman:.6f} → {final_spearman:.6f}")

# Overall performance assessment
print("\n" + "=" * 80)
print("OVERALL PERFORMANCE ASSESSMENT")
print("=" * 80)

scores = []
if diff_r2 >= -0.05:  # Within 5% is acceptable
    scores.append(1)
    print(f"\n✓ R² Score: {best_r2:.4f} - {'EXCELLENT' if diff_r2 >= 0 else 'ACCEPTABLE'}")
else:
    scores.append(0)
    print(f"\n✗ R² Score: {best_r2:.4f} - NEEDS IMPROVEMENT ({abs(diff_r2):.4f} below Elena)")

if diff_pearson >= -0.05:
    scores.append(1)
    print(f"✓ Pearson: {final_pearson:.4f} - {'EXCELLENT' if diff_pearson >= 0 else 'ACCEPTABLE'}")
else:
    scores.append(0)
    print(f"✗ Pearson: {final_pearson:.4f} - NEEDS IMPROVEMENT ({abs(diff_pearson):.4f} below Elena)")

if diff_spearman >= -0.05:
    scores.append(1)
    print(f"✓ Spearman: {final_spearman:.4f} - {'EXCELLENT' if diff_spearman >= 0 else 'ACCEPTABLE'}")
else:
    scores.append(0)
    print(f"✗ Spearman: {final_spearman:.4f} - NEEDS IMPROVEMENT ({abs(diff_spearman):.4f} below Elena)")

overall_score = sum(scores) / len(scores) * 100
print(f"\n{'=' * 80}")
print(f"OVERALL SCORE: {overall_score:.0f}% ({sum(scores)}/{len(scores)} metrics meet benchmark)")
print(f"{'=' * 80}")

# Save detailed report
report_file = "training_results_report.txt"
with open(report_file, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("COMPLETE TRAINING RESULTS - olive-shadow-2\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("YOUR MODEL RESULTS:\n")
    f.write(f"  R² Score (best):          {best_r2:.6f} at epoch {best_r2_epoch}\n")
    f.write(f"  R² Score (final):         {final_r2:.6f}\n")
    f.write(f"  Pearson Correlation:      {final_pearson:.6f}\n")
    f.write(f"  Spearman Correlation:     {final_spearman:.6f}\n")
    f.write(f"  Validation Loss (best):   {best_val_loss:.4f} at epoch {best_val_loss_epoch}\n")
    f.write(f"  Training Loss (final):    {final_train_loss:.4f}\n\n")
    
    f.write("ELENA'S BENCHMARK:\n")
    f.write(f"  R² Score:                 {elena_r2:.6f}\n")
    f.write(f"  Pearson Correlation:      {elena_pearson:.6f}\n")
    f.write(f"  Spearman Correlation:     {elena_spearman:.6f}\n")
    f.write(f"  MSE:                      {elena_mse:.4f}\n\n")
    
    f.write("DIFFERENCES:\n")
    f.write(f"  R² Score:                 {diff_r2:+.6f} {status_r2}\n")
    f.write(f"  Pearson:                  {diff_pearson:+.6f} {status_pearson}\n")
    f.write(f"  Spearman:                 {diff_spearman:+.6f} {status_spearman}\n")
    f.write(f"  Validation Loss:          {diff_loss:+.4f} {status_loss}\n\n")
    
    f.write(f"Overall Score: {overall_score:.0f}% ({sum(scores)}/{len(scores)} metrics meet benchmark)\n")

print(f"\n✓ Detailed report saved: {report_file}")

# Save CSV summary
summary_data = {
    'Metric': ['R² Score (best)', 'R² Score (final)', 'Pearson', 'Spearman', 
               'Val Loss (best)', 'Train Loss (final)', 'Total Epochs', 'Best R² Epoch', 'Best Val Loss Epoch'],
    'Your Model': [best_r2, final_r2, final_pearson, final_spearman,
                   best_val_loss, final_train_loss, total_epochs, best_r2_epoch, best_val_loss_epoch],
    'Elena': [elena_r2, '-', elena_pearson, elena_spearman,
              elena_mse, '-', 750, '-', '-'],
    'Difference': [diff_r2, '-', diff_pearson, diff_spearman,
                   diff_loss, '-', total_epochs-750, '-', '-']
}

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('training_summary.csv', index=False)
print(f"✓ Summary table saved: training_summary.csv")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print("\nGenerated files:")
print("  1. training_results_report.txt (detailed text report)")
print("  2. training_summary.csv (metrics comparison table)")
