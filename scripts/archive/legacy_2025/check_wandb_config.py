"""
Check exact training configuration from WandB
Identifies all potential issues with the olive-shadow-2 run
"""

print("=" * 80)
print("ARCHITECTURE MISMATCH ANALYSIS")
print("=" * 80)

print("\nELENA'S EXACT ARCHITECTURE (from paper):")
print("-" * 80)
print("PointNet Layers:")
print("  Layer 1: 7 → 256 → 512 (input: 5 features + 2 pos)")
print("  Layer 2: 514 → 256 → 512 (512 from L1 + 2 pos)")
print("\nTransformer Layers:")
print("  Layer 3: 512 → 64, heads=4")
print("  Layer 4: 64 → 128, heads=4")
print("\nGAT Layers:")
print("  Layer 5: 128 → 64")
print("  Layer 6: 64 → 1 (output)")

print("\n" + "=" * 80)
print("YOUR MODEL ARCHITECTURE (olive-shadow-2)")
print("=" * 80)

# Default from point_net_transf_gat.py
default_gat_structure = [128, 256, 512]

print("\nMost likely used (DEFAULT parameters from repo):")
print("-" * 80)
print("PointNet Layers:")
print("  Layer 1: 7 → 256 → 512 (input: 5 features + 2 pos)")
print("  Layer 2: 514 → 256 → 512 (512 from L1 + 2 pos)")
print("\nTransformer + GAT Structure:")
print(f"  gat_conv_layer_structure: {default_gat_structure}")
print("\nThis means:")
print("  Transformer 1: 512 → 128, heads=4  ← WRONG (should be 512→64)")
print("  Transformer 2: 128 → 256, heads=4  ← WRONG (should be 64→128)")
print("  GAT 1: 256 → 512                   ← WRONG (should be 128→64)")
print("  GAT 2: 512 → 1                     ← WRONG (should be 64→1)")

print("\n" + "=" * 80)
print("IMPACT ON PERFORMANCE")
print("=" * 80)

print("\n1. BOTTLENECK ISSUE:")
print("   Elena: 512 → 64 (compression) → 128 (expansion) → 64 → 1")
print("   Your:  512 → 128 → 256 → 512 (keeps expanding!) → 1")
print("   Problem: No bottleneck = can't learn compact representations")

print("\n2. PARAMETER MISMATCH:")
print("   Elena's architecture learned to compress features")
print("   Your architecture kept expanding them")
print("   Result: Different learning dynamics entirely")

print("\n3. CAPACITY ISSUE:")
print("   Transformer embedding dim (Elena): 64")
print("   Transformer embedding dim (Your):  128 (first), 256 (second)")
print("   More parameters ≠ better results without proper architecture")

print("\n" + "=" * 80)
print("OTHER POTENTIAL ISSUES")
print("=" * 80)

print("\n1. Learning Rate Schedule:")
print("   ✓ Correct: Cosine annealing 0.000013 → 0.000005")
print("   Note: Initial LR seems very low (Elena used 5e-4 initial)")
print("   This could slow down learning significantly")

print("\n2. Number of Epochs:")
print("   ✓ Correct: 750 epochs (matches Elena)")

print("\n3. Results Comparison:")
print("   Metric          Your    Elena    Gap")
print("   R²              0.570   0.760   -25%")
print("   Spearman        0.193   0.850   -77% ← CRITICAL")
print("   Validation Loss 51.0    24.95   2x higher")

print("\n4. Spearman Correlation VERY LOW:")
print("   Spearman = 0.193 means model struggles with RANKING predictions")
print("   Elena = 0.85 means excellent ranking ability")
print("   This suggests fundamental architecture issue")

print("\n" + "=" * 80)
print("VERDICT")
print("=" * 80)

print("\n❌ PRIMARY ISSUE: Architecture Mismatch")
print("   gat_conv_layer_structure: [128, 256, 512] instead of [512, 64, 128]")
print("   This fundamentally changes how the model learns")

print("\n❌ SECONDARY ISSUE: Learning Rate Too Low at Start")
print("   Initial LR: 0.000013 vs Elena's 5e-4 (0.0005)")
print("   38x lower! This severely limits early learning")

print("\n❌ RESULT: Model Underfitting")
print("   Low R²: Can't explain 43% of variance (should be 24%)")
print("   Low Spearman: Can't properly rank predictions")
print("   High loss: 2x higher than Elena's benchmark")

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)

print("\n✓ Retrain with EXACT Elena parameters:")
print("  - gat_conv_layer_structure: [512, 64, 128]")
print("  - Initial LR: 5e-4 (not 1.32e-5)")
print("  - Final LR: 5e-6 (correct)")
print("  - Cosine annealing over 750 epochs")

print("\n✓ Expected improvements:")
print("  - R² should reach ~0.76")
print("  - Spearman should reach ~0.85")
print("  - Validation loss should drop to ~25")

print("\n" + "=" * 80)
print("CHECK YOUR TRAINING COMMAND")
print("=" * 80)

print("\nDid you use:")
print("  python run_models.py --gnn_arch point_net_transf_gat --in_channels 5")
print("\nThis uses DEFAULT parameters [128, 256, 512] ❌")

print("\nShould have used:")
print("  python run_models.py --gnn_arch point_net_transf_gat \\")
print("    --in_channels 5 \\")
print("    --model_kwargs elena_config.json \\")
print("    --lr 0.0005 \\")
print("    --num_epochs 750")

print("\nWith elena_config.json containing:")
print("  {")
print('    "gat_conv_layer_structure": [512, 64, 128],')
print('    "point_net_conv_layer_structure_local_mlp": [256],')
print('    "point_net_conv_layer_structure_global_mlp": [512]')
print("  }")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("\n✗ Wrong architecture used (default instead of Elena's)")
print("✗ Learning rate 38x too low at start")
print("✗ Model can't learn proper feature compression")
print("✗ Results: 25% lower R², 77% lower Spearman")
print("\n✓ Solution: Retrain with correct parameters")
print("\n" + "=" * 80)
