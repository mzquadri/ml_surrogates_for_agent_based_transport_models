"""
Train a proper deep ensemble: M identical T8 models with different random seeds.

Each model uses the SAME:
  - Architecture: PointNetTransfGAT
  - Hyperparameters: dropout=0.2, lr=5e-4, batch_size=8, etc. (identical to T8)
  - Data split: 80/10/10 with shuffle_seed=42 (hardcoded in gnn_io.py)

Each model uses a DIFFERENT:
  - Random seed for weight initialization and training stochasticity

Usage:
  python run_deep_ensemble.py                    # Train all 5 members
  python run_deep_ensemble.py --members 3        # Train only first 3
  python run_deep_ensemble.py --start_from 3     # Resume from member 3 (skip 1,2)
  python run_deep_ensemble.py --seeds 42 123 456 # Custom seeds

Output:
  code/data/TR-C_Benchmarks/deep_ensemble_seed{S}/  for each seed S
"""

import os
import sys
import subprocess
import time
import argparse

# Default seeds for M=5 ensemble members
DEFAULT_SEEDS = [42, 137, 256, 389, 512]

# T8 configuration (exact same hyperparameters)
T8_CONFIG = {
    "gnn_arch": "point_net_transf_gat",
    "project_name": "TR-C_Benchmarks",
    "in_channels": "5",
    "use_all_features": "False",
    "lr": "0.0005",
    "use_dropout": "True",
    "dropout": "0.2",
    "batch_size": "8",
    "gradient_accumulation_steps": "3",
    "use_weighted_loss": "False",
    "early_stopping_patience": "25",
    "num_epochs": "1000",
    "use_gradient_clipping": "True",
}


def train_member(seed, run_models_path, dry_run=False):
    """Train a single ensemble member with the given seed."""

    model_desc = f"deep_ensemble_seed{seed}"

    cmd = [
        sys.executable,
        run_models_path,
        "--seed",
        str(seed),
        "--unique_model_description",
        model_desc,
    ]

    # Add all T8 config
    for key, value in T8_CONFIG.items():
        cmd.extend([f"--{key}", value])

    print(f"\n{'=' * 70}")
    print(f"  DEEP ENSEMBLE — Training member with seed={seed}")
    print(f"  Output dir: TR-C_Benchmarks/{model_desc}/")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'=' * 70}\n")

    if dry_run:
        print("  [DRY RUN] Skipping actual training.")
        return True

    start_time = time.time()

    result = subprocess.run(
        cmd,
        cwd=os.path.dirname(run_models_path),
    )

    elapsed = (time.time() - start_time) / 60

    if result.returncode == 0:
        print(f"\n  Seed {seed} completed successfully in {elapsed:.1f} minutes.")
        return True
    else:
        print(
            f"\n  ERROR: Seed {seed} failed after {elapsed:.1f} minutes (return code {result.returncode})"
        )
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Train deep ensemble (M independently seeded T8 models)"
    )
    parser.add_argument(
        "--members",
        type=int,
        default=5,
        help="Number of ensemble members to train (default: 5)",
    )
    parser.add_argument(
        "--start_from",
        type=int,
        default=1,
        help="Start from member N (1-indexed, for resuming)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Custom seeds (overrides --members)",
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="Print commands without training"
    )
    args = parser.parse_args()

    seeds = args.seeds if args.seeds else DEFAULT_SEEDS[: args.members]

    run_models_path = os.path.join(os.path.dirname(__file__), "run_models.py")
    if not os.path.exists(run_models_path):
        print(f"ERROR: run_models.py not found at {run_models_path}")
        sys.exit(1)

    print(f"Deep Ensemble Training Plan:")
    print(f"  Members: {len(seeds)}")
    print(f"  Seeds: {seeds}")
    print(f"  Architecture: PointNetTransfGAT (identical to T8)")
    print(f"  Starting from member: {args.start_from}")
    print()

    results = {}
    total_start = time.time()

    for i, seed in enumerate(seeds, 1):
        if i < args.start_from:
            print(f"  Skipping member {i} (seed={seed})")
            continue

        success = train_member(seed, run_models_path, dry_run=args.dry_run)
        results[seed] = success

        if not success and not args.dry_run:
            print(
                f"\n  WARNING: Member {i} (seed={seed}) failed. Continuing with next..."
            )

    total_elapsed = (time.time() - total_start) / 60

    print(f"\n{'=' * 70}")
    print(f"  DEEP ENSEMBLE TRAINING SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total time: {total_elapsed:.1f} minutes")
    for seed, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  Seed {seed}: {status}")

    failed = [s for s, ok in results.items() if not ok]
    if failed:
        print(f"\n  To retry failed members:")
        print(
            f"    python run_deep_ensemble.py --seeds {' '.join(str(s) for s in failed)}"
        )

    print(f"\n  Next step: Run evaluation")
    print(f"    python ../evaluation/evaluate_deep_ensemble.py")


if __name__ == "__main__":
    main()
