"""
Run GNN model training with configurable architecture and hyperparameters.

'dataset_path' and 'base_dir' need to be adjusted to the correct paths.
All the other parameters can be passed as command line arguments. Run `python run_models.py --help` to see the list of available arguments.

Example usage with default architecture, dropout, and most significant features found using ablation tests:
`python ml_surrogates_for_agent_based_transport_models\scripts\training\run_models.py --in_channels 5 --use_all_features False --num_epochs 500 --lr 0.003 --early_stopping_patience 25 --use_dropout True --dropout 0.3`
"""
import os
import sys
import json
import argparse
from datetime import datetime, timezone

import torch
# torch.serialization.add_safe_globals([torch_geometric.data.data.DataEdgeAttr])

# Add the 'scripts' directory to Python Path
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from training.help_functions import *
from gnn.help_functions import GNN_Loss, compute_baseline_of_mean_target, compute_baseline_of_no_policies

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Please adjust as needed
dataset_path = os.path.join(project_root, 'data', 'train_data', 'dist_not_connected_10k_1pct')
base_dir = os.path.join(project_root, 'data')


def _load_run_counters(counter_file_path: str) -> dict:
    """Load the persistent run counters JSON file, or return an empty dict if missing/invalid."""
    if not os.path.exists(counter_file_path):
        return {}
    try:
        with open(counter_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        # Corrupt or unreadable file; start fresh
        return {}


def _save_run_counters(counter_file_path: str, counters: dict) -> None:
    """Persist the run counters JSON file."""
    os.makedirs(os.path.dirname(counter_file_path), exist_ok=True)
    with open(counter_file_path, 'w', encoding='utf-8') as f:
        json.dump(counters, f, indent=2)


def get_next_sequential_description(base_storage_dir: str, base_name: str) -> str:
    """
    Return a unique description with an auto-incremented suffix for each script execution.
    Example: base_name='TR-C_Benchmarks' -> 'TR-C_Benchmarks_1', then '_2', etc.

    The counter is persisted in base_storage_dir/.run_counters.json.
    """
    os.makedirs(base_storage_dir, exist_ok=True)
    counter_file = os.path.join(base_storage_dir, '.run_counters.json')
    counters = _load_run_counters(counter_file)

    last_val = counters.get(base_name, 0)
    next_val = last_val + 1
    counters[base_name] = next_val
    _save_run_counters(counter_file, counters)

    return f"{base_name}_{next_val}"


def annotate_plot_with_timestamp(ax=None,
                                 when=None,
                                 loc='lower right',
                                 fmt='%Y-%m-%d %H:%M:%S %Z',
                                 fontsize=8,
                                 alpha=0.7,
                                 pad=0.02,
                                 use_utc=True):
    """
    Annotate a Matplotlib plot with the current date/time.

    Parameters:
      - ax: Matplotlib Axes. If None, uses current axes.
      - when: datetime to render. If None, uses now (UTC if use_utc else local).
      - loc: 'lower right' | 'lower left' | 'upper right' | 'upper left'
      - fmt: datetime strftime format
      - fontsize: font size for the timestamp
      - alpha: text transparency
      - pad: padding from the axes edge (axes fraction)
      - use_utc: whether to use UTC time (True) or local time (False)
    """
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()

    if when is None:
        when = datetime.now(timezone.utc) if use_utc else datetime.now()

    loc = (loc or '').lower().strip()
    x = 1 - pad if 'right' in loc else pad
    y = 1 - pad if 'upper' in loc else pad
    ha = 'right' if 'right' in loc else 'left'
    va = 'top' if 'upper' in loc else 'bottom'

    # Ensure timezone displays if using UTC
    if use_utc and when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)

    ax.text(
        x, y,
        when.strftime(fmt),
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=fontsize,
        alpha=alpha,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.5, boxstyle='round,pad=0.2')
    )


def main():
    try:
        datalist = []
        batch_num = 1
        while True:  # Change this to "and batch_num < 10" for a faster run
            print(f"Processing batch number: {batch_num}")
            batch_file = os.path.join(dataset_path, f'datalist_batch_{batch_num}.pt')

            print(batch_file)

            if not os.path.exists(batch_file):
                break
            batch_data = torch.load(batch_file, map_location='cpu', weights_only=False)
            if isinstance(batch_data, list):
                datalist.extend(batch_data)
            batch_num += 1
        print(f"Loaded {len(datalist)} items into datalist")

        # Temp fix, rerun data_preprocessing to solve.
        for i, data in enumerate(datalist):
            data.num_nodes = data.x.shape[0]

    except Exception as e:
        print(f"An error occurred: {str(e)}")

    parser = argparse.ArgumentParser(description="Run GNN model training with configurable parameters.")
    parser.add_argument("--gnn_arch", type=str, default="trans_conv",
                        help="The GNN architecture to use.",
                        choices=["point_net_transf_gat", "gat", "gcn", "gcn2", "trans_conv", "pnc", "fc_nn", "graphSAGE", "eign", "xgboost", "trans_encoder"])

    # CHANGED: Renamed from --project_name to --unique_model_description with dynamic default sequencing.
    # If the user does not provide a value, the script will generate TR-C_Benchmarks_<N> automatically.
    parser.add_argument("--unique_model_description", type=str, default=None,
                        help="Unique descriptor for this series/project (was --project_name). "
                             "If omitted, a sequential name like 'TR-C_Benchmarks_1' will be generated.")

    # CHANGED: The previous --unique_model_description (per-run description) is now --run_name to avoid conflict.
    parser.add_argument("--run_name", type=str, default="trans_conv_5_features",
                        help="A unique description for this specific run within the project/series.")

    parser.add_argument("--in_channels", type=int, default=5, help="The number of input channels.")
    parser.add_argument("--use_all_features", type=str_to_bool, default=False, help="Whether to use all features.")
    parser.add_argument("--out_channels", type=int, default=1, help="The number of output channels.")
    parser.add_argument("--model_kwargs", type=str, default=None,
                        help='Additional model parameters (as defined in the class) in JSON format (path to the file). '
                             'If not provided, defaults params will be used.')
    parser.add_argument("--loss_fct", type=str, default="mse", help="The loss function to use. Supported: mse, l1.")
    parser.add_argument("--use_weighted_loss", type=str_to_bool, default=False, help="Whether to use weighted loss (based on vol_base_case) or not.")
    parser.add_argument("--predict_mode_stats", type=str_to_bool, default=False, help="Whether to predict mode stats or not.")
    parser.add_argument("--use_bootstrapping", type=str_to_bool, default=False, help="Whether to use bootstrapping for train-validation split.")
    parser.add_argument("--num_epochs", type=int, default=1000, help="Number of epochs to train for.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=0.001, help="The learning rate for the model.")
    parser.add_argument("--early_stopping_patience", type=int, default=25, help="The early stopping patience.")
    parser.add_argument("--use_dropout", type=str_to_bool, default=False, help="Whether to use dropout.")
    parser.add_argument("--dropout", type=float, default=0.3, help="The dropout rate.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=3, help="After how many steps the gradient should be updated.")
    parser.add_argument("--use_gradient_clipping", type=str_to_bool, default=True, help="Whether to use gradient clipping.")
    parser.add_argument("--device_nr", type=int, default=0, help="The device number (0 or 1 for Retina Roaster's two GPUs).")
    parser.add_argument("--continue_training", type=str_to_bool, default=False, help="Whether to continue training from a checkpoint.")
    parser.add_argument("--base_checkpoint_path", type=str, default=None, help="Path to the checkpoint to continue training from.")

    args = vars(parser.parse_args())
    set_random_seeds()

    # Compute dynamic default for unique_model_description if not provided
    DEFAULT_BASE_NAME = "TR-C_Benchmarks"
    if not args.get("unique_model_description"):
        # No value provided: generate sequential default like TR-C_Benchmarks_1, _2, ...
        args["unique_model_description"] = get_next_sequential_description(base_storage_dir=base_dir,
                                                                           base_name=DEFAULT_BASE_NAME)
    elif args["unique_model_description"] == DEFAULT_BASE_NAME:
        # User explicitly set to the base default; still apply sequencing to enable run tracking
        args["unique_model_description"] = get_next_sequential_description(base_storage_dir=base_dir,
                                                                           base_name=DEFAULT_BASE_NAME)
    # Backward-compatibility: some downstream helpers (e.g., setup_wandb) may expect 'project_name'
    args["project_name"] = args["unique_model_description"]

    try:
        gpus = get_available_gpus()
        best_gpu = select_best_gpu(gpus)
        set_cuda_visible_device(best_gpu)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create directory for the run
        # Structure: data/<unique_model_description>/<run_name>/
        run_root_dir = os.path.join(base_dir, args['unique_model_description'])
        unique_run_dir = os.path.join(run_root_dir, args['run_name'])
        os.makedirs(unique_run_dir, exist_ok=True)

        # Update paths to reflect the new structure
        model_save_path, path_to_save_dataloader = get_paths(
            base_dir=run_root_dir,
            unique_model_description=args['run_name'],
            model_save_path='trained_model/model.pth'
        )

        train_dl, valid_dl, scalers_train, scalers_validation = prepare_data_with_graph_features(
            datalist=datalist,
            batch_size=args['batch_size'],
            path_to_save_dataloader=path_to_save_dataloader,
            use_all_features=args['use_all_features'],
            use_bootstrapping=args['use_bootstrapping'],
            is_eign=(args['gnn_arch'] == "eign")
        )

        # Create WandB config
        config = setup_wandb(args)

        if args["model_kwargs"] is not None:
            with open(args["model_kwargs'], 'r', encoding='utf-8') as f:
                model_kwargs = json.load(f)
        else:
            model_kwargs = {}

        # Create model instance
        gnn_instance = create_gnn_model(
            gnn_arch=config.gnn_arch,
            config=config,
            model_kwargs=model_kwargs,
            device=device
        )

        gnn_instance = gnn_instance.to(device)
        loss_fct = GNN_Loss(config.loss_fct, datalist[0].x.shape[0], device, config.use_weighted_loss)

        early_stopping = EarlyStopping(patience=config.early_stopping_patience, verbose=True)
        best_val_loss, best_epoch = gnn_instance.train_model(
            config=config,
            loss_fct=loss_fct,
            optimizer=torch.optim.AdamW(gnn_instance.parameters(), lr=config.lr, weight_decay=1e-4) if config.gnn_arch != "xgboost" else None,
            train_dl=train_dl,
            valid_dl=valid_dl,
            device=device,
            early_stopping=early_stopping,
            model_save_path=model_save_path,
            scalers_train=scalers_train,
            scalers_validation=scalers_validation
        )

        print(f'Best model saved to {model_save_path} with validation loss: {best_val_loss} at epoch {best_epoch}')

        # Example (optional) usage for graph timestamping:
        # After you generate a Matplotlib figure elsewhere in your pipeline, call:
        #   annotate_plot_with_timestamp(ax=plt.gca(), use_utc=True)
        # Then save/show the figure as usual.

    except Exception as e:
        print(f"Error: {e}")
        print("Falling back to CPU.")
        os.environ['CUDA_VISIBLE_DEVICES'] = ""


if __name__ == '__main__':
    main()