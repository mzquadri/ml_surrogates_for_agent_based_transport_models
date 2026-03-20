# Copilot Instructions for ML Surrogates Project

## Project Overview

This is a **Graph Neural Network (GNN) surrogate model** for predicting traffic policy effects on the Paris road network. The model learns to approximate MATSim agent-based simulations (~2 hours) in seconds, enabling rapid policy exploration.

**Key Paper:** https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5182100

## Architecture

### Directory Structure
- `scripts/gnn/models/` - GNN architectures (all inherit from `base_gnn.py`)
- `scripts/training/` - Training pipeline with `run_models.py` as entry point
- `scripts/data_preprocessing/` - MATSim simulation data → PyG format conversion
- `scripts/evaluation/` - Model testing, visualization, benchmarking notebooks
- `scripts/misc/` - Uncertainty quantification (MC Dropout), feature importance
- `data/TR-C_Benchmarks/` - Trained model checkpoints and results (8 trials)

### Core GNN Pattern
All models inherit from `BaseGNN` and must implement:
```python
class MyModel(BaseGNN):
    def define_layers(self):  # Architecture definition
    def forward(self, data):  # Forward pass with PyG Data object
```
Reference implementation: [scripts/gnn/models/point_net_transf_gat.py](scripts/gnn/models/point_net_transf_gat.py)

### Data Format
- Input: PyG `Data` objects with dual line graph representation (roads as nodes)
- Features defined in `EdgeFeatures` enum ([process_simulations_for_gnn.py](scripts/data_preprocessing/process_simulations_for_gnn.py#L53-L63)):
  - `VOL_BASE_CASE`, `CAPACITY_BASE_CASE`, `CAPACITY_REDUCTION`, `FREESPEED`, `LENGTH`
- Target (`y`): Traffic volume change from baseline

## Development Workflows

### Environment Setup
```bash
conda env create -f traffic-gnn.yml
conda activate traffic-gnn
```

### Training (Local)
```bash
cd scripts/training
python run_models.py --gnn_arch point_net_transf_gat --in_channels 5 \
    --num_epochs 500 --lr 0.003 --use_dropout True --dropout 0.3
```
Adjust `dataset_path` and `base_dir` in `run_models.py` before running.

### Training (HPC/Slurm)
Use `run_models.sbatch` with container mounts. Checkpoints save every 20 epochs.

### Experiment Tracking
All runs log to **WandB** (`project_name` arg). Local artifacts save to:
```
data/{project_name}/{unique_model_description}/
├── trained_model/model.pth
├── trained_model/checkpoints/
└── data_created_during_training/{scalers, test_dl.pt}
```

## Code Conventions

### Python Path Setup
Scripts add parent directories to `sys.path` for cross-module imports:
```python
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)
```

### Model Hyperparameters
Pass architecture-specific kwargs via JSON file:
```bash
--model_kwargs "path/to/config.json"
```
Default 5 features (ablation-tested): volume, capacity (base + reduction), speed, length.

### Loss Function
Custom `GNN_Loss` class supports weighted loss (by `vol_base_case`):
```python
loss = GNN_Loss(loss_fct='mse', num_nodes=N, device=device, weighted=True)
```

### Reproducibility
Always call `set_random_seeds(42)` before training. Seeds set for: Python, NumPy, PyTorch (CPU/GPU), cuDNN.

## Testing & Validation

### Verification Scripts
- `verify_all_8_trials.py` - Batch validation of all trained models
- Model evaluation: load `test_dl.pt`, run inference with `model.eval()`, compute metrics

### Key Metrics
```python
from gnn.help_functions import compute_r2_torch, mc_dropout_predict
from scipy.stats import spearmanr, pearsonr
```

### Uncertainty Quantification
MC Dropout implemented in `gnn/help_functions.py::mc_dropout_predict()`. Models must have `use_dropout=True`.

## Critical Files

| Purpose | File |
|---------|------|
| Training entry | [scripts/training/run_models.py](scripts/training/run_models.py) |
| Base GNN class | [scripts/gnn/models/base_gnn.py](scripts/gnn/models/base_gnn.py) |
| Best model | [scripts/gnn/models/point_net_transf_gat.py](scripts/gnn/models/point_net_transf_gat.py) |
| Data preprocessing | [scripts/data_preprocessing/process_simulations_for_gnn.py](scripts/data_preprocessing/process_simulations_for_gnn.py) |
| Loss & metrics | [scripts/gnn/help_functions.py](scripts/gnn/help_functions.py) |
| Feature engineering | [scripts/data_preprocessing/help_functions.py](scripts/data_preprocessing/help_functions.py) |
