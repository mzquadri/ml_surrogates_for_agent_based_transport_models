# Local Development and Career Workflow

This setup keeps the reproducibility environment stable while adding practical tools for data, ML, testing, and portfolio work.

## Laptop profile

- Intel i7-1165G7, 4 cores / 8 threads
- 8 GB RAM
- NVIDIA MX330 with 2 GB VRAM; the current `thesis-env` uses CPU PyTorch
- Suitable for inference, UQ analysis, dashboards, classical ML, and small deep-learning experiments
- Not suitable for full 1,000-simulation GNN retraining or several local servers at once

## Thesis workflow

The primary repository's ignored `data/` path is a Windows junction to the canonical local artifact store in `ml_surrogates_thesis_final/code/data`. It does not duplicate the multi-gigabyte files and is not committed.

```powershell
conda activate thesis-env
python scripts/analysis/generate_thesis_intelligence.py --include-local-graphs
streamlit run thesis_dashboard/app.py
python -m pytest -p no:cacheprovider tests
ruff check thesis_dashboard scripts/analysis scripts/check_repository.py tests
pyright
python scripts/check_repository.py
```

The generation pipeline analyzes the full T7/T8 and Deep Ensemble arrays, validates schemas, recomputes model/UQ metrics, distinguishes calibration protocols, and writes only safe aggregate outputs. The Streamlit app reads that bundle and uses a deterministic 12,000-row sample only for optional scatter rendering; samples are never included in downloads.

Replay inference is locally feasible but was not required for the aggregate audit. New policy scenarios are not possible because raw MATSim scenario outputs are absent. Never load untrusted `.pt`, `.pth`, or `.pkl` files because these formats can execute pickle payloads, and never commit or upload the ignored `data/` junction.

## Isolated global tools

```powershell
dvc --version
mlflow --version
pre-commit --version
ast-grep --version
```

- DVC: version large private data and models. Do not run `dvc push` until a private, approved remote is configured.
- MLflow: compare future experiments locally. Start only when needed with `mlflow ui --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db`.
- pre-commit: run project checks before commits.
- ast-grep: learn deterministic AST-aware code search and refactoring.

These tools are installed in isolated environments and do not alter Torch or thesis dependencies.
DVC analytics and MLflow telemetry are disabled in the user-level configuration. MLflow is bound to localhost in the example command.

## Weekly practice loop

1. Pick one small question and define an output metric before coding.
2. Create a separate Git repository and environment; never develop new projects in Conda `base`.
3. Build a thin vertical slice: load data, validate it, create a baseline, test one function, and document one result.
4. Run Ruff, Pyright, and pytest before every commit.
5. Add MLflow only when comparing repeated experiments and DVC only when Git cannot safely hold the artifacts.
6. Publish a README with the problem, data license, method, reproducible commands, measured result, limitations, and screenshots.

The separate `career-data-lab` project in the workspace is the practice template for this loop.
