# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Crack image recognition system (裂纹图像识别系统) for the BJTU "Machine Learning & Python Programming" course research project. The system classifies road surface images as cracked (Positive) or non-cracked (Negative) using traditional ML, unsupervised clustering, and a custom CNN.

## Environment Setup

- **Python 3.10**, conda environment name: `bjtu_ml`
- Create environment: `conda env create -f environment.yml`
- Activate: `conda activate bjtu_ml`
- Register Jupyter kernel: `python -m ipykernel install --user --name=bjtu_ml --display-name "Python (bjtu_ml)"`
- Strip notebook outputs before committing: `nbstripout --install`
- Data path configured via `.env` (copy from `.env.example`), variable: `CRACK_DATA_ROOT`
- PyTorch installs with CUDA 11.8 by default; auto-falls back to CPU if no GPU. Devices without NVIDIA GPU should install CPU-only torch instead.

## Commands

```bash
# Run all tests
pytest tests/ -v

# Run a single test
pytest tests/test_smoke.py::test_crack_cnn_instantiation -v

# Lint and format (Ruff replaces Black + Flake8 + isort)
ruff check src/ tests/
ruff format src/ tests/

# Verify data path config
python -c "from src.config import DATA_ROOT, check_data_exists; print(DATA_ROOT); check_data_exists()"
```

## Architecture

The project follows a **notebook-driven workflow**: Jupyter notebooks in `notebooks/` are the primary research artifacts, calling reusable Python modules from `src/`.

### Notebook Pipeline (sequential)

`01_data_exploration` → `02_traditional_ml` → `03_unsupervised` → `04_deep_learning` → `05_comparison`

### Key `src/` Modules

- **`src/config.py`** — Central configuration. Loads `.env` via `python-dotenv`, resolves `DATA_ROOT`/`POSITIVE_DIR`/`NEGATIVE_DIR`, auto-detects `DEVICE` (cuda/cpu). All other modules import paths and device from here.
- **`src/data_utils.py`** — Image preprocessing (CLAHE, Gaussian, median filters) and feature extraction (HOG, LBP, GLCM, edge density). Also provides `load_dataset()` and `split_dataset()`.
- **`src/models/traditional.py`** — Wraps sklearn DecisionTree/SVM/KNN behind a unified `TraditionalClassifier` interface with a `compare_classifiers()` function.
- **`src/models/unsupervised.py`** — Wraps K-Means, GMM, DBSCAN (+ Agglomerative, Spectral) with a `UnsupervisedPipeline` for running all methods and comparing results.
- **`src/models/cnn.py`** — `CrackCNN`: lightweight CNN for binary classification. Design constraint: 500K–2M parameters, BatchNorm + Dropout. `get_cnn_model()` auto-loads to DEVICE.
- **`src/evaluation/metrics.py`** — Re-exports sklearn metrics (accuracy, precision, recall, F1, confusion matrix, classification report).
- **`src/training/losses.py`** — Cross-entropy loss factory; extensible for Focal Loss etc.
- **`src/training/optimizers.py`** — Placeholder for hyperparameter search and LR scheduling.
- **`src/plot_config.py`** — `set_chinese_font()` configures matplotlib to use Microsoft YaHei. Call at the top of every notebook before plotting.

### Data Flow

```
.env (CRACK_DATA_ROOT)
  → src/config.py (DATA_ROOT, POSITIVE_DIR, NEGATIVE_DIR)
    → src/data_utils.py (load_dataset, preprocessing, feature extraction)
      → src/models/* (training and prediction)
        → src/evaluation/metrics.py (scoring)
```

## Code Conventions

- **Ruff** is the sole formatter/linter (line-length 88, target Python 3.10, LF line endings).
- `.gitattributes` enforces `eol=lf` for all text files — do not commit CRLF.
- Commit messages follow conventional prefixes: `feat:`, `fix:`, `docs:`, `refactor:`, `env:`.
- Notebook outputs are stripped by `nbstripout` on commit — do not commit notebooks with outputs during development.
- `.env` is gitignored; each collaborator configures their own `CRACK_DATA_ROOT`.
- Model weights (`*.pth`, `*.pkl`) and datasets are never committed.
- When adding dependencies: install via conda or pip, then manually edit `environment.yml` — never use `conda env export` (it destroys the PyTorch CUDA `+cu118` suffix and `--extra-index-url`).
- Code comments, docstrings, and variable names are in Chinese or mixed Chinese/English to match the team's working language.
- Many `src/` functions are currently stubs (`raise NotImplementedError`) — this is intentional scaffolding awaiting implementation.
