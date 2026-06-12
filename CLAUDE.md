# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Crack image recognition system (裂纹图像识别系统) for the BJTU "Machine Learning & Python Programming" course research project. The system classifies road surface images as cracked (Positive) or non-cracked (Negative) using traditional ML, unsupervised clustering, and a custom CNN.

All implementation code is self-contained within Jupyter notebooks — there is no separate `src/` module system.

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
# Lint and format Python files (Ruff replaces Black + Flake8 + isort)
ruff check .
ruff format .

# Verify data path config (.env)
python -c "
import os; from pathlib import Path; from dotenv import load_dotenv
load_dotenv('.env')
print(os.getenv('CRACK_DATA_ROOT'))
"
```

## Architecture

The project follows a **notebook-only architecture**: all Python code is inline within Jupyter notebooks. Each notebook is a self-contained deliverable that can be run independently.

### Notebook Pipeline

```
01_数据处理与特征工程.ipynb     — Shared infrastructure (data loading, preprocessing, feature extraction)
02_传统监督学习对比.ipynb       — 7 classifiers (DT/SVM/NB/RF/LR/XGBoost/LightGBM)
03_深度学习对比.ipynb           — CNN + loss comparison + hyperparameter grid search
04_无监督学习对比.ipynb         — 5 clustering methods (KMeans/GMM/DBSCAN/Agglomerative/Spectral)
05_Gradio接口规范.ipynb         — Unified Gradio interface specification
```

### Key Design Decisions

- **Notebooks are self-contained**: each notebook contains all code it needs (preprocessing, feature extraction, model definition, evaluation) — no cross-notebook imports.
- **Gradio interfaces are reserved**: each notebook ends with Gradio callback function signatures (stubs with `raise NotImplementedError`), ready for Phase 2 visualization system development.
- **Chinese documentation**: all markdown explanations and code comments in Chinese.
- **PDF compliance**: each notebook's structure maps to the "机器学习理论部分" five aspects (data processing, model selection, loss measurement, parameter optimization, model validation).
- **Pretrained models**: 16 models saved to `outputs/models/` (7 traditional + 6 CNN + 3 unsupervised) for the Gradio visualization system. Traditional models store best hyperparams from GridSearchCV; CNN models store weights from 15-epoch training with proper train/val/test split. DBSCAN excluded — no valid 2-cluster params found on this dataset.
- **Data pipeline**: features are extracted per-image after train/test split (not before), preventing accidental global-statistics leakage. `prepare_data()` in NB05 provides a unified pipeline for fair cross-model comparison.
- **FocalLoss**: `alpha=None` (default) skips class balancing — correct for this balanced dataset. Old implementation compressed all gradients to 25% (F1≈0.09); fixed June 2026.

### Model Coverage

| Category | Models | Notebook |
|----------|--------|----------|
| Tree | Decision Tree | 02 |
| Kernel | SVM (linear/rbf/poly) | 02 |
| Probabilistic | Naive Bayes (Gaussian) | 02 |
| Bagging | Random Forest | 02 |
| Linear | Logistic Regression | 02 |
| Boosting | XGBoost, LightGBM | 02 |
| Deep Learning | CrackCNN (~1.17M params) | 03 |
| Clustering | KMeans, GMM, DBSCAN, Agglomerative, Spectral | 04 |

### Loss Functions (CNN only, 03 notebook)

CrossEntropy, Focal Loss (focal_gamma2: γ=2 无类别权重; focal_gamma3: γ=3 无类别权重; focal_balanced: α=0.5, γ=2), Label Smoothing CE, Dice Loss — 6 total configurations. FocalLoss `alpha=None` (default) omits class balancing, suitable for this balanced dataset (20k/20k).

### Data Split Methods (01 notebook)

Hold-out (70/30, 80/20, 90/10), K-Fold CV (5-fold, 10-fold), Stratified sampling.

## Code Conventions

- **Ruff** is the sole formatter/linter (line-length 88, target Python 3.10, LF line endings).
- `.gitattributes` enforces `eol=lf` for all text files — do not commit CRLF.
- Commit messages follow conventional prefixes: `feat:`, `fix:`, `docs:`, `refactor:`, `env:`.
- Notebook outputs are stripped by `nbstripout` on commit — do not commit notebooks with outputs during development.
- `.env` is gitignored; each collaborator configures their own `CRACK_DATA_ROOT`.
- Model weights (`*.pth`, `*.pkl`) and datasets are never committed.
- When adding dependencies: install via conda or pip, then manually edit `environment.yml` — never use `conda env export` (it destroys the PyTorch CUDA `+cu118` suffix and `--extra-index-url`).
- Code comments, docstrings, and variable names are in Chinese or mixed Chinese/English.
- The `_backup/` directory contains the legacy `src/` and `tests/` code; it is a reference only and should not be modified.
