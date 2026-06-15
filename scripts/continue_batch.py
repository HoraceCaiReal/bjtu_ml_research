#!/usr/bin/env python
"""Continuation script — resume batch verification from existing results."""

import sys, os, json, time, gc, traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gradio_app import (  # noqa: E402
    prepare_data, _run_traditional, _run_cnn, _run_unsupervised,
    TRAD_DIR, CNN_DIR, UNSUP_DIR, DEVICE, _DATA_CACHE,
)

# Use existing batch directory
BATCH_DIR = PROJECT_ROOT / "outputs/batch_verify/20260613_043410"
PLOTS_DIR = BATCH_DIR / "plots"
assert BATCH_DIR.exists(), f"Batch dir not found: {BATCH_DIR}"

BASELINE = dict(max_samples=2000, random_seed=42, split_method="holdout",
                split_ratio=0.7, use_stratify=True,
                features=["hog", "lbp", "glcm", "edge_density"])
PREPROCESSING_ALL = ["none", "clahe", "gaussian", "median", "clahe+gaussian", "clahe+median"]

ALL_RESULTS = []  # incremental

def _sanitize(s):
    for ch in [":", "<", ">", '"', "|", "?", "*"]:
        s = s.replace(ch, "_")
    return s

def _make_data(prep_list):
    return prepare_data(max_samples=BASELINE["max_samples"],
        random_seed=BASELINE["random_seed"], split_method=BASELINE["split_method"],
        split_ratio=BASELINE["split_ratio"], preprocessing=prep_list,
        features=BASELINE["features"], use_stratify=BASELINE["use_stratify"])

def _save_plots(result, combo_id):
    d = PLOTS_DIR / _sanitize(combo_id)
    d.mkdir(parents=True, exist_ok=True)
    saved = []
    for key, fname in [("cm_fig","confusion_matrix.png"),("roc_fig","roc_curve.png"),
                        ("pr_fig","pr_curve.png"),("fi_fig","feature_importance.png"),
                        ("prob_fig","prob_distribution.png"),("extra_fig","extra.png"),
                        ("sil_fig","silhouette.png")]:
        fig = result.get(key)
        if fig is not None and hasattr(fig, "savefig"):
            try:
                fig.savefig(d / fname, dpi=100, bbox_inches="tight"); saved.append(fname)
                plt.close(fig)
            except Exception: pass
    metrics = result.get("metrics", {}) or {}
    with open(d / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(dict(metrics=metrics, status=str(result.get("status","")),
                       metrics_md=str(result.get("metrics_md",""))), f, ensure_ascii=False, indent=2, default=str)
    return saved

def _is_done(combo_id):
    return (PLOTS_DIR / _sanitize(combo_id) / "metrics.json").exists()

def run_one(combo_id, combo, run_fn):
    if _is_done(combo_id):
        print(f"  ⏭ skip: {combo_id}")
        return None
    t0 = time.time()
    try:
        result = run_fn(combo)
    except Exception as e:
        print(f"  ❌ FAIL: {type(e).__name__}: {str(e)[:150]}")
        result = {"status": f"ERROR: {e}", "metrics_md": "", "metrics": {}}
    elapsed = time.time() - t0
    _save_plots(result, combo_id)
    # Also save to CSV
    m = result.get("metrics", {}) or {}
    record = dict(combo_id=combo_id, **combo, **m,
                  elapsed_sec=round(elapsed,1), error="" if "ERROR" not in str(result.get("status","")) else str(result.get("status",""))[:200])
    ALL_RESULTS.append(record)
    if len(ALL_RESULTS) % 5 == 0:
        import pandas as pd
        df = pd.DataFrame(ALL_RESULTS)
        # Load existing, append
        csv_path = BATCH_DIR / "results.csv"
        if csv_path.exists():
            try:
                existing = pd.read_csv(csv_path, encoding="utf-8-sig", on_bad_lines="skip")
                df = pd.concat([existing, df], ignore_index=True)
            except Exception: pass
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return result

def run_all():
    print(f"Continue batch to: {BATCH_DIR}")
    print(f"Already completed: counting...")
    existing = len(list(PLOTS_DIR.glob("*/metrics.json")))
    print(f"  {existing} runs already done")

    total_new = 0
    remaining_models = ["xgboost", "lightgbm", "cnn", "kmeans", "gmm", "dbscan", "agglomerative", "spectral"]

    # ── XGBoost ──────────────────────────────────────
    model = "xgboost"
    if model in remaining_models:
        print(f"\n{'='*50}\n🚀 {model}\n{'='*50}")
        for prep in PREPROCESSING_ALL:
            for obj in ["binary:logistic", "binary:hinge"]:
                for opt in ["pretrained", "manual"]:
                    tag = f"prep={prep}_objective={obj}" if opt == "pretrained" else f"manual_prep={prep}_objective={obj}"
                    cid = f"{model}/{tag}"
                    combo = dict(model=model, preprocessing=prep, optimization=opt,
                                 validation="holdout", step4_value=obj, step4_key="objective", tag=tag)
                    if _is_done(cid): continue
                    print(f"  [{total_new+1}] {cid}")
                    _clear_cache()
                    prep_list = [prep] if prep != "none" else ["none"]
                    data = _make_data(prep_list)
                    params = dict(n_estimators=100, max_depth=6, learning_rate=0.1,
                                  subsample=0.8, objective=obj, random_state=42)
                    result = _run_traditional(model, params, data, opt, 3, "f1", "holdout", 42)
                    _save_plots(result, cid)
                    total_new += 1

        # kfold, grid, random, scoring variants
        for tag, opt, val, obj, scoring in [
            ("kfold_baseline", "pretrained", "kfold", "binary:logistic", "f1"),
            ("grid_baseline", "grid_search", "holdout", "binary:logistic", "f1"),
            ("random_baseline", "random_search", "holdout", "binary:logistic", "f1"),
            ("grid_kfold", "grid_search", "kfold", "binary:logistic", "f1"),
            ("grid_scoring=accuracy", "grid_search", "holdout", "binary:logistic", "accuracy"),
            ("grid_scoring=roc_auc", "grid_search", "holdout", "binary:logistic", "roc_auc"),
            ("grid_worst_prep", "grid_search", "holdout", "binary:logistic", "f1"),
            ("random_worst_prep", "random_search", "holdout", "binary:logistic", "f1"),
        ]:
            cid = f"{model}/{tag}"
            if _is_done(cid): continue
            prep = "none" if "worst" in tag else "clahe+median"
            print(f"  [{total_new+1}] {cid} | {opt}")
            _clear_cache()
            data = _make_data([prep] if prep != "none" else ["none"])
            params = dict(n_estimators=100, max_depth=6, learning_rate=0.1,
                          subsample=0.8, objective=obj, random_state=42)
            result = _run_traditional(model, params, data, opt, 3, scoring, val, 42)
            _save_plots(result, cid)
            total_new += 1

    # ── LightGBM ─────────────────────────────────────
    model = "lightgbm"
    if model in remaining_models:
        print(f"\n{'='*50}\n🚀 {model}\n{'='*50}")
        for prep in PREPROCESSING_ALL:
            for obj in ["binary", "cross_entropy"]:
                for opt in ["pretrained", "manual"]:
                    tag = f"prep={prep}_objective={obj}" if opt == "pretrained" else f"manual_prep={prep}_objective={obj}"
                    cid = f"{model}/{tag}"
                    if _is_done(cid): continue
                    print(f"  [{total_new+1}] {cid}")
                    _clear_cache()
                    data = _make_data([prep] if prep != "none" else ["none"])
                    params = dict(n_estimators=100, max_depth=6, num_leaves=31,
                                  learning_rate=0.1, objective=obj, random_state=42)
                    result = _run_traditional(model, params, data, opt, 3, "f1", "holdout", 42)
                    _save_plots(result, cid)
                    total_new += 1

        for tag, opt, val, obj, scoring in [
            ("kfold_baseline", "pretrained", "kfold", "binary", "f1"),
            ("grid_baseline", "grid_search", "holdout", "binary", "f1"),
            ("random_baseline", "random_search", "holdout", "binary", "f1"),
            ("grid_kfold", "grid_search", "kfold", "binary", "f1"),
            ("grid_scoring=accuracy", "grid_search", "holdout", "binary", "accuracy"),
            ("grid_scoring=roc_auc", "grid_search", "holdout", "binary", "roc_auc"),
            ("grid_worst_prep", "grid_search", "holdout", "binary", "f1"),
            ("random_worst_prep", "random_search", "holdout", "binary", "f1"),
        ]:
            cid = f"{model}/{tag}"
            if _is_done(cid): continue
            prep = "none" if "worst" in tag else "clahe+median"
            print(f"  [{total_new+1}] {cid} | {opt}")
            _clear_cache()
            data = _make_data([prep] if prep != "none" else ["none"])
            params = dict(n_estimators=100, max_depth=6, num_leaves=31,
                          learning_rate=0.1, objective=obj, random_state=42)
            result = _run_traditional(model, params, data, opt, 3, scoring, val, 42)
            _save_plots(result, cid)
            total_new += 1

    # ── CNN ──────────────────────────────────────────
    print(f"\n{'='*50}\n🚀 CNN\n{'='*50}")
    import torch
    loss_variants = [
        ("cross_entropy", None, None, "CE"),
        ("focal", None, 2.0, "FocalG2"),
        ("focal", None, 3.0, "FocalG3"),
        ("focal", 0.5, 2.0, "FocalBal"),
        ("label_smoothing", None, None, "LabelSmooth"),
        ("dice", None, None, "Dice"),
    ]
    for prep in PREPROCESSING_ALL:
        for lf, alpha, gamma, lf_name in loss_variants:
            cid = f"cnn/prep={prep}_loss={lf_name}_adam"
            if _is_done(cid): continue
            print(f"  [{total_new+1}] {cid}")
            _clear_cache()
            data = _make_data([prep] if prep != "none" else ["none"])
            params = dict(loss_fn=lf, focal_alpha=alpha, focal_gamma=gamma if gamma else 2.0,
                          label_smoothing_epsilon=0.1, optimizer="adam", learning_rate=0.001,
                          dropout_rate=0.5, batch_size=64, epochs=15, early_stopping_patience=10,
                          input_size=128, weight_decay=1e-4)
            result = _run_cnn(params, data, "pretrained", 42, 6, "f1")
            _save_plots(result, cid)
            total_new += 1
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    # CNN manual/grid/random (subset)
    for tag, opt, lf_name, alpha, gamma in [
        ("manual_CE_adam", "manual", "CE", None, None),
        ("grid_CE_adam", "grid_search", "CE", None, None),
        ("random_CE_adam", "random_search", "CE", None, None),
        ("random_FocalG2_adam", "random_search", "FocalG2", None, 2.0),
        ("random_LabelSmooth_adam", "random_search", "LabelSmooth", None, None),
        ("random_Dice_adam", "random_search", "Dice", None, None),
        ("grid_worst_prep_CE", "grid_search", "CE", None, None),
        ("random_worst_prep_CE", "random_search", "CE", None, None),
    ]:
        cid = f"cnn/{tag}"
        if _is_done(cid): continue
        prep = "none" if "worst" in tag else "clahe+median"
        print(f"  [{total_new+1}] {cid} | {opt}")
        _clear_cache()
        data = _make_data([prep] if prep != "none" else ["none"])
        lf = "cross_entropy" if "CE" in lf_name else ("focal" if "Focal" in lf_name else ("label_smoothing" if "Label" in lf_name else "dice"))
        params = dict(loss_fn=lf, focal_alpha=alpha, focal_gamma=gamma if gamma else 2.0,
                      label_smoothing_epsilon=0.1, optimizer="adam", learning_rate=0.001,
                      dropout_rate=0.5, batch_size=64, epochs=15, early_stopping_patience=10,
                      input_size=128, weight_decay=1e-4)
        result = _run_cnn(params, data, opt, 42, 6, "f1")
        _save_plots(result, cid)
        total_new += 1
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ── Unsupervised ────────────────────────────────
    unsup_configs = {
        "kmeans": [("lloyd", "algorithm"), ("elkan", "algorithm")],
        "gmm": [("full", "cov_type"), ("tied", "cov_type"), ("diag", "cov_type"), ("spherical", "cov_type")],
        "dbscan": [(None, "none")],
        "agglomerative": [("ward", "linkage"), ("complete", "linkage"), ("average", "linkage"), ("single", "linkage")],
        "spectral": [("rbf", "affinity"), ("nearest_neighbors", "affinity")],
    }
    val_modes = ["internal_external", "internal_only", "external_only"]

    for method, step4_variants in unsup_configs.items():
        print(f"\n{'='*50}\n🚀 {method}\n{'='*50}")
        for prep in PREPROCESSING_ALL:
            for s4_val, s4_label in step4_variants:
                for opt in ["pretrained", "manual"]:
                    s4_str = s4_val if s4_val else "default"
                    tag = f"prep={prep}_{s4_label}={s4_str}" if opt == "pretrained" else f"manual_prep={prep}_{s4_label}={s4_str}"
                    cid = f"{method}/{tag}"
                    if _is_done(cid): continue
                    print(f"  [{total_new+1}] {cid}")
                    _clear_cache()
                    data = _make_data([prep] if prep != "none" else ["none"])
                    params = dict(n_clusters=2)
                    if method == "kmeans":
                        params["algorithm"] = s4_val if s4_val else "lloyd"
                    elif method == "gmm":
                        params["covariance_type"] = s4_val if s4_val else "full"
                    elif method == "dbscan":
                        params["eps"] = 0.5; params["min_samples"] = 5
                    elif method == "agglomerative":
                        params["linkage"] = s4_val if s4_val else "ward"
                    elif method == "spectral":
                        params["affinity"] = s4_val if s4_val else "rbf"
                    result = _run_unsupervised(method, params, data, opt, 42, 5, "internal_external")
                    _save_plots(result, cid)
                    total_new += 1

        # Validation modes + grid/random
        for val in ["internal_only", "external_only"]:
            cid = f"{method}/val={val}"
            if _is_done(cid): continue
            print(f"  [{total_new+1}] {cid}")
            _clear_cache()
            data = _make_data(["clahe", "median"])
            params = dict(n_clusters=2)
            if method == "kmeans": params["algorithm"] = "lloyd"
            elif method == "gmm": params["covariance_type"] = "full"
            elif method == "dbscan": params["eps"] = 0.5; params["min_samples"] = 5
            elif method == "agglomerative": params["linkage"] = "ward"
            elif method == "spectral": params["affinity"] = "rbf"
            result = _run_unsupervised(method, params, data, "pretrained", 42, 5, val)
            _save_plots(result, cid)
            total_new += 1

        for opt in ["grid_search", "random_search"]:
            cid = f"{method}/{opt}_baseline"
            if _is_done(cid): continue
            print(f"  [{total_new+1}] {cid} | {opt}")
            _clear_cache()
            data = _make_data(["clahe", "median"])
            params = dict(n_clusters=2)
            if method == "kmeans": params["algorithm"] = "lloyd"
            elif method == "gmm": params["covariance_type"] = "full"
            elif method == "dbscan": params["eps"] = 0.5; params["min_samples"] = 5
            elif method == "agglomerative": params["linkage"] = "ward"
            elif method == "spectral": params["affinity"] = "rbf"
            result = _run_unsupervised(method, params, data, opt, 42, 5, "internal_external")
            _save_plots(result, cid)
            total_new += 1

        cid = f"{method}/grid_worst_prep"
        if not _is_done(cid):
            print(f"  [{total_new+1}] {cid}")
            _clear_cache()
            data = _make_data(["none"])
            params = dict(n_clusters=2)
            if method == "kmeans": params["algorithm"] = "lloyd"
            elif method == "gmm": params["covariance_type"] = "full"
            elif method == "dbscan": params["eps"] = 0.5; params["min_samples"] = 5
            elif method == "agglomerative": params["linkage"] = "ward"
            elif method == "spectral": params["affinity"] = "rbf"
            result = _run_unsupervised(method, params, data, "grid_search", 42, 5, "internal_external")
            _save_plots(result, cid)
            total_new += 1

    print(f"\n\n{'='*70}")
    print(f"DONE! New runs: {total_new}")
    existing = len(list(PLOTS_DIR.glob("*/metrics.json")))
    print(f"Total runs now: {existing}")
    print("="*70)


def _clear_cache():
    _DATA_CACHE.clear()
    gc.collect()
    import torch
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == "__main__":
    run_all()
