#!/usr/bin/env python
"""Optimized batch — run best-combination verification for remaining models.

Leverages existing 189 runs from 20260613_043410 for DT/SVM/NB/RF/LR.
Only runs remaining models: XGBoost, LightGBM, CNN, and 5 unsupervised methods.
Focus: 2 preps (best clahe+median + worst none) × all Step4 × key strategies.
"""

import sys, os, json, time, gc
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gradio_app import prepare_data, _run_traditional, _run_cnn, _run_unsupervised, _DATA_CACHE, DEVICE

BATCH_DIR = PROJECT_ROOT / "outputs/batch_verify/20260613_043410"
PLOTS_DIR = BATCH_DIR / "plots"

BASELINE = dict(max_samples=2000, random_seed=42, split_method="holdout",
                split_ratio=0.7, use_stratify=True,
                features=["hog", "lbp", "glcm", "edge_density"])
PREPS = ["clahe+median", "none"]  # best + worst

def _s(s):  # sanitize path
    for ch in [":","<",">",'"',"|","?","*"]: s = s.replace(ch, "_")
    return s

def _done(cid): return (PLOTS_DIR / _s(cid) / "metrics.json").exists()

def _mk_data(prep):
    return prepare_data(max_samples=2000, random_seed=42, split_method="holdout",
        split_ratio=0.7, preprocessing=[prep] if prep != "none" else ["none"],
        features=["hog","lbp","glcm","edge_density"], use_stratify=True)

def _save(result, cid):
    d = PLOTS_DIR / _s(cid); d.mkdir(parents=True, exist_ok=True)
    for key, fn in [("cm_fig","cm.png"),("roc_fig","roc.png"),("pr_fig","pr.png"),
                     ("fi_fig","fi.png"),("prob_fig","prob.png"),("extra_fig","extra.png"),
                     ("sil_fig","sil.png")]:
        fig = result.get(key)
        if fig and hasattr(fig,"savefig"):
            try: fig.savefig(d/fn,dpi=100,bbox_inches="tight"); plt.close(fig)
            except: pass
    m = result.get("metrics",{}) or {}
    with open(d/"metrics.json","w",encoding="utf-8") as f:
        json.dump(dict(metrics=m,status=str(result.get("status",""))),f,ensure_ascii=False,indent=2,default=str)

def _clr():
    _DATA_CACHE.clear(); gc.collect()
    import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None

total = 0
def log(msg): print(f"  [{total}] {msg}")

# ═══════════════════════════════════════════════════════
print("="*60)
print("Optimized batch — remaining models (best+worst prep)")
print(f"Output: {BATCH_DIR}")
print(f"Existing runs: {len(list(PLOTS_DIR.glob('*/metrics.json')))}")
print("="*60)

# ── XGBoost ──────────────────────────────────────────
print("\n🚀 XGBoost")
for prep in PREPS:
    for obj in ["binary:logistic", "binary:hinge"]:
        for opt in ["pretrained", "manual"]:
            cid = f"xgboost/prep={prep}_obj={obj}_{opt}"
            if _done(cid): continue
            total += 1; print(f"  [{total}] {cid}")
            _clr()
            data = _mk_data(prep)
            p = dict(n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8,
                     objective=obj, random_state=42)
            _save(_run_traditional("xgboost",p,data,opt,3,"f1","holdout",42), cid)

# XGBoost grid/random/kfold (baseline only)
for cid, opt, val, obj, scoring in [
    ("xgboost/grid_baseline","grid_search","holdout","binary:logistic","f1"),
    ("xgboost/random_baseline","random_search","holdout","binary:logistic","f1"),
    ("xgboost/kfold_baseline","pretrained","kfold","binary:logistic","f1"),
    ("xgboost/grid_kfold","grid_search","kfold","binary:logistic","f1"),
    ("xgboost/grid_acc","grid_search","holdout","binary:logistic","accuracy"),
    ("xgboost/grid_auc","grid_search","holdout","binary:logistic","roc_auc"),
]:
    if _done(cid): continue
    total += 1; print(f"  [{total}] {cid} | {opt}")
    _clr()
    data = _mk_data("clahe+median")
    p = dict(n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8,
             objective=obj, random_state=42)
    _save(_run_traditional("xgboost",p,data,opt,3 if val=="holdout" else 5,scoring,val,42), cid)

# ── LightGBM ─────────────────────────────────────────
print("\n🚀 LightGBM")
for prep in PREPS:
    for obj in ["binary", "cross_entropy"]:
        for opt in ["pretrained", "manual"]:
            cid = f"lightgbm/prep={prep}_obj={obj}_{opt}"
            if _done(cid): continue
            total += 1; print(f"  [{total}] {cid}")
            _clr()
            data = _mk_data(prep)
            p = dict(n_estimators=100, max_depth=6, num_leaves=31, learning_rate=0.1,
                     objective=obj, random_state=42)
            _save(_run_traditional("lightgbm",p,data,opt,3,"f1","holdout",42), cid)

for cid, opt, val, obj, scoring in [
    ("lightgbm/grid_baseline","grid_search","holdout","binary","f1"),
    ("lightgbm/random_baseline","random_search","holdout","binary","f1"),
    ("lightgbm/kfold_baseline","pretrained","kfold","binary","f1"),
    ("lightgbm/grid_kfold","grid_search","kfold","binary","f1"),
    ("lightgbm/grid_acc","grid_search","holdout","binary","accuracy"),
    ("lightgbm/grid_auc","grid_search","holdout","binary","roc_auc"),
]:
    if _done(cid): continue
    total += 1; print(f"  [{total}] {cid} | {opt}")
    _clr()
    data = _mk_data("clahe+median")
    p = dict(n_estimators=100, max_depth=6, num_leaves=31, learning_rate=0.1,
             objective=obj, random_state=42)
    _save(_run_traditional("lightgbm",p,data,opt,3 if val=="holdout" else 5,scoring,val,42), cid)

# ── CNN ─────────────────────────────────────────────
print("\n🚀 CNN")
import torch
LOSSES = [("cross_entropy","CE",None,None), ("focal","FocalG2",None,2.0),
          ("focal","FocalG3",None,3.0), ("focal","FocalBal",0.5,2.0),
          ("label_smoothing","LabelSmooth",None,None), ("dice","Dice",None,None)]

for prep in PREPS:
    for lf, lname, alpha, gamma in LOSSES:
        cid = f"cnn/prep={prep}_loss={lname}_adam"
        if _done(cid): continue
        total += 1; print(f"  [{total}] {cid}")
        _clr()
        data = _mk_data(prep)
        p = dict(loss_fn=lf, focal_alpha=alpha, focal_gamma=gamma or 2.0,
                 label_smoothing_epsilon=0.1, optimizer="adam", learning_rate=0.001,
                 dropout_rate=0.5, batch_size=64, epochs=15, early_stopping_patience=10,
                 input_size=128, weight_decay=1e-4)
        _save(_run_cnn(p,data,"pretrained",42,6,"f1"), cid)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

# CNN key extra runs
for cid, opt, lf, lname, alpha, gamma in [
    ("cnn/manual_CE_adam","manual","cross_entropy","CE",None,None),
    ("cnn/grid_CE_adam","grid_search","cross_entropy","CE",None,None),
    ("cnn/random_CE_adam","random_search","cross_entropy","CE",None,None),
    ("cnn/random_FocalG2_adam","random_search","focal","FocalG2",None,2.0),
    ("cnn/random_LabelSmooth_adam","random_search","label_smoothing","LabelSmooth",None,None),
    ("cnn/random_Dice_adam","random_search","dice","Dice",None,None),
]:
    if _done(cid): continue
    total += 1; print(f"  [{total}] {cid} | {opt}")
    _clr()
    data = _mk_data("clahe+median")
    p = dict(loss_fn=lf, focal_alpha=alpha, focal_gamma=gamma or 2.0,
             label_smoothing_epsilon=0.1, optimizer="adam", learning_rate=0.001,
             dropout_rate=0.5, batch_size=64, epochs=15, early_stopping_patience=10,
             input_size=128, weight_decay=1e-4)
    _save(_run_cnn(p,data,opt,42,6,"f1"), cid)
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ── Unsupervised ─────────────────────────────────────
UNSUP = {
    "kmeans": [("lloyd","alg"),("elkan","alg")],
    "gmm": [("full","cov"),("tied","cov"),("diag","cov"),("spherical","cov")],
    "dbscan": [(None,"default")],
    "agglomerative": [("ward","link"),("complete","link"),("average","link"),("single","link")],
    "spectral": [("rbf","aff"),("nearest_neighbors","aff")],
}

for method, variants in UNSUP.items():
    print(f"\n🚀 {method}")
    for prep in PREPS:
        for s4v, s4l in variants:
            cid = f"{method}/prep={prep}_{s4l}={s4v or 'default'}_pretrained"
            if _done(cid): continue
            total += 1; print(f"  [{total}] {cid}")
            _clr()
            data = _mk_data(prep)
            p = dict(n_clusters=2)
            if method=="kmeans": p["algorithm"]=s4v or "lloyd"
            elif method=="gmm": p["covariance_type"]=s4v or "full"
            elif method=="dbscan": p["eps"]=0.5; p["min_samples"]=5
            elif method=="agglomerative": p["linkage"]=s4v or "ward"
            elif method=="spectral": p["affinity"]=s4v or "rbf"
            _save(_run_unsupervised(method,p,data,"pretrained",42,5,"internal_external"), cid)

    # Grid + random baseline
    for opt in ["grid_search","random_search"]:
        cid = f"{method}/{opt}_baseline"
        if _done(cid): continue
        total += 1; print(f"  [{total}] {cid} | {opt}")
        _clr()
        data = _mk_data("clahe+median")
        p = dict(n_clusters=2)
        if method=="kmeans": p["algorithm"]="lloyd"
        elif method=="gmm": p["covariance_type"]="full"
        elif method=="dbscan": p["eps"]=0.5; p["min_samples"]=5
        elif method=="agglomerative": p["linkage"]="ward"
        elif method=="spectral": p["affinity"]="rbf"
        _save(_run_unsupervised(method,p,data,opt,42,5,"internal_external"), cid)

    # Validation modes
    for val in ["internal_only","external_only"]:
        cid = f"{method}/val={val}"
        if _done(cid): continue
        total += 1; print(f"  [{total}] {cid}")
        _clr()
        data = _mk_data("clahe+median")
        p = dict(n_clusters=2)
        if method=="kmeans": p["algorithm"]="lloyd"
        elif method=="gmm": p["covariance_type"]="full"
        elif method=="dbscan": p["eps"]=0.5; p["min_samples"]=5
        elif method=="agglomerative": p["linkage"]="ward"
        elif method=="spectral": p["affinity"]="rbf"
        _save(_run_unsupervised(method,p,data,"pretrained",42,5,val), cid)

existing = len(list(PLOTS_DIR.glob("*/metrics.json")))
print(f"\n{'='*60}")
print(f"DONE! New runs: {total}, Total: {existing}")
print(f"{'='*60}")
