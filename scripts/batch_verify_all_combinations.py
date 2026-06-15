#!/usr/bin/env python
"""
批量组合验证脚本 — 运行所有 298 种方法组合并记录结果。

覆盖 PDF 指导书五环节：数据处理、模型选择、损失衡量、参数优化、模型验证。
直接调用 src/gradio_app.py 后端函数，不启动 Gradio UI。

用法: conda run -n bjtu_ml python scripts/batch_verify_all_combinations.py
"""

import sys
import os
import json
import time
import gc
import traceback
from pathlib import Path
from datetime import datetime
from itertools import product
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── 项目路径 ────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ── 导入后端函数 ────────────────────────────────────────
from gradio_app import (  # noqa: E402
    prepare_data,
    _run_traditional,
    _run_cnn,
    _run_unsupervised,
    TRAD_DIR,
    CNN_DIR,
    UNSUP_DIR,
    DEVICE,
    _DATA_CACHE,
)

# ── 输出目录 ────────────────────────────────────────────
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
BATCH_DIR = PROJECT_ROOT / "outputs" / "batch_verify" / TIMESTAMP
PLOTS_DIR = BATCH_DIR / "plots"
RESULTS_DIR = BATCH_DIR / "results"
for d in [BATCH_DIR, PLOTS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"输出目录: {BATCH_DIR}")
print(f"计算设备: {DEVICE}")
print("=" * 70)

# ══════════════════════════════════════════════════════════
# 固定基线参数
# ══════════════════════════════════════════════════════════

BASELINE = {
    "max_samples": 2000,
    "random_seed": 42,
    "split_method": "holdout",
    "split_ratio": 0.7,
    "use_stratify": True,
    "features": ["hog", "lbp", "glcm", "edge_density"],
}

PREPROCESSING_ALL = ["none", "clahe", "gaussian", "median", "clahe+gaussian", "clahe+median"]
PREPROCESSING_DEFAULT = "clahe+median"

# ══════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════

ALL_RESULTS = []  # 全局结果收集


def _make_data(preprocessing_list: List[str], random_seed: int = 42) -> dict:
    """封装 prepare_data，使用基线参数。"""
    return prepare_data(
        max_samples=BASELINE["max_samples"],
        random_seed=random_seed,
        split_method=BASELINE["split_method"],
        split_ratio=BASELINE["split_ratio"],
        preprocessing=preprocessing_list,
        features=BASELINE["features"],
        use_stratify=BASELINE["use_stratify"],
    )


def _sanitize_path(combo_id: str) -> str:
    """替换 Windows 路径非法字符。"""
    for ch in [":", "<", ">", '"', "|", "?", "*"]:
        combo_id = combo_id.replace(ch, "_")
    return combo_id


def _save_plots(result: dict, combo_id: str):
    """保存图表到磁盘。"""
    safe_id = _sanitize_path(combo_id)
    plot_dir = PLOTS_DIR / safe_id
    plot_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    plot_keys = {
        "cm_fig": "confusion_matrix.png",
        "roc_fig": "roc_curve.png",
        "pr_fig": "pr_curve.png",
        "fi_fig": "feature_importance.png",
        "prob_fig": "prob_distribution.png",
        "extra_fig": "extra.png",
        "sil_fig": "silhouette.png",
    }
    for key, fname in plot_keys.items():
        fig = result.get(key)
        if fig is not None and hasattr(fig, "savefig"):
            try:
                fig.savefig(plot_dir / fname, dpi=100, bbox_inches="tight")
                saved.append(fname)
                plt.close(fig)
            except Exception:
                pass
    # 保存指标 JSON
    metrics = result.get("metrics", {}) if isinstance(result.get("metrics"), dict) else {}
    try:
        metrics_md = result.get("metrics_md", "")
        status = result.get("status", "")
    except Exception:
        metrics_md = ""
        status = ""
    with open(plot_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "metrics": metrics,
            "status": str(status),
            "metrics_md": str(metrics_md),
        }, f, ensure_ascii=False, indent=2, default=str)
    return saved


def _record(combo: dict, metrics: dict, status: str, elapsed: float,
            plots_saved: List[str], error: Optional[str] = None):
    """记录一条结果到全局列表。"""
    record = {
        **combo,
        **metrics,
        "status": status[:200].replace("\n", " ").replace("\r", ""),
        "elapsed_sec": round(elapsed, 1),
        "plots_saved": ",".join(plots_saved),
        "error": error or "",
        "timestamp": datetime.now().isoformat(),
    }
    ALL_RESULTS.append(record)
    # 每 10 条增量保存 CSV
    if len(ALL_RESULTS) % 10 == 0:
        _save_csv()


def _save_csv():
    """保存当前结果到 CSV。"""
    if not ALL_RESULTS:
        return
    df = pd.DataFrame(ALL_RESULTS)
    csv_path = BATCH_DIR / "results.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    # 也存 JSON 备份
    json_path = BATCH_DIR / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ALL_RESULTS, f, ensure_ascii=False, indent=2, default=str)


def _is_already_done(combo: dict) -> bool:
    """检查组合是否已完成（metrics.json 已存在）。"""
    combo_id = _sanitize_path(combo["combo_id"])
    metrics_path = PLOTS_DIR / combo_id / "metrics.json"
    return metrics_path.exists()


def _safe_run(combo: dict, run_fn, skip_if_done: bool = True):
    """安全执行一次运行，捕获异常并记录。"""
    combo_id = combo["combo_id"]
    safe_id = _sanitize_path(combo_id)

    if skip_if_done and _is_already_done(combo):
        print(f"  ⏭ 跳过(已完成): {combo_id}")
        return None

    t0 = time.time()
    error = None
    result = {}
    try:
        result = run_fn(combo)
    except Exception as e:
        error = f"{type(e).__name__}: {str(e)[:300]}\n{traceback.format_exc()[-500:]}"
        print(f"  ❌ 失败: {error[:120]}")
        result = {"status": error, "metrics_md": "", "metrics": {}}
    elapsed = time.time() - t0

    plots_saved = _save_plots(result, combo_id)

    # 提取指标
    metrics = result.get("metrics", {}) if isinstance(result.get("metrics"), dict) else {}
    # 兼容无监督返回的额外指标
    for extra_key in ["silhouette", "davies_bouldin", "calinski_harabasz",
                       "ari", "nmi", "sil", "db", "ch"]:
        if extra_key in result and extra_key not in metrics:
            metrics[extra_key] = result[extra_key]

    _record(combo, metrics, result.get("status", ""), elapsed, plots_saved, error)
    return result


def _clear_cache():
    """清空 prepare_data 内存缓存。"""
    _DATA_CACHE.clear()
    gc.collect()
    if DEVICE.type == "cuda":
        import torch
        torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════
# 组合生成器
# ══════════════════════════════════════════════════════════

def generate_traditional_combos(model_name: str, step4_key: str,
                                 step4_options: List[str],
                                 step4_label: str) -> List[dict]:
    """为传统模型生成所有组合。

    Parameters
    ----------
    model_name : 内部模型名 (decision_tree, svm, ...)
    step4_key : 参数字典中的键名 (criterion, kernel, penalty, objective, ...)
    step4_options : Step4 变体列表
    step4_label : 显示标签
    """
    combos = []
    default_step4 = step4_options[0]

    def add(prep, opt, val, s4, scoring, tag):
        combo_id = f"{model_name}/{tag}"
        combos.append({
            "combo_id": combo_id,
            "model": model_name,
            "category": "traditional",
            "preprocessing": prep,
            "optimization": opt,
            "validation": val,
            "step4_key": step4_key,
            "step4_value": s4,
            "scoring": scoring,
            "tag": tag,
        })

    prep_list = PREPROCESSING_ALL

    # (1) 所有预处理 × pretrained × holdout × 所有 Step4 变体（数据处理对比 + 损失对比）
    for prep, s4 in product(prep_list, step4_options):
        add(prep, "pretrained", "holdout", s4, "f1", f"prep={prep}_{step4_label}={s4}")

    # (2) 所有预处理 × manual × holdout × 所有 Step4 变体
    for prep, s4 in product(prep_list, step4_options):
        add(prep, "manual", "holdout", s4, "f1", f"manual_prep={prep}_{step4_label}={s4}")

    # (3) baseline 预处理 × grid × holdout × 默认 Step4
    add(PREPROCESSING_DEFAULT, "grid_search", "holdout", default_step4, "f1",
        "grid_baseline")

    # (4) baseline 预处理 × random × holdout × 默认 Step4
    add(PREPROCESSING_DEFAULT, "random_search", "holdout", default_step4, "f1",
        "random_baseline")

    # (5) baseline × pretrained × kfold × 默认 Step4（验证方法对比）
    add(PREPROCESSING_DEFAULT, "pretrained", "kfold", default_step4, "f1",
        "kfold_baseline")

    # (6) baseline × grid × kfold × 默认 Step4
    add(PREPROCESSING_DEFAULT, "grid_search", "kfold", default_step4, "f1",
        "grid_kfold")

    # (7) grid × holdout × 不同评分指标（acc, roc_auc）
    for scoring in ["accuracy", "roc_auc"]:
        add(PREPROCESSING_DEFAULT, "grid_search", "holdout", default_step4, scoring,
            f"grid_scoring={scoring}")

    # (8) grid/random on worst preprocessing (none) for comparison
    add("none", "grid_search", "holdout", default_step4, "f1", "grid_worst_prep")
    add("none", "random_search", "holdout", default_step4, "f1", "random_worst_prep")

    return combos


def generate_cnn_combos() -> List[dict]:
    """为 CNN 生成所有组合。"""
    combos = []
    loss_fns = ["cross_entropy", "focal", "label_smoothing", "dice"]
    focal_configs = [
        ("focal_gamma2", {"focal_alpha": None, "focal_gamma": 2.0}),
        ("focal_gamma3", {"focal_alpha": None, "focal_gamma": 3.0}),
        ("focal_balanced", {"focal_alpha": 0.5, "focal_gamma": 2.0}),
    ]
    optimizers = ["adam", "sgd"]

    def add(prep, opt_strat, loss_fn, focal_cfg, optimizer, tag):
        combo_id = f"cnn/{tag}"
        combos.append({
            "combo_id": combo_id,
            "model": "cnn",
            "category": "cnn",
            "preprocessing": prep,
            "optimization": opt_strat,
            "validation": "holdout",
            "loss_fn": loss_fn,
            "focal_alpha": focal_cfg["focal_alpha"] if focal_cfg else None,
            "focal_gamma": focal_cfg["focal_gamma"] if focal_cfg else 2.0,
            "optimizer": optimizer,
            "tag": tag,
        })

    # All CNN loss variants for loss function display names
    all_loss_variants = []
    for lf in loss_fns:
        if lf == "focal":
            for fname, fcfg in focal_configs:
                all_loss_variants.append((lf, fcfg, fname))
        else:
            all_loss_variants.append((lf, None, lf))

    # (1) 所有预处理 × pretrained × 所有损失函数 × adam（数据处理 + 损失对比）
    for prep, (lf, fcfg, lf_name) in product(PREPROCESSING_ALL, all_loss_variants):
        add(prep, "pretrained", lf, fcfg, "adam",
            f"prep={prep}_loss={lf_name}_adam")

    # (2) baseline × pretrained × 所有损失 × sgd（优化器对比）
    for lf, fcfg, lf_name in all_loss_variants:
        add(PREPROCESSING_DEFAULT, "pretrained", lf, fcfg, "sgd",
            f"loss={lf_name}_sgd")

    # (3) baseline × manual × CE × adam
    add(PREPROCESSING_DEFAULT, "manual", "cross_entropy", None, "adam",
        "manual_CE_adam")

    # (4) baseline × grid × CE × adam
    add(PREPROCESSING_DEFAULT, "grid_search", "cross_entropy", None, "adam",
        "grid_CE_adam")

    # (5) baseline × random × CE × adam
    add(PREPROCESSING_DEFAULT, "random_search", "cross_entropy", None, "adam",
        "random_CE_adam")

    # (6) baseline × random × 3 key losses (FocalG2, LabelSmooth, Dice)
    for lf, fcfg, lf_name in all_loss_variants:
        if lf_name in ("focal_gamma2", "label_smoothing", "dice"):
            add(PREPROCESSING_DEFAULT, "random_search", lf, fcfg, "adam",
                f"random_{lf_name}_adam")

    # (7) Grid on worst preprocessing
    add("none", "grid_search", "cross_entropy", None, "adam", "grid_worst_prep_CE")
    add("none", "random_search", "cross_entropy", None, "adam", "random_worst_prep_CE")

    return combos


def generate_unsupervised_combos(method: str, step4_key: Optional[str],
                                  step4_options: Optional[List[str]],
                                  step4_label: str) -> List[dict]:
    """为无监督方法生成所有组合。"""
    combos = []
    default_s4 = step4_options[0] if step4_options else None

    val_methods = ["internal_external", "internal_only", "external_only"]

    def add(prep, opt, val, s4, tag):
        combo_id = f"{method}/{tag}"
        combos.append({
            "combo_id": combo_id,
            "model": method,
            "category": "unsupervised",
            "preprocessing": prep,
            "optimization": opt,
            "validation": val,
            "step4_key": step4_key or "",
            "step4_value": str(s4) if s4 else "",
            "tag": tag,
        })

    prep_list = PREPROCESSING_ALL
    s4_list = step4_options if step4_options else [None]

    # (1) 所有预处理 × pretrained × internal_external × 所有 Step4
    for prep, s4 in product(prep_list, s4_list):
        s4_str = str(s4) if s4 else "default"
        add(prep, "pretrained", "internal_external", s4,
            f"prep={prep}_{step4_label}={s4_str}")

    # (2) 所有预处理 × manual × internal_external × 所有 Step4
    for prep, s4 in product(prep_list, s4_list):
        s4_str = str(s4) if s4 else "default"
        add(prep, "manual", "internal_external", s4,
            f"manual_prep={prep}_{step4_label}={s4_str}")

    # (3) baseline × pretrained × 3 种评估范围
    for val in val_methods:
        if val == "internal_external":
            continue  # 已包含在上面
        add(PREPROCESSING_DEFAULT, "pretrained", val, default_s4,
            f"val={val}")

    # (4) baseline × grid × internal_external
    add(PREPROCESSING_DEFAULT, "grid_search", "internal_external", default_s4,
        "grid_baseline")

    # (5) baseline × random × internal_external
    add(PREPROCESSING_DEFAULT, "random_search", "internal_external", default_s4,
        "random_baseline")

    # (6) baseline × grid × 另2种评估范围
    for val in ["internal_only", "external_only"]:
        add(PREPROCESSING_DEFAULT, "grid_search", val, default_s4,
            f"grid_val={val}")

    # (7) worst preprocessing × grid
    add("none", "grid_search", "internal_external", default_s4, "grid_worst_prep")

    return combos


# ══════════════════════════════════════════════════════════
# 模型执行器
# ══════════════════════════════════════════════════════════

def _gen_trad_params(combo: dict) -> dict:
    """根据 combo 构建传统模型参数字典。"""
    m = combo["model"]
    seed = BASELINE["random_seed"]
    params = {"random_state": seed}

    if m == "decision_tree":
        params.update({"max_depth": 15, "min_samples_split": 5,
                       "criterion": combo["step4_value"]})
    elif m == "svm":
        params.update({"C": 1.0, "kernel": combo["step4_value"],
                       "gamma": "scale"})
    elif m == "naive_bayes":
        params.update({"var_smoothing": 1e-9})
    elif m == "random_forest":
        params.update({"n_estimators": 100, "max_depth": 20,
                       "min_samples_split": 5, "criterion": combo["step4_value"]})
    elif m == "logistic_regression":
        penalty = combo["step4_value"]
        params.update({"C": 1.0, "penalty": penalty})
        if penalty == "l1":
            params["solver"] = "liblinear"
        elif penalty == "elasticnet":
            params["solver"] = "saga"
            params["l1_ratio"] = 0.5
        else:
            params["solver"] = "lbfgs"
    elif m == "xgboost":
        params.update({"n_estimators": 100, "max_depth": 6,
                       "learning_rate": 0.1, "subsample": 0.8,
                       "objective": combo["step4_value"]})
    elif m == "lightgbm":
        params.update({"n_estimators": 100, "max_depth": 6,
                       "num_leaves": 31, "learning_rate": 0.1,
                       "objective": combo["step4_value"]})
    return params


def _gen_cnn_params(combo: dict) -> dict:
    """根据 combo 构建 CNN 参数字典。"""
    return {
        "loss_fn": combo["loss_fn"],
        "focal_alpha": combo["focal_alpha"],
        "focal_gamma": combo["focal_gamma"],
        "label_smoothing_epsilon": 0.1,
        "optimizer": combo["optimizer"],
        "learning_rate": 0.001,
        "dropout_rate": 0.5,
        "batch_size": 64,
        "epochs": 15,
        "early_stopping_patience": 10,
        "input_size": 128,
        "weight_decay": 1e-4,
    }


def _gen_unsup_params(combo: dict) -> dict:
    """根据 combo 构建无监督参数字典。"""
    m = combo["model"]
    params = {"n_clusters": 2}
    if m == "kmeans":
        alg = combo["step4_value"] if combo["step4_value"] != "default" else "lloyd"
        params["algorithm"] = alg if alg else "lloyd"
    elif m == "gmm":
        cov = combo["step4_value"] if combo["step4_value"] != "default" else "full"
        params["covariance_type"] = cov if cov else "full"
    elif m == "dbscan":
        params["eps"] = 0.5
        params["min_samples"] = 5
    elif m == "agglomerative":
        link = combo["step4_value"] if combo["step4_value"] != "default" else "ward"
        params["linkage"] = link if link else "ward"
    elif m == "spectral":
        aff = combo["step4_value"] if combo["step4_value"] != "default" else "rbf"
        params["affinity"] = aff if aff else "rbf"
    return params


def run_traditional_combo(combo: dict) -> dict:
    """执行一个传统模型组合。"""
    m = combo["model"]
    prep_list = [combo["preprocessing"]] if combo["preprocessing"] != "none" else ["none"]

    # 准备数据
    data = _make_data(prep_list, BASELINE["random_seed"])

    params = _gen_trad_params(combo)
    opt = combo["optimization"]
    val = combo["validation"]
    cv_folds = 5 if val == "kfold" else 3
    scoring = combo.get("scoring", "f1")
    n_iter = 30

    return _run_traditional(
        model_name=m, params=params, data=data,
        optimization=opt, cv_folds=cv_folds,
        scoring=scoring, validation_method=val,
        random_seed=BASELINE["random_seed"],
        n_iter=n_iter,
    )


def run_cnn_combo(combo: dict) -> dict:
    """执行一个 CNN 组合。"""
    prep_list = [combo["preprocessing"]] if combo["preprocessing"] != "none" else ["none"]
    data = _make_data(prep_list, BASELINE["random_seed"])
    params = _gen_cnn_params(combo)
    n_iter = 6
    scoring = "f1"

    return _run_cnn(
        params=params, data=data,
        optimization=combo["optimization"],
        random_seed=BASELINE["random_seed"],
        n_iter=n_iter,
        scoring_metric=scoring,
    )


def run_unsupervised_combo(combo: dict) -> dict:
    """执行一个无监督组合。"""
    prep_list = [combo["preprocessing"]] if combo["preprocessing"] != "none" else ["none"]
    data = _make_data(prep_list, BASELINE["random_seed"])
    params = _gen_unsup_params(combo)

    return _run_unsupervised(
        method=combo["model"],
        params=params,
        data=data,
        optimization=combo["optimization"],
        random_seed=BASELINE["random_seed"],
        n_iter=5,
        unsup_val_method=combo["validation"],
    )


# ══════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("🔍 裂纹图像识别系统 — 全组合批量验证")
    print(f"开始时间: {datetime.now().isoformat()}")
    print("=" * 70)

    # ── 生成所有组合 ──────────────────────────────────
    all_combos = []

    # 传统模型 (7)
    trad_models = {
        "decision_tree": ("criterion", ["gini", "entropy", "log_loss"], "criterion"),
        "svm": ("kernel", ["rbf", "linear", "poly"], "kernel"),
        "naive_bayes": (None, [None], "none"),
        "random_forest": ("criterion", ["gini", "entropy", "log_loss"], "criterion"),
        "logistic_regression": ("penalty", ["l2", "l1", "elasticnet"], "penalty"),
        "xgboost": ("objective", ["binary:logistic", "binary:hinge"], "objective"),
        "lightgbm": ("objective", ["binary", "cross_entropy"], "objective"),
    }

    for m, (s4_key, s4_opts, s4_label) in trad_models.items():
        combos = generate_traditional_combos(m, s4_key, s4_opts, s4_label)
        all_combos.extend(combos)
        print(f"  {m}: {len(combos)} 组合 ({s4_label} × {s4_opts})")

    # CNN
    cnn_combos = generate_cnn_combos()
    all_combos.extend(cnn_combos)
    print(f"  cnn: {len(cnn_combos)} 组合")

    # 无监督 (5)
    unsup_models = {
        "kmeans": ("algorithm", ["lloyd", "elkan"], "algorithm"),
        "gmm": ("covariance_type", ["full", "tied", "diag", "spherical"], "cov_type"),
        "dbscan": (None, [None], "none"),
        "agglomerative": ("linkage", ["ward", "complete", "average", "single"], "linkage"),
        "spectral": ("affinity", ["rbf", "nearest_neighbors"], "affinity"),
    }

    for m, (s4_key, s4_opts, s4_label) in unsup_models.items():
        combos = generate_unsupervised_combos(m, s4_key, s4_opts, s4_label)
        all_combos.extend(combos)
        print(f"  {m}: {len(combos)} 组合 ({s4_label} × {s4_opts})")

    print(f"\n📊 总组合数: {len(all_combos)}")
    print("=" * 70)

    # ── 保存组合清单 ──────────────────────────────────
    with open(BATCH_DIR / "combo_list.json", "w", encoding="utf-8") as f:
        json.dump(all_combos, f, ensure_ascii=False, indent=2, default=str)

    # ── 按预计耗时分批执行 ────────────────────────────
    # 快速组合先跑 (pretrained + manual)
    fast_combos = [c for c in all_combos
                   if c["optimization"] in ("pretrained", "manual")]
    # 慢速组合后跑 (grid_search + random_search)
    slow_combos = [c for c in all_combos
                   if c["optimization"] in ("grid_search", "random_search")]

    print(f"\n⚡ 快速组合 (pretrained/manual): {len(fast_combos)} 个")
    print(f"🐢 慢速组合 (grid/random): {len(slow_combos)} 个")

    # ── 阶段1：快速组合 ──────────────────────────────
    print("\n" + "=" * 70)
    print("🚀 阶段 1: 快速组合 (pretrained / manual)")
    print("=" * 70)

    for i, combo in enumerate(fast_combos):
        cid = combo["combo_id"]
        model = combo["model"]
        opt = combo["optimization"]
        prep = combo.get("preprocessing", "?")
        tag = combo.get("tag", "")

        print(f"\n[{i+1}/{len(fast_combos)}] {cid} | {opt} | prep={prep}")

        _clear_cache()

        if combo["category"] == "traditional":
            _safe_run(combo, run_traditional_combo)
        elif combo["category"] == "cnn":
            _safe_run(combo, run_cnn_combo)
        elif combo["category"] == "unsupervised":
            _safe_run(combo, run_unsupervised_combo)

        # 每 50 个打印进度
        if (i + 1) % 50 == 0:
            _save_csv()
            success = sum(1 for r in ALL_RESULTS if not r.get("error"))
            fail = sum(1 for r in ALL_RESULTS if r.get("error"))
            print(f"  📊 进度: {i+1}/{len(fast_combos)} | ✅ {success} | ❌ {fail}")

    _save_csv()
    print(f"\n✅ 阶段1完成! 已运行: {len(ALL_RESULTS)}")

    # ── 阶段2：慢速组合 ──────────────────────────────
    print("\n" + "=" * 70)
    print("🐢 阶段 2: 慢速组合 (grid_search / random_search)")
    print("=" * 70)

    for i, combo in enumerate(slow_combos):
        cid = combo["combo_id"]
        model = combo["model"]
        opt = combo["optimization"]
        tag = combo.get("tag", "")

        print(f"\n[{i+1}/{len(slow_combos)}] {cid} | {opt}")

        _clear_cache()

        if combo["category"] == "traditional":
            _safe_run(combo, run_traditional_combo)
        elif combo["category"] == "cnn":
            _safe_run(combo, run_cnn_combo)
        elif combo["category"] == "unsupervised":
            _safe_run(combo, run_unsupervised_combo)

        if (i + 1) % 10 == 0:
            _save_csv()
            success = sum(1 for r in ALL_RESULTS if not r.get("error"))
            fail = sum(1 for r in ALL_RESULTS if r.get("error"))
            print(f"  📊 进度: {i+1}/{len(slow_combos)} | ✅ {success} | ❌ {fail}")

    # ── 最终保存 ──────────────────────────────────────
    _save_csv()

    success = sum(1 for r in ALL_RESULTS if not r.get("error"))
    fail = sum(1 for r in ALL_RESULTS if r.get("error"))

    print("\n" + "=" * 70)
    print(f"🏁 全部完成!")
    print(f"   总运行: {len(ALL_RESULTS)}")
    print(f"   成功: {success} ✅")
    print(f"   失败: {fail} ❌")
    print(f"   结果目录: {BATCH_DIR}")
    print(f"   结束时间: {datetime.now().isoformat()}")
    print("=" * 70)

    # ── 快速摘要 ──────────────────────────────────────
    if success > 0:
        df = pd.DataFrame(ALL_RESULTS)
        valid = df[df["error"] == ""]
        if len(valid) > 0:
            print("\n📈 各模型汇总:")
            for model in valid["model"].unique():
                mdf = valid[valid["model"] == model]
                if "f1" in mdf.columns:
                    best = mdf.loc[mdf["f1"].idxmax()]
                    print(f"  {model}: 最佳F1={best['f1']:.4f} ({best['tag']})")
                elif "silhouette" in mdf.columns:
                    # 对于无监督，找最佳轮廓系数
                    sil_col = "silhouette" if "silhouette" in mdf.columns else None
                    if sil_col:
                        best = mdf.loc[mdf[sil_col].idxmax()]
                        print(f"  {model}: 最佳Sil={best[sil_col]:.4f} ({best['tag']})")

    # 保存最终摘要
    summary = {
        "timestamp": TIMESTAMP,
        "total_runs": len(ALL_RESULTS),
        "success": success,
        "fail": fail,
        "output_dir": str(BATCH_DIR),
    }
    with open(BATCH_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    return BATCH_DIR


if __name__ == "__main__":
    main()
