"""
Gradio 可视化界面全链路集成测试
=================================
调用 run_pipeline() 测试 51 种选项组合，带数据缓存优化。
输出: outputs/integration_test_<timestamp>/{results.json, plots/}

用法:
    conda activate bjtu_ml
    python -u tests/test_integration.py
"""

import sys
import os
import json
import time
import re
import traceback
from pathlib import Path
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 项目路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 动态加载 gradio_app（src/ 无 __init__.py）
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "gradio_app", str(PROJECT_ROOT / "src" / "gradio_app.py"),
)
_ga = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ga)

run_pipeline = _ga.run_pipeline
prepare_data = _ga.prepare_data
_run_traditional = _ga._run_traditional
_run_cnn = _ga._run_cnn
_run_unsupervised = _ga._run_unsupervised

# ============================================================
# 1. 共享默认参数（匹配 UI 默认值）
# ============================================================

DATA_DEFAULTS = dict(
    max_samples=800,
    random_seed=42,
    split_method="holdout",
    split_ratio=0.7,
    preprocessing=["clahe", "median"],
    features=["lbp", "glcm", "edge_density"],  # 精简特征集（108维），加速 A-E 组测试
)

# F 组使用完整特征集
FULL_FEATURES = ["hog", "lbp", "glcm", "edge_density"]

PIPELINE_DEFAULTS = dict(
    # Step 3: 超参数
    dt_max_depth=15,
    dt_min_samples_split=5,
    svm_C=1.0,
    nb_var_smoothing=1e-9,
    rf_n_estimators=100,
    rf_max_depth=20,
    rf_min_samples_split=5,
    lr_C=1.0,
    xgb_n_estimators=100,
    xgb_max_depth=6,
    xgb_subsample=0.8,
    lgbm_n_estimators=100,
    lgbm_max_depth=6,
    lgbm_num_leaves=31,
    cnn_dropout=0.5,
    cnn_batch_size=64,
    cnn_epochs=30,
    cnn_early_stopping=10,
    unsup_n_clusters=2,
    unsup_eps=0.5,
    unsup_min_samples=5,
    # Step 4: 损失函数/优化器
    dt_criterion="gini",
    svm_kernel="linear",
    svm_gamma="scale",
    rf_criterion="gini",
    lr_penalty="l2",
    lr_solver="lbfgs",
    xgb_objective="binary:logistic",
    xgb_learning_rate=0.1,
    lgbm_objective="binary",
    lgbm_learning_rate=0.1,
    cnn_loss_fn="cross_entropy",
    cnn_focal_alpha="None",
    cnn_focal_gamma=2.0,
    cnn_label_smoothing_epsilon=0.1,
    cnn_optimizer="adam",
    cnn_learning_rate=0.001,
    kmeans_algorithm="lloyd",
    gmm_covariance_type="full",
    agg_linkage="ward",
    spec_affinity="rbf",
    # Step 5: 参数优化 + 验证
    optimization_strategy="pretrained",
    cv_folds_opt=3,
    n_iter=30,
    validation_method="holdout",
    scoring_metric="f1",
)

# ============================================================
# 2. 51 个测试用例
# ============================================================

TRAD_MODELS = {"decision_tree", "svm", "naive_bayes", "random_forest",
               "logistic_regression", "xgboost", "lightgbm"}
UNSUP_MODELS = {"kmeans", "gmm", "dbscan", "agglomerative", "spectral"}

TEST_CASES = [
    # ── A 组: 传统模型 pretrained 基线 (7) ──
    {"id": "A1_dt_pretrained", "model_name": "decision_tree"},
    {"id": "A2_svm_pretrained", "model_name": "svm"},
    {"id": "A3_nb_pretrained", "model_name": "naive_bayes"},
    {"id": "A4_rf_pretrained", "model_name": "random_forest"},
    {"id": "A5_lr_pretrained", "model_name": "logistic_regression"},
    {"id": "A6_xgb_pretrained", "model_name": "xgboost"},
    {"id": "A7_lgbm_pretrained", "model_name": "lightgbm"},
    # ── B 组: 传统模型分类选项变化 (8) ──
    {"id": "B1_dt_entropy", "model_name": "decision_tree", "dt_criterion": "entropy"},
    {"id": "B2_dt_log_loss", "model_name": "decision_tree", "dt_criterion": "log_loss"},
    {"id": "B3_svm_rbf", "model_name": "svm", "svm_kernel": "rbf"},
    {"id": "B4_svm_poly", "model_name": "svm", "svm_kernel": "poly"},
    {"id": "B5_rf_entropy", "model_name": "random_forest", "rf_criterion": "entropy"},
    {
        "id": "B6_lr_l1_liblinear",
        "model_name": "logistic_regression",
        "lr_penalty": "l1",
        "lr_solver": "liblinear",
    },
    {"id": "B7_xgb_hinge", "model_name": "xgboost", "xgb_objective": "binary:hinge"},
    {
        "id": "B8_lgbm_cross_entropy",
        "model_name": "lightgbm",
        "lgbm_objective": "cross_entropy",
    },
    # ── C 组: 传统模型优化策略 + 验证方法 (6) ──
    {"id": "C1_dt_manual", "model_name": "decision_tree", "optimization_strategy": "manual"},
    {
        "id": "C2_dt_grid_search",
        "model_name": "decision_tree",
        "optimization_strategy": "grid_search",
    },
    {"id": "C3_svm_manual", "model_name": "svm", "optimization_strategy": "manual"},
    {
        "id": "C4_rf_random_search",
        "model_name": "random_forest",
        "optimization_strategy": "random_search",
    },
    {"id": "C5_rf_kfold", "model_name": "random_forest", "validation_method": "kfold"},
    {
        "id": "C6_dt_grid_roc_auc",
        "model_name": "decision_tree",
        "optimization_strategy": "grid_search",
        "scoring_metric": "roc_auc",
    },
    # ── D 组: CNN 损失函数 + 优化器 (9) ──
    {"id": "D1_cnn_ce_adam", "model_name": "cnn", "cnn_loss_fn": "cross_entropy"},
    {
        "id": "D2_cnn_focal_g2_adam",
        "model_name": "cnn",
        "cnn_loss_fn": "focal",
        "cnn_focal_gamma": 2.0,
        "cnn_focal_alpha": "None",
    },
    {
        "id": "D3_cnn_focal_g3_adam",
        "model_name": "cnn",
        "cnn_loss_fn": "focal",
        "cnn_focal_gamma": 3.0,
        "cnn_focal_alpha": "None",
    },
    {
        "id": "D4_cnn_focal_bal_adam",
        "model_name": "cnn",
        "cnn_loss_fn": "focal",
        "cnn_focal_gamma": 2.0,
        "cnn_focal_alpha": "0.5",
    },
    {"id": "D5_cnn_ls_adam", "model_name": "cnn", "cnn_loss_fn": "label_smoothing"},
    {"id": "D6_cnn_dice_adam", "model_name": "cnn", "cnn_loss_fn": "dice"},
    {
        "id": "D7_cnn_ce_sgd",
        "model_name": "cnn",
        "cnn_loss_fn": "cross_entropy",
        "cnn_optimizer": "sgd",
    },
    {
        "id": "D8_cnn_focal_sgd",
        "model_name": "cnn",
        "cnn_loss_fn": "focal",
        "cnn_focal_gamma": 2.0,
        "cnn_focal_alpha": "None",
        "cnn_optimizer": "sgd",
    },
    {
        "id": "D9_cnn_dice_sgd",
        "model_name": "cnn",
        "cnn_loss_fn": "dice",
        "cnn_optimizer": "sgd",
    },
    # ── E 组: 无监督聚类 (11) ──
    {"id": "E1_kmeans_lloyd", "model_name": "kmeans", "kmeans_algorithm": "lloyd"},
    {"id": "E2_kmeans_elkan", "model_name": "kmeans", "kmeans_algorithm": "elkan"},
    {"id": "E3_gmm_full", "model_name": "gmm", "gmm_covariance_type": "full"},
    {"id": "E4_gmm_diag", "model_name": "gmm", "gmm_covariance_type": "diag"},
    {"id": "E5_gmm_spherical", "model_name": "gmm", "gmm_covariance_type": "spherical"},
    {"id": "E6_gmm_tied", "model_name": "gmm", "gmm_covariance_type": "tied"},
    {"id": "E7_dbscan_manual", "model_name": "dbscan", "optimization_strategy": "manual"},
    {"id": "E8_agg_ward", "model_name": "agglomerative", "agg_linkage": "ward"},
    {"id": "E9_agg_complete", "model_name": "agglomerative", "agg_linkage": "complete"},
    {"id": "E10_spec_rbf", "model_name": "spectral", "spec_affinity": "rbf"},
    {
        "id": "E11_spec_nn",
        "model_name": "spectral",
        "spec_affinity": "nearest_neighbors",
    },
    # ── F 组: 预处理方式 × 模型类型搭配 (10) — 使用完整特征集 ──
    {
        "id": "F1_rf_none",
        "model_name": "random_forest",
        "preprocessing": ["none"],
        "features": FULL_FEATURES,
    },
    {
        "id": "F2_rf_clahe",
        "model_name": "random_forest",
        "preprocessing": ["clahe"],
        "features": FULL_FEATURES,
    },
    {
        "id": "F3_rf_gaussian",
        "model_name": "random_forest",
        "preprocessing": ["gaussian"],
        "features": FULL_FEATURES,
    },
    {
        "id": "F4_rf_clahe_gaussian",
        "model_name": "random_forest",
        "preprocessing": ["clahe+gaussian"],
        "features": FULL_FEATURES,
    },
    {
        "id": "F5_rf_clahe_median",
        "model_name": "random_forest",
        "preprocessing": ["clahe+median"],
        "features": FULL_FEATURES,
    },
    {
        "id": "F6_rf_all",
        "model_name": "random_forest",
        "preprocessing": ["clahe", "gaussian", "median"],
        "features": FULL_FEATURES,
    },
    {
        "id": "F7_cnn_none",
        "model_name": "cnn",
        "cnn_loss_fn": "cross_entropy",
        "preprocessing": ["none"],
    },
    {
        "id": "F8_cnn_clahe_gaussian",
        "model_name": "cnn",
        "cnn_loss_fn": "cross_entropy",
        "preprocessing": ["clahe+gaussian"],
    },
    {
        "id": "F9_km_none",
        "model_name": "kmeans",
        "preprocessing": ["none"],
        "features": FULL_FEATURES,
    },
    {
        "id": "F10_km_clahe_gaussian",
        "model_name": "kmeans",
        "preprocessing": ["clahe+gaussian"],
        "features": FULL_FEATURES,
    },
]


# ============================================================
# 3. 辅助函数
# ============================================================


def build_params(overrides: dict):
    """从覆盖值构建完整的传统/CNN/无监督参数字典。"""
    cfg = {**PIPELINE_DEFAULTS, **overrides}
    seed = cfg.get("random_seed", 42)

    trad_params = {
        "decision_tree": {
            "max_depth": int(cfg["dt_max_depth"]),
            "min_samples_split": int(cfg["dt_min_samples_split"]),
            "criterion": cfg["dt_criterion"],
            "random_state": seed,
        },
        "svm": {
            "C": float(cfg["svm_C"]),
            "kernel": cfg["svm_kernel"],
            "gamma": cfg["svm_gamma"],
            "random_state": seed,
        },
        "naive_bayes": {"var_smoothing": float(cfg["nb_var_smoothing"])},
        "random_forest": {
            "n_estimators": int(cfg["rf_n_estimators"]),
            "max_depth": int(cfg["rf_max_depth"]),
            "min_samples_split": int(cfg["rf_min_samples_split"]),
            "criterion": cfg["rf_criterion"],
            "random_state": seed,
        },
        "logistic_regression": {
            "C": float(cfg["lr_C"]),
            "penalty": cfg["lr_penalty"],
            "solver": cfg["lr_solver"],
            "random_state": seed,
        },
        "xgboost": {
            "n_estimators": int(cfg["xgb_n_estimators"]),
            "max_depth": int(cfg["xgb_max_depth"]),
            "subsample": float(cfg["xgb_subsample"]),
            "objective": cfg["xgb_objective"],
            "learning_rate": float(cfg["xgb_learning_rate"]),
            "random_state": seed,
        },
        "lightgbm": {
            "n_estimators": int(cfg["lgbm_n_estimators"]),
            "max_depth": int(cfg["lgbm_max_depth"]),
            "num_leaves": int(cfg["lgbm_num_leaves"]),
            "objective": cfg["lgbm_objective"],
            "learning_rate": float(cfg["lgbm_learning_rate"]),
            "random_state": seed,
        },
    }

    focal_alpha = cfg["cnn_focal_alpha"]
    cnn_params = {
        "loss_fn": cfg["cnn_loss_fn"],
        "focal_alpha": None if focal_alpha == "None" else float(focal_alpha),
        "focal_gamma": float(cfg["cnn_focal_gamma"]),
        "label_smoothing_epsilon": float(cfg["cnn_label_smoothing_epsilon"]),
        "optimizer": cfg["cnn_optimizer"],
        "learning_rate": float(cfg["cnn_learning_rate"]),
        "dropout_rate": float(cfg["cnn_dropout"]),
        "batch_size": int(cfg["cnn_batch_size"]),
        "epochs": int(cfg["cnn_epochs"]),
        "early_stopping_patience": int(cfg["cnn_early_stopping"]),
    }

    unsup_params = {
        "n_clusters": int(cfg["unsup_n_clusters"]),
        "eps": float(cfg["unsup_eps"]),
        "min_samples": int(cfg["unsup_min_samples"]),
        "algorithm": cfg["kmeans_algorithm"],
        "covariance_type": cfg["gmm_covariance_type"],
        "linkage": cfg["agg_linkage"],
        "affinity": cfg["spec_affinity"],
    }

    return trad_params, cnn_params, unsup_params, cfg


def get_data_cache_key(overrides: dict) -> tuple:
    """生成数据缓存键（基于影响数据的参数）。"""
    preproc = tuple(sorted(overrides.get("preprocessing", DATA_DEFAULTS["preprocessing"])))
    feats = tuple(sorted(overrides.get("features", DATA_DEFAULTS["features"])))
    return (
        overrides.get("max_samples", DATA_DEFAULTS["max_samples"]),
        overrides.get("random_seed", DATA_DEFAULTS["random_seed"]),
        overrides.get("split_method", DATA_DEFAULTS["split_method"]),
        overrides.get("split_ratio", DATA_DEFAULTS["split_ratio"]),
        preproc,
        feats,
    )


def get_cached_data(overrides: dict, cache: dict) -> dict:
    """获取数据（带缓存）。"""
    key = get_data_cache_key(overrides)
    if key not in cache:
        preproc = overrides.get("preprocessing", DATA_DEFAULTS["preprocessing"])
        feats = overrides.get("features", DATA_DEFAULTS["features"])
        cache[key] = prepare_data(
            max_samples=overrides.get("max_samples", DATA_DEFAULTS["max_samples"]),
            random_seed=overrides.get("random_seed", DATA_DEFAULTS["random_seed"]),
            split_method=overrides.get("split_method", DATA_DEFAULTS["split_method"]),
            split_ratio=overrides.get("split_ratio", DATA_DEFAULTS["split_ratio"]),
            preprocessing=list(preproc),
            features=list(feats),
        )
    return cache[key]


def run_single_test(test_id: str, overrides: dict, data_cache: dict) -> dict:
    """运行单个测试用例，返回结构化结果。"""
    model_name = overrides.get("model_name", "random_forest")
    seed = overrides.get("random_seed", PIPELINE_DEFAULTS.get("random_seed", 42))

    t0 = time.time()
    try:
        # 获取数据（缓存）
        data = get_cached_data(overrides, data_cache)

        # 构建参数
        trad_params, cnn_params, unsup_params, cfg = build_params(overrides)
        optimization = overrides.get("optimization_strategy",
                                     PIPELINE_DEFAULTS["optimization_strategy"])
        cv_folds = overrides.get("cv_folds_opt", PIPELINE_DEFAULTS["cv_folds_opt"])
        scoring = overrides.get("scoring_metric", PIPELINE_DEFAULTS["scoring_metric"])
        validation = overrides.get("validation_method",
                                   PIPELINE_DEFAULTS["validation_method"])

        # 分发执行
        if model_name in TRAD_MODELS:
            result = _run_traditional(
                model_name, trad_params[model_name], data,
                optimization, int(cv_folds), scoring, validation, int(seed),
            )
        elif model_name == "cnn":
            result = _run_cnn(cnn_params, data, optimization, int(seed))
        elif model_name in UNSUP_MODELS:
            result = _run_unsupervised(
                model_name, unsup_params, data, optimization, int(seed),
            )
        else:
            raise ValueError(f"未知模型: {model_name}")

        elapsed = time.time() - t0
        status = result.get("status", "")
        metrics_md = result.get("metrics_md", "")
        is_error = status.startswith("❌") if status else False

        return {
            "test_id": test_id,
            "passed": not is_error,
            "elapsed": round(elapsed, 2),
            "status": status,
            "metrics_md": metrics_md,
            "metrics": extract_metrics(metrics_md),
            "result_dict": result,
            "error": None,
        }
    except Exception as e:
        elapsed = time.time() - t0
        tb = traceback.format_exc()
        return {
            "test_id": test_id,
            "passed": False,
            "elapsed": round(elapsed, 2),
            "status": f"❌ 异常: {e}",
            "metrics_md": "",
            "metrics": {},
            "result_dict": None,
            "error": tb,
        }


def save_figures(test_id: str, result_dict: dict | None, plots_dir: Path) -> list:
    """将 matplotlib Figure 保存为 PNG，返回路径列表。"""
    if result_dict is None:
        return []

    test_dir = plots_dir / test_id
    test_dir.mkdir(parents=True, exist_ok=True)

    fig_keys = {
        "cm_fig": "confusion_matrix.png",
        "roc_fig": "roc_curve.png",
        "pr_fig": "pr_curve.png",
        "fi_fig": "feature_importance.png",
        "prob_fig": "prob_distribution.png",
        "extra_fig": "extra.png",
        "sil_fig": "silhouette.png",
    }

    saved = []
    for key, fname in fig_keys.items():
        fig = result_dict.get(key)
        if fig is not None:
            fpath = test_dir / fname
            fig.savefig(str(fpath), dpi=100, bbox_inches="tight")
            plt.close(fig)
            saved.append(str(fpath.relative_to(plots_dir.parent)))
        else:
            plt.close("all")
    return saved


def extract_metrics(metrics_md: str) -> dict:
    """从 markdown 指标表中提取数值。"""
    metrics = {}
    pattern = (
        r"\|\s*(准确率|精确率|召回率|F1分数|ROC-AUC|"
        r"轮廓系数|Davies-Bouldin|Calinski-Harabasz|ARI|NMI)"
        r"\s*\|\s*([\d.nan-]+)"
    )
    for m in re.finditer(pattern, metrics_md):
        name = m.group(1)
        try:
            metrics[name] = float(m.group(2))
        except ValueError:
            metrics[name] = m.group(2)
    return metrics


# ============================================================
# 4. 主函数
# ============================================================


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "outputs" / f"integration_test_{timestamp}"
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    total = len(TEST_CASES)
    print(f"\n{'='*60}")
    print(f"  Gradio 全链路集成测试 — {total} 个用例")
    print(f"  输出目录: {output_dir}")
    print(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    data_cache = {}
    all_results = []
    passed_count = 0
    failed_count = 0

    for i, tc in enumerate(TEST_CASES, 1):
        test_id = tc["id"]
        overrides = {k: v for k, v in tc.items() if k != "id"}

        print(f"[{i}/{total}] ⏳ {test_id} ...", flush=True)

        result = run_single_test(test_id, overrides, data_cache)

        # 保存图表
        plot_paths = save_figures(test_id, result["result_dict"], plots_dir)
        result["plot_paths"] = plot_paths

        # 释放 Figure 对象
        result["result_dict"] = None

        icon = "✅" if result["passed"] else "❌"
        print(f"       {icon} ({result['elapsed']:.1f}s)", flush=True)

        if result["passed"]:
            passed_count += 1
        else:
            failed_count += 1
            if result["error"]:
                lines = result["error"].strip().split("\n")
                for line in lines[-3:]:
                    print(f"       ⚠️  {line}", flush=True)
            elif "❌" in result["status"]:
                err_line = result["status"].split("\n")[-2] if "\n" in result["status"] else ""
                print(f"       ⚠️  {err_line[:120]}", flush=True)

        all_results.append(result)

    # 保存 results.json
    results_json = {
        "timestamp": timestamp,
        "total": total,
        "passed": passed_count,
        "failed": failed_count,
        "pass_rate": f"{passed_count/total*100:.1f}%",
        "results": [
            {
                "test_id": r["test_id"],
                "passed": r["passed"],
                "elapsed": r["elapsed"],
                "metrics": r["metrics"],
                "plot_paths": r.get("plot_paths", []),
                "status_preview": (r["status"] or "")[:200],
                "metrics_md_preview": (r["metrics_md"] or "")[:300],
            }
            for r in all_results
        ],
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"  测试完成: {passed_count}/{total} 通过, {failed_count} 失败")
    print(f"  通过率: {passed_count/total*100:.1f}%")
    print(f"  结果文件: {results_path}")
    print(f"  图表目录: {plots_dir}")
    print(f"  数据缓存命中: {len(data_cache)} 个唯一数据配置")
    print(f"{'='*60}\n")

    # 打印失败测试详情
    if failed_count > 0:
        print("失败测试详情:\n")
        for r in all_results:
            if not r["passed"]:
                print(f"  {r['test_id']}:")
                if r["error"]:
                    lines = r["error"].strip().split("\n")
                    for line in lines[-5:]:
                        print(f"    {line}")
                else:
                    print(f"    {r['status'][:300]}")
                print()


if __name__ == "__main__":
    main()
