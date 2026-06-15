"""
修复无监督组合的metrics解析并重跑U01-U04
"""
import sys
import json
import time
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib
matplotlib.use('Agg')

from gradio_app import run_pipeline

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "combo_verify"


def save_fig(fig, name):
    if fig is None:
        return None
    path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(str(path), dpi=150, bbox_inches='tight')
    import matplotlib.pyplot as plt
    plt.close(fig)
    return str(path)


def parse_unsup_metrics(metrics_md):
    """解析无监督模型的metrics_md（3列格式：指标类别|指标|值）"""
    metrics = {}
    for line in metrics_md.split('\n'):
        if '|' not in line:
            continue
        parts = [x.strip() for x in line.split('|') if x.strip()]
        if len(parts) < 3:
            continue
        # parts[0]=类别, parts[1]=指标名, parts[2]=值
        name, val_str = parts[1], parts[2]
        # 跳过表头
        if val_str == '值' or name == '指标':
            continue
        try:
            val = float(val_str)
            key_map = {
                '轮廓系数': 'silhouette',
                'Davies-Bouldin': 'davies_bouldin',
                'Calinski-Harabasz': 'calinski_harabasz',
                'ARI': 'ari',
                'NMI': 'nmi',
            }
            key = key_map.get(name, name)
            metrics[key] = val
        except ValueError:
            pass
    return metrics


def run_unsup(combo_id, description, **kwargs):
    print(f"\n{'='*60}")
    print(f"[{combo_id}] {description}")
    print(f"{'='*60}")

    defaults = {
        'split_method': 'holdout', 'split_ratio': 0.7, 'use_stratify': True,
        'preprocessing': 'clahe+median',
        'features': ['hog', 'lbp', 'glcm', 'edge_density'],
        'max_samples': 2000, 'model_name': 'kmeans',
        'dt_max_depth': 15, 'dt_min_samples_split': 5,
        'svm_C': 1.0, 'nb_var_smoothing': 1e-9,
        'rf_n_estimators': 100, 'rf_max_depth': 20, 'rf_min_samples_split': 5,
        'lr_C': 1.0,
        'xgb_n_estimators': 100, 'xgb_max_depth': 6, 'xgb_subsample': 0.8,
        'lgbm_n_estimators': 100, 'lgbm_max_depth': 6, 'lgbm_num_leaves': 31,
        'cnn_dropout': 0.5, 'cnn_batch_size': 64, 'cnn_epochs': 30,
        'cnn_early_stopping': 10, 'cnn_input_size': 128, 'cnn_weight_decay': 1e-4,
        'unsup_n_clusters': 2, 'unsup_eps': 0.5, 'unsup_min_samples': 5,
        'dt_criterion': 'gini', 'svm_kernel': 'rbf', 'svm_gamma': 'scale',
        'rf_criterion': 'gini',
        'lr_penalty': 'l2', 'lr_solver': 'lbfgs', 'lr_l1_ratio': 0.5,
        'xgb_objective': 'binary:logistic', 'xgb_learning_rate': 0.1,
        'lgbm_objective': 'binary', 'lgbm_learning_rate': 0.1,
        'cnn_loss_fn': 'cross_entropy', 'cnn_focal_alpha': 'None',
        'cnn_focal_gamma': 2.0, 'cnn_label_smoothing_epsilon': 0.1,
        'cnn_optimizer': 'adam', 'cnn_learning_rate': 0.001,
        'kmeans_algorithm': 'lloyd', 'gmm_covariance_type': 'full',
        'agg_linkage': 'ward', 'spec_affinity': 'rbf',
        'optimization_strategy': 'pretrained',
        'cv_folds_opt': 3, 'n_iter': 30,
        'validation_method': 'holdout', 'scoring_metric': 'f1',
        'unsup_val_method': 'internal_external',
        'random_seed': 42,
    }
    defaults.update(kwargs)

    t0 = time.time()
    try:
        result = run_pipeline(**defaults)
        elapsed = time.time() - t0

        status, metrics_md, cm_fig, roc_fig, pr_fig, fi_fig, prob_fig, extra_fig, sil_fig = result

        # 保存图表
        fig_paths = {}
        fig_map = {
            'extra': extra_fig, 'sil': sil_fig,
        }
        for suffix, fig in fig_map.items():
            p = save_fig(fig, f"{combo_id}_{suffix}")
            if p:
                fig_paths[suffix] = p

        metrics = parse_unsup_metrics(metrics_md)

        combo_result = {
            'combo_id': combo_id,
            'description': description,
            'elapsed': round(elapsed, 1),
            'metrics': metrics,
            'status': status,
            'fig_paths': fig_paths,
            'params': {k: v for k, v in kwargs.items()},
            'error': None,
        }
        print(f"  ✅ 完成 ({elapsed:.1f}s): {metrics}")

    except Exception as e:
        elapsed = time.time() - t0
        combo_result = {
            'combo_id': combo_id,
            'description': description,
            'elapsed': round(elapsed, 1),
            'metrics': {},
            'status': f"ERROR: {e}",
            'fig_paths': {},
            'params': {k: v for k, v in kwargs.items()},
            'error': str(e),
        }
        print(f"  ❌ 失败 ({elapsed:.1f}s): {e}")
        import traceback
        traceback.print_exc()

    return combo_result


if __name__ == '__main__':
    results = []

    results.append(run_unsup("U01", "KMeans | clahe+median",
                             model_name='kmeans', optimization_strategy='pretrained',
                             unsup_n_clusters=2))
    results.append(run_unsup("U02", "GMM | clahe+median",
                             model_name='gmm', optimization_strategy='pretrained',
                             unsup_n_clusters=2))
    results.append(run_unsup("U03", "Agglomerative ward | clahe+median",
                             model_name='agglomerative', optimization_strategy='manual',
                             unsup_n_clusters=2, agg_linkage='ward'))
    results.append(run_unsup("U04", "Spectral rbf | clahe+median",
                             model_name='spectral', optimization_strategy='manual',
                             unsup_n_clusters=2, spec_affinity='rbf'))

    # 加上无监督预处理none的对比
    results.append(run_unsup("U05", "KMeans | none预处理",
                             model_name='kmeans', optimization_strategy='pretrained',
                             unsup_n_clusters=2, preprocessing='none'))

    with open(OUTPUT_DIR / 'unsup_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n无监督结果已保存")
