"""
批量组合验证脚本 — 直接调用 gradio_app.run_pipeline()
走与用户在可视化界面完全一致的通路。
"""
import sys
import json
import time
import os
import traceback
from pathlib import Path

# 确保项目根目录在 sys.path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib
matplotlib.use('Agg')

from gradio_app import run_pipeline

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "combo_verify"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 收集所有结果
all_results = []

def save_fig(fig, name):
    """保存matplotlib图到文件"""
    if fig is None:
        return None
    path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(str(path), dpi=150, bbox_inches='tight')
    import matplotlib.pyplot as plt
    plt.close(fig)
    return str(path)


def run_combo(combo_id, description, **kwargs):
    """运行一个组合，返回结构化结果"""
    print(f"\n{'='*60}")
    print(f"[{combo_id}] {description}")
    print(f"{'='*60}")

    # run_pipeline 的完整参数列表（与 Gradio UI 完全一致）
    defaults = {
        'split_method': 'holdout',
        'split_ratio': 0.7,
        'use_stratify': True,
        'preprocessing': 'clahe+median',
        'features': ['hog', 'lbp', 'glcm', 'edge_density'],
        'max_samples': 2000,
        'model_name': 'random_forest',
        # DT
        'dt_max_depth': 15, 'dt_min_samples_split': 5,
        # SVM
        'svm_C': 1.0,
        # NB
        'nb_var_smoothing': 1e-9,
        # RF
        'rf_n_estimators': 100, 'rf_max_depth': 20, 'rf_min_samples_split': 5,
        # LR
        'lr_C': 1.0,
        # XGB
        'xgb_n_estimators': 100, 'xgb_max_depth': 6, 'xgb_subsample': 0.8,
        # LGBM
        'lgbm_n_estimators': 100, 'lgbm_max_depth': 6, 'lgbm_num_leaves': 31,
        # CNN
        'cnn_dropout': 0.5, 'cnn_batch_size': 64, 'cnn_epochs': 30,
        'cnn_early_stopping': 10, 'cnn_input_size': 128, 'cnn_weight_decay': 1e-4,
        # Unsup
        'unsup_n_clusters': 2, 'unsup_eps': 0.5, 'unsup_min_samples': 5,
        # Step 4: 损失/优化器
        'dt_criterion': 'gini',
        'svm_kernel': 'rbf', 'svm_gamma': 'scale',
        'rf_criterion': 'gini',
        'lr_penalty': 'l2', 'lr_solver': 'lbfgs', 'lr_l1_ratio': 0.5,
        'xgb_objective': 'binary:logistic', 'xgb_learning_rate': 0.1,
        'lgbm_objective': 'binary', 'lgbm_learning_rate': 0.1,
        'cnn_loss_fn': 'cross_entropy', 'cnn_focal_alpha': 'None',
        'cnn_focal_gamma': 2.0, 'cnn_label_smoothing_epsilon': 0.1,
        'cnn_optimizer': 'adam', 'cnn_learning_rate': 0.001,
        'kmeans_algorithm': 'lloyd', 'gmm_covariance_type': 'full',
        'agg_linkage': 'ward', 'spec_affinity': 'rbf',
        # Step 5
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
            'cm': cm_fig, 'roc': roc_fig, 'pr': pr_fig,
            'fi': fi_fig, 'prob': prob_fig, 'extra': extra_fig, 'sil': sil_fig,
        }
        for suffix, fig in fig_map.items():
            p = save_fig(fig, f"{combo_id}_{suffix}")
            if p:
                fig_paths[suffix] = p

        # 提取metrics（从metrics_md解析）
        # metrics已经由调用方在result中，但run_pipeline返回的是tuple，需要从metrics_md解析
        metrics = {}
        for line in metrics_md.split('\n'):
            if '|' in line and '准确率' in line:
                vals = [x.strip() for x in line.split('|') if x.strip()]
                if len(vals) >= 2:
                    metrics['accuracy'] = float(vals[1])
            elif '|' in line and '精确率' in line:
                vals = [x.strip() for x in line.split('|') if x.strip()]
                if len(vals) >= 2:
                    metrics['precision'] = float(vals[1])
            elif '|' in line and '召回率' in line:
                vals = [x.strip() for x in line.split('|') if x.strip()]
                if len(vals) >= 2:
                    metrics['recall'] = float(vals[1])
            elif '|' in line and 'F1分数' in line:
                vals = [x.strip() for x in line.split('|') if x.strip()]
                if len(vals) >= 2:
                    metrics['f1'] = float(vals[1])
            elif '|' in line and 'ROC-AUC' in line:
                vals = [x.strip() for x in line.split('|') if x.strip()]
                if len(vals) >= 2:
                    metrics['roc_auc'] = float(vals[1])
            # 无监督指标
            elif '|' in line and '轮廓系数' in line:
                vals = [x.strip() for x in line.split('|') if x.strip()]
                if len(vals) >= 2:
                    metrics['silhouette'] = float(vals[1])
            elif '|' in line and 'Davies-Bouldin' in line:
                vals = [x.strip() for x in line.split('|') if x.strip()]
                if len(vals) >= 2:
                    metrics['davies_bouldin'] = float(vals[1])
            elif '|' in line and 'Calinski-Harabasz' in line:
                vals = [x.strip() for x in line.split('|') if x.strip()]
                if len(vals) >= 2:
                    try:
                        metrics['calinski_harabasz'] = float(vals[1])
                    except:
                        pass
            elif '|' in line and 'ARI' in line:
                vals = [x.strip() for x in line.split('|') if x.strip()]
                if len(vals) >= 2:
                    metrics['ari'] = float(vals[1])
            elif '|' in line and 'NMI' in line:
                vals = [x.strip() for x in line.split('|') if x.strip()]
                if len(vals) >= 2:
                    metrics['nmi'] = float(vals[1])

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
        traceback.print_exc()

    all_results.append(combo_result)
    return combo_result


# ============================================================
# 精简组合方案：覆盖5维度 + 好/差性能，约20组
# ============================================================

if __name__ == '__main__':
    t_start = time.time()

    # ========== 1. 数据预处理对比（4组，固定RF pretrained）==========
    # none: 预期最差
    run_combo("P01", "预处理=none | RF pretrained",
              preprocessing='none', model_name='random_forest',
              optimization_strategy='pretrained')
    # clahe
    run_combo("P02", "预处理=clahe | RF pretrained",
              preprocessing='clahe', model_name='random_forest',
              optimization_strategy='pretrained')
    # median
    run_combo("P03", "预处理=median | RF pretrained",
              preprocessing='median', model_name='random_forest',
              optimization_strategy='pretrained')
    # clahe+median: 预期最佳
    run_combo("P04", "预处理=clahe+median | RF pretrained",
              preprocessing='clahe+median', model_name='random_forest',
              optimization_strategy='pretrained')

    # ========== 2. 模型选择对比（8组，复用P04的RF结果）==========
    # 传统模型 - pretrained策略
    run_combo("M01", "DT pretrained | clahe+median",
              model_name='decision_tree', optimization_strategy='pretrained')
    run_combo("M02", "SVM pretrained | clahe+median",
              model_name='svm', optimization_strategy='pretrained')
    run_combo("M03", "NB pretrained | clahe+median",
              model_name='naive_bayes', optimization_strategy='pretrained')
    # P04 = RF (已跑，复用)
    run_combo("M04", "LR pretrained | clahe+median",
              model_name='logistic_regression', optimization_strategy='pretrained')
    run_combo("M05", "XGBoost pretrained | clahe+median",
              model_name='xgboost', optimization_strategy='pretrained')
    run_combo("M06", "LightGBM pretrained | clahe+median",
              model_name='lightgbm', optimization_strategy='pretrained')
    # CNN - pretrained (CrossEntropy)
    run_combo("M07", "CNN CE pretrained | clahe+median",
              model_name='cnn', optimization_strategy='pretrained',
              cnn_loss_fn='cross_entropy')

    # ========== 3. 损失函数对比（关键几组）==========
    # DT: gini vs entropy
    run_combo("L01", "DT criterion=entropy | clahe+median",
              model_name='decision_tree', optimization_strategy='pretrained',
              dt_criterion='entropy')
    # SVM: rbf vs linear vs poly
    run_combo("L02", "SVM kernel=linear | clahe+median",
              model_name='svm', optimization_strategy='pretrained',
              svm_kernel='linear')
    run_combo("L03", "SVM kernel=poly | clahe+median",
              model_name='svm', optimization_strategy='pretrained',
              svm_kernel='poly')
    # RF: gini vs entropy (P04已有gini)
    run_combo("L04", "RF criterion=entropy | clahe+median",
              model_name='random_forest', optimization_strategy='pretrained',
              rf_criterion='entropy')
    # LR: l2(P04) vs l1 vs elasticnet
    run_combo("L05", "LR penalty=l1 | clahe+median",
              model_name='logistic_regression', optimization_strategy='pretrained',
              lr_penalty='l1', lr_solver='liblinear')
    # XGB: binary:logistic(P04) vs binary:hinge
    run_combo("L06", "XGB objective=binary:hinge | clahe+median",
              model_name='xgboost', optimization_strategy='pretrained',
              xgb_objective='binary:hinge')
    # CNN: CE(M07) vs Focal γ=2 vs Focal γ=3 vs LabelSmoothing vs Dice
    run_combo("L07", "CNN Focal γ=2 pretrained | clahe+median",
              model_name='cnn', optimization_strategy='pretrained',
              cnn_loss_fn='focal', cnn_focal_alpha='None', cnn_focal_gamma=2.0)
    run_combo("L08", "CNN Focal γ=3 pretrained | clahe+median",
              model_name='cnn', optimization_strategy='pretrained',
              cnn_loss_fn='focal', cnn_focal_alpha='None', cnn_focal_gamma=3.0)
    run_combo("L09", "CNN LabelSmoothing pretrained | clahe+median",
              model_name='cnn', optimization_strategy='pretrained',
              cnn_loss_fn='label_smoothing', cnn_label_smoothing_epsilon=0.1)
    run_combo("L10", "CNN Dice pretrained | clahe+median",
              model_name='cnn', optimization_strategy='pretrained',
              cnn_loss_fn='dice')

    # ========== 4. 参数优化策略对比（4组，固定RF）==========
    # P04 = RF pretrained (已跑，复用)
    run_combo("O01", "RF manual | clahe+median",
              model_name='random_forest', optimization_strategy='manual')
    run_combo("O02", "RF grid_search | clahe+median",
              model_name='random_forest', optimization_strategy='grid_search',
              cv_folds_opt=3, scoring_metric='f1')
    run_combo("O03", "RF random_search | clahe+median",
              model_name='random_forest', optimization_strategy='random_search',
              cv_folds_opt=3, n_iter=10, scoring_metric='f1')

    # ========== 5. 模型验证方法对比（2组）==========
    # holdout (P04已有)
    run_combo("V01", "RF pretrained kfold | clahe+median",
              model_name='random_forest', optimization_strategy='pretrained',
              validation_method='kfold', cv_folds_opt=5)

    # ========== 6. 无监督方法（4组）==========
    run_combo("U01", "KMeans | clahe+median",
              model_name='kmeans', optimization_strategy='pretrained',
              unsup_n_clusters=2)
    run_combo("U02", "GMM | clahe+median",
              model_name='gmm', optimization_strategy='pretrained',
              unsup_n_clusters=2)
    run_combo("U03", "Agglomerative ward | clahe+median",
              model_name='agglomerative', optimization_strategy='manual',
              unsup_n_clusters=2, agg_linkage='ward')
    run_combo("U04", "Spectral rbf | clahe+median",
              model_name='spectral', optimization_strategy='manual',
              unsup_n_clusters=2, spec_affinity='rbf')

    # ========== 7. 边缘情况：差性能组合 ==========
    run_combo("E01", "NB + none预处理 (预期差)",
              model_name='naive_bayes', optimization_strategy='pretrained',
              preprocessing='none')
    run_combo("E02", "LR elasticnet (saga) | clahe+median",
              model_name='logistic_regression', optimization_strategy='pretrained',
              lr_penalty='elasticnet', lr_solver='saga', lr_l1_ratio=0.5)
    run_combo("E03", "LGBM cross_entropy | clahe+median",
              model_name='lightgbm', optimization_strategy='pretrained',
              lgbm_objective='cross_entropy')

    # ========== 保存结果 ==========
    total_time = time.time() - t_start
    print(f"\n\n{'='*60}")
    print(f"全部完成! 总耗时: {total_time:.1f}s")
    print(f"成功: {sum(1 for r in all_results if not r['error'])}")
    print(f"失败: {sum(1 for r in all_results if r['error'])}")
    print(f"{'='*60}")

    # 去除不可序列化的内容
    for r in all_results:
        r.pop('status', None)

    with open(OUTPUT_DIR / 'all_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"结果已保存到: {OUTPUT_DIR / 'all_results.json'}")
