"""Test 4 system improvements"""
import sys; sys.path.insert(0, 'src')
import matplotlib; matplotlib.use('Agg')
from gradio_app import run_pipeline

BASE = dict(
    split_method='holdout', split_ratio=0.7, use_stratify=True,
    preprocessing='clahe+median', features=['hog','lbp','glcm','edge_density'],
    max_samples=1000,
    dt_max_depth=15, dt_min_samples_split=5, svm_C=1.0, nb_var_smoothing=1e-9,
    rf_n_estimators=100, rf_max_depth=20, rf_min_samples_split=5, lr_C=1.0,
    xgb_n_estimators=100, xgb_max_depth=6, xgb_subsample=0.8,
    lgbm_n_estimators=100, lgbm_max_depth=6, lgbm_num_leaves=31,
    cnn_dropout=0.5, cnn_batch_size=64, cnn_epochs=5, cnn_early_stopping=3,
    cnn_input_size=128, cnn_weight_decay=1e-4,
    unsup_n_clusters=2, unsup_eps=0.5, unsup_min_samples=5,
    dt_criterion='gini', svm_kernel='rbf', svm_gamma='scale', rf_criterion='gini',
    lr_penalty='l2', lr_solver='lbfgs', lr_l1_ratio=0.5,
    xgb_objective='binary:logistic', xgb_learning_rate=0.1,
    lgbm_objective='binary', lgbm_learning_rate=0.1,
    cnn_loss_fn='cross_entropy', cnn_focal_alpha='None', cnn_focal_gamma=2.0,
    cnn_label_smoothing_epsilon=0.1, cnn_optimizer='adam', cnn_learning_rate=0.001,
    kmeans_algorithm='lloyd', gmm_covariance_type='full',
    agg_linkage='ward', spec_affinity='rbf',
    optimization_strategy='pretrained', cv_folds_opt=3, n_iter=10,
    validation_method='holdout', scoring_metric='f1',
    unsup_val_method='internal_external', random_seed=42,
)

results = []

# Test 1: KFold std output - check for plus-minus (± U+00B1) in metrics md
print('Running Test 1: KFold...')
r = run_pipeline(**{**BASE, 'model_name': 'random_forest', 'validation_method': 'kfold'})
_, md1, *_ = r
# Check if kfold summary uses mean+std format (has more complex decimal pattern than raw)
t1 = '±' in md1
results.append(('KFold std (plus-minus sign)', t1, ''))

# Test 2: Binary Hinge N/A - use MANUAL mode so hinge really takes effect
print('Running Test 2: Hinge...')
r2 = run_pipeline(**{**BASE, 'model_name': 'xgboost', 'xgb_objective': 'binary:hinge',
                     'optimization_strategy': 'manual'})
s2, _, _, roc2, pr2, _, prob2, _, _ = r2
t2a = 'binary:hinge' in s2  # warning present
t2b = roc2 is not None and pr2 is not None and prob2 is not None  # N/A figs
results.append(('Hinge warning', t2a, ''))
results.append(('Hinge N/A figs (roc/pr/prob not None)', t2b,
               f'roc={roc2 is not None} pr={pr2 is not None} prob={prob2 is not None}'))

# Test 3: Unsupervised warning
print('Running Test 3: Unsupervised...')
r3 = run_pipeline(**{**BASE, 'model_name': 'kmeans'})
s3 = r3[0]
t3 = '聚类质量极低' in s3  # 聚类质量极低
results.append(('Unsup quality warning', t3, ''))

# Test 4: Import OK
results.append(('No syntax/import errors', True, ''))

# Write results
passed = all(r[1] for r in results)
with open('scripts/test_result.txt', 'w', encoding='utf-8') as f:
    for name, ok, detail in results:
        line = f'[{"PASS" if ok else "FAIL"}] {name} {detail}'
        f.write(line + '\n')
        print(line)
    summary = 'ALL PASSED' if passed else 'SOME FAILED'
    f.write(f'\n{summary}\n')
    print(f'\n{summary}')
    f.write(f'exit_code={"0" if passed else "1"}')
