"""Count combinations from batch_verify_all_combinations.py."""
import sys
sys.path.insert(0, 'src')

# Extract the combo generation functions
with open('scripts/batch_verify_all_combinations.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Split before main() and exec only the function defs
exec_code = content.split("def main()")[0]
exec(exec_code)

trad_models = {
    'decision_tree': ('criterion', ['gini', 'entropy', 'log_loss'], 'criterion'),
    'svm': ('kernel', ['rbf', 'linear', 'poly'], 'kernel'),
    'naive_bayes': (None, [None], 'none'),
    'random_forest': ('criterion', ['gini', 'entropy', 'log_loss'], 'criterion'),
    'logistic_regression': ('penalty', ['l2', 'l1', 'elasticnet'], 'penalty'),
    'xgboost': ('objective', ['binary:logistic', 'binary:hinge'], 'objective'),
    'lightgbm': ('objective', ['binary', 'cross_entropy'], 'objective'),
}

total = 0
for m, (s4_key, s4_opts, s4_label) in trad_models.items():
    combos = generate_traditional_combos(m, s4_key, s4_opts, s4_label)
    total += len(combos)
    print(f"  {m}: {len(combos)}")

cnn_combos = generate_cnn_combos()
total += len(cnn_combos)
print(f"  cnn: {len(cnn_combos)}")

unsup_models = {
    'kmeans': ('algorithm', ['lloyd', 'elkan'], 'algorithm'),
    'gmm': ('covariance_type', ['full', 'tied', 'diag', 'spherical'], 'cov_type'),
    'dbscan': (None, [None], 'none'),
    'agglomerative': ('linkage', ['ward', 'complete', 'average', 'single'], 'linkage'),
    'spectral': ('affinity', ['rbf', 'nearest_neighbors'], 'affinity'),
}
for m, (s4_key, s4_opts, s4_label) in unsup_models.items():
    combos = generate_unsupervised_combos(m, s4_key, s4_opts, s4_label)
    total += len(combos)
    print(f"  {m}: {len(combos)}")

print(f"\nTotal: {total} combinations")

# Count by category
all_combos = []
for m, (s4_key, s4_opts, s4_label) in trad_models.items():
    all_combos.extend(generate_traditional_combos(m, s4_key, s4_opts, s4_label))
all_combos.extend(generate_cnn_combos())
for m, (s4_key, s4_opts, s4_label) in unsup_models.items():
    all_combos.extend(generate_unsupervised_combos(m, s4_key, s4_opts, s4_label))

fast = [c for c in all_combos if c["optimization"] in ("pretrained", "manual")]
slow = [c for c in all_combos if c["optimization"] in ("grid_search", "random_search")]
print(f"Fast (pretrained/manual): {len(fast)}")
print(f"Slow (grid/random): {len(slow)}")

# By model
from collections import Counter
model_counts = Counter(c["model"] for c in all_combos)
print("\nBy model:")
for model, count in model_counts.most_common():
    print(f"  {model}: {count}")
