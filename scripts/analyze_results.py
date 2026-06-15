#!/usr/bin/env python
"""分析批量验证结果并生成汇总统计。"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

BATCH_DIR = sys.argv[1] if len(sys.argv) > 1 else None

if BATCH_DIR is None:
    # Find latest batch directory
    batch_root = Path("outputs/batch_verify")
    dirs = sorted(batch_root.glob("20*"))
    BATCH_DIR = str(dirs[-1]) if dirs else None
    if BATCH_DIR is None:
        print("ERROR: No batch directory found")
        sys.exit(1)

print(f"Analyzing: {BATCH_DIR}")

# Load results
csv_path = Path(BATCH_DIR) / "results.csv"
df = pd.read_csv(csv_path)
print(f"Total runs: {len(df)}")

# Filter out errors
error_mask = df["error"].notna() & (df["error"] != "") & (df["error"] != "nan")
valid = df[~error_mask].copy()
print(f"Successful: {len(valid)}")
print(f"Failed: {error_mask.sum()}")

# =========================================
# 1. Data Processing Comparison
# =========================================
print("\n" + "="*70)
print("2. DATA PROCESSING COMPARISON")
print("="*70)

# Group by preprocessing for supervised models (pretrained, default step4)
trad = valid[valid["category"] == "traditional"]
trad_pretrained = trad[trad["optimization"] == "pretrained"]

# Get baseline step4 for each model
default_step4 = {
    "decision_tree": "gini", "svm": "rbf", "naive_bayes": None,
    "random_forest": "gini", "logistic_regression": "l2",
    "xgboost": "binary:logistic", "lightgbm": "binary",
}

for model in ["decision_tree", "svm", "naive_bayes", "random_forest",
              "logistic_regression", "xgboost", "lightgbm"]:
    mdf = trad_pretrained[trad_pretrained["model"] == model]
    if len(mdf) == 0:
        continue
    ds4 = default_step4[model]
    if ds4:
        mdf = mdf[mdf["step4_value"] == ds4]

    print(f"\n{model}:")
    for prep in ["none", "clahe", "gaussian", "median", "clahe+gaussian", "clahe+median"]:
        pdf = mdf[mdf["preprocessing"] == prep]
        if len(pdf) > 0 and "f1" in pdf.columns:
            f1_val = pdf["f1"].mean()
            print(f"  {prep:20s}: F1={f1_val:.4f}")

# Average across all supervised models per preprocessing
print("\n--- Average across supervised models ---")
for prep in ["none", "clahe", "gaussian", "median", "clahe+gaussian", "clahe+median"]:
    pdf = trad_pretrained[trad_pretrained["preprocessing"] == prep]
    if len(pdf) > 0 and "f1" in pdf.columns:
        avg_f1 = pdf["f1"].mean()
        print(f"  {prep:20s}: avg F1={avg_f1:.4f} (n={len(pdf)})")

# =========================================
# 2. Model Comparison
# =========================================
print("\n" + "="*70)
print("3. MODEL COMPARISON")
print("="*70)

baseline_prep = "clahe+median"

# Supervised models on baseline
print(f"\nSupervised models (prep={baseline_prep}, pretrained, default step4):")
sup_baseline = valid[
    (valid["preprocessing"] == baseline_prep) &
    (valid["optimization"] == "pretrained") &
    (valid["category"].isin(["traditional", "cnn"]))
]

for _, row in sup_baseline.sort_values("f1", ascending=False).iterrows():
    print(f"  {row['model']:25s} | F1={row.get('f1', 'N/A'):.4f} "
          f"| Acc={row.get('accuracy', 'N/A'):.4f} "
          f"| AUC={row.get('roc_auc', 'N/A'):.4f} "
          f"| tag={row.get('tag', '')}")

# Unsupervised methods
print(f"\nUnsupervised methods (prep={baseline_prep}, pretrained):")
unsup = valid[(valid["category"] == "unsupervised") & (valid["preprocessing"] == baseline_prep)]
for _, row in unsup.iterrows():
    sil = row.get("silhouette", row.get("sil", "N/A"))
    ari = row.get("ari", "N/A")
    print(f"  {row['model']:25s} | Sil={sil} | ARI={ari}")

# =========================================
# 3. Loss Function Comparison
# =========================================
print("\n" + "="*70)
print("4. LOSS FUNCTION COMPARISON")
print("="*70)

# CNN loss comparison
cnn = valid[(valid["model"] == "cnn") & (valid["preprocessing"] == baseline_prep) &
            (valid["optimization"] == "pretrained")]
print("\nCNN loss functions (baseline, pretrained, adam):")
cnn_adam = cnn[cnn.get("optimizer", cnn.get("tag", "")).str.contains("adam", na=False) | cnn["tag"].str.contains("adam", na=False)]
if len(cnn_adam) == 0:
    cnn_adam = cnn
for _, row in cnn_adam.sort_values("f1", ascending=False).iterrows():
    print(f"  {row.get('loss_fn', row.get('tag', '')):40s} | F1={row.get('f1', 'N/A'):.4f} "
          f"| Acc={row.get('accuracy', 'N/A'):.4f} | AUC={row.get('roc_auc', 'N/A'):.4f}")

# Traditional model step4 comparison
print("\nTraditional model loss/objective comparison (baseline, pretrained):")
for model in ["decision_tree", "svm", "random_forest", "logistic_regression",
              "xgboost", "lightgbm"]:
    mdf = trad_pretrained[(trad_pretrained["model"] == model) &
                          (trad_pretrained["preprocessing"] == baseline_prep)]
    step4_vals = mdf["step4_value"].unique()
    if len(step4_vals) > 1:
        print(f"\n  {model}:")
        for _, row in mdf.sort_values("f1", ascending=False).iterrows():
            print(f"    {row['step4_value']:25s} | F1={row.get('f1', 'N/A'):.4f}")

# =========================================
# 4. Optimization Strategy Comparison
# =========================================
print("\n" + "="*70)
print("5. OPTIMIZATION STRATEGY COMPARISON")
print("="*70)

for model in ["decision_tree", "random_forest", "xgboost"]:
    mdf = valid[(valid["model"] == model) & (valid["preprocessing"] == baseline_prep)]
    print(f"\n{model}:")
    for opt in ["pretrained", "manual", "grid_search", "random_search"]:
        odf = mdf[mdf["optimization"] == opt]
        if len(odf) > 0:
            f1_mean = odf["f1"].mean() if "f1" in odf.columns else "N/A"
            elapsed = odf["elapsed_sec"].mean()
            print(f"  {opt:20s} | F1={f1_mean} | time={elapsed:.0f}s (n={len(odf)})")

# =========================================
# 5. Validation Method Comparison
# =========================================
print("\n" + "="*70)
print("6. VALIDATION METHOD COMPARISON")
print("="*70)

for model in ["decision_tree", "svm", "naive_bayes", "random_forest",
              "logistic_regression", "xgboost", "lightgbm"]:
    mdf = valid[(valid["model"] == model) & (valid["preprocessing"] == baseline_prep) &
                (valid["optimization"] == "pretrained")]
    holdout_df = mdf[mdf["validation"] == "holdout"]
    kfold_df = mdf[mdf["validation"] == "kfold"]
    if len(holdout_df) > 0 and len(kfold_df) > 0:
        h_f1 = holdout_df["f1"].mean()
        k_f1 = kfold_df["f1"].mean()
        print(f"  {model:25s} | Holdout F1={h_f1:.4f} | KFold F1={k_f1:.4f} | Δ={k_f1-h_f1:+.4f}")

# =========================================
# 6. Best Combinations
# =========================================
print("\n" + "="*70)
print("7. BEST COMBINATIONS")
print("="*70)

# Best supervised
print("\nTop 10 supervised (F1):")
sup = valid[valid["category"].isin(["traditional", "cnn"])]
for _, row in sup.sort_values("f1", ascending=False).head(10).iterrows():
    print(f"  {row['model']:20s} | {row['tag']:40s} | F1={row['f1']:.4f}")

# Best unsupervised
print("\nBest unsupervised (Silhouette):")
unsup_valid = valid[valid["category"] == "unsupervised"]
for _, row in unsup_valid.sort_values("silhouette", ascending=False).head(5).iterrows():
    print(f"  {row['model']:20s} | {row['tag']:40s} | Sil={row.get('silhouette', 'N/A')} "
          f"| ARI={row.get('ari', 'N/A')}")

# Worst combinations (for comparison)
print("\nWorst supervised (F1):")
for _, row in sup.sort_values("f1", ascending=True).head(5).iterrows():
    print(f"  {row['model']:20s} | {row['tag']:40s} | F1={row['f1']:.4f}")

print("\n✅ Analysis complete!")
