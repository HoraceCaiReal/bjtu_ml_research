#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GMM 降维后 ARI=0.91 现象的诊断证据生成
========================================
F16 发现 GMM 在 PCA 降维后 ARI 暴涨至 0.91（K-Means 同条件下仍为 0.10）。
本脚本生成三组证据数据，供 draw_F17 绘制综合证据图，严谨论证该结果可信：

  证据1：GMM 多种子稳定性（10 个 seed，证明非偶然局部最优）
  证据2：PCA 维度-ARI 曲线（GMM vs K-Means，证明信息在 3-10 主成分）
  证据3：GMM 协方差类型对比（full/diag/spherical/tied，证明椭球簇假设）

数据源：outputs/models/unsupervised/pca_analysis.npz（gen_unsup_ch_pca_data.py 产出）

【输出】
  outputs/results/gmm_evidence.npz   三组诊断数据
"""
import sys
import warnings
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

PROJECT = Path(__file__).resolve().parent.parent
UNSUP_DIR = PROJECT / "outputs" / "models" / "unsupervised"
RESULTS_DIR = PROJECT / "outputs" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("加载 PCA 降维数据...")
d = np.load(UNSUP_DIR / "pca_analysis.npz", allow_pickle=True)
X_red = d["X_pca_reduced"]          # (2000, 51) PCA 降维后
y = d["true_labels"].astype(int)    # 真实标签
print(f"  X_red{X_red.shape}, 标签平衡: 有裂纹={int(y.sum())}, 无裂纹={int((y==0).sum())}")

# ------------------------------------------------------------------
# 证据1：GMM 多种子稳定性
# ------------------------------------------------------------------
print("\n[证据1] GMM(full) 多种子稳定性...")
N_SEEDS = 10
seed_aris = []
for seed in range(N_SEEDS):
    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=seed)
    lab = gmm.fit_predict(X_red)
    seed_aris.append(adjusted_rand_score(y, lab))
seed_aris = np.array(seed_aris)
print(f"  {N_SEEDS} 个 seed: ARI 均值={seed_aris.mean():.4f}, "
      f"标准差={seed_aris.std():.4f}, 范围=[{seed_aris.min():.4f}, {seed_aris.max():.4f}]")

# ------------------------------------------------------------------
# 证据2：PCA 维度-ARI 曲线
# ------------------------------------------------------------------
print("\n[证据2] PCA 维度-ARI 曲线 (GMM vs K-Means)...")
dims = [2, 5, 10, 20, 30, 51]
gmm_dim_aris, km_dim_aris = [], []
for k in dims:
    Xk = X_red[:, :k]  # X_red 已是按方差降序的主成分
    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=42).fit(Xk)
    km = KMeans(n_clusters=2, random_state=42, n_init="auto").fit(Xk)
    gmm_dim_aris.append(adjusted_rand_score(y, gmm.predict(Xk)))
    km_dim_aris.append(adjusted_rand_score(y, km.predict(Xk)))
    print(f"  维度={k:>3}: GMM ARI={gmm_dim_aris[-1]:.4f}, KMeans ARI={km_dim_aris[-1]:.4f}")

# ------------------------------------------------------------------
# 证据3：GMM 协方差类型对比
# ------------------------------------------------------------------
print("\n[证据3] GMM 协方差类型对比 (seed=42, 51维)...")
cov_types = ["full", "tied", "diag", "spherical"]
cov_aris = []
for cov in cov_types:
    gmm = GaussianMixture(n_components=2, covariance_type=cov, random_state=42)
    lab = gmm.fit_predict(X_red)
    cov_aris.append(adjusted_rand_score(y, lab))
    print(f"  {cov:10s}: ARI={cov_aris[-1]:.4f}")

# ------------------------------------------------------------------
# 保存
# ------------------------------------------------------------------
out_path = RESULTS_DIR / "gmm_evidence.npz"
np.savez(
    out_path,
    # 证据1
    seed_aris=seed_aris,
    n_seeds=N_SEEDS,
    # 证据2
    pca_dims=np.array(dims),
    gmm_dim_aris=np.array(gmm_dim_aris),
    km_dim_aris=np.array(km_dim_aris),
    # 证据3
    cov_types=np.array(cov_types),
    cov_aris=np.array(cov_aris),
)
print(f"\n[已保存] {out_path}")
print("DONE")
