#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
无监督学习补充数据生成：CH 分数 + PCA 降维
=============================================
为 PPT 无监督部分补全 Calinski-Harabasz (CH) 指数，并新增 PCA 数据降维
相关数据，供 generate_ppt_charts.py 绘制 F13-F16 图表。

【口径】与 _pretrain_all.py 第 600-735 行完全同源，保证与现有 F6 图
(unsupervised_comparison.csv) 的 silhouette/ari/nmi 可对比：
  - 数据：load_dataset(DATA_ROOT, per_class=1000) → 2000 样本
  - 特征：extract_features_reduced（降维HOG + LBP + GLCM + 边缘密度）
  - 标准化：StandardScaler
  - 聚类：K-Means / GMM / Agglomerative(ward) / Spectral(rbf)，random_state=42
  - 评估：eval_clustering（含 silhouette/DB/CH/ARI/NMI，对噪声点做 mask）

【输出】
  outputs/results/unsupervised_comparison_full.csv   原始特征空间 4 方法全指标(补 CH/DB)
  outputs/models/unsupervised/pca_analysis.npz       PCA 2D/方差/降维数据 + 降维后聚类指标
  outputs/results/unsupervised_pca_comparison.csv    原始 vs PCA降维 的 silhouette/CH/ARI

运行：
    conda activate bjtu_ml
    python scripts/gen_unsup_ch_pca_data.py
"""
import os
import sys
import json
import warnings
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import cv2
import joblib
from dotenv import load_dotenv
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import (
    KMeans, AgglomerativeClustering, SpectralClustering,
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score,
)
from sklearn.decomposition import PCA

# ------------------------------------------------------------------
# 路径
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
UNSUP_DIR = MODELS_DIR / "unsupervised"
RESULTS_DIR = OUTPUT_DIR / "results"
for d in (UNSUP_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_ROOT / ".env")
_DR = os.getenv("CRACK_DATA_ROOT")
DATA_ROOT = (Path(_DR).expanduser() if _DR else PROJECT_ROOT / "data").resolve()
SEED = 42
PER_CLASS = 1000  # 与 _pretrain_all.py UNSUP_SAMPLES 一致 → 2000 样本
PCA_VAR_TARGET = 0.90  # PCA 降维保留 90% 方差

print(f"DATA_ROOT: {DATA_ROOT}")
print(f"样本: {PER_CLASS * 2} (per_class={PER_CLASS})")

# ------------------------------------------------------------------
# 数据加载（与 _pretrain_all.py:89-113 一致）
# ------------------------------------------------------------------
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _imread_gray(path):
    buf = np.fromfile(str(path), dtype=np.uint8)
    if buf is None or buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


def load_dataset(data_root, per_class=None):
    def _load_dir(directory, label):
        imgs, lbls = [], []
        for p in sorted(directory.iterdir())[:per_class]:
            if p.suffix.lower() in IMAGE_EXTS:
                img = _imread_gray(p)
                if img is not None:
                    imgs.append(img)
                    lbls.append(label)
        return imgs, lbls

    pos_imgs, pos_lbls = _load_dir(data_root / "Positive", 1)
    neg_imgs, neg_lbls = _load_dir(data_root / "Negative", 0)
    all_imgs = pos_imgs + neg_imgs
    labels = np.array(pos_lbls + neg_lbls, dtype=np.int64)
    shapes = {img.shape for img in all_imgs}
    if len(shapes) == 1:
        images = np.stack(all_imgs)
    else:
        images = np.array(all_imgs, dtype=object)
    return images, labels


# ------------------------------------------------------------------
# 特征提取（与 _pretrain_all.py:607-624 一致：降维HOG + LBP + GLCM + 边缘）
# ------------------------------------------------------------------
def extract_features_reduced(img):
    img_u8 = img.astype(np.uint8) if img.dtype != np.uint8 else img
    # 降维 HOG
    h = hog(img_u8, orientations=6, pixels_per_cell=(16, 16),
            cells_per_block=(2, 2), feature_vector=True)
    # LBP
    n_bins = 8 * 7 + 3
    lbp_img = local_binary_pattern(img_u8, 8, 1, method="uniform")
    lbp_hist, _ = np.histogram(lbp_img, bins=n_bins, range=(0, n_bins), density=True)
    # GLCM（降维：1 距离 / 0 角度）
    glcm = graycomatrix(img_u8, distances=[1], angles=[0],
                        levels=256, symmetric=True, normed=True)
    glcm_feats = [graycoprops(glcm, p)[0, 0] for p in
                  ["contrast", "correlation", "energy", "homogeneity"]]
    # 边缘密度
    edges = cv2.Canny(img_u8, 50, 150)
    edge_den = float(np.count_nonzero(edges)) / edges.size
    return np.concatenate([h, lbp_hist, glcm_feats, [edge_den]])


# ------------------------------------------------------------------
# 聚类评估（与 _pretrain_all.py:638-670 一致，含 CH/DB 与噪声 mask）
# ------------------------------------------------------------------
def eval_clustering(y_pred, y_true, X_data):
    unique_labels = set(y_pred)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = int(np.sum(y_pred == -1))

    mask = y_pred != -1
    if mask.sum() >= 2 and len(set(y_pred[mask])) >= 2:
        sil = silhouette_score(X_data[mask], y_pred[mask])
        db = davies_bouldin_score(X_data[mask], y_pred[mask])
        ch = calinski_harabasz_score(X_data[mask], y_pred[mask])
    else:
        sil, db, ch = float("nan"), float("nan"), float("nan")

    if n_noise > 0:
        ari = adjusted_rand_score(y_true[mask], y_pred[mask])
        nmi = normalized_mutual_info_score(y_true[mask], y_pred[mask])
    else:
        ari = adjusted_rand_score(y_true, y_pred)
        nmi = normalized_mutual_info_score(y_true, y_pred)

    return {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "silhouette": round(float(sil), 4) if not np.isnan(sil) else None,
        "davies_bouldin": round(float(db), 4) if not np.isnan(db) else None,
        "calinski_harabasz": round(float(ch), 2) if not np.isnan(ch) else None,
        "ari": round(float(ari), 4),
        "nmi": round(float(nmi), 4),
    }


# 4 种方法名（统一展示名，与现有 F6 的 _load_unsup 映射一致）
METHOD_ORDER = [
    ("K-Means", "K-Means"),
    ("GMM", "GMM"),
    ("Agglomerative", "层次聚类(ward)"),
    ("Spectral", "谱聚类(rbf)"),
]


def run_four_cluster_methods(X_scaled, y_true):
    """跑 K-Means/GMM/Agglomerative/Spectral，返回 {展示名: (labels, metrics_dict)}。"""
    out = {}
    # 1. K-Means
    km = KMeans(n_clusters=2, random_state=SEED, n_init="auto")
    labels = km.fit_predict(X_scaled)
    out["K-Means"] = (labels, eval_clustering(labels, y_true, X_scaled))
    print(f"  K-Means: {out['K-Means'][1]}")

    # 2. GMM
    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=SEED)
    labels = gmm.fit_predict(X_scaled)
    out["GMM"] = (labels, eval_clustering(labels, y_true, X_scaled))
    print(f"  GMM: {out['GMM'][1]}")

    # 3. Agglomerative(ward)
    agg = AgglomerativeClustering(n_clusters=2, linkage="ward")
    labels = agg.fit_predict(X_scaled)
    out["Agglomerative"] = (labels, eval_clustering(labels, y_true, X_scaled))
    print(f"  Agglomerative(ward): {out['Agglomerative'][1]}")

    # 4. Spectral(rbf)
    spec = SpectralClustering(n_clusters=2, affinity="rbf", random_state=SEED, n_init=10)
    labels = spec.fit_predict(X_scaled)
    out["Spectral"] = (labels, eval_clustering(labels, y_true, X_scaled))
    print(f"  Spectral(rbf): {out['Spectral'][1]}")

    return out


# ==================================================================
# 主流程
# ==================================================================
def main():
    print("\n[1/4] 加载数据...")
    images, labels = load_dataset(DATA_ROOT, per_class=PER_CLASS)
    print(f"  样本数: {len(labels)}, 正样本: {int(labels.sum())}, 负样本: {int((labels == 0).sum())}")

    print("\n[2/4] 提取降维特征（同 _pretrain_all.py 口径）...")
    X = np.stack([extract_features_reduced(img) for img in images])
    print(f"  特征维度: {X.shape[1]}")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\n[3/4] 原始特征空间聚类（补 CH/DB）...")
    orig_results = run_four_cluster_methods(X_scaled, labels)

    # ---- 输出 1：原始特征全指标 CSV ----
    rows = []
    for key, disp in METHOD_ORDER:
        m = orig_results[key][1]
        rows.append({
            "method": disp,
            "n_clusters": m["n_clusters"],
            "n_noise": m["n_noise"],
            "silhouette": m["silhouette"],
            "davies_bouldin": m["davies_bouldin"],
            "calinski_harabasz": m["calinski_harabasz"],
            "ari": m["ari"],
            "nmi": m["nmi"],
        })
    df_full = pd.DataFrame(rows)
    full_csv = RESULTS_DIR / "unsupervised_comparison_full.csv"
    df_full.to_csv(full_csv, index=False, encoding="utf-8-sig")
    print(f"\n  [已保存] {full_csv}")

    # ---- PCA 分析 ----
    print("\n[4/4] PCA 降维分析...")
    # 全量 PCA 取前 50 主成分的方差信息
    pca_full = PCA(n_components=min(50, X_scaled.shape[1], X_scaled.shape[0]),
                   random_state=SEED)
    pca_full.fit(X_scaled)
    explained_var_ratio = pca_full.explained_variance_ratio_
    explained_var_cumsum = np.cumsum(explained_var_ratio)

    # 2D 投影（可视化用）
    pca_2d = PCA(n_components=2, random_state=SEED)
    X_pca_2d = pca_2d.fit_transform(X_scaled)

    # 降到保留 90% 方差的维度
    n_keep = int(np.searchsorted(explained_var_cumsum, PCA_VAR_TARGET) + 1)
    n_keep = max(2, min(n_keep, X_scaled.shape[1]))
    pca_red = PCA(n_components=n_keep, random_state=SEED)
    X_pca_reduced = pca_red.fit_transform(X_scaled)
    var_kept = float(pca_red.explained_variance_ratio_.sum())
    print(f"  PCA 降维: {X_scaled.shape[1]} → {n_keep} 维 (保留方差 {var_kept:.2%})")

    # 降维后聚类
    print("  降维特征空间聚类...")
    red_results = run_four_cluster_methods(X_pca_reduced, labels)

    # 用层次聚类的标签作为 F14 散点图的"聚类标签"（CH 最高 / ARI 最优代表）
    best_method_key = max(
        orig_results.keys(),
        key=lambda k: orig_results[k][1]["calinski_harabasz"] or -1,
    )
    cluster_labels_for_plot = orig_results[best_method_key][0]
    print(f"  F14 散点采用聚类标签来源: {dict(METHOD_ORDER)[best_method_key]} "
          f"(CH 最高={orig_results[best_method_key][1]['calinski_harabasz']})")

    # ---- 输出 2：PCA 分析 npz ----
    npz_path = UNSUP_DIR / "pca_analysis.npz"
    np.savez(
        npz_path,
        X_pca_2d=X_pca_2d,
        true_labels=labels,
        cluster_labels=cluster_labels_for_plot,
        cluster_method=dict(METHOD_ORDER)[best_method_key],
        explained_var_ratio=explained_var_ratio,
        explained_var_cumsum=explained_var_cumsum,
        X_pca_reduced=X_pca_reduced,
        n_components_reduced=n_keep,
        var_kept=var_kept,
        orig_feat_dim=X_scaled.shape[1],
    )
    print(f"  [已保存] {npz_path}")

    # ---- 输出 3：原始 vs PCA 降维 聚类对比 CSV ----
    cmp_rows = []
    for key, disp in METHOD_ORDER:
        om = orig_results[key][1]
        rm = red_results[key][1]
        cmp_rows.append({
            "method": disp,
            "silhouette_orig": om["silhouette"],
            "silhouette_pca": rm["silhouette"],
            "calinski_harabasz_orig": om["calinski_harabasz"],
            "calinski_harabasz_pca": rm["calinski_harabasz"],
            "ari_orig": om["ari"],
            "ari_pca": rm["ari"],
        })
    df_cmp = pd.DataFrame(cmp_rows)
    cmp_csv = RESULTS_DIR / "unsupervised_pca_comparison.csv"
    df_cmp.to_csv(cmp_csv, index=False, encoding="utf-8-sig")
    print(f"  [已保存] {cmp_csv}")

    print("\n" + "=" * 60)
    print("DONE — 已生成:")
    print(f"  {full_csv}")
    print(f"  {npz_path}")
    print(f"  {cmp_csv}")
    print("=" * 60)


if __name__ == "__main__":
    main()
