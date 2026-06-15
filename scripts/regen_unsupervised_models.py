"""
重新生成无监督预训练模型，使用 Gradio UI 默认特征管道。

解决 BUG-002/003: 旧模型使用 4120 维特征训练，与当前特征管道不兼容。
重新训练使用 UI 默认的全部 4 种特征 (HOG+LBP+GLCM+edge_density)。

注意: 预训练模型仅与特定特征维度兼容。当用户在 UI 中选择不同特征集时，
BUG-002 修复的 graceful fallback 会自动回退到 manual 模式。

用法:
    conda activate bjtu_ml
    python scripts/regen_unsupervised_models.py
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "src"))

import numpy as np
import joblib
from pathlib import Path

# 从 gradio_app 导入数据管线和路径常量
from gradio_app import prepare_data, UNSUP_DIR, PROJECT_ROOT

SEED = 42
MAX_SAMPLES = 400  # 与集成测试一致 (200 正 + 200 负)


def main():
    feat_dim_expected = None  # 不设硬性断言，由 UI 默认特征决定

    print("=" * 60)
    print("重新生成无监督预训练模型 (UI 默认特征)")
    print("=" * 60)

    # 1. 准备数据 — 使用 Gradio 界面默认配置 (HOG+LBP+GLCM+edge_density)
    data = prepare_data(
        max_samples=MAX_SAMPLES,
        random_seed=SEED,
        split_method="holdout",
        split_ratio=0.7,
        preprocessing=["clahe", "median"],
        features=["hog", "lbp", "glcm", "edge_density"],
    )

    X_all = np.vstack([data["X_train"], data["X_test"]])
    y_all = np.concatenate([data["y_train"], data["y_test"]])

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    feat_dim = X_scaled.shape[1]
    print(f"\n特征维度: {feat_dim}")
    print(f"样本数: {X_scaled.shape[0]}")

    UNSUP_DIR.mkdir(parents=True, exist_ok=True)

    # 2. KMeans
    print("\n--- 训练 KMeans ---")
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=2, random_state=SEED, n_init="auto", algorithm="lloyd")
    labels_km = km.fit_predict(X_scaled)
    joblib.dump(km, UNSUP_DIR / "kmeans_best.joblib")
    print(f"  KMeans 簇分布: {dict(zip(*np.unique(labels_km, return_counts=True)))}")
    print(f"  保存到: {UNSUP_DIR / 'kmeans_best.joblib'}")

    # 3. GMM (使用 diag 协方差以避免 26352 维下 full/tied 协方差矩阵过大)
    print("\n--- 训练 GMM (diag covariance) ---")
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=2, covariance_type="diag", random_state=SEED)
    labels_gmm = gmm.fit_predict(X_scaled)
    joblib.dump(gmm, UNSUP_DIR / "gmm_best.joblib")
    print(f"  GMM 簇分布: {dict(zip(*np.unique(labels_gmm, return_counts=True)))}")
    print(f"  GMM n_features_in_: {gmm.n_features_in_}")
    print(f"  保存到: {UNSUP_DIR / 'gmm_best.joblib'}")

    # 4. Agglomerative
    print("\n--- 训练 Agglomerative ---")
    from sklearn.cluster import AgglomerativeClustering
    agg = AgglomerativeClustering(n_clusters=2, linkage="ward")
    labels_agg = agg.fit_predict(X_scaled)
    joblib.dump(agg, UNSUP_DIR / "agglomerative_best.joblib")
    print(f"  Agglomerative 簇分布: {dict(zip(*np.unique(labels_agg, return_counts=True)))}")
    print(f"  保存到: {UNSUP_DIR / 'agglomerative_best.joblib'}")

    # 5. Spectral
    print("\n--- 训练 Spectral ---")
    from sklearn.cluster import SpectralClustering
    spec = SpectralClustering(n_clusters=2, affinity="rbf", random_state=SEED, n_init=10)
    labels_spec = spec.fit_predict(X_scaled)
    joblib.dump(spec, UNSUP_DIR / "spectral_best.joblib")
    print(f"  Spectral 簇分布: {dict(zip(*np.unique(labels_spec, return_counts=True)))}")
    print(f"  保存到: {UNSUP_DIR / 'spectral_best.joblib'}")

    # 6. 验证
    print("\n--- 验证模型维度 ---")
    for name in ["kmeans", "gmm", "agglomerative", "spectral"]:
        # 安全说明: 加载本脚本刚保存的模型，来源完全可信。
        m = joblib.load(UNSUP_DIR / f"{name}_best.joblib")
        if hasattr(m, "predict"):
            test_pred = m.predict(X_scaled[:5])
            print(f"  {name}: predict OK, shape={test_pred.shape}")
        elif hasattr(m, "labels_"):
            print(f"  {name}: labels_ shape={m.labels_.shape} (无 predict)")
        else:
            print(f"  {name}: fit_predict only")

    print("\n[DONE] All unsupervised pretrained models regenerated.")


if __name__ == "__main__":
    main()
