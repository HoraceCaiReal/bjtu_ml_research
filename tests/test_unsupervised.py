"""无监督学习模块单元测试。"""

import numpy as np

from src.models.unsupervised import (
    AgglomerativeClusterer,
    DBSCANClusterer,
    GMMClusterer,
    KMeansClusterer,
    SpectralClusterer,
    UnsupervisedPipeline,
)


def _sample_feature_matrix():
    """构造两个明显分离的簇，方便验证聚类结果。"""
    rng = np.random.default_rng(42)
    # 簇 0：中心 (-5, -5)
    cluster_0 = rng.normal(loc=-5.0, scale=0.5, size=(15, 2))
    # 簇 1：中心 (5, 5)
    cluster_1 = rng.normal(loc=5.0, scale=0.5, size=(15, 2))
    X = np.vstack([cluster_0, cluster_1])
    y = np.array([0] * 15 + [1] * 15, dtype=np.int64)
    return X, y


def test_kmeans_fit_predict():
    """验证 K-Means 聚类器可训练并预测。"""
    X, _ = _sample_feature_matrix()
    model = KMeansClusterer(n_clusters=2).fit(X)

    labels = model.labels_
    pred = model.predict(X)
    centers = model.get_centers()

    assert labels.shape == (30,)
    assert pred.shape == (30,)
    assert centers.shape == (2, 2)
    # 两个簇的样本数应该大致均衡
    assert 10 <= np.sum(labels == 0) <= 20
    assert 10 <= np.sum(labels == 1) <= 20


def test_kmeans_raises_before_fit():
    """验证未训练时调用方法抛出异常。"""
    model = KMeansClusterer(n_clusters=2)
    with np.testing.assert_raises(RuntimeError):
        model.predict(np.random.rand(10, 2))
    with np.testing.assert_raises(RuntimeError):
        model.get_centers()


def test_gmm_fit_predict_proba():
    """验证 GMM 聚类器可训练、预测硬标签和软概率。"""
    X, _ = _sample_feature_matrix()
    model = GMMClusterer(n_components=2).fit(X)

    labels = model.labels_
    pred = model.predict(X)
    proba = model.predict_proba(X)

    assert labels.shape == (30,)
    assert pred.shape == (30,)
    assert proba.shape == (30, 2)
    # 每行概率和为 1
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_gmm_raises_before_fit():
    """验证未训练时调用方法抛出异常。"""
    model = GMMClusterer(n_components=2)
    with np.testing.assert_raises(RuntimeError):
        model.predict(np.random.rand(10, 2))
    with np.testing.assert_raises(RuntimeError):
        model.predict_proba(np.random.rand(10, 2))


def test_dbscan_fit():
    """验证 DBSCAN 聚类器可训练并检测噪声。"""
    X, _ = _sample_feature_matrix()
    # 用较大 eps 确保能发现两个簇
    model = DBSCANClusterer(eps=2.0, min_samples=3).fit(X)

    labels = model.labels_
    pred = model.predict(X)

    assert labels.shape == (30,)
    assert pred.shape == (30,)
    # 两个分离的簇，噪声点应该很少
    assert model.n_noise_ <= 5


def test_dbscan_noise_detection():
    """验证 DBSCAN 能将离群点标记为噪声。"""
    rng = np.random.default_rng(42)
    # 密集簇
    cluster_0 = rng.normal(loc=0.0, scale=0.1, size=(10, 2))
    cluster_1 = rng.normal(loc=5.0, scale=0.1, size=(10, 2))
    # 远距离噪声点
    noise = np.array([[100.0, 100.0], [-100.0, -100.0]])
    X = np.vstack([cluster_0, cluster_1, noise])

    model = DBSCANClusterer(eps=0.5, min_samples=3).fit(X)

    assert model.n_noise_ >= 2  # 两个极端噪声点
    assert model.labels_[-1] == -1
    assert model.labels_[-2] == -1


def test_dbscan_raises_before_fit():
    """验证未训练时调用 predict 抛出异常。"""
    model = DBSCANClusterer(eps=0.5)
    with np.testing.assert_raises(RuntimeError):
        model.predict(np.random.rand(10, 2))


def test_agglomerative_fit():
    """验证 Agglomerative 聚类器可训练。"""
    X, _ = _sample_feature_matrix()
    model = AgglomerativeClusterer(n_clusters=2, linkage="ward").fit(X)

    labels = model.labels_

    assert labels.shape == (30,)
    assert np.unique(labels).size == 2


def test_agglomerative_different_linkages():
    """验证不同链接方式均可运行。"""
    X, _ = _sample_feature_matrix()
    for linkage in ("ward", "complete", "average", "single"):
        model = AgglomerativeClusterer(n_clusters=2, linkage=linkage).fit(X)
        assert model.labels_.shape == (30,)
        assert np.unique(model.labels_).size == 2


def test_spectral_fit():
    """验证 Spectral 聚类器可训练。"""
    X, _ = _sample_feature_matrix()
    model = SpectralClusterer(n_clusters=2).fit(X)

    labels = model.labels_

    assert labels.shape == (30,)
    assert np.unique(labels).size == 2


def test_pipeline_run_all():
    """验证统一 pipeline 可运行所有方法并返回正确结构。"""
    X, y = _sample_feature_matrix()
    pipeline = UnsupervisedPipeline()
    results = pipeline.run_all(X, y_true=y)

    expected_methods = {"kmeans", "gmm", "dbscan", "agglomerative", "spectral"}
    assert set(results) == expected_methods

    for name, result in results.items():
        assert "labels" in result
        assert "metrics" in result
        assert "n_clusters" in result
        assert result["labels"].shape == (30,)

        metrics = result["metrics"]
        # 内部指标
        assert "silhouette" in metrics
        assert "davies_bouldin" in metrics
        assert "calinski_harabasz" in metrics
        # 外部指标（传入了 y_true）
        assert "ari" in metrics
        assert "nmi" in metrics


def test_pipeline_without_labels():
    """验证不传入 y_true 时 pipeline 不计算外部指标。"""
    X, _ = _sample_feature_matrix()
    pipeline = UnsupervisedPipeline()
    results = pipeline.run_all(X)

    for result in results.values():
        metrics = result["metrics"]
        assert "silhouette" in metrics
        # 无 y_true 时不计算外部指标
        assert "ari" not in metrics
        assert "nmi" not in metrics


def test_pipeline_summary():
    """验证 summary() 方法输出字符串。"""
    X, _ = _sample_feature_matrix()
    pipeline = UnsupervisedPipeline()
    pipeline.run_all(X)
    text = pipeline.summary()

    assert isinstance(text, str)
    assert "轮廓系数" in text
    assert "DB指数" in text
    assert "CH指数" in text


def test_pipeline_invalid_method_raises():
    """验证 pipeline 中传入不支持的方法时抛出异常。"""
    X, _ = _sample_feature_matrix()
    pipeline = UnsupervisedPipeline(core_methods=("invalid_method",))
    with np.testing.assert_raises(ValueError):
        pipeline.run_all(X)


def test_pipeline_summary_before_run():
    """验证未运行聚类时调用 summary 抛出异常。"""
    pipeline = UnsupervisedPipeline()
    with np.testing.assert_raises(RuntimeError):
        pipeline.summary()
