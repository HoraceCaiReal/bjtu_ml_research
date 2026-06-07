"""传统机器学习模块单元测试。"""

import numpy as np

from src.models.traditional import (
    DecisionTreeClassifier,
    KNNClassifier,
    SVMClassifier,
    compare_classifiers,
    extract_features,
)


def _sample_feature_matrix():
    """构造线性可分的小型特征数据。"""
    X = np.array(
        [
            [0.0, 0.1, 0.0],
            [0.1, 0.0, 0.2],
            [0.2, 0.1, 0.1],
            [1.0, 1.1, 1.0],
            [1.1, 1.0, 1.2],
            [1.2, 1.1, 1.1],
        ],
        dtype=np.float64,
    )
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    return X, y


def test_extract_features_returns_1d_vector():
    """验证四类图像特征能拼接成一维向量。"""
    image = np.random.default_rng(42).integers(0, 256, size=(64, 64), dtype=np.uint8)
    features = extract_features(image)

    assert features.ndim == 1
    assert features.size > 0
    assert np.all(np.isfinite(features))


def test_decision_tree_fit_predict():
    """验证决策树分类器可训练并预测。"""
    X, y = _sample_feature_matrix()
    model = DecisionTreeClassifier(random_state=42).fit(X, y)

    pred = model.predict(X)
    proba = model.predict_proba(X)

    assert pred.shape == y.shape
    assert proba.shape == (len(y), 2)


def test_svm_fit_predict():
    """验证 SVM 分类器可训练并输出概率。"""
    X, y = _sample_feature_matrix()
    model = SVMClassifier(kernel="linear", probability=True, random_state=42).fit(X, y)

    pred = model.predict(X)
    proba = model.predict_proba(X)

    assert pred.shape == y.shape
    assert proba.shape == (len(y), 2)


def test_knn_fit_predict():
    """验证 KNN 分类器可训练并输出概率。"""
    X, y = _sample_feature_matrix()
    model = KNNClassifier(n_neighbors=1).fit(X, y)

    pred = model.predict(X)
    proba = model.predict_proba(X)

    assert pred.shape == y.shape
    assert proba.shape == (len(y), 2)


def test_compare_classifiers_metrics():
    """验证三分类器对比函数返回统一指标。"""
    X, y = _sample_feature_matrix()
    results = compare_classifiers(X, y, X, y)

    assert set(results) == {"decision_tree", "svm", "knn"}
    for metrics in results.values():
        assert set(metrics) == {"accuracy", "precision", "recall", "f1"}
        assert all(0.0 <= value <= 1.0 for value in metrics.values())
