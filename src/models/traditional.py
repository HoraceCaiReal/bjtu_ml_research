"""
传统机器学习模型
包含：DecisionTreeClassifier、SVMClassifier、KNNClassifier
特征提取：HOG + LBP + GLCM + 边缘密度
"""

from typing import Any, Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier as SklearnKNNClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier

from src.data_utils import (
    extract_edge_density,
    extract_glcm_features,
    extract_hog_features,
    extract_lbp_features,
)


def extract_features(image: np.ndarray) -> np.ndarray:
    """
    提取图像的多特征向量。

    组合以下四种特征：
    - HOG（方向梯度直方图）：捕获裂缝方向性
    - LBP（局部二值模式）：捕获局部纹理模式
    - GLCM（灰度共生矩阵）：纹理统计特性（对比度/相关性/能量/同质性）
    - 边缘密度：Canny 边缘像素占比

    Parameters
    ----------
    image : np.ndarray
        输入灰度图像。

    Returns
    -------
    np.ndarray
        拼接后的特征向量。
    """
    image = image.astype(np.uint8) if image.dtype != np.uint8 else image
    hog_features = extract_hog_features(image)
    lbp_features = extract_lbp_features(image)
    glcm_features = extract_glcm_features(image)
    edge_density = np.array([extract_edge_density(image)], dtype=np.float64)
    return np.concatenate(
        [hog_features, lbp_features, glcm_features, edge_density]
    ).astype(np.float64)


class TraditionalClassifier:
    """传统机器学习分类器基类。

    支持三种分类器：决策树、SVM、KNN。
    """

    def __init__(self, model_type: str = "decision_tree", **kwargs: Any) -> None:
        """
        Parameters
        ----------
        model_type : str
            "decision_tree" | "svm" | "knn"
        **kwargs
            传递给具体分类器的参数。
        """
        self.model_type = model_type
        self.model = None
        self.kwargs = kwargs

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TraditionalClassifier":
        """训练分类器。"""
        if self.model_type == "decision_tree":
            default_kwargs = {"random_state": 42}
            default_kwargs.update(self.kwargs)
            self.model = SklearnDecisionTreeClassifier(**default_kwargs)
        elif self.model_type == "svm":
            default_kwargs = {"kernel": "rbf", "probability": True, "random_state": 42}
            default_kwargs.update(self.kwargs)
            self.model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("classifier", SVC(**default_kwargs)),
                ]
            )
        elif self.model_type == "knn":
            default_kwargs = {"n_neighbors": 5, "algorithm": "ball_tree"}
            default_kwargs.update(self.kwargs)
            self.model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("classifier", SklearnKNNClassifier(**default_kwargs)),
                ]
            )
        else:
            raise ValueError(f"不支持的传统分类器类型: {self.model_type}")

        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别标签。"""
        if self.model is None:
            raise RuntimeError("模型尚未训练，请先调用 fit()。")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测类别概率（SVM 需 Platt scaling 或使用 decision_function）。"""
        if self.model is None:
            raise RuntimeError("模型尚未训练，请先调用 fit()。")
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        raise RuntimeError(f"{self.model_type} 不支持 predict_proba。")

    def get_params(self) -> Dict[str, Any]:
        """返回模型参数。"""
        return {"model_type": self.model_type, **self.kwargs}


class DecisionTreeClassifier(TraditionalClassifier):
    """决策树分类器（封装 sklearn.tree.DecisionTreeClassifier）。"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(model_type="decision_tree", **kwargs)


class SVMClassifier(TraditionalClassifier):
    """SVM 分类器（封装 sklearn.svm.SVC）。"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(model_type="svm", **kwargs)


class KNNClassifier(TraditionalClassifier):
    """KNN 分类器（封装 sklearn.neighbors.KNeighborsClassifier）。"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(model_type="knn", **kwargs)


def compare_classifiers(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """
    对比三种分类器（决策树、SVM、KNN）在相同特征集上的表现。

    Returns
    -------
    Dict[str, Dict[str, float]]
        {分类器名: {metric: value}} 格式的对比结果。
    """
    classifiers = {
        "decision_tree": DecisionTreeClassifier(),
        "svm": SVMClassifier(),
        "knn": KNNClassifier(n_neighbors=3),
    }

    results: Dict[str, Dict[str, float]] = {}
    for name, classifier in classifiers.items():
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        results[name] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        }

    return results
