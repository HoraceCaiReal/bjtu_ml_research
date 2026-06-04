"""
传统机器学习模型
包含：DecisionTreeClassifier、SVMClassifier、KNNClassifier
特征提取：HOG + LBP + GLCM + 边缘密度
"""

from typing import Any, Dict, Tuple

import numpy as np


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
    raise NotImplementedError(
        "TODO: 整合 HOG + LBP + GLCM + 边缘密度 四特征"
    )


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
        raise NotImplementedError("TODO: 根据 model_type 训练对应分类器")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别标签。"""
        raise NotImplementedError("TODO: 实现预测")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测类别概率（SVM 需 Platt scaling 或使用 decision_function）。"""
        raise NotImplementedError("TODO: 实现概率预测")

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
    raise NotImplementedError("TODO: 实现三分类器对比实验")