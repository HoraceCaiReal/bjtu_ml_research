"""
无监督学习模型
核心方法：K-Means、GMM（高斯混合模型）、DBSCAN
补充方法：Agglomerative（层次聚类）、Spectral（谱聚类）
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.cluster import (
    DBSCAN as SklearnDBSCAN,
)
from sklearn.cluster import (
    AgglomerativeClustering as SklearnAgglomerative,
)
from sklearn.cluster import (
    KMeans as SklearnKMeans,
)
from sklearn.cluster import (
    SpectralClustering as SklearnSpectral,
)
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture as SklearnGMM
from sklearn.neighbors import KNeighborsClassifier


class KMeansClusterer:
    """K-Means 聚类器（封装 sklearn.cluster.KMeans）。

    基于中心的聚类方法，适合球形簇、计算效率高。
    """

    def __init__(self, n_clusters: int = 2, **kwargs: Any) -> None:
        """
        Parameters
        ----------
        n_clusters : int
            聚类数量。
        **kwargs
            传递给 sklearn.cluster.KMeans 的额外参数。
        """
        self.n_clusters = n_clusters
        self.model = None
        self.labels_ = None
        self.kwargs = kwargs

    def fit(self, X: np.ndarray) -> "KMeansClusterer":
        """训练 K-Means 模型。

        Parameters
        ----------
        X : np.ndarray
            特征矩阵 (n_samples, n_features)。

        Returns
        -------
        KMeansClusterer
            返回 self，支持链式调用。
        """
        default_kwargs = {"random_state": 42, "n_init": "auto"}
        default_kwargs.update(self.kwargs)
        self.model = SklearnKMeans(n_clusters=self.n_clusters, **default_kwargs)
        self.model.fit(X)
        self.labels_ = self.model.labels_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测聚类标签。

        Parameters
        ----------
        X : np.ndarray
            特征矩阵 (n_samples, n_features)。

        Returns
        -------
        np.ndarray
            聚类标签 (n_samples,)。
        """
        if self.model is None:
            raise RuntimeError("模型尚未训练，请先调用 fit()。")
        return self.model.predict(X).astype(np.int64)

    def get_centers(self) -> np.ndarray:
        """返回聚类中心。

        Returns
        -------
        np.ndarray
            聚类中心矩阵 (n_clusters, n_features)。
        """
        if self.model is None:
            raise RuntimeError("模型尚未训练，请先调用 fit()。")
        return self.model.cluster_centers_


class GMMClusterer:
    """高斯混合模型聚类器（封装 sklearn.mixture.GaussianMixture）。

    基于概率分布的软聚类方法，适合椭圆形簇、可输出属于各类的概率。
    """

    def __init__(self, n_components: int = 2, **kwargs: Any) -> None:
        """
        Parameters
        ----------
        n_components : int
            高斯成分数量。
        **kwargs
            传递给 sklearn.mixture.GaussianMixture 的额外参数。
        """
        self.n_components = n_components
        self.model = None
        self.labels_ = None
        self.probabilities_ = None
        self.kwargs = kwargs

    def fit(self, X: np.ndarray) -> "GMMClusterer":
        """训练 GMM 模型。

        Parameters
        ----------
        X : np.ndarray
            特征矩阵 (n_samples, n_features)。

        Returns
        -------
        GMMClusterer
            返回 self，支持链式调用。
        """
        default_kwargs = {"random_state": 42}
        default_kwargs.update(self.kwargs)
        self.model = SklearnGMM(n_components=self.n_components, **default_kwargs)
        self.model.fit(X)
        self.labels_ = self.model.predict(X).astype(np.int64)
        self.probabilities_ = self.model.predict_proba(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测硬聚类标签。

        Parameters
        ----------
        X : np.ndarray
            特征矩阵 (n_samples, n_features)。

        Returns
        -------
        np.ndarray
            聚类标签 (n_samples,)。
        """
        if self.model is None:
            raise RuntimeError("模型尚未训练，请先调用 fit()。")
        return self.model.predict(X).astype(np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """返回属于各成分的概率（软聚类）。

        Parameters
        ----------
        X : np.ndarray
            特征矩阵 (n_samples, n_features)。

        Returns
        -------
        np.ndarray
            概率矩阵 (n_samples, n_components)。
        """
        if self.model is None:
            raise RuntimeError("模型尚未训练，请先调用 fit()。")
        return self.model.predict_proba(X)


class DBSCANClusterer:
    """DBSCAN 聚类器（封装 sklearn.cluster.DBSCAN）。

    基于密度的聚类方法，可发现任意形状簇，自动识别噪声点。
    对新数据的预测基于最近邻分类器（排除噪声点训练）。
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 5, **kwargs: Any) -> None:
        """
        Parameters
        ----------
        eps : float
            邻域半径。
        min_samples : int
            核心点的最小邻域样本数。
        **kwargs
            传递给 sklearn.cluster.DBSCAN 的额外参数。
        """
        self.eps = eps
        self.min_samples = min_samples
        self.model = None
        self.labels_ = None
        self.n_noise_ = 0
        self.kwargs = kwargs
        self._nn_classifier = None

    def fit(self, X: np.ndarray) -> "DBSCANClusterer":
        """训练 DBSCAN 模型。

        Parameters
        ----------
        X : np.ndarray
            特征矩阵 (n_samples, n_features)。

        Returns
        -------
        DBSCANClusterer
            返回 self，支持链式调用。
        """
        default_kwargs = {"n_jobs": -1}
        default_kwargs.update(self.kwargs)
        self.model = SklearnDBSCAN(
            eps=self.eps, min_samples=self.min_samples, **default_kwargs
        )
        self.labels_ = self.model.fit_predict(X).astype(np.int64)
        self.n_noise_ = int(np.sum(self.labels_ == -1))

        # 用非噪声点训练 1-NN 分类器，支持对新数据的 predict
        mask = self.labels_ != -1
        if np.any(mask) and np.unique(self.labels_[mask]).size > 1:
            self._nn_classifier = KNeighborsClassifier(n_neighbors=1)
            self._nn_classifier.fit(X[mask], self.labels_[mask])

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测聚类标签（噪声点标签为 -1）。

        若传入的数据与训练数据形状相同，直接返回存储的标签；
        否则通过最近邻分类器预测。

        Parameters
        ----------
        X : np.ndarray
            特征矩阵 (n_samples, n_features)。

        Returns
        -------
        np.ndarray
            聚类标签 (n_samples,)，噪声点为 -1。
        """
        if self.model is None:
            raise RuntimeError("模型尚未训练，请先调用 fit()。")
        if self.labels_ is not None and X.shape == self.labels_.shape:
            # 推断传入的是训练数据本身
            return self.labels_
        if self._nn_classifier is not None:
            return self._nn_classifier.predict(X).astype(np.int64)
        raise RuntimeError(
            "DBSCAN 无法对新数据进行预测（所有训练点均为噪声或只有一个簇）。"
        )


class AgglomerativeClusterer:
    """层次聚类器（封装 sklearn.cluster.AgglomerativeClustering）。

    自底向上的凝聚层次聚类，适合探索数据的层次结构。
    """

    def __init__(
        self, n_clusters: int = 2, linkage: str = "ward", **kwargs: Any
    ) -> None:
        """
        Parameters
        ----------
        n_clusters : int
            聚类数量。
        linkage : str
            链接方式："ward" | "complete" | "average" | "single"。
        **kwargs
            传递给 sklearn.cluster.AgglomerativeClustering 的额外参数。
        """
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.model = None
        self.labels_ = None
        self.kwargs = kwargs

    def fit(self, X: np.ndarray) -> "AgglomerativeClusterer":
        """训练层次聚类模型。

        Parameters
        ----------
        X : np.ndarray
            特征矩阵 (n_samples, n_features)。

        Returns
        -------
        AgglomerativeClusterer
            返回 self，支持链式调用。
        """
        default_kwargs = {}
        default_kwargs.update(self.kwargs)
        self.model = SklearnAgglomerative(
            n_clusters=self.n_clusters, linkage=self.linkage, **default_kwargs
        )
        self.labels_ = self.model.fit_predict(X).astype(np.int64)
        return self


class SpectralClusterer:
    """谱聚类器（封装 sklearn.cluster.SpectralClustering）。

    基于图论的聚类方法，适合非凸形状的簇。
    """

    def __init__(self, n_clusters: int = 2, **kwargs: Any) -> None:
        """
        Parameters
        ----------
        n_clusters : int
            聚类数量。
        **kwargs
            传递给 sklearn.cluster.SpectralClustering 的额外参数。
        """
        self.n_clusters = n_clusters
        self.model = None
        self.labels_ = None
        self.kwargs = kwargs

    def fit(self, X: np.ndarray) -> "SpectralClusterer":
        """训练谱聚类模型。

        Parameters
        ----------
        X : np.ndarray
            特征矩阵 (n_samples, n_features)。

        Returns
        -------
        SpectralClusterer
            返回 self，支持链式调用。
        """
        default_kwargs = {"random_state": 42, "affinity": "rbf", "n_init": 10}
        default_kwargs.update(self.kwargs)
        self.model = SklearnSpectral(n_clusters=self.n_clusters, **default_kwargs)
        self.labels_ = self.model.fit_predict(X).astype(np.int64)
        return self


class UnsupervisedPipeline:
    """无监督学习统一 pipeline。

    支持一键运行多种聚类方法并收集结果对比。
    """

    METHODS: Dict[str, type] = {
        "kmeans": KMeansClusterer,
        "gmm": GMMClusterer,
        "dbscan": DBSCANClusterer,
        "agglomerative": AgglomerativeClusterer,
        "spectral": SpectralClusterer,
    }

    # 各方法的默认构造参数
    _DEFAULT_CONFIG: Dict[str, Dict[str, Any]] = {
        "kmeans": {"n_clusters": 2, "random_state": 42, "n_init": "auto"},
        "gmm": {"n_components": 2, "random_state": 42},
        "dbscan": {"eps": 0.5, "min_samples": 5},
        "agglomerative": {"n_clusters": 2, "linkage": "ward"},
        "spectral": {"n_clusters": 2, "random_state": 42, "affinity": "rbf"},
    }

    def __init__(
        self,
        core_methods: Tuple[str, ...] = ("kmeans", "gmm", "dbscan"),
        supplementary_methods: Tuple[str, ...] = ("agglomerative", "spectral"),
    ) -> None:
        """
        Parameters
        ----------
        core_methods : Tuple[str, ...]
            核心方法列表。
        supplementary_methods : Tuple[str, ...]
            补充方法列表。
        """
        self.core_methods = core_methods
        self.supplementary_methods = supplementary_methods
        self.results: Dict[str, Dict[str, Any]] = {}

    def run_all(
        self, X: np.ndarray, y_true: Optional[np.ndarray] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        依次运行核心方法与补充方法，返回聚类结果对比。

        Parameters
        ----------
        X : np.ndarray
            特征矩阵 (n_samples, n_features)。
        y_true : Optional[np.ndarray]
            真实标签；传入时额外计算 ARI 和 NMI 等外部指标。

        Returns
        -------
        Dict[str, Dict[str, Any]]
            {方法名: {labels, metrics, ...}} 格式的对比结果。
        """
        all_methods = list(self.core_methods) + list(self.supplementary_methods)
        self.results = {}

        for name in all_methods:
            if name not in self.METHODS:
                raise ValueError(f"不支持的聚类方法: {name}")

            # 实例化并训练
            config = self._DEFAULT_CONFIG.get(name, {})
            clusterer_class = self.METHODS[name]
            clusterer = clusterer_class(**config)
            clusterer.fit(X)

            labels = clusterer.labels_

            # 计算内部聚类评价指标
            metrics: Dict[str, float] = {}
            unique_labels = np.unique(labels)

            # silhouette_score 需要至少 2 个簇且至少有非噪声标签
            valid_for_silhouette = (
                unique_labels.size >= 2 and len(unique_labels[unique_labels >= 0]) >= 2
            )
            if valid_for_silhouette and X.shape[0] > 1:
                try:
                    # 排除噪声点（标签为 -1）计算 silhouette
                    if -1 in unique_labels:
                        mask = labels != -1
                        if np.sum(mask) > 1 and np.unique(labels[mask]).size >= 2:
                            metrics["silhouette"] = float(
                                silhouette_score(X[mask], labels[mask])
                            )
                        else:
                            metrics["silhouette"] = float("nan")
                    else:
                        metrics["silhouette"] = float(silhouette_score(X, labels))
                except Exception:
                    metrics["silhouette"] = float("nan")
            else:
                metrics["silhouette"] = float("nan")

            # Davies-Bouldin（需要非噪声数据）
            try:
                if -1 in unique_labels:
                    mask = labels != -1
                    if np.sum(mask) > 1 and np.unique(labels[mask]).size >= 2:
                        metrics["davies_bouldin"] = float(
                            davies_bouldin_score(X[mask], labels[mask])
                        )
                    else:
                        metrics["davies_bouldin"] = float("nan")
                else:
                    if unique_labels.size >= 2:
                        metrics["davies_bouldin"] = float(
                            davies_bouldin_score(X, labels)
                        )
                    else:
                        metrics["davies_bouldin"] = float("nan")
            except Exception:
                metrics["davies_bouldin"] = float("nan")

            # Calinski-Harabasz
            try:
                if -1 in unique_labels:
                    mask = labels != -1
                    if np.sum(mask) > 1 and np.unique(labels[mask]).size >= 2:
                        metrics["calinski_harabasz"] = float(
                            calinski_harabasz_score(X[mask], labels[mask])
                        )
                    else:
                        metrics["calinski_harabasz"] = float("nan")
                else:
                    if unique_labels.size >= 2:
                        metrics["calinski_harabasz"] = float(
                            calinski_harabasz_score(X, labels)
                        )
                    else:
                        metrics["calinski_harabasz"] = float("nan")
            except Exception:
                metrics["calinski_harabasz"] = float("nan")

            # 噪声点统计（仅 DBSCAN）
            n_noise = int(np.sum(labels == -1))
            metrics["n_noise"] = n_noise if n_noise > 0 else 0

            # 外部评价指标（需要真实标签）
            if y_true is not None:
                # ARI 和 NMI 仅对非噪声点计算
                if -1 in unique_labels:
                    mask = labels != -1
                    if np.sum(mask) > 1 and np.unique(labels[mask]).size >= 2:
                        metrics["ari"] = float(
                            adjusted_rand_score(y_true[mask], labels[mask])
                        )
                        metrics["nmi"] = float(
                            normalized_mutual_info_score(y_true[mask], labels[mask])
                        )
                    else:
                        metrics["ari"] = float("nan")
                        metrics["nmi"] = float("nan")
                else:
                    metrics["ari"] = float(adjusted_rand_score(y_true, labels))
                    metrics["nmi"] = float(normalized_mutual_info_score(y_true, labels))

            self.results[name] = {
                "labels": labels,
                "metrics": metrics,
                "n_clusters": int(unique_labels.size),
            }

        return self.results

    def summary(self) -> str:
        """打印聚类结果对比摘要。"""
        if not self.results:
            raise RuntimeError("尚未运行聚类，请先调用 run_all()。")

        lines = ["=" * 72]
        lines.append(
            f"{'方法':<20} {'轮廓系数':>10} {'DB指数':>10} {'CH指数':>14} {'噪声点':>8}"
        )
        lines.append("-" * 72)

        for name, result in self.results.items():
            m = result["metrics"]
            lines.append(
                f"{name:<20} "
                f"{m.get('silhouette', float('nan')):>10.4f} "
                f"{m.get('davies_bouldin', float('nan')):>10.4f} "
                f"{m.get('calinski_harabasz', float('nan')):>14.2f} "
                f"{m.get('n_noise', 0):>8d}"
            )

        lines.append("=" * 72)
        return "\n".join(lines)
