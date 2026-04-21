"""
无监督学习模型
包含：K-Means 聚类
"""

from sklearn.cluster import KMeans


def get_kmeans(n_clusters=2, **kwargs):
    """获取 K-Means 模型"""
    return KMeans(n_clusters=n_clusters, **kwargs)
