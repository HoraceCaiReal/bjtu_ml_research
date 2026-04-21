"""
传统机器学习模型
包含：决策树 (Decision Tree)、支持向量机 (SVM)
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


def get_decision_tree(**kwargs):
    """获取决策树模型"""
    return DecisionTreeClassifier(**kwargs)


def get_svm(**kwargs):
    """获取 SVM 模型"""
    return SVC(**kwargs)
