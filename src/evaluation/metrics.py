"""
评价指标模块
"""

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,  # noqa: F401
    calinski_harabasz_score,  # noqa: F401
    classification_report,  # noqa: F401
    confusion_matrix,
    davies_bouldin_score,  # noqa: F401
    f1_score,
    normalized_mutual_info_score,  # noqa: F401
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,  # noqa: F401
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """计算二分类常用指标。"""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: tuple = ("负样本", "正样本"),
) -> plt.Figure:
    """绘制混淆矩阵。"""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    acc = accuracy_score(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels([f"预测 {class_names[0]}", f"预测 {class_names[1]}"])
    ax.set_yticklabels([f"实际 {class_names[0]}", f"实际 {class_names[1]}"])

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )

    ax.set_title(f"混淆矩阵 (准确率: {acc:.4f})")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label: str = "模型",
) -> plt.Figure:
    """绘制 ROC 曲线。"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(
        fpr,
        tpr,
        color="tomato",
        linewidth=2,
        label=f"{label} (AUC = {roc_auc:.4f})",
    )
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", alpha=0.5, label="随机猜测")
    ax.set_xlabel("假阳性率 (FPR)")
    ax.set_ylabel("真阳性率 (TPR)")
    ax.set_title("ROC 曲线")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig
