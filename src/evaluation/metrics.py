"""
评价指标模块
职责：统一封装模型验证所需的各类评价指标
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# TODO: 根据任务需要添加 ROC-AUC、IoU 等指标
