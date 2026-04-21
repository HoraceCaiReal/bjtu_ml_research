"""
评价指标模块
职责：统一封装模型验证所需的各类评价指标
"""

from sklearn.metrics import (
    accuracy_score,  # noqa: F401
    classification_report,  # noqa: F401
    confusion_matrix,  # noqa: F401
    f1_score,  # noqa: F401
    precision_score,  # noqa: F401
    recall_score,  # noqa: F401
)


# TODO: 根据任务需要添加 ROC-AUC、IoU 等指标
