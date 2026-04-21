"""
损失函数模块
职责：封装各类损失函数，便于在综合展示系统中切换对比
"""

import torch.nn as nn


def get_cross_entropy_loss(**kwargs):
    """交叉熵损失（分类任务默认）"""
    return nn.CrossEntropyLoss(**kwargs)


# TODO: 根据任务需要添加更多损失函数，如 Focal Loss 等
