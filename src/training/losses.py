"""
损失函数模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_cross_entropy_loss(**kwargs):
    """交叉熵损失（分类任务默认）。"""
    return nn.CrossEntropyLoss(**kwargs)


class FocalLoss(nn.Module):
    """Focal Loss，应对样本不均衡。

    FL = -alpha * (1 - p_t)^gamma * log(p_t)

    Parameters
    ----------
    alpha : float
        正样本权重。二元分类时设为正类权重，如 0.25。
    gamma : float
        聚焦参数，越大越关注难分样本，默认 2.0。
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        p_t = torch.exp(-ce_loss)
        focal_weight = (1 - p_t) ** self.gamma

        # 对正类加权
        alpha_weight = torch.where(
            targets == 1,
            torch.tensor(self.alpha, device=inputs.device),
            torch.tensor(1 - self.alpha, device=inputs.device),
        )
        return (alpha_weight * focal_weight * ce_loss).mean()
