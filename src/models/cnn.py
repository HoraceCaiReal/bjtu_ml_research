"""
深度学习模型
包含：卷积神经网络 (CNN)
"""

import torch
import torch.nn as nn

from src.config import DEVICE


class CrackCNN(nn.Module):
    """
    裂纹识别 CNN 基类
    TODO: 根据任务需求设计网络结构
    """
    def __init__(self, num_classes=2):
        super(CrackCNN, self).__init__()
        # TODO: 定义网络层
        pass

    def forward(self, x):
        # TODO: 定义前向传播
        pass


def get_cnn_model(**kwargs):
    """获取 CNN 模型实例，自动加载到 GPU/CPU"""
    model = CrackCNN(**kwargs)
    model = model.to(DEVICE)
    print(f"模型已加载到设备: {DEVICE}")
    return model
