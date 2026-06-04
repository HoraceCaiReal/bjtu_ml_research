"""
深度学习模型
包含：自建小型 CNN（CrackCNN）用于裂纹图像二分类
"""

import torch.nn as nn

from src.config import DEVICE


class CrackCNN(nn.Module):
    """自建小型卷积神经网络，用于 2 分类（有裂纹 / 无裂纹）。

    设计原则：轻量级，适合 ~20000 张图像级别的数据集，
    避免过拟合，训练时间可控。

    预期结构（待实验确定具体层数和参数）：
    - Conv Block 1: Conv2d → BatchNorm → ReLU → MaxPool
    - Conv Block 2: Conv2d → BatchNorm → ReLU → MaxPool
    - Conv Block 3: Conv2d → BatchNorm → ReLU → MaxPool
    - Global Average Pooling 或 Flatten
    - FC Block 1: Linear → ReLU → Dropout
    - FC Block 2: Linear → 输出 2 类
    """

    def __init__(self, num_classes: int = 2, input_channels: int = 1) -> None:
        """
        Parameters
        ----------
        num_classes : int
            分类类别数（默认 2：有裂纹/无裂纹）。
        input_channels : int
            输入通道数（灰度图 = 1，RGB = 3）。
        """
        super().__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        # TODO: 替换为真实的网络层定义
        # 设计约束：
        #   - 总参数量控制在 500K - 2M 之间
        #   - 使用 BatchNorm 加速收敛
        #   - 使用 Dropout 防止过拟合
        #   - 输入尺寸建议从 128x128 或 224x224 开始实验
        #
        # 占位层：确保模型可实例化以便调试和检查参数量，
        # 实现时请替换为 Conv2d → BN → ReLU → MaxPool 等真实层。
        self.features = nn.Identity()
        self.classifier = nn.Linear(input_channels, num_classes)

    def forward(self, x):
        """
        前向传播。

        Parameters
        ----------
        x : torch.Tensor, shape (N, C, H, W)
            输入图像批次。

        Returns
        -------
        torch.Tensor, shape (N, num_classes)
            类别 logits。
        """
        # TODO: 实现真实的前向传播逻辑
        # 占位实现：展平后直接通过线性层，仅用于验证模型可实例化
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def get_cnn_model(num_classes: int = 2, input_channels: int = 1, **kwargs):
    """获取 CrackCNN 模型实例，自动加载到 GPU/CPU。

    Parameters
    ----------
    num_classes : int
        分类类别数。
    input_channels : int
        输入通道数。

    Returns
    -------
    CrackCNN
        已加载到 DEVICE 的模型实例。
    """
    model = CrackCNN(num_classes=num_classes, input_channels=input_channels, **kwargs)
    model = model.to(DEVICE)
    print(f"CrackCNN 已加载到设备: {DEVICE}")
    return model
