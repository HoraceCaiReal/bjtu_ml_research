"""
深度学习模型：自建小型 CNN（CrackCNN）用于裂纹图像二分类
"""

import torch.nn as nn

from src.config import DEVICE


class CrackCNN(nn.Module):
    """自建小型 CNN，4 个卷积块 + 全局平均池化 + 分类头，参数量约 1.17M。

    输入：(N, 1, H, W)
    输出：(N, 2) 类别 logits
    推荐输入尺寸 ≥ 64×64。
    """

    def __init__(
        self,
        num_classes: int = 2,
        input_channels: int = 1,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.dropout_rate = dropout_rate

        # Block 1: 1 → 32
        self.block1 = self._make_block(input_channels, 32)
        # Block 2: 32 → 64
        self.block2 = self._make_block(32, 64)
        # Block 3: 64 → 128
        self.block3 = self._make_block(64, 128)
        # Block 4: 128 → 256
        self.block4 = self._make_block(128, 256)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(256, num_classes)

    @staticmethod
    def _make_block(in_ch: int, out_ch: int) -> nn.Sequential:
        """构建一个卷积块：Conv→BN→ReLU → Conv→BN→ReLU → MaxPool(2)。"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


def get_cnn_model(
    num_classes: int = 2,
    input_channels: int = 1,
    dropout_rate: float = 0.5,
    **kwargs,
):
    """获取 CrackCNN 模型实例并加载到 DEVICE。"""
    model = CrackCNN(
        num_classes=num_classes,
        input_channels=input_channels,
        dropout_rate=dropout_rate,
        **kwargs,
    )
    model = model.to(DEVICE)
    print(f"CrackCNN 已加载到设备: {DEVICE}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {total_params:,}")
    return model
