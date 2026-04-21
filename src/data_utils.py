"""
数据工具模块
职责：数据集读取、划分、预处理、增强
说明：此模块供 notebooks 调用，避免在 notebook 中写过长数据处理代码
"""

import os
import glob
from typing import Tuple, List


def load_image_paths(data_dir: str) -> Tuple[List[str], List[int]]:
    """
    读取裂纹图像数据集路径
    
    Args:
        data_dir: 数据根目录，包含 Negative/ 和 Positive/
        
    Returns:
        image_paths: 图像文件路径列表
        labels: 对应标签列表（0=无裂纹, 1=有裂纹）
    """
    # TODO: 实现数据读取逻辑
    pass


def split_dataset(image_paths, labels, test_size=0.2, val_size=0.1, random_state=42):
    """
    划分训练/验证/测试集
    
    Returns:
        六元组: X_train, X_val, X_test, y_train, y_val, y_test
    """
    # TODO: 实现划分逻辑（可对比多种划分策略）
    pass
