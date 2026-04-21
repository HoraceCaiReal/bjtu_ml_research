"""
项目全局配置

用法：
    from src.config import DATA_ROOT, DEVICE
    data_path = DATA_ROOT / "Positive"
    tensor = tensor.to(DEVICE)
"""

import os
from pathlib import Path

import torch
from dotenv import load_dotenv

# 加载 .env 文件（如果存在）
load_dotenv()

# 数据集根目录：优先读取环境变量，否则使用默认路径
_DATA_ROOT = os.getenv("CRACK_DATA_ROOT", "data")
DATA_ROOT = Path(_DATA_ROOT).resolve()

# 正/负样本目录
POSITIVE_DIR = DATA_ROOT / "Positive"
NEGATIVE_DIR = DATA_ROOT / "Negative"

# 自动检测 GPU/CPU：有 CUDA 可用则使用 GPU，否则回退到 CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check_data_exists() -> bool:
    """检查数据集路径是否存在且包含图像文件"""
    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"数据集根目录不存在: {DATA_ROOT}\n"
            f"请在项目根目录创建 .env 文件，并设置 CRACK_DATA_ROOT=你的实际路径\n"
            f"参考 .env.example 模板。"
        )
    return True
