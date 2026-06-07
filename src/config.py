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

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 加载项目根目录下的 .env 文件（如果存在）
load_dotenv(PROJECT_ROOT / ".env")

# 数据集根目录：优先读取环境变量，否则使用项目根目录下的 data/
_DATA_ROOT = os.getenv("CRACK_DATA_ROOT")
if _DATA_ROOT:
    _data_root = Path(_DATA_ROOT).expanduser()
    DATA_ROOT = (
        _data_root if _data_root.is_absolute() else PROJECT_ROOT / _data_root
    ).resolve()
else:
    DATA_ROOT = PROJECT_ROOT / "data"

# 正/负样本目录
POSITIVE_DIR = DATA_ROOT / "Positive"
NEGATIVE_DIR = DATA_ROOT / "Negative"

# 自动检测 GPU/CPU：有 CUDA 可用则使用 GPU，否则回退到 CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check_data_exists() -> bool:
    """检查数据集路径是否存在且包含 Positive/Negative 子目录与图像文件。"""
    required_dirs = {
        "数据集根目录": DATA_ROOT,
        "正样本目录": POSITIVE_DIR,
        "负样本目录": NEGATIVE_DIR,
    }
    missing_dirs = [name for name, path in required_dirs.items() if not path.exists()]

    image_files = []
    if DATA_ROOT.exists():
        for pattern in ("*.jpg", "*.jpeg", "*.png"):
            image_files.extend(DATA_ROOT.rglob(pattern))

    missing_parts = []
    if missing_dirs:
        missing_parts.append("缺失目录：" + "、".join(missing_dirs))
    if not image_files:
        missing_parts.append("未找到 .jpg/.jpeg/.png 图像文件")

    if missing_parts:
        raise FileNotFoundError(
            "数据集检查未通过：\n"
            + "\n".join(f"- {part}" for part in missing_parts)
            + "\n\n请在项目根目录创建 .env 文件，并设置 CRACK_DATA_ROOT=你的实际路径\n"
            + "参考 .env.example 模板。"
        )

    return True
