"""
数据处理与特征提取工具
包含：图像预处理（CLAHE、高斯滤波、中值滤波）
      特征提取（HOG、LBP、GLCM、边缘密度）
"""

from pathlib import Path
from typing import Tuple

import numpy as np


# ========== 图像预处理方法 ==========

def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    对图像应用 CLAHE（自适应直方图均衡化）增强裂缝对比度。

    Parameters
    ----------
    image : np.ndarray
        输入灰度图像。
    clip_limit : float
        对比度裁剪阈值。
    tile_grid_size : Tuple[int, int]
        分块大小。

    Returns
    -------
    np.ndarray
        CLAHE 增强后的图像。
    """
    raise NotImplementedError("TODO: 实现 CLAHE 对比度增强")


def apply_gaussian_filter(
    image: np.ndarray,
    kernel_size: Tuple[int, int] = (5, 5),
    sigma: float = 1.0,
) -> np.ndarray:
    """
    对图像应用高斯滤波去噪。

    Parameters
    ----------
    image : np.ndarray
        输入灰度图像。
    kernel_size : Tuple[int, int]
        高斯核尺寸（奇数）。
    sigma : float
        高斯核标准差。

    Returns
    -------
    np.ndarray
        滤波后的图像。
    """
    raise NotImplementedError("TODO: 实现高斯滤波")


def apply_median_filter(
    image: np.ndarray,
    kernel_size: int = 5,
) -> np.ndarray:
    """
    对图像应用中值滤波去噪（适合去除椒盐噪声）。

    Parameters
    ----------
    image : np.ndarray
        输入灰度图像。
    kernel_size : int
        滤波核尺寸（奇数）。

    Returns
    -------
    np.ndarray
        滤波后的图像。
    """
    raise NotImplementedError("TODO: 实现中值滤波")


# ========== 特征提取方法 ==========

def extract_hog_features(
    image: np.ndarray,
    orientations: int = 9,
    pixels_per_cell: Tuple[int, int] = (8, 8),
    cells_per_block: Tuple[int, int] = (2, 2),
) -> np.ndarray:
    """
    提取 HOG（方向梯度直方图）特征。

    HOG 捕获图像的梯度方向分布，对裂缝的方向性特征敏感。

    Parameters
    ----------
    image : np.ndarray
        输入灰度图像。
    orientations : int
        梯度方向分箱数。
    pixels_per_cell : Tuple[int, int]
        每个 cell 的像素数。
    cells_per_block : Tuple[int, int]
        每个 block 的 cell 数。

    Returns
    -------
    np.ndarray
        HOG 特征向量。
    """
    raise NotImplementedError("TODO: 实现 HOG 特征提取")


def extract_lbp_features(
    image: np.ndarray,
    radius: int = 1,
    n_points: int = 8,
) -> np.ndarray:
    """
    提取 LBP（局部二值模式）特征。

    LBP 捕获局部纹理模式，对路面的微观纹理变化敏感。

    Parameters
    ----------
    image : np.ndarray
        输入灰度图像。
    radius : int
        LBP 半径。
    n_points : int
        邻域采样点数。

    Returns
    -------
    np.ndarray
        LBP 特征向量（直方图）。
    """
    raise NotImplementedError("TODO: 实现 LBP 特征提取")


def extract_glcm_features(
    image: np.ndarray,
    distances: Tuple[int, ...] = (1, 3, 5),
    angles: Tuple[float, ...] = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
) -> np.ndarray:
    """
    提取 GLCM（灰度共生矩阵）纹理统计特征。

    GLCM 刻画像素灰度级的空间共生关系，裂缝区域的纹理统计
    特性（对比度、相关性、能量、同质性）与正常路面差异显著。

    Parameters
    ----------
    image : np.ndarray
        输入灰度图像。
    distances : Tuple[int, ...]
        像素距离。
    angles : Tuple[float, ...]
        方向角（弧度）。

    Returns
    -------
    np.ndarray
        GLCM 特征向量（包含 contrast, correlation, energy, homogeneity）。
    """
    raise NotImplementedError("TODO: 实现 GLCM 特征提取")


def extract_edge_density(
    image: np.ndarray,
    low_threshold: float = 50,
    high_threshold: float = 150,
) -> float:
    """
    计算边缘密度特征。

    裂缝区域经 Canny 边缘检测后边缘像素密度显著高于正常路面，
    是区分裂缝/非裂缝的强特征。

    Parameters
    ----------
    image : np.ndarray
        输入灰度图像。
    low_threshold : float
        Canny 低阈值。
    high_threshold : float
        Canny 高阈值。

    Returns
    -------
    float
        边缘密度（边缘像素数 / 总像素数）。
    """
    raise NotImplementedError("TODO: 实现边缘密度特征提取")


# ========== 数据集工具函数 ==========

def load_dataset(data_root: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载全部数据集，返回图像数组和标签数组。

    Parameters
    ----------
    data_root : Path
        数据集根目录（包含 Positive/ 和 Negative/ 子目录）。

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (images, labels)，标签 1 表示有裂纹，0 表示无裂纹。
    """
    raise NotImplementedError("TODO: 实现数据集加载")


def split_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    将数据集划分为训练集、验证集和测试集。

    Parameters
    ----------
    images : np.ndarray
        图像数组。
    labels : np.ndarray
        标签数组。
    train_ratio : float
        训练集比例。
    val_ratio : float
        验证集比例（测试集 = 1 - train_ratio - val_ratio）。
    random_seed : int
        随机种子。

    Returns
    -------
    Tuple 包含 (X_train, X_val, X_test, y_train, y_val, y_test)。
    """
    raise NotImplementedError("TODO: 实现数据集划分")