"""
数据处理与特征提取工具
包含：图像预处理（CLAHE、高斯滤波、中值滤波）
      特征提取（HOG、LBP、GLCM、边缘密度）
"""

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern
from sklearn.model_selection import train_test_split


def _imread_gray(path: Path) -> "np.ndarray | None":
    """以灰度模式读取图像，兼容 Windows 中文路径。"""
    buf = np.fromfile(str(path), dtype=np.uint8)
    if buf is None or buf.size == 0:
        return None
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return img


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
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


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
    return cv2.GaussianBlur(image, kernel_size, sigma)


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
    return cv2.medianBlur(image, kernel_size)


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
    features = hog(
        image,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        feature_vector=True,
    )
    return features


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
    n_bins = n_points * (n_points - 1) + 3  # uniform LBP 直方图分箱数
    lbp_image = local_binary_pattern(image, n_points, radius, method="uniform")
    hist, _ = np.histogram(lbp_image, bins=n_bins, range=(0, n_bins), density=True)
    return hist


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
    # 将图像量化到 0-255 的 uint8 范围
    img_uint8 = image.astype(np.uint8) if image.dtype != np.uint8 else image

    props = []
    for distance in distances:
        for angle in angles:
            glcm = graycomatrix(
                img_uint8,
                distances=[distance],
                angles=[angle],
                levels=256,
                symmetric=True,
                normed=True,
            )
            contrast = graycoprops(glcm, "contrast")[0, 0]
            correlation = graycoprops(glcm, "correlation")[0, 0]
            energy = graycoprops(glcm, "energy")[0, 0]
            homogeneity = graycoprops(glcm, "homogeneity")[0, 0]
            props.extend([contrast, correlation, energy, homogeneity])

    return np.array(props, dtype=np.float64)


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
    edges = cv2.Canny(image, low_threshold, high_threshold)
    return float(np.count_nonzero(edges)) / edges.size


# ========== 数据集工具函数 ==========


def load_dataset(
    data_root: Path,
    max_samples: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载数据集，返回图像数组和标签数组。

    Parameters
    ----------
    data_root : Path
        数据集根目录（包含 Positive/ 和 Negative/ 子目录）。
    max_samples : int or None
        每类最多加载的图像数量。None 表示加载全部。
        用于可视化/调试时快速加载少量样本。

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (images, labels)，标签 1 表示有裂纹，0 表示无裂纹。
    """
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

    def _load_dir(directory: Path, label: int):
        imgs, lbls = [], []
        paths = sorted(directory.iterdir())
        for path in paths[:max_samples]:
            if path.suffix.lower() in IMAGE_EXTS:
                img = _imread_gray(path)
                if img is not None:
                    imgs.append(img)
                    lbls.append(label)
        return imgs, lbls

    pos_imgs, pos_lbls = _load_dir(data_root / "Positive", label=1)
    neg_imgs, neg_lbls = _load_dir(data_root / "Negative", label=0)

    all_imgs = pos_imgs + neg_imgs
    labels = np.array(pos_lbls + neg_lbls, dtype=np.int64)

    # 图像尺寸相同时堆叠为标准数组 (N, H, W)，否则回退 object 数组
    shapes = {img.shape for img in all_imgs}
    if len(shapes) == 1:
        images = np.stack(all_imgs)
    else:
        images = np.array(all_imgs, dtype=object)
    return images, labels


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
    # 第一次划分：分离训练集和（验证集 + 测试集）
    val_test_ratio = 1.0 - train_ratio
    X_train, X_val_test, y_train, y_val_test = train_test_split(
        images,
        labels,
        test_size=val_test_ratio,
        random_state=random_seed,
        stratify=labels,
    )

    # 第二次划分：从剩余部分分离验证集和测试集
    test_ratio_in_remainder = val_ratio / (val_ratio + (1.0 - train_ratio - val_ratio))

    # 小数据集时 stratify 可能导致某类样本不足，回退到不分层划分
    try:
        X_val, X_test, y_val, y_test = train_test_split(
            X_val_test,
            y_val_test,
            test_size=test_ratio_in_remainder,
            random_state=random_seed,
            stratify=y_val_test,
        )
    except ValueError:
        X_val, X_test, y_val, y_test = train_test_split(
            X_val_test,
            y_val_test,
            test_size=test_ratio_in_remainder,
            random_state=random_seed,
        )

    return X_train, X_val, X_test, y_train, y_val, y_test
