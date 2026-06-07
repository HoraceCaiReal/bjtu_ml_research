"""
data_utils 模块单元测试
使用合成小图像验证各函数的输入输出形状与值域。
运行方式：pytest tests/test_data_utils.py -v
"""

import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.data_utils import (
    apply_clahe,
    apply_gaussian_filter,
    apply_median_filter,
    extract_edge_density,
    extract_glcm_features,
    extract_hog_features,
    extract_lbp_features,
    load_dataset,
    split_dataset,
)


@pytest.fixture
def sample_gray_image():
    """生成一张 64×64 的随机灰度图像（uint8）。"""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(64, 64), dtype=np.uint8)


# ========== 图像预处理测试 ==========


class TestPreprocessing:
    """测试图像预处理函数。"""

    def test_clahe_output_shape(self, sample_gray_image):
        result = apply_clahe(sample_gray_image)
        assert result.shape == sample_gray_image.shape

    def test_clahe_output_dtype(self, sample_gray_image):
        result = apply_clahe(sample_gray_image)
        assert result.dtype == np.uint8

    def test_clahe_custom_params(self, sample_gray_image):
        result = apply_clahe(sample_gray_image, clip_limit=3.0, tile_grid_size=(4, 4))
        assert result.shape == sample_gray_image.shape

    def test_gaussian_output_shape(self, sample_gray_image):
        result = apply_gaussian_filter(sample_gray_image, kernel_size=(5, 5))
        assert result.shape == sample_gray_image.shape

    def test_gaussian_output_dtype(self, sample_gray_image):
        result = apply_gaussian_filter(sample_gray_image)
        assert result.dtype == np.uint8

    def test_median_output_shape(self, sample_gray_image):
        result = apply_median_filter(sample_gray_image, kernel_size=5)
        assert result.shape == sample_gray_image.shape

    def test_median_output_dtype(self, sample_gray_image):
        result = apply_median_filter(sample_gray_image, kernel_size=3)
        assert result.dtype == np.uint8


# ========== 特征提取测试 ==========


class TestFeatureExtraction:
    """测试特征提取函数。"""

    def test_hog_output_is_1d(self, sample_gray_image):
        features = extract_hog_features(sample_gray_image)
        assert features.ndim == 1
        assert len(features) > 0

    def test_hog_custom_params(self, sample_gray_image):
        features = extract_hog_features(
            sample_gray_image, orientations=8, pixels_per_cell=(4, 4)
        )
        assert features.ndim == 1

    def test_lbp_output_is_1d(self, sample_gray_image):
        features = extract_lbp_features(sample_gray_image)
        assert features.ndim == 1
        n_bins = 8 * (8 - 1) + 3  # 默认 n_points=8
        assert len(features) == n_bins

    def test_lbp_sum_approx_one(self, sample_gray_image):
        features = extract_lbp_features(sample_gray_image)
        # density=True 的直方图积分约等于 1
        assert abs(features.sum() - 1.0) < 0.1

    def test_glcm_output_shape(self, sample_gray_image):
        distances = (1, 3)
        angles = (0, np.pi / 4)
        features = extract_glcm_features(
            sample_gray_image, distances=distances, angles=angles
        )
        expected_len = len(distances) * len(angles) * 4  # 4 props
        assert features.shape == (expected_len,)

    def test_glcm_values_finite(self, sample_gray_image):
        features = extract_glcm_features(sample_gray_image)
        assert np.all(np.isfinite(features))

    def test_edge_density_range(self, sample_gray_image):
        density = extract_edge_density(sample_gray_image)
        assert 0.0 <= density <= 1.0

    def test_edge_density_uniform_image(self):
        uniform_img = np.ones((64, 64), dtype=np.uint8) * 128
        density = extract_edge_density(uniform_img)
        assert density == 0.0  # 均匀图像无边


# ========== 数据加载测试 ==========


class TestDataLoading:
    """测试数据集加载与划分函数。"""

    @pytest.fixture
    def temp_dataset_dir(self):
        """创建临时数据集目录。"""
        tmpdir = Path(tempfile.mkdtemp())
        for label_dir, label_value in [("Positive", 1), ("Negative", 0)]:
            d = tmpdir / label_dir
            d.mkdir()
            for i in range(4):
                img = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
                cv2.imwrite(str(d / f"{i:03d}.png"), img)
        yield tmpdir
        shutil.rmtree(tmpdir)

    def test_load_dataset_counts(self, temp_dataset_dir):
        images, labels = load_dataset(temp_dataset_dir)
        assert len(images) == 8
        assert len(labels) == 8

    def test_load_dataset_labels(self, temp_dataset_dir):
        _, labels = load_dataset(temp_dataset_dir)
        assert np.sum(labels == 1) == 4
        assert np.sum(labels == 0) == 4

    def test_split_dataset_ratios(self, temp_dataset_dir):
        images, labels = load_dataset(temp_dataset_dir)
        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
            images, labels, train_ratio=0.5, val_ratio=0.25
        )
        total = len(y_train) + len(y_val) + len(y_test)
        assert total == len(labels)

    def test_split_dataset_returns_six_arrays(self, temp_dataset_dir):
        images, labels = load_dataset(temp_dataset_dir)
        result = split_dataset(images, labels)
        assert len(result) == 6

    def test_split_dataset_reproducible(self, temp_dataset_dir):
        images, labels = load_dataset(temp_dataset_dir)
        r1 = split_dataset(images, labels, random_seed=42)
        r2 = split_dataset(images, labels, random_seed=42)
        for a, b in zip(r1, r2):
            np.testing.assert_array_equal(a, b)
