"""
裂纹图像识别系统 — Gradio 可视化训练链路

单页面链路式设计：用户依次选择数据处理 → 模型 → 超参数 → 损失函数 → 优化策略 → 运行。
所有代码自包含，不依赖 Notebook 或 _backup/src/ 模块。
"""

# ============================================================
# 1. 导入与基础配置
# ============================================================
import matplotlib
matplotlib.use('Agg')

import os
import sys
import json
import time
import copy
from pathlib import Path
from typing import Tuple, Dict, Optional

import cv2
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, KFold,
    GridSearchCV, RandomizedSearchCV,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import gradio as gr

# ============================================================
# 2. 项目路径与设备配置
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
load_dotenv(PROJECT_ROOT / ".env")

_DATA_ROOT = os.getenv("CRACK_DATA_ROOT")
if _DATA_ROOT:
    _dr = Path(_DATA_ROOT).expanduser()
    DATA_ROOT = (_dr if _dr.is_absolute() else PROJECT_ROOT / _dr).resolve()
else:
    DATA_ROOT = PROJECT_ROOT / "data"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
TRAD_DIR = MODEL_DIR / "traditional"
CNN_DIR = MODEL_DIR / "cnn"
UNSUP_DIR = MODEL_DIR / "unsupervised"
SCALER_DIR = OUTPUT_DIR / "scalers"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 中文字体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

print(f"项目根目录: {PROJECT_ROOT}")
print(f"数据集根目录: {DATA_ROOT}")
print(f"计算设备: {DEVICE}")

# ============================================================
# 3. NB01 工具函数 — 图像读取、预处理、特征提取
# ============================================================

def _imread_gray(path: Path) -> Optional[np.ndarray]:
    """以灰度模式读取图像，兼容 Windows 中文路径。"""
    buf = np.fromfile(str(path), dtype=np.uint8)
    if buf is None or buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


def load_dataset(data_root: Path, max_samples: int = None):
    """加载数据集，返回 (images, labels)。标签 1=Positive, 0=Negative。"""
    def _load_dir(directory: Path, label: int, limit: int = None):
        imgs, lbls = [], []
        paths = sorted(directory.iterdir())
        for path in paths[:limit]:
            if path.suffix.lower() in IMAGE_EXTS:
                img = _imread_gray(path)
                if img is not None:
                    imgs.append(img); lbls.append(label)
        return imgs, lbls

    n_per = max_samples // 2 if max_samples else None
    pos_imgs, pos_lbls = _load_dir(data_root / "Positive", 1, n_per)
    neg_imgs, neg_lbls = _load_dir(data_root / "Negative", 0, n_per)
    all_imgs = pos_imgs + neg_imgs
    labels = np.array(pos_lbls + neg_lbls, dtype=np.int64)
    shapes = {img.shape for img in all_imgs}
    images = np.stack(all_imgs) if len(shapes) == 1 else np.array(all_imgs, dtype=object)
    return images, labels


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0,
                tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe_obj.apply(image)


def apply_gaussian_filter(image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5),
                          sigma: float = 1.0) -> np.ndarray:
    return cv2.GaussianBlur(image, kernel_size, sigma)


def apply_median_filter(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    return cv2.medianBlur(image, kernel_size)


def extract_hog_features(image: np.ndarray, orientations: int = 9,
                         pixels_per_cell: Tuple[int, int] = (8, 8),
                         cells_per_block: Tuple[int, int] = (2, 2)) -> np.ndarray:
    return hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell,
               cells_per_block=cells_per_block, feature_vector=True)


def extract_lbp_features(image: np.ndarray, radius: int = 1,
                         n_points: int = 8) -> np.ndarray:
    n_bins = n_points * (n_points - 1) + 3
    lbp_image = local_binary_pattern(image, n_points, radius, method="uniform")
    hist, _ = np.histogram(lbp_image, bins=n_bins, range=(0, n_bins), density=True)
    return hist


def extract_glcm_features(image: np.ndarray,
                          distances: Tuple[int, ...] = (1, 3, 5),
                          angles: Tuple[float, ...] = (0, np.pi/4, np.pi/2, 3*np.pi/4),
                          ) -> np.ndarray:
    img_u8 = image.astype(np.uint8) if image.dtype != np.uint8 else image
    props = []
    for d in distances:
        for a in angles:
            glcm = graycomatrix(img_u8, distances=[d], angles=[a],
                                levels=256, symmetric=True, normed=True)
            props.extend([
                graycoprops(glcm, "contrast")[0, 0],
                graycoprops(glcm, "correlation")[0, 0],
                graycoprops(glcm, "energy")[0, 0],
                graycoprops(glcm, "homogeneity")[0, 0],
            ])
    return np.array(props, dtype=np.float64)


def extract_edge_density(image: np.ndarray, low_threshold: float = 50,
                         high_threshold: float = 150) -> float:
    edges = cv2.Canny(image, low_threshold, high_threshold)
    return float(np.count_nonzero(edges)) / edges.size


def extract_features_separate(image: np.ndarray) -> dict:
    return {
        "hog": extract_hog_features(image),
        "lbp": extract_lbp_features(image),
        "glcm": extract_glcm_features(image),
        "edge_density": np.array([extract_edge_density(image)]),
    }


def _subsample_balanced(images, labels, max_samples, random_seed):
    rng = np.random.default_rng(random_seed)
    n_per_class = max_samples // 2
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    sp = rng.choice(pos_idx, min(n_per_class, len(pos_idx)), replace=False)
    sn = rng.choice(neg_idx, min(n_per_class, len(neg_idx)), replace=False)
    idx = np.concatenate([sp, sn])
    return images[idx], labels[idx]


# ============================================================
# 4. 数据管线 prepare_data（带缓存）
# ============================================================

_DATA_CACHE = {}  # key: (max_samples, tuple(preprocessing), tuple(features), seed)

def prepare_data(
    max_samples: int = 1000,
    random_seed: int = 42,
    split_method: str = "holdout",
    split_ratio: float = 0.7,
    preprocessing: list = None,
    features: list = None,
    use_stratify: bool = True,
    **kwargs,
) -> dict:
    """统一数据管线：加载→预处理→提特征→划分。支持内存缓存。"""
    if preprocessing is None:
        preprocessing = ["clahe", "median"]
    if features is None:
        features = ["hog", "lbp", "glcm", "edge_density"]

    # 缓存检查
    _cache_key = (max_samples, tuple(preprocessing), tuple(features), random_seed,
                  split_method, split_ratio, use_stratify)
    if _cache_key in _DATA_CACHE:
        print(f"prepare_data: 命中缓存 ({_DATA_CACHE[_cache_key]['config']['n_samples']} 样本)")
        return _DATA_CACHE[_cache_key]

    n_per_class = max_samples // 2
    images, labels = load_dataset(DATA_ROOT, max_samples=n_per_class)

    pipeline_map = {
        "none": lambda img: img,
        "clahe": lambda img: apply_clahe(img),
        "gaussian": lambda img: apply_gaussian_filter(img),
        "median": lambda img: apply_median_filter(img),
        "clahe+gaussian": lambda img: apply_gaussian_filter(apply_clahe(img)),
        "clahe+median": lambda img: apply_median_filter(apply_clahe(img)),
    }

    def compose_preprocess(img):
        for p in preprocessing:
            if p in pipeline_map and p != "none":
                img = pipeline_map[p](img)
        return img

    feat_map = {
        "hog": extract_hog_features,
        "lbp": extract_lbp_features,
        "glcm": extract_glcm_features,
        "edge_density": lambda img: np.array([extract_edge_density(img)]),
    }

    def extract_selected(img):
        parts = [feat_map[f](img) for f in features if f in feat_map]
        return np.concatenate(parts)

    # 构建特征名列表（用于特征重要性图）
    _sample_img = compose_preprocess(images[0])
    feature_names = []
    _offset = 0
    for f in features:
        if f not in feat_map:
            continue
        _fv = feat_map[f](_sample_img)
        _dim = len(_fv) if hasattr(_fv, '__len__') else 1
        if _dim <= 4:
            feature_names.extend([f"{f}"] * _dim)
        else:
            feature_names.extend([f"{f}[{i}]" for i in range(_dim)])
        _offset += _dim

    processed = np.array([compose_preprocess(img) for img in images])
    X_all = np.stack([extract_selected(img) for img in processed])
    y_all = labels

    test_size = round(1.0 - split_ratio, 4)
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=test_size,
        random_state=random_seed, stratify=y_all if use_stratify else None,
    )

    result = {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "raw_images": images, "raw_labels": labels,
        "feature_names": feature_names,
        "config": {
            "preprocessing": preprocessing, "features": features,
            "split_method": split_method, "split_ratio": split_ratio,
            "n_samples": len(labels), "feature_dim": X_train.shape[1],
        },
    }

    _DATA_CACHE[_cache_key] = result
    print(f"prepare_data: {len(y_train)} train / {len(y_test)} test, "
          f"dim={X_train.shape[1]}, preproc={preprocessing}, feats={features}")

    return result


# ============================================================
# 5. 模型定义 — CrackCNN + 损失函数
# ============================================================

class CrackCNN(nn.Module):
    """4 个卷积块 + 全局平均池化 + 分类头，~1.17M 参数。"""
    def __init__(self, num_classes=2, input_channels=1, dropout_rate=0.5):
        super().__init__()
        self.block1 = self._make_block(input_channels, 32)
        self.block2 = self._make_block(32, 64)
        self.block3 = self._make_block(64, 128)
        self.block4 = self._make_block(128, 256)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(256, num_classes)

    @staticmethod
    def _make_block(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        for block in [self.block1, self.block2, self.block3, self.block4]:
            x = block(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.classifier(x)


class FocalLoss(nn.Module):
    """Focal Loss: FL = -(1-p_t)^gamma * log(p_t)。alpha=None 时不加类别权重（适合均衡数据集）。"""
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        p_t = torch.exp(-ce_loss)
        focal_weight = (1 - p_t) ** self.gamma
        if self.alpha is not None:
            alpha_weight = torch.where(
                targets == 1,
                torch.tensor(self.alpha, device=inputs.device),
                torch.tensor(1 - self.alpha, device=inputs.device),
            )
            return (alpha_weight * focal_weight * ce_loss).mean()
        return (focal_weight * ce_loss).mean()


class LabelSmoothingCE(nn.Module):
    """Label Smoothing Cross Entropy。"""
    def __init__(self, epsilon=0.1, num_classes=2):
        super().__init__()
        self.epsilon = epsilon
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, self.num_classes).float()
        targets_smooth = targets_one_hot * (1 - self.epsilon) + self.epsilon / self.num_classes
        return (-targets_smooth * log_probs).sum(dim=1).mean()


class DiceLoss(nn.Module):
    """Dice Loss for binary classification。"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        probs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, inputs.size(1)).float()
        intersection = (probs * targets_one_hot).sum(dim=0)
        union = probs.sum(dim=0) + targets_one_hot.sum(dim=0)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class CrackDataset(Dataset):
    """PyTorch Dataset：resize 到 input_size×input_size，归一化到 [0,1]，
    可选预处理（CLAHE等）通过 preprocess_fn 在加载时应用。"""
    def __init__(self, images, labels, input_size=128, preprocess_fn=None):
        self.images = images
        self.labels = labels
        self.input_size = input_size
        self.preprocess_fn = preprocess_fn

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.preprocess_fn is not None:
            img = self.preprocess_fn(img)
        img = cv2.resize(img, (self.input_size, self.input_size))
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0
        return img_tensor, torch.tensor(self.labels[idx], dtype=torch.long)


# ============================================================
# 6. 图表绘制函数
# ============================================================

def _plot_confusion_matrix(y_true, y_pred, class_names=("无裂缝", "有裂缝")):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    acc = accuracy_score(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels([f"预测{class_names[0]}", f"预测{class_names[1]}"])
    ax.set_yticklabels([f"实际{class_names[0]}", f"实际{class_names[1]}"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16,
                    fontweight="bold", color="white" if cm[i, j] > cm.max()/2 else "black")
    ax.set_title(f"混淆矩阵 (准确率: {acc:.4f})")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig


def _plot_roc_curve(y_true, y_prob, label="模型"):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, color="tomato", linewidth=2, label=f"{label} (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", alpha=0.5, label="随机猜测")
    ax.set_xlabel("假阳性率 (FPR)"); ax.set_ylabel("真阳性率 (TPR)")
    ax.set_title("ROC 曲线"); ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def _plot_pr_curve(y_true, y_prob):
    precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(recalls, precisions, color="steelblue", linewidth=2, label=f"AP={ap:.4f}")
    ax.axhline(y=np.mean(y_true), color="gray", linestyle="--", alpha=0.5, label="Baseline")
    ax.set_xlabel("召回率 (Recall)"); ax.set_ylabel("精确率 (Precision)")
    ax.set_title("Precision-Recall 曲线"); ax.legend(loc="lower left"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def _plot_prob_distribution(y_prob, y_true):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    pos_prob = y_prob[y_true == 1]; neg_prob = y_prob[y_true == 0]
    ax.hist(neg_prob, bins=30, alpha=0.6, color="#2ecc71", label="无裂缝 (实际)", edgecolor="white")
    ax.hist(pos_prob, bins=30, alpha=0.6, color="#e74c3c", label="有裂缝 (实际)", edgecolor="white")
    ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=1.5, label="决策阈值 0.5")
    ax.set_xlabel("预测概率 (正类)"); ax.set_ylabel("样本数")
    ax.set_title("预测概率分布"); ax.legend()
    plt.tight_layout()
    return fig


def _plot_training_curves(history: dict):
    """history: {train_loss:[], val_loss:[], train_acc:[], val_acc:[]}"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    epochs = range(1, len(history.get("train_loss", [])) + 1)
    ax1.plot(epochs, history.get("train_loss", []), "b-", label="训练 Loss")
    ax1.plot(epochs, history.get("val_loss", []), "r-", label="验证 Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Loss 曲线")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(epochs, history.get("train_acc", []), "b-", label="训练 Acc")
    ax2.plot(epochs, history.get("val_acc", []), "r-", label="验证 Acc")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.set_title("Accuracy 曲线")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def _plot_feature_importance(importances, feature_names, top_n=20):
    """绘制特征重要性图，默认只显示 Top N 个特征。"""
    if importances is None or len(importances) == 0:
        return None
    # 只取 Top N
    n_show = min(top_n, len(importances))
    idx = np.argsort(importances)[-n_show:]  # 取最大的 n_show 个
    fig_height = max(3, n_show * 0.25)
    fig, ax = plt.subplots(figsize=(5.5, fig_height))
    ax.barh(range(n_show), importances[idx], color=plt.cm.Greens(np.linspace(0.3, 0.9, n_show)))
    ax.set_yticks(range(n_show))
    ax.set_yticklabels([feature_names[i] for i in idx])
    ax.set_xlabel("重要性")
    ax.set_title(f"特征重要性 (Top {n_show})")
    plt.tight_layout()
    return fig


def _plot_pca_clusters(X_2d, labels_pred, labels_true, method_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for lbl in np.unique(labels_pred):
        mask = labels_pred == lbl
        ax1.scatter(X_2d[mask, 0], X_2d[mask, 1], s=10, alpha=0.6, label=f"簇 {lbl}")
    ax1.set_title(f"{method_name} — 聚类结果"); ax1.legend(markerscale=3)
    for lbl in [0, 1]:
        mask = labels_true == lbl
        ax2.scatter(X_2d[mask, 0], X_2d[mask, 1], s=10, alpha=0.6,
                    label="有裂缝" if lbl == 1 else "无裂缝")
    ax2.set_title("真实标签"); ax2.legend(markerscale=3)
    plt.tight_layout()
    return fig


def _plot_silhouette(X_scaled, labels_pred):
    """绘制 Silhouette 轮廓图。需要至少2个簇且每个簇至少2个样本。"""
    try:
        from sklearn.metrics import silhouette_samples
        n_labels = len(set(labels_pred))
        if n_labels < 2 or n_labels >= len(labels_pred):
            return None
        sample_silhouette_values = silhouette_samples(X_scaled, labels_pred)
        fig, ax = plt.subplots(figsize=(5.5, 5))
        y_lower = 10
        for i in range(n_labels):
            ith_values = sample_silhouette_values[labels_pred == i]
            ith_values.sort()
            size_i = ith_values.shape[0]
            y_upper = y_lower + size_i
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_values, alpha=0.7)
            ax.text(-0.05, y_lower + 0.5 * size_i, str(i))
            y_lower = y_upper + 10
        ax.axvline(x=np.mean(sample_silhouette_values), color="red", linestyle="--")
        ax.set_xlabel("轮廓系数"); ax.set_ylabel("簇"); ax.set_title("Silhouette 轮廓图")
        plt.tight_layout()
        return fig
    except Exception:
        return None


def _plot_not_available(reason: str = "当前设置下不可用") -> plt.Figure:
    """生成 N/A 占位图，用于无法生成的图表位。"""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.text(0.5, 0.5, f"N/A\n{reason}", ha="center", va="center",
            fontsize=16, color="gray", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.tight_layout()
    return fig


# ============================================================
# 7. 链路执行引擎
# ============================================================

def _build_traditional_model(model_name, params):
    """构建传统 ML 模型实例。"""
    if model_name == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=params.get("max_depth", 15),
            criterion=params.get("criterion", "gini"),
            min_samples_split=params.get("min_samples_split", 5),
            random_state=params.get("random_state", 42),
        )
    elif model_name == "svm":
        return SVC(
            kernel=params.get("kernel", "rbf"),
            C=params.get("C", 1.0),
            gamma=params.get("gamma", "scale"),
            probability=True,
            random_state=params.get("random_state", 42),
        )
    elif model_name == "naive_bayes":
        return GaussianNB(var_smoothing=params.get("var_smoothing", 1e-9))
    elif model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 20),
            criterion=params.get("criterion", "gini"),
            min_samples_split=params.get("min_samples_split", 5),
            random_state=params.get("random_state", 42),
            n_jobs=-1,
        )
    elif model_name == "logistic_regression":
        lr_kwargs = {
            "C": params.get("C", 1.0),
            "penalty": params.get("penalty", "l2"),
            "solver": params.get("solver", "lbfgs"),
            "max_iter": 2000,
            "random_state": params.get("random_state", 42),
        }
        # elasticnet 正则化需要 l1_ratio 参数
        if lr_kwargs["penalty"] == "elasticnet":
            lr_kwargs["l1_ratio"] = params.get("l1_ratio", 0.5)
        return LogisticRegression(**lr_kwargs)
    elif model_name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 6),
            learning_rate=params.get("learning_rate", 0.1),
            subsample=params.get("subsample", 0.8),
            objective=params.get("objective", "binary:logistic"),
            random_state=params.get("random_state", 42),
            n_jobs=-1, verbosity=0,
        )
    elif model_name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 6),
            learning_rate=params.get("learning_rate", 0.1),
            num_leaves=params.get("num_leaves", 31),
            objective=params.get("objective", "binary"),
            random_state=params.get("random_state", 42),
            n_jobs=-1, verbose=-1,
        )
    else:
        raise ValueError(f"未知传统模型: {model_name}")


def _get_param_grid(model_name):
    """返回每个模型的 GridSearchCV 参数网格。"""
    grids = {
        "decision_tree": {
            "max_depth": [3, 5, 10, 15, 20, None],
            "criterion": ["gini", "entropy"],
            "min_samples_split": [2, 5, 10],
        },
        "svm": {
            "kernel": ["linear", "rbf", "poly"],
            "C": [0.1, 1, 10],
            "gamma": ["scale", "auto"],
        },
        "naive_bayes": {
            "var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3],
        },
        "random_forest": {
            "n_estimators": [50, 100, 200, 500],
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10],
        },
        # 逻辑回归需按 solver/penalty 兼容性分组（list-of-dicts 格式）
        # lbfgs: 仅 l2; liblinear: l1+l2; saga: l1+l2+elasticnet
        "logistic_regression": [
            {"C": [0.01, 0.1, 1, 10, 100],
             "penalty": ["l2"],
             "solver": ["lbfgs", "liblinear", "saga"]},
            {"C": [0.01, 0.1, 1, 10, 100],
             "penalty": ["l1"],
             "solver": ["liblinear", "saga"]},
            {"C": [0.01, 0.1, 1, 10, 100],
             "penalty": ["elasticnet"],
             "solver": ["saga"],
             "l1_ratio": [0.25, 0.5, 0.75]},
        ],
        "xgboost": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 6, 9],
            "learning_rate": [0.01, 0.1, 0.3],
            "subsample": [0.8, 1.0],
        },
        "lightgbm": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 6, 9],
            "num_leaves": [31, 63, 127],
            "learning_rate": [0.01, 0.1, 0.3],
        },
    }
    return grids.get(model_name, {})


def _run_traditional(model_name, params, data, optimization, cv_folds, scoring,
                     validation_method, random_seed, n_iter=30):
    """执行传统 ML 模型链路。"""
    from sklearn.pipeline import Pipeline

    status_msgs = []
    X_tr, X_te = data["X_train"], data["X_test"]
    y_tr, y_te = data["y_train"], data["y_test"]

    # SVM/NB/LR 需要 StandardScaler；DT/RF/XGBoost/LightGBM 不需要
    needs_pipeline = model_name in ("svm", "naive_bayes", "logistic_regression")

    model_path = TRAD_DIR / f"{model_name}_best.joblib"
    best_params = params.copy()

    t0 = time.time()
    if optimization == "pretrained":
        if model_path.exists():
            # 安全说明: 加载本项目自身训练产出的 sklearn 模型，来源可信。
            model = joblib.load(model_path)
            # 用当前数据重新 fit（若为 Pipeline 则内部含 scaler）
            model.fit(X_tr, y_tr)
            status_msgs.append(f"✅ 已加载预训练模型: {model_path.name}")
        else:
            status_msgs.append(f"⚠️ 预训练模型未找到: {model_path.name}，改用 GridSearchCV")
            optimization = "grid_search"

    if optimization == "manual":
        if needs_pipeline:
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", _build_traditional_model(model_name, params)),
            ])
        else:
            model = _build_traditional_model(model_name, params)
        model.fit(X_tr, y_tr)
        status_msgs.append("✅ 使用手动参数训练完成")

    elif optimization in ("grid_search", "random_search"):
        param_grid = _get_param_grid(model_name)
        if not param_grid:
            status_msgs.append(f"⚠️ {model_name} 无预定义参数网格，使用手动参数")
            if needs_pipeline:
                model = Pipeline([("scaler", StandardScaler()),
                                  ("clf", _build_traditional_model(model_name, params))])
            else:
                model = _build_traditional_model(model_name, params)
            model.fit(X_tr, y_tr)
        else:
            if needs_pipeline:
                base_model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", _build_traditional_model(model_name, params)),
                ])
                # 兼容 list-of-dicts 格式（如逻辑回归按 solver/penalty 分组）
                if isinstance(param_grid, list):
                    pg = [{f"clf__{k}": v for k, v in d.items()} for d in param_grid]
                else:
                    pg = {f"clf__{k}": v for k, v in param_grid.items()}
            else:
                base_model = _build_traditional_model(model_name, params)
                pg = param_grid
            if optimization == "grid_search":
                search = GridSearchCV(base_model, pg, cv=cv_folds,
                                      scoring=scoring, n_jobs=-1, verbose=0)
            else:
                search = RandomizedSearchCV(base_model, pg, n_iter=n_iter,
                                            cv=cv_folds, scoring=scoring, n_jobs=-1,
                                            random_state=random_seed)
            search.fit(X_tr, y_tr)
            model = search.best_estimator_
            best_params.update(search.best_params_)
            status_msgs.append(f"✅ {optimization}完成, 最优参数: {search.best_params_}")

    elapsed = time.time() - t0
    status_msgs.append(f"⏱ 训练耗时: {elapsed:.1f}s")

    # 验证评估
    y_pred = model.predict(X_te)
    try:
        y_prob = model.predict_proba(X_te)[:, 1]
    except Exception:
        y_prob = np.zeros_like(y_pred, dtype=float)

    # binary:hinge 不输出概率，ROC-AUC/PR 曲线不可用，给出用户提示
    _hinge_no_prob = (model_name == "xgboost"
                      and params.get("objective") == "binary:hinge")
    if _hinge_no_prob:
        status_msgs.append(
            "⚠️ binary:hinge 仅输出硬标签，无法产生概率估计，"
            "ROC-AUC 和 PR 曲线不可用。如需概率输出请改用 binary:logistic。"
        )

    cv_text = ""
    cv_metrics = {}  # kfold 均值±标准差，用于替换摘要指标
    if validation_method == "kfold":
        from sklearn.model_selection import cross_validate as _cv
        X_all_cv = np.vstack([X_tr, X_te])
        y_all_cv = np.concatenate([y_tr, y_te])
        scoring_list = ["accuracy", "f1", "precision", "recall", "roc_auc"]
        cv_res = _cv(model, X_all_cv, y_all_cv, cv=cv_folds,
                     scoring=scoring_list, n_jobs=-1)
        # 构建 kfold 均值±标准差
        for sc in scoring_list:
            key = f"test_{sc}"
            m, s = cv_res[key].mean(), cv_res[key].std()
            cv_metrics[sc] = f"{m:.4f}±{s:.4f}"
        # 构建每折指标表格
        cv_rows = []
        for fold_i in range(cv_folds):
            row = [f"Fold {fold_i+1}"]
            for sc in scoring_list:
                key = f"test_{sc}"
                row.append(f"{cv_res[key][fold_i]:.4f}")
            cv_rows.append("| " + " | ".join(row) + " |")
        # 均值±标准差
        mean_row = ["**均值±std**"]
        for sc in scoring_list:
            key = f"test_{sc}"
            m, s = cv_res[key].mean(), cv_res[key].std()
            mean_row.append(f"**{m:.4f}±{s:.4f}**")
        cv_rows.append("| " + " | ".join(mean_row) + " |")
        header = "| Fold | " + " | ".join(scoring_list) + " |"
        sep = "|------|" + "|".join(["------"] * len(scoring_list)) + "|"
        cv_table = "\n".join([header, sep] + cv_rows)
        cv_text = f"\n\n#### 📊 {cv_folds} 折交叉验证\n\n{cv_table}"

    metrics = {
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "precision": float(precision_score(y_te, y_pred, zero_division=0)),
        "recall": float(recall_score(y_te, y_pred, zero_division=0)),
        "f1": float(f1_score(y_te, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_te, y_prob)) if y_prob.max() > 0 else 0.0,
    }

    # 图表
    cm_fig = _plot_confusion_matrix(y_te, y_pred)
    if _hinge_no_prob:
        roc_fig = _plot_not_available("binary:hinge 无概率输出")
        pr_fig = _plot_not_available("binary:hinge 无概率输出")
    else:
        roc_fig = _plot_roc_curve(y_te, y_prob, model_name) if y_prob.max() > 0 else None
        pr_fig = _plot_pr_curve(y_te, y_prob) if y_prob.max() > 0 else None

    # 特征重要性（仅树模型）
    fi_fig = None
    tree_models = ("decision_tree", "random_forest", "xgboost", "lightgbm")
    if model_name in tree_models:
        try:
            imp = model.feature_importances_
            # 使用实际特征名（如果有），否则用通用名
            feat_names = data.get("feature_names")
            if feat_names is None or len(feat_names) != X_tr.shape[1]:
                feat_names = [f"特征{i}" for i in range(X_tr.shape[1])]
            fi_fig = _plot_feature_importance(imp[:len(feat_names)], feat_names)
        except Exception:
            pass

    prob_fig = _plot_prob_distribution(y_prob, y_te) if not _hinge_no_prob and y_prob.max() > 0 else (
        _plot_not_available("binary:hinge 无概率输出") if _hinge_no_prob else None)

    # 构建摘要指标表（kfold 时使用均值±标准差格式）
    _acc = cv_metrics.get("accuracy", f"{metrics['accuracy']:.4f}")
    _prec = cv_metrics.get("precision", f"{metrics['precision']:.4f}")
    _rec = cv_metrics.get("recall", f"{metrics['recall']:.4f}")
    _f1 = cv_metrics.get("f1", f"{metrics['f1']:.4f}")
    _auc = cv_metrics.get("roc_auc", f"{metrics['roc_auc']:.4f}")

    metrics_md = (
        f"### 📈 评估指标 ({model_name})\n\n"
        f"| 指标 | 值 |\n|------|------|\n"
        f"| 准确率 | {_acc} |\n"
        f"| 精确率 | {_prec} |\n"
        f"| 召回率 | {_rec} |\n"
        f"| F1分数 | {_f1} |\n"
        f"| ROC-AUC | {_auc} |\n"
        f"{cv_text}\n"
        f"\n⏱ 耗时: {elapsed:.1f}s | 最优参数: {best_params}"
    )

    return {
        "status": "\n".join(status_msgs),
        "metrics_md": metrics_md,
        "metrics": metrics,
        "cm_fig": cm_fig,
        "roc_fig": roc_fig,
        "pr_fig": pr_fig,
        "fi_fig": fi_fig,
        "prob_fig": prob_fig,
        "extra_fig": None,
    }


def _run_cnn(params, data, optimization, random_seed, n_iter=6, scoring_metric="f1"):
    """执行 CNN 模型链路。"""
    status_msgs = []
    loss_fn_name = params.get("loss_fn", "cross_entropy")

    # 预处理函数
    preproc_list = data["config"]["preprocessing"]
    def cnn_preprocess(img):
        for p in preproc_list:
            if p == "clahe":
                img = apply_clahe(img)
            elif p == "gaussian":
                img = apply_gaussian_filter(img)
            elif p == "median":
                img = apply_median_filter(img)
        return img

    raw_images = data["raw_images"]
    raw_labels = data["raw_labels"]

    # 划分训练/测试集（CNN 用原始图像，不用手工特征）
    test_size = round(1.0 - data["config"]["split_ratio"], 4)
    train_imgs, test_imgs, y_tr, y_te = train_test_split(
        raw_images, raw_labels, test_size=test_size,
        random_state=random_seed, stratify=raw_labels,
    )
    # 再从训练集分验证集
    train_imgs, val_imgs, y_tr, y_val = train_test_split(
        train_imgs, y_tr, test_size=0.15,
        random_state=random_seed, stratify=y_tr,
    )

    input_size = params.get("input_size", 128)
    train_ds = CrackDataset(train_imgs, y_tr, input_size=input_size, preprocess_fn=cnn_preprocess)
    val_ds = CrackDataset(val_imgs, y_val, input_size=input_size, preprocess_fn=cnn_preprocess)
    test_ds = CrackDataset(test_imgs, y_te, input_size=input_size, preprocess_fn=cnn_preprocess)

    bs = params.get("batch_size", 64)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)

    dropout = params.get("dropout_rate", 0.5)
    lr = float(params.get("learning_rate", 0.001))
    epochs = params.get("epochs", 30)
    patience = params.get("early_stopping_patience", 10)
    weight_decay = float(params.get("weight_decay", 1e-4))

    # 确定预训练模型路径
    cnn_path = None
    if loss_fn_name == "cross_entropy":
        cnn_path = CNN_DIR / "crackcnn_cross_entropy_best.pth"
    elif loss_fn_name == "focal":
        alpha = params.get("focal_alpha")
        gamma = params.get("focal_gamma", 2.0)
        if alpha is None and gamma == 2.0:
            cnn_path = CNN_DIR / "crackcnn_focal_gamma2_best.pth"
        elif alpha is None and gamma == 3.0:
            cnn_path = CNN_DIR / "crackcnn_focal_gamma3_best.pth"
        elif alpha == 0.5 and gamma == 2.0:
            cnn_path = CNN_DIR / "crackcnn_focal_balanced_best.pth"
    elif loss_fn_name == "label_smoothing":
        cnn_path = CNN_DIR / "crackcnn_label_smoothing_best.pth"
    elif loss_fn_name == "dice":
        cnn_path = CNN_DIR / "crackcnn_dice_best.pth"

    model = CrackCNN(dropout_rate=dropout).to(DEVICE)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    t0 = time.time()
    if optimization == "pretrained" and cnn_path and cnn_path.exists():
        state_dict = torch.load(cnn_path, map_location=DEVICE, weights_only=True)
        # 兼容旧版权重文件 (c1/c2/c3/c4/cls → block1/block2/block3/block4/classifier)
        _LEGACY_KEY_MAP = {
            "c1": "block1", "c2": "block2", "c3": "block3", "c4": "block4",
            "cls": "classifier",
        }
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k
            for old_prefix, new_prefix in _LEGACY_KEY_MAP.items():
                if k.startswith(f"{old_prefix}."):
                    new_key = f"{new_prefix}.{k[len(old_prefix)+1:]}"
                    break
            new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict)
        model.eval()
        status_msgs.append(f"✅ 已加载预训练模型: {cnn_path.name}")
    else:
        if optimization == "pretrained":
            status_msgs.append("⚠️ 预训练模型未找到，使用完整训练")
            optimization = "manual"
        else:
            status_msgs.append("🔄 开始 CNN 完整训练...")

        # 构建损失函数 (搜索和训练共用)
        if loss_fn_name == "cross_entropy":
            criterion = nn.CrossEntropyLoss()
        elif loss_fn_name == "focal":
            alpha_val = params.get("focal_alpha")
            gamma_val = float(params.get("focal_gamma", 2.0))
            if alpha_val is not None:
                alpha_val = float(alpha_val)
            criterion = FocalLoss(alpha=alpha_val, gamma=gamma_val)
        elif loss_fn_name == "label_smoothing":
            criterion = LabelSmoothingCE(epsilon=float(params.get("label_smoothing_epsilon", 0.1)))
        elif loss_fn_name == "dice":
            criterion = DiceLoss()
        else:
            criterion = nn.CrossEntropyLoss()

        # ---- CNN 参数搜索 ----
        if optimization in ("grid_search", "random_search"):
            cnn_grid = {
                "learning_rate": [1e-4, 5e-4, 1e-3],
                "dropout_rate": [0.3, 0.5, 0.7],
                "batch_size": [32, 64],
            }
            import itertools as _it
            all_combos = [dict(zip(cnn_grid, v))
                          for v in _it.product(*cnn_grid.values())]
            if optimization == "random_search":
                rng = np.random.default_rng(random_seed)
                n_iter_search = min(int(n_iter), len(all_combos))
                indices = rng.choice(len(all_combos), n_iter_search, replace=False)
                all_combos = [all_combos[i] for i in indices]

            search_epochs = 10
            status_msgs.append(f"🔍 CNN {optimization}: {len(all_combos)} 组参数 × {search_epochs} epochs...")
            best_score, best_combo = -1, all_combos[0]
            for ci, combo in enumerate(all_combos):
                _m = CrackCNN(dropout_rate=combo["dropout_rate"]).to(DEVICE)
                _opt = torch.optim.Adam(_m.parameters(), lr=combo["learning_rate"],
                                        weight_decay=weight_decay)
                _bs = combo["batch_size"]
                _tl = DataLoader(train_ds, batch_size=_bs, shuffle=True)
                _vl = DataLoader(val_ds, batch_size=_bs, shuffle=False)
                _m.train()
                for _ep in range(search_epochs):
                    for _inp, _tgt in _tl:
                        _inp, _tgt = _inp.to(DEVICE), _tgt.to(DEVICE)
                        _opt.zero_grad()
                        _out = _m(_inp)
                        _loss = criterion(_out, _tgt)
                        _loss.backward(); _opt.step()
                _m.eval()
                _preds, _tgts = [], []
                with torch.no_grad():
                    for _inp, _tgt in _vl:
                        _inp = _inp.to(DEVICE)
                        _out = _m(_inp)
                        _, _p = _out.max(1)
                        _preds.extend(_p.cpu().numpy())
                        _tgts.extend(_tgt.numpy())
                if scoring_metric == "accuracy":
                    _score = accuracy_score(_tgts, _preds)
                elif scoring_metric == "precision":
                    _score = precision_score(_tgts, _preds, zero_division=0)
                elif scoring_metric == "recall":
                    _score = recall_score(_tgts, _preds, zero_division=0)
                elif scoring_metric == "roc_auc":
                    _score = roc_auc_score(_tgts, _preds)
                else:  # f1
                    _score = f1_score(_tgts, _preds, zero_division=0)
                status_msgs.append(f"  [{ci+1}/{len(all_combos)}] lr={combo['learning_rate']}, "
                                   f"drop={combo['dropout_rate']}, bs={combo['batch_size']} "
                                   f"→ val_{scoring_metric}={_score:.4f}")
                if _score > best_score:
                    best_score = _score
                    best_combo = combo
                del _m, _opt
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            lr = float(best_combo["learning_rate"])
            dropout = float(best_combo["dropout_rate"])
            bs = int(best_combo["batch_size"])
            train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)
            model = CrackCNN(dropout_rate=dropout).to(DEVICE)
            status_msgs.append(f"✅ CNN 搜索完成: 最优 val_{scoring_metric}={best_score:.4f}, "
                               f"lr={lr}, dropout={dropout}, bs={bs}")

        opt_name = params.get("optimizer", "adam")
        if opt_name == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                        weight_decay=weight_decay)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward(); optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                _, pred = outputs.max(1)
                total += targets.size(0); correct += pred.eq(targets).sum().item()
            train_loss = running_loss / total
            train_acc = correct / total
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
                    _, pred = outputs.max(1)
                    val_total += targets.size(0); val_correct += pred.eq(targets).sum().item()
            val_loss_epoch = val_loss / val_total
            val_acc_epoch = val_correct / val_total
            history["val_loss"].append(val_loss_epoch)
            history["val_acc"].append(val_acc_epoch)
            scheduler.step(val_loss_epoch)

            if val_loss_epoch < best_val_loss:
                best_val_loss = val_loss_epoch
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                status_msgs.append(f"⏹ 早停于 Epoch {epoch}/{epochs}")
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        elapsed = time.time() - t0
        status_msgs.append(f"✅ CNN 训练完成 | ⏱ {elapsed:.1f}s | 最佳 Val Loss: {best_val_loss:.4f}")
        model.eval()

    # 评估
    all_preds, all_probs, all_targets = [], [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, pred = outputs.max(1)
            all_preds.extend(pred.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_targets.extend(targets.numpy())

    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    y_true = np.array(all_targets)
    elapsed = time.time() - t0

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }

    cm_fig = _plot_confusion_matrix(y_true, y_pred)
    roc_fig = _plot_roc_curve(y_true, y_prob, f"CNN({loss_fn_name})")
    pr_fig = _plot_pr_curve(y_true, y_prob)
    prob_fig = _plot_prob_distribution(y_prob, y_true)
    train_fig = _plot_training_curves(history) if history["train_loss"] else None

    metrics_md = (
        f"### 📈 评估指标 (CNN, loss={loss_fn_name})\n\n"
        f"| 指标 | 值 |\n|------|------|\n"
        f"| 准确率 | {metrics['accuracy']:.4f} |\n"
        f"| 精确率 | {metrics['precision']:.4f} |\n"
        f"| 召回率 | {metrics['recall']:.4f} |\n"
        f"| F1分数 | {metrics['f1']:.4f} |\n"
        f"| ROC-AUC | {metrics['roc_auc']:.4f} |\n"
        f"\n⏱ 耗时: {elapsed:.1f}s | 优化器: {params.get('optimizer','adam')} | "
        f"lr={lr} | dropout={dropout} | input_size={input_size} | wd={weight_decay}"
    )

    return {
        "status": "\n".join(status_msgs),
        "metrics_md": metrics_md,
        "metrics": metrics,
        "cm_fig": cm_fig,
        "roc_fig": roc_fig,
        "pr_fig": pr_fig,
        "fi_fig": None,
        "prob_fig": prob_fig,
        "extra_fig": train_fig,
    }


def _run_unsupervised(method, params, data, optimization, random_seed, n_iter=5,
                      unsup_val_method="internal_external"):
    """执行无监督聚类链路。"""
    status_msgs = []
    X_all = np.vstack([data["X_train"], data["X_test"]])
    y_all = np.concatenate([data["y_train"], data["y_test"]])

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    n_clusters = params.get("n_clusters", 2)
    model = None
    labels_pred = None

    t0 = time.time()
    if optimization == "pretrained":
        model_path = UNSUP_DIR / f"{method}_best.joblib"
        if model_path.exists():
            # 安全说明: joblib 加载本项目自身训练产出的 sklearn 模型，来源可信。
            # sklearn 模型无安全替代序列化方案（官方推荐 joblib），当前场景可接受。
            model = joblib.load(model_path)
            # 优先使用 predict()（对当前数据预测），而非 labels_（原始训练集标签）
            if hasattr(model, "predict"):
                try:
                    labels_pred = model.predict(X_scaled)
                    status_msgs.append(f"✅ 已加载预训练模型: {model_path.name}")
                except Exception as e:
                    # 维度不匹配等异常 → 回退到 manual 模式
                    status_msgs.append(
                        f"⚠️ 预训练模型 {model_path.name} 不兼容当前特征 "
                        f"({type(e).__name__})，回退到现场训练"
                    )
                    optimization = "manual"
                    model = None
            else:
                # Agglomerative/Spectral 无 predict() 方法，必须重新训练
                status_msgs.append(
                    f"⚠️ {method} 不支持 predict()，使用现场训练"
                )
                optimization = "manual"
                model = None
        else:
            status_msgs.append(f"⚠️ 预训练模型未找到: {method}，现场训练")
            optimization = "manual"

    if optimization != "pretrained" or model is None:
        # ---- 无监督参数搜索 ----
        if optimization in ("grid_search", "random_search"):
            from sklearn.cluster import KMeans as _KM, DBSCAN as _DB
            from sklearn.mixture import GaussianMixture as _GMM
            from sklearn.cluster import AgglomerativeClustering as _AG
            from sklearn.cluster import SpectralClustering as _SC

            search_grids = {
                "kmeans": {"n_clusters": [2, 3, 4, 5, 6, 8, 10]},
                "gmm": {"covariance_type": ["full", "tied", "diag", "spherical"]},
                "dbscan": {"eps": [0.3, 0.5, 0.8, 1.0, 1.5], "min_samples": [3, 5, 10, 20]},
                "agglomerative": {"linkage": ["ward", "complete", "average", "single"]},
                "spectral": {"affinity": ["rbf", "nearest_neighbors"]},
            }
            grid = search_grids.get(method, {})
            if grid:
                import itertools as _it
                keys = list(grid.keys())
                combos = [dict(zip(keys, v)) for v in _it.product(*grid.values())]
                if optimization == "random_search":
                    rng = np.random.default_rng(random_seed)
                    n_pick = min(int(n_iter), len(combos))
                    combos = [combos[i] for i in rng.choice(len(combos), n_pick, replace=False)]

                status_msgs.append(f"🔍 {method} {optimization}: {len(combos)} 组参数...")
                best_sil, best_combo, best_labels = -1, combos[0], None

                for ci, combo in enumerate(combos):
                    try:
                        if method == "kmeans":
                            _m = _KM(n_clusters=combo["n_clusters"], random_state=random_seed, n_init="auto")
                        elif method == "gmm":
                            _m = _GMM(n_components=n_clusters, covariance_type=combo["covariance_type"],
                                      random_state=random_seed)
                        elif method == "dbscan":
                            _m = _DB(eps=combo["eps"], min_samples=combo["min_samples"])
                        elif method == "agglomerative":
                            _m = _AG(n_clusters=n_clusters, linkage=combo["linkage"])
                        elif method == "spectral":
                            _m = _SC(n_clusters=n_clusters, affinity=combo["affinity"],
                                     random_state=random_seed, n_init=10)
                        else:
                            continue

                        _labels = _m.fit_predict(X_scaled)
                        n_clust = len(set(_labels) - {-1})
                        if n_clust < 2:
                            status_msgs.append(f"  [{ci+1}] {combo} → 仅 {n_clust} 簇，跳过")
                            continue
                        _sil = silhouette_score(X_scaled, _labels)
                        status_msgs.append(f"  [{ci+1}/{len(combos)}] {combo} → sil={_sil:.4f}, 簇数={n_clust}")
                        if _sil > best_sil:
                            best_sil = _sil
                            best_combo = combo
                            best_labels = _labels
                            model = _m
                    except Exception as _e:
                        status_msgs.append(f"  [{ci+1}] {combo} → 失败: {_e}")

                if best_labels is not None:
                    labels_pred = best_labels
                    status_msgs.append(f"✅ {method} 搜索完成: 最优 sil={best_sil:.4f}, 参数={best_combo}")
                else:
                    status_msgs.append("⚠️ 搜索未找到有效参数组合，回退到手动参数")
                    optimization = "manual"

        # ---- 手动参数训练 ----
        if optimization == "manual" or labels_pred is None:
            if method == "kmeans":
                from sklearn.cluster import KMeans
                model = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init="auto",
                               algorithm=params.get("algorithm", "lloyd"))
            elif method == "gmm":
                from sklearn.mixture import GaussianMixture
                model = GaussianMixture(n_components=n_clusters,
                                        covariance_type=params.get("covariance_type", "full"),
                                        random_state=random_seed)
            elif method == "dbscan":
                from sklearn.cluster import DBSCAN
                model = DBSCAN(eps=float(params.get("eps", 0.5)),
                               min_samples=int(params.get("min_samples", 5)))
            elif method == "agglomerative":
                from sklearn.cluster import AgglomerativeClustering
                model = AgglomerativeClustering(n_clusters=n_clusters,
                                                linkage=params.get("linkage", "ward"))
            elif method == "spectral":
                from sklearn.cluster import SpectralClustering
                model = SpectralClustering(n_clusters=n_clusters,
                                           affinity=params.get("affinity", "rbf"),
                                           random_state=random_seed, n_init=10)
            else:
                raise ValueError(f"未知聚类方法: {method}")

            labels_pred = model.fit_predict(X_scaled)

            # DBSCAN: 检测并处理全噪声情况
            if method == "dbscan" and len(set(labels_pred)) <= 1 and all(lbl == -1 for lbl in labels_pred):
                from sklearn.neighbors import NearestNeighbors
                k = int(params.get("min_samples", 5))
                nn = NearestNeighbors(n_neighbors=k)
                nn.fit(X_scaled)
                distances, _ = nn.kneighbors(X_scaled)
                k_distances = np.sort(distances[:, -1])
                auto_eps = float(np.percentile(k_distances, 75))
                model = DBSCAN(eps=auto_eps, min_samples=k)
                labels_pred = model.fit_predict(X_scaled)
                status_msgs.append(f"⚠️ 默认 eps 过小，自动调整为 eps={auto_eps:.3f}")

            status_msgs.append(f"✅ {method} 训练完成")

    elapsed = time.time() - t0

    # 内部指标
    try:
        sil = silhouette_score(X_scaled, labels_pred) if len(set(labels_pred)) > 1 else float("nan")
        db = davies_bouldin_score(X_scaled, labels_pred) if len(set(labels_pred)) > 1 else float("nan")
        ch = calinski_harabasz_score(X_scaled, labels_pred) if len(set(labels_pred)) > 1 else float("nan")
    except Exception:
        sil = db = ch = float("nan")

    # 外部指标
    try:
        ari = adjusted_rand_score(y_all, labels_pred)
        nmi = normalized_mutual_info_score(y_all, labels_pred)
    except Exception:
        ari = nmi = float("nan")

    # 聚类质量警告：当 Sil<0.1 且 ARI<0.15 时提醒用户
    _sil_val = float(sil) if not (isinstance(sil, float) and (sil != sil)) else -1.0
    _ari_val = float(ari) if not (isinstance(ari, float) and (ari != ari)) else -1.0
    if _sil_val < 0.1 and _ari_val < 0.15:
        status_msgs.append(
            f"⚠️ 聚类质量极低（Silhouette={_sil_val:.4f}, ARI={_ari_val:.4f}），"
            "无监督方法可能不适合此任务。建议尝试监督学习方法。"
        )

    # PCA 降维可视化
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)

    cluster_fig = _plot_pca_clusters(X_2d, labels_pred, y_all, method)
    sil_fig = _plot_silhouette(X_scaled, labels_pred)

    # 根据评估范围构建指标表
    _internal_rows = (
        f"| 内部 | 轮廓系数 | {sil:.4f} |\n"
        f"| 内部 | Davies-Bouldin | {db:.4f} |\n"
        f"| 内部 | Calinski-Harabasz | {ch:.2f} |\n"
    )
    _external_rows = (
        f"| 外部 | ARI | {ari:.4f} |\n"
        f"| 外部 | NMI | {nmi:.4f} |\n"
    )
    _show_internal = unsup_val_method in ("internal_external", "internal_only")
    _show_external = unsup_val_method in ("internal_external", "external_only")
    _rows = ""
    if _show_internal:
        _rows += _internal_rows
    if _show_external:
        _rows += _external_rows

    metrics_md = (
        f"### 📈 聚类评估 ({method})\n\n"
        f"| 指标类别 | 指标 | 值 |\n|------|------|------|\n"
        f"{_rows}"
        f"\n⏱ 耗时: {elapsed:.1f}s | 簇数: {len(set(labels_pred))}"
    )

    return {
        "status": "\n".join(status_msgs),
        "metrics_md": metrics_md,
        "metrics": {
            "silhouette": float(sil) if not (isinstance(sil, float) and (sil != sil)) else None,
            "davies_bouldin": float(db) if not (isinstance(db, float) and (db != db)) else None,
            "calinski_harabasz": float(ch) if not (isinstance(ch, float) and (ch != ch)) else None,
            "ari": float(ari) if not (isinstance(ari, float) and (ari != ari)) else None,
            "nmi": float(nmi) if not (isinstance(nmi, float) and (nmi != nmi)) else None,
            "n_clusters_found": len(set(labels_pred)),
        },
        "cm_fig": None,
        "roc_fig": None,
        "pr_fig": None,
        "fi_fig": None,
        "prob_fig": None,
        "extra_fig": cluster_fig,
        "sil_fig": sil_fig,
    }


# ============================================================
# 8. Gradio 统一入口：run_pipeline()
# ============================================================

def run_pipeline(
    # Step 1: 数据处理
    split_method, split_ratio, use_stratify,
    preprocessing, features, max_samples,
    # Step 2: 模型选择
    model_name,
    # Step 3: 模型超参数 (所有模型 — 仅选中模型的参数生效)
    dt_max_depth, dt_min_samples_split,
    svm_C,
    nb_var_smoothing,
    rf_n_estimators, rf_max_depth, rf_min_samples_split,
    lr_C,
    xgb_n_estimators, xgb_max_depth, xgb_subsample,
    lgbm_n_estimators, lgbm_max_depth, lgbm_num_leaves,
    cnn_dropout, cnn_batch_size, cnn_epochs, cnn_early_stopping,
    cnn_input_size, cnn_weight_decay,
    unsup_n_clusters, unsup_eps, unsup_min_samples,
    # Step 4: 损失函数/优化器 (所有模型)
    dt_criterion,
    svm_kernel, svm_gamma,
    rf_criterion,
    lr_penalty, lr_solver, lr_l1_ratio,
    xgb_objective, xgb_learning_rate,
    lgbm_objective, lgbm_learning_rate,
    cnn_loss_fn, cnn_focal_alpha, cnn_focal_gamma,
    cnn_label_smoothing_epsilon, cnn_optimizer, cnn_learning_rate,
    kmeans_algorithm, gmm_covariance_type,
    agg_linkage, spec_affinity,
    # Step 5: 参数优化 + 验证 + 指标
    optimization_strategy, cv_folds_opt, n_iter,
    validation_method, scoring_metric,
    unsup_val_method,
    # 通用
    random_seed,
    progress=gr.Progress(track_tqdm=False),
):
    """Gradio 统一入口：执行完整训练链路。"""
    t_total = time.time()
    status_msgs = []

    try:
        # ---- 准备数据 ----
        progress(0.05, desc="加载数据...")
        status_msgs.append("⏳ 正在加载数据...")
        if preprocessing is None:
            preprocessing = "clahe+median"
        if features is None:
            features = ["hog", "lbp", "glcm", "edge_density"]

        # preprocessing 是 Radio 单选 (str)，转为 list 供 prepare_data 使用
        preprocessing_list = [preprocessing] if isinstance(preprocessing, str) else list(preprocessing)

        data = prepare_data(
            max_samples=int(max_samples),
            random_seed=int(random_seed),
            split_method=split_method,
            split_ratio=float(split_ratio),
            preprocessing=preprocessing_list,
            features=list(features),
            use_stratify=use_stratify,
        )
        progress(0.2, desc="数据准备完成，构建模型参数...")
        status_msgs.append(f"✅ 数据准备完成: {data['config']['n_samples']} 样本, "
                           f"{data['X_train'].shape[1]} 维特征")

        # ---- 构建模型参数 ----
        # 传统模型参数
        trad_params_map = {
            "decision_tree": {
                "max_depth": int(dt_max_depth),
                "min_samples_split": int(dt_min_samples_split),
                "criterion": dt_criterion,
                "random_state": int(random_seed),
            },
            "svm": {
                "C": float(svm_C),
                "kernel": svm_kernel,
                "gamma": svm_gamma,
                "random_state": int(random_seed),
            },
            "naive_bayes": {
                "var_smoothing": float(nb_var_smoothing),
            },
            "random_forest": {
                "n_estimators": int(rf_n_estimators),
                "max_depth": int(rf_max_depth),
                "min_samples_split": int(rf_min_samples_split),
                "criterion": rf_criterion,
                "random_state": int(random_seed),
            },
            "logistic_regression": {
                "C": float(lr_C),
                "penalty": lr_penalty,
                "solver": lr_solver,
                "l1_ratio": float(lr_l1_ratio),
                "random_state": int(random_seed),
            },
            "xgboost": {
                "n_estimators": int(xgb_n_estimators),
                "max_depth": int(xgb_max_depth),
                "subsample": float(xgb_subsample),
                "objective": xgb_objective,
                "learning_rate": float(xgb_learning_rate),
                "random_state": int(random_seed),
            },
            "lightgbm": {
                "n_estimators": int(lgbm_n_estimators),
                "max_depth": int(lgbm_max_depth),
                "num_leaves": int(lgbm_num_leaves),
                "objective": lgbm_objective,
                "learning_rate": float(lgbm_learning_rate),
                "random_state": int(random_seed),
            },
        }

        cnn_params = {
            "loss_fn": cnn_loss_fn,
            "focal_alpha": cnn_focal_alpha if cnn_focal_alpha != "None" else None,
            "focal_gamma": float(cnn_focal_gamma) if cnn_focal_gamma is not None else 2.0,
            "label_smoothing_epsilon": float(cnn_label_smoothing_epsilon),
            "optimizer": cnn_optimizer,
            "learning_rate": float(cnn_learning_rate),
            "dropout_rate": float(cnn_dropout),
            "batch_size": int(cnn_batch_size),
            "epochs": int(cnn_epochs),
            "early_stopping_patience": int(cnn_early_stopping),
            "input_size": int(cnn_input_size) if cnn_input_size else 128,
            "weight_decay": float(cnn_weight_decay) if cnn_weight_decay else 1e-4,
        }

        unsup_params = {
            "n_clusters": int(unsup_n_clusters) if unsup_n_clusters is not None else 2,
            "eps": float(unsup_eps) if unsup_eps is not None else 0.5,
            "min_samples": int(unsup_min_samples) if unsup_min_samples is not None else 5,
            "algorithm": kmeans_algorithm,
            "covariance_type": gmm_covariance_type,
            "linkage": agg_linkage,
            "affinity": spec_affinity,
        }

        # ---- 分发执行 ----
        progress(0.3, desc=f"开始训练 {model_name}...")
        trad_models = ["decision_tree", "svm", "naive_bayes", "random_forest",
                       "logistic_regression", "xgboost", "lightgbm"]
        unsup_models = ["kmeans", "gmm", "dbscan", "agglomerative", "spectral"]

        if model_name in trad_models:
            result = _run_traditional(
                model_name, trad_params_map[model_name], data,
                optimization_strategy, int(cv_folds_opt) if cv_folds_opt else 3,
                scoring_metric, validation_method, int(random_seed),
                n_iter=int(n_iter) if n_iter else 30,
            )
        elif model_name == "cnn":
            result = _run_cnn(cnn_params, data, optimization_strategy, int(random_seed),
                            n_iter=int(n_iter) if n_iter else 6,
                            scoring_metric=scoring_metric)
        elif model_name in unsup_models:
            result = _run_unsupervised(
                model_name, unsup_params, data, optimization_strategy, int(random_seed),
                n_iter=int(n_iter) if n_iter else 5,
                unsup_val_method=unsup_val_method,
            )
        else:
            raise ValueError(f"未知模型: {model_name}")

        progress(0.95, desc="生成可视化图表...")
        total_elapsed = time.time() - t_total
        result["status"] = "\n".join(status_msgs) + f"\n\n⏱ 总耗时: {total_elapsed:.1f}s\n" + result["status"]

        progress(1.0, desc="完成!")
        return (
            result["status"],
            result["metrics_md"],
            result["cm_fig"],
            result["roc_fig"],
            result["pr_fig"],
            result["fi_fig"],
            result["prob_fig"],
            result.get("extra_fig"),
            result.get("sil_fig"),
        )

    except Exception as e:
        import traceback
        err_msg = f"❌ 运行出错: {str(e)}\n\n```\n{traceback.format_exc()}\n```"
        return (err_msg, "", None, None, None, None, None, None, None)


# ============================================================
# 9. Gradio UI 构建
# ============================================================

MODEL_CHOICES = [
    "──── 传统监督学习 ────",
    "决策树 (Decision Tree)",
    "SVM (支持向量机)",
    "朴素贝叶斯 (Naive Bayes)",
    "随机森林 (Random Forest)",
    "逻辑回归 (Logistic Regression)",
    "XGBoost",
    "LightGBM",
    "──── 深度学习 ────",
    "CNN (CrackCNN)",
    "──── 无监督聚类 ────",
    "K-Means",
    "GMM (高斯混合模型)",
    "DBSCAN",
    "层次聚类 (Agglomerative)",
    "谱聚类 (Spectral)",
]

# 模型名到内部key的映射
MODEL_KEY_MAP = {
    "决策树 (Decision Tree)": "decision_tree",
    "SVM (支持向量机)": "svm",
    "朴素贝叶斯 (Naive Bayes)": "naive_bayes",
    "随机森林 (Random Forest)": "random_forest",
    "逻辑回归 (Logistic Regression)": "logistic_regression",
    "XGBoost": "xgboost",
    "LightGBM": "lightgbm",
    "CNN (CrackCNN)": "cnn",
    "K-Means": "kmeans",
    "GMM (高斯混合模型)": "gmm",
    "DBSCAN": "dbscan",
    "层次聚类 (Agglomerative)": "agglomerative",
    "谱聚类 (Spectral)": "spectral",
}

TRAD_MODEL_KEYS = {"decision_tree", "svm", "naive_bayes", "random_forest",
                   "logistic_regression", "xgboost", "lightgbm"}
UNSUP_MODEL_KEYS = {"kmeans", "gmm", "dbscan", "agglomerative", "spectral"}
TREE_MODELS = {"decision_tree", "random_forest", "xgboost", "lightgbm"}


def _is_trad(key): return key in TRAD_MODEL_KEYS
def _is_cnn(key): return key == "cnn"
def _is_unsup(key): return key in UNSUP_MODEL_KEYS
def _is_dbscan(key): return key == "dbscan"
def _is_supervised(key): return key in TRAD_MODEL_KEYS or key == "cnn"


def _model_visibility(model_key):
    """根据模型名返回各组件的 gr.update 可见性。"""
    v = lambda cond: gr.update(visible=cond)
    return {
        # Step 3: 传统模型参数
        "dt_params": v(model_key == "decision_tree"),
        "svm_params": v(model_key == "svm"),
        "nb_params": v(model_key == "naive_bayes"),
        "rf_params": v(model_key == "random_forest"),
        "lr_params": v(model_key == "logistic_regression"),
        "xgb_params": v(model_key == "xgboost"),
        "lgbm_params": v(model_key == "lightgbm"),
        # Step 3: CNN 参数
        "cnn_params": v(model_key == "cnn"),
        # Step 3: 无监督参数
        "unsup_n_clusters": v(_is_unsup(model_key) and not _is_dbscan(model_key)),
        "unsup_dbscan": v(_is_dbscan(model_key)),
        # Step 4: 传统模型 loss
        "dt_loss": v(model_key == "decision_tree"),
        "svm_loss": v(model_key == "svm"),
        "nb_loss_info": v(model_key == "naive_bayes"),
        "rf_loss": v(model_key == "random_forest"),
        "lr_loss": v(model_key == "logistic_regression"),
        "xgb_loss": v(model_key == "xgboost"),
        "lgbm_loss": v(model_key == "lightgbm"),
        # Step 4: CNN loss
        "cnn_loss": v(model_key == "cnn"),
        "cnn_focal": v(model_key == "cnn"),
        "cnn_ls": v(model_key == "cnn"),
        # Step 4: 无监督 loss
        "kmeans_loss": v(model_key == "kmeans"),
        "gmm_loss": v(model_key == "gmm"),
        "dbscan_loss_info": v(model_key == "dbscan"),
        "agg_loss": v(model_key == "agglomerative"),
        "spec_loss": v(model_key == "spectral"),
        # Step 5: 验证方法
        "supervised_val": v(_is_supervised(model_key) and not _is_cnn(model_key)),
        "unsup_val": v(_is_unsup(model_key)),
        # Step 5: 优化目标 (监督模型 + 网格/随机搜索时显示)
        "scoring_metric": v(_is_supervised(model_key)),
        # Step 5: 无监督优化提示
        "unsup_opt_info": v(_is_unsup(model_key)),
    }


def create_interface():
    """创建 Gradio Blocks 界面。"""
    theme = gr.themes.Soft(primary_hue="blue")

    with gr.Blocks(title="裂纹图像识别系统") as app:
        gr.Markdown("""
        # 🔍 裂纹图像识别系统 — 交互式训练链路
        ### 北京交通大学《机器学习与Python编程》研究性专题
        """)

        with gr.Row():
            # ============ 左侧控制面板 ============
            with gr.Column(scale=1, min_width=380):
                gr.Markdown("### 📊 Step 1: 数据处理")
                gr.Markdown(
                    "> 配置数据加载、预处理和特征提取方式。"
                    "预处理可降噪增强裂纹边缘；特征类型越多信息越丰富，但训练越慢。"
                )

                split_method = gr.Dropdown(
                    choices=["holdout"], value="holdout", label="划分方法")
                split_ratio = gr.Slider(0.5, 0.9, 0.7, step=0.05, label="训练集比例",
                                        visible=True)
                use_stratify = gr.Checkbox(True, label="分层抽样")

                preprocessing = gr.Radio(
                    choices=["none", "clahe", "gaussian", "median",
                             "clahe+gaussian", "clahe+median"],
                    value="clahe+median", label="预处理方法",
                    info="选择一种预处理管线（互斥）")

                features = gr.CheckboxGroup(
                    choices=["hog", "lbp", "glcm", "edge_density"],
                    value=["hog", "lbp", "glcm", "edge_density"], label="特征类型")
                max_samples = gr.Slider(200, 4000, 2000, step=200, label="样本数上限")

                gr.Markdown("---")
                gr.Markdown("### 🤖 Step 2: 模型选择")
                gr.Markdown(
                    "> 选择分类或聚类模型。传统方法训练快、可解释性强，"
                    "适合快速实验；CNN 学习能力更强但需要更多样本和训练时间；"
                    "无监督聚类无需标签即可发现数据模式。"
                )

                model_choice = gr.Dropdown(
                    choices=MODEL_CHOICES,
                    value="随机森林 (Random Forest)", label="模型",
                    filterable=False)

                # ---- Step 3 & 4: 模型参数容器 ----
                gr.Markdown("---")
                gr.Markdown("### 🔧 Step 3: 模型超参数")
                gr.Markdown(
                    "> 调整模型结构参数。默认值通常可行；"
                    "增大复杂度（深度/树数）可能提升拟合能力但增加过拟合风险，"
                    "建议从小值开始逐步尝试。"
                )

                # 决策树参数
                with gr.Group(visible=False) as dt_params:
                    dt_max_depth = gr.Slider(3, 50, 15, step=1, label="max_depth")
                    dt_min_samples_split = gr.Slider(2, 20, 5, step=1, label="min_samples_split")
                # SVM参数
                with gr.Group(visible=False) as svm_params:
                    svm_C = gr.Number(1.0, label="C (正则化)", precision=2)
                # 朴素贝叶斯参数
                with gr.Group(visible=False) as nb_params:
                    nb_var_smoothing = gr.Number(1e-9, label="var_smoothing", precision=10)
                # 随机森林参数（默认模型，初始可见）
                with gr.Group(visible=True) as rf_params:
                    rf_n_estimators = gr.Slider(50, 500, 100, step=10, label="n_estimators")
                    rf_max_depth = gr.Slider(3, 50, 20, step=1, label="max_depth")
                    rf_min_samples_split = gr.Slider(2, 20, 5, step=1, label="min_samples_split")
                # 逻辑回归参数
                with gr.Group(visible=False) as lr_params:
                    lr_C = gr.Number(1.0, label="C (正则化)", precision=2)
                # XGBoost参数
                with gr.Group(visible=False) as xgb_params:
                    xgb_n_estimators = gr.Slider(50, 300, 100, step=10, label="n_estimators")
                    xgb_max_depth = gr.Slider(3, 12, 6, step=1, label="max_depth")
                    xgb_subsample = gr.Slider(0.5, 1.0, 0.8, step=0.05, label="subsample")
                # LightGBM参数
                with gr.Group(visible=False) as lgbm_params:
                    lgbm_n_estimators = gr.Slider(50, 300, 100, step=10, label="n_estimators")
                    lgbm_max_depth = gr.Slider(3, 12, 6, step=1, label="max_depth")
                    lgbm_num_leaves = gr.Slider(15, 127, 31, step=4, label="num_leaves")
                # CNN参数
                with gr.Group(visible=False) as cnn_params:
                    cnn_dropout = gr.Slider(0.0, 0.9, 0.5, step=0.05, label="Dropout 比例")
                    cnn_batch_size = gr.Slider(16, 256, 64, step=16, label="Batch Size")
                    cnn_epochs = gr.Slider(5, 100, 30, step=5, label="最大 Epochs")
                    cnn_early_stopping = gr.Slider(3, 30, 10, step=1, label="早停耐心值")
                    cnn_input_size = gr.Dropdown(
                        choices=[64, 128, 256], value=128, label="输入图像尺寸 (input_size)")
                    cnn_weight_decay = gr.Number(1e-4, label="Weight Decay (L2正则)", precision=5)
                # 无监督参数
                with gr.Group(visible=False) as unsup_n_clusters:
                    unsup_n_clusters_val = gr.Slider(2, 10, 2, step=1, label="聚类数 (n_clusters)")
                with gr.Group(visible=False) as unsup_dbscan:
                    unsup_eps = gr.Slider(0.1, 2.0, 0.5, step=0.1, label="DBSCAN eps")
                    unsup_min_samples = gr.Slider(2, 20, 5, step=1, label="DBSCAN min_samples")

                gr.Markdown("---")
                gr.Markdown("### 📉 Step 4: 损失函数 / 优化器")
                gr.Markdown(
                    "> 配置损失函数和优化器。不同损失函数影响模型学习偏好；"
                    "预训练模式下此部分设置不生效（使用模型训练时的内置参数）。"
                )

                pretrained_loss_hint = gr.Markdown(
                    "💡 **预训练模式**：loss/核函数/目标函数设置不生效，使用模型训练时的内置参数。"
                    "切换为 manual/grid_search/random_search 模式可自定义。",
                    visible=False,
                )

                # DT loss
                with gr.Group(visible=False) as dt_loss:
                    dt_criterion = gr.Dropdown(["gini", "entropy", "log_loss"], value="gini", label="criterion (分裂准则)")
                # SVM loss
                with gr.Group(visible=False) as svm_loss:
                    svm_kernel = gr.Dropdown(["linear", "rbf", "poly"], value="rbf", label="kernel (核函数)")
                    svm_gamma = gr.Dropdown(["scale", "auto"], value="scale", label="gamma")
                # NB (无显式损失)
                with gr.Group(visible=False) as nb_loss_info:
                    gr.Markdown("*生成式模型，无显式损失函数；通过极大似然估计参数。*")
                # RF loss（默认模型，初始可见）
                with gr.Group(visible=True) as rf_loss:
                    rf_criterion = gr.Dropdown(["gini", "entropy", "log_loss"], value="gini", label="criterion (分裂准则)")
                # LR loss
                with gr.Group(visible=False) as lr_loss:
                    lr_penalty = gr.Dropdown(["l1", "l2", "elasticnet"], value="l2", label="penalty (正则化)")
                    lr_solver = gr.Dropdown(["lbfgs", "liblinear", "saga"], value="lbfgs", label="solver (优化器)")
                    lr_l1_ratio = gr.Slider(0.0, 1.0, 0.5, step=0.05,
                                            label="l1_ratio (elasticnet 混合比例)",
                                            visible=False)
                # XGBoost loss
                with gr.Group(visible=False) as xgb_loss:
                    xgb_objective = gr.Dropdown(["binary:logistic", "binary:hinge"], value="binary:logistic", label="objective (目标函数)")
                    xgb_learning_rate = gr.Slider(0.01, 0.5, 0.1, step=0.01, label="learning_rate (学习率)")
                    xgb_hinge_warning = gr.Markdown(
                        "⚠️ **注意**：`binary:hinge` 仅输出硬标签 (0/1)，无法产生概率估计，"
                        "ROC-AUC 和 PR 曲线将不可用。",
                        visible=False,
                    )
                # LightGBM loss
                with gr.Group(visible=False) as lgbm_loss:
                    lgbm_objective = gr.Dropdown(["binary", "cross_entropy"], value="binary", label="objective (目标函数)")
                    lgbm_learning_rate = gr.Slider(0.01, 0.5, 0.1, step=0.01, label="learning_rate (学习率)")
                # CNN loss
                with gr.Group(visible=False) as cnn_loss:
                    cnn_loss_fn = gr.Dropdown(["cross_entropy", "focal", "label_smoothing", "dice"], value="cross_entropy", label="损失函数")
                    cnn_focal_alpha = gr.Dropdown(["None", "0.25", "0.5"], value="None", label="Focal α (None=无类别权重)", visible=False)
                    cnn_focal_gamma = gr.Slider(0.0, 5.0, 2.0, step=0.5, label="Focal γ", visible=False)
                    cnn_label_smoothing_epsilon = gr.Slider(0.0, 0.3, 0.1, step=0.05, label="Label Smoothing ε", visible=False)
                    cnn_optimizer = gr.Dropdown(["adam", "sgd"], value="adam", label="优化器")
                    cnn_learning_rate = gr.Number(0.001, label="学习率", precision=5)
                # KMeans loss
                with gr.Group(visible=False) as kmeans_loss:
                    kmeans_algorithm = gr.Dropdown(["lloyd", "elkan"], value="lloyd", label="algorithm (优化算法)")
                # GMM loss
                with gr.Group(visible=False) as gmm_loss:
                    gmm_covariance_type = gr.Dropdown(["full", "tied", "diag", "spherical"], value="full", label="covariance_type (协方差类型)")
                    gr.Markdown("⚠️ *`full` 协方差在高维特征下模型文件约 778MB，加载较慢。*")
                # DBSCAN (无显式损失)
                with gr.Group(visible=False) as dbscan_loss_info:
                    gr.Markdown("*基于密度的聚类，无显式损失函数；通过密度可达性定义簇。*")
                # Agglomerative loss
                with gr.Group(visible=False) as agg_loss:
                    agg_linkage = gr.Dropdown(["ward", "complete", "average", "single"], value="ward", label="linkage (链接准则)")
                # Spectral loss
                with gr.Group(visible=False) as spec_loss:
                    spec_affinity = gr.Dropdown(["rbf", "nearest_neighbors"], value="rbf", label="affinity (相似度图)")

                gr.Markdown("---")
                gr.Markdown("### ⚡ Step 5: 参数优化 + 验证 + 指标")
                gr.Markdown(
                    "> 选择参数优化策略和验证方法。推荐先用 pretrained 快速评估模型效果，"
                    "确认方向后再用 grid_search 精细调参。"
                )

                optimization_strategy = gr.Radio(
                    choices=["pretrained", "manual", "grid_search", "random_search"],
                    value="pretrained",
                    label="参数优化策略",
                    info="pretrained=加载预训练模型 | manual=手动参数 | grid_search=GridSearchCV | random_search=RandomizedSearchCV",
                )
                with gr.Group(visible=False) as opt_search_params:
                    cv_folds_opt = gr.Slider(2, 10, 3, step=1, label="CV 折数")
                    n_iter = gr.Slider(10, 100, 30, step=10, label="随机搜索迭代次数 (仅RandomSearch)")

                with gr.Group(visible=True) as supervised_val:
                    validation_method = gr.Radio(
                        choices=["holdout", "kfold"], value="holdout", label="验证方法")
                with gr.Group(visible=False) as unsup_val:
                    unsup_val_method = gr.Radio(
                        choices=["internal_external", "internal_only", "external_only"],
                        value="internal_external", label="评估指标范围")

                with gr.Group(visible=False) as scoring_metric_grp:
                    scoring_metric = gr.Dropdown(
                        choices=["f1", "accuracy", "roc_auc", "precision", "recall"],
                        value="f1", label="优化目标指标 (GridSearch/RandomSearch时使用)")

                with gr.Group(visible=False) as unsup_opt_info:
                    gr.Markdown("*💡 聚类方法的 GridSearch/RandomSearch 使用 **轮廓系数 (Silhouette)** 作为搜索目标（非监督指标）。*")

                random_seed = gr.Number(42, label="随机种子", precision=0)

                run_btn = gr.Button("▶ 运行训练链路", variant="primary", size="lg")

            # ============ 右侧结果区 ============
            with gr.Column(scale=2):
                status_output = gr.Markdown("### 训练状态\n\n*等待运行...*")
                metrics_output = gr.Markdown("")

                with gr.Row():
                    cm_plot = gr.Plot(label="混淆矩阵")
                    roc_plot = gr.Plot(label="ROC 曲线")

                with gr.Row():
                    pr_plot = gr.Plot(label="PR 曲线")
                    fi_plot = gr.Plot(label="特征重要性 / 训练曲线 / PCA聚类")

                with gr.Row():
                    prob_plot = gr.Plot(label="预测概率分布")
                    extra_plot = gr.Plot(label="训练曲线 / 特征重要性")

                with gr.Row():
                    extra_plot2 = gr.Plot(label="Silhouette / PCA 聚类")

        # ============================================================
        # 事件绑定
        # ============================================================

        # --- 优化策略联动 ---
        def _on_opt_change(strategy):
            is_search = strategy in ("grid_search", "random_search")
            is_pretrained = strategy == "pretrained"
            # loss 参数在 pretrained 模式下禁用
            loss_interactive = gr.update(interactive=not is_pretrained)
            return (
                gr.update(visible=is_search),              # opt_search_params
                gr.update(visible=is_search),              # scoring_metric_grp
                gr.update(visible=is_pretrained),          # pretrained_loss_hint
                # 传统模型 loss 参数
                loss_interactive,  # dt_criterion
                loss_interactive,  # svm_kernel
                loss_interactive,  # svm_gamma
                loss_interactive,  # rf_criterion
                loss_interactive,  # lr_penalty
                loss_interactive,  # lr_solver
                loss_interactive,  # xgb_objective
                loss_interactive,  # xgb_learning_rate
                loss_interactive,  # lgbm_objective
                loss_interactive,  # lgbm_learning_rate
                # 无监督 loss 参数
                loss_interactive,  # kmeans_algorithm
                loss_interactive,  # gmm_covariance_type
                loss_interactive,  # agg_linkage
                loss_interactive,  # spec_affinity
            )
        optimization_strategy.change(
            fn=_on_opt_change, inputs=[optimization_strategy],
            outputs=[opt_search_params, scoring_metric_grp, pretrained_loss_hint,
                     dt_criterion, svm_kernel, svm_gamma, rf_criterion,
                     lr_penalty, lr_solver,
                     xgb_objective, xgb_learning_rate,
                     lgbm_objective, lgbm_learning_rate,
                     kmeans_algorithm, gmm_covariance_type,
                     agg_linkage, spec_affinity],
        )

        # --- CNN loss 联动 ---
        def _on_cnn_loss_change(loss):
            return (
                gr.update(visible=(loss == "focal")),
                gr.update(visible=(loss == "focal")),
                gr.update(visible=(loss == "label_smoothing")),
            )
        cnn_loss_fn.change(
            fn=_on_cnn_loss_change, inputs=[cnn_loss_fn],
            outputs=[cnn_focal_alpha, cnn_focal_gamma, cnn_label_smoothing_epsilon],
        )

        # --- LR penalty 联动：elasticnet 时显示 l1_ratio，并过滤 solver ---
        def _on_lr_penalty_change(penalty):
            if penalty == "l2":
                solver_choices = ["lbfgs", "liblinear", "saga"]
                solver_value = "lbfgs"
            elif penalty == "l1":
                solver_choices = ["liblinear", "saga"]
                solver_value = "liblinear"
            else:  # elasticnet
                solver_choices = ["saga"]
                solver_value = "saga"
            return (
                gr.update(visible=(penalty == "elasticnet")),
                gr.update(choices=solver_choices, value=solver_value),
            )
        lr_penalty.change(
            fn=_on_lr_penalty_change, inputs=[lr_penalty],
            outputs=[lr_l1_ratio, lr_solver],
        )

        # --- XGBoost objective 联动：binary:hinge 时显示警告并移除 roc_auc ---
        def _on_xgb_objective_change(objective):
            is_hinge = (objective == "binary:hinge")
            if is_hinge:
                scoring_choices = ["f1", "accuracy", "precision", "recall"]
                scoring_value = "f1"
            else:
                scoring_choices = ["f1", "accuracy", "roc_auc", "precision", "recall"]
                scoring_value = None  # 保持当前值，不强制切换
            return (
                gr.update(visible=is_hinge),
                gr.update(choices=scoring_choices, value=scoring_value),
            )
        xgb_objective.change(
            fn=_on_xgb_objective_change, inputs=[xgb_objective],
            outputs=[xgb_hinge_warning, scoring_metric],
        )

        # --- SVM kernel 联动：linear 时隐藏 gamma ---
        def _on_svm_kernel_change(kernel):
            return gr.update(visible=(kernel != "linear"))
        svm_kernel.change(
            fn=_on_svm_kernel_change, inputs=[svm_kernel],
            outputs=[svm_gamma],
        )

        # --- 模型切换联动 (最复杂的部分) ---
        def on_model_change(choice):
            if choice is None or choice.startswith("────"):
                key = "decision_tree"
            else:
                key = MODEL_KEY_MAP.get(choice, "decision_tree")
            vis = _model_visibility(key)

            # 构建输出列表 (按组件的定义顺序)
            return [
                # dt, svm, nb, rf, lr, xgb, lgbm params
                vis["dt_params"], vis["svm_params"], vis["nb_params"],
                vis["rf_params"], vis["lr_params"], vis["xgb_params"], vis["lgbm_params"],
                # cnn params
                vis["cnn_params"],
                # unsup params
                vis["unsup_n_clusters"], vis["unsup_dbscan"],
                # dt, svm, nb, rf, lr, xgb, lgbm loss
                vis["dt_loss"], vis["svm_loss"], vis["nb_loss_info"],
                vis["rf_loss"], vis["lr_loss"], vis["xgb_loss"], vis["lgbm_loss"],
                # cnn loss
                vis["cnn_loss"],
                # kmeans, gmm, dbscan, agg, spec loss
                vis["kmeans_loss"], vis["gmm_loss"], vis["dbscan_loss_info"],
                vis["agg_loss"], vis["spec_loss"],
                # validation
                vis["supervised_val"], vis["unsup_val"],
                vis["scoring_metric"],
                vis["unsup_opt_info"],
            ]

        # 收集所有需要联动控制的组件
        model_linked_outputs = [
            dt_params, svm_params, nb_params, rf_params, lr_params, xgb_params, lgbm_params,
            cnn_params,
            unsup_n_clusters, unsup_dbscan,
            dt_loss, svm_loss, nb_loss_info, rf_loss, lr_loss, xgb_loss, lgbm_loss,
            cnn_loss,
            kmeans_loss, gmm_loss, dbscan_loss_info, agg_loss, spec_loss,
            supervised_val, unsup_val,
            scoring_metric_grp,
            unsup_opt_info,
        ]

        model_choice.change(
            fn=on_model_change,
            inputs=[model_choice],
            outputs=model_linked_outputs,
        )

        # --- 运行按钮 ---
        all_inputs = [
            split_method, split_ratio, use_stratify,
            preprocessing, features, max_samples,
            model_choice,
            dt_max_depth, dt_min_samples_split,
            svm_C, nb_var_smoothing,
            rf_n_estimators, rf_max_depth, rf_min_samples_split,
            lr_C,
            xgb_n_estimators, xgb_max_depth, xgb_subsample,
            lgbm_n_estimators, lgbm_max_depth, lgbm_num_leaves,
            cnn_dropout, cnn_batch_size, cnn_epochs, cnn_early_stopping,
            cnn_input_size, cnn_weight_decay,
            unsup_n_clusters_val, unsup_eps, unsup_min_samples,
            dt_criterion,
            svm_kernel, svm_gamma,
            rf_criterion,
            lr_penalty, lr_solver, lr_l1_ratio,
            xgb_objective, xgb_learning_rate,
            lgbm_objective, lgbm_learning_rate,
            cnn_loss_fn, cnn_focal_alpha, cnn_focal_gamma,
            cnn_label_smoothing_epsilon, cnn_optimizer, cnn_learning_rate,
            kmeans_algorithm, gmm_covariance_type,
            agg_linkage, spec_affinity,
            optimization_strategy, cv_folds_opt, n_iter,
            validation_method, scoring_metric,
            unsup_val_method,
            random_seed,
        ]

        all_outputs = [
            status_output, metrics_output,
            cm_plot, roc_plot, pr_plot, fi_plot, prob_plot, extra_plot, extra_plot2,
        ]

        # 包装 run_pipeline 以将模型显示名转为内部key
        def _run_wrapper(*args):
            # args[7] 是 model_choice
            args_list = list(args)
            choice = args_list[6]
            if choice and not choice.startswith("────"):
                args_list[6] = MODEL_KEY_MAP.get(choice, "random_forest")
            else:
                args_list[6] = "random_forest"
            return run_pipeline(*args_list)

        run_btn.click(
            fn=_run_wrapper,
            inputs=all_inputs,
            outputs=all_outputs,
        )

    return app


# ============================================================
# 10. 主入口
# ============================================================

def main():
    app = create_interface()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
