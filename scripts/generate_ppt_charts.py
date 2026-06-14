#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PPT 演讲图表生成脚本
=====================
根据《裂纹图像识别系统_演讲大纲》生成 PPT 所需数据图表，输出到 reports/ppt图表输出/。

【约束】所有模型训练/评估结果数据均来自项目已有流程：
  - 已持久化数据：直接读取 outputs/results/*.csv、outputs/combo_verify/*.json、
    outputs/models/cnn/*_history.json。
  - 4 项未持久化对比实验（划分方式/特征组合/CNN优化器/CNN超参数网格）：
    忠实复用 notebook 01/03 中已有的对比函数，底层训练/评估全部调用
    src/gradio_app.py 的既有函数（load_dataset / apply_* / extract_* / CrackCNN /
    CrackDataset）。本脚本不新增任何训练/评估方法，仅做画图与必要的数据获取胶水。

运行：
    conda activate bjtu_ml  (或直接用 bjtu_ml 的 python.exe)
    python scripts/generate_ppt_charts.py [zero|light|all]   # 默认 all
"""

import os
import sys
import copy
import json
import warnings
from itertools import product
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 中文字体（与 notebook 一致）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 11

# 让 src 可被 import
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from sklearn.model_selection import train_test_split, StratifiedKFold  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, precision_score, recall_score, f1_score,
)

# 复用项目已有流程（src/gradio_app.py）
from src import gradio_app as ga  # noqa: E402

# ------------------------------------------------------------------
# 全局配置
# ------------------------------------------------------------------
OUT_DIR = PROJECT_ROOT / "reports" / "ppt图表输出"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
COMBO_DIR = PROJECT_ROOT / "outputs" / "combo_verify"
BATCH_DIR = PROJECT_ROOT / "outputs" / "batch_verify" / "20260613_043410"

DEVICE = getattr(ga, "DEVICE", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# 配色（按类别）
COLOR_TRAD = "#3498db"   # 传统监督 - 蓝
COLOR_TREE = "#2ecc71"   # 树/集成 - 绿
COLOR_CNN = "#e67e22"    # CNN - 橙
COLOR_UNSUP = "#9b59b6"  # 无监督 - 紫
PALETTE = {
    "传统监督": COLOR_TRAD, "树模型与集成": COLOR_TREE,
    "CNN": COLOR_CNN, "无监督": COLOR_UNSUP,
}


# ==================================================================
# 一、notebook 01 忠实搬运：辅助函数 + 划分/特征对比实验
# （来源：notebooks/01_数据处理与特征工程.ipynb cell 15/16/19/27/29）
# ==================================================================
def split_dataset(images, labels, train_ratio=0.7, val_ratio=0.15, random_seed=42):
    """留出法 + 分层抽样三路划分（notebook01 cell15，移植自 src/data_utils.py）。"""
    val_test_ratio = 1.0 - train_ratio
    X_train, X_temp, y_train, y_temp = train_test_split(
        images, labels, test_size=val_test_ratio,
        random_state=random_seed, stratify=labels,
    )
    test_ratio_in_temp = (1.0 - train_ratio - val_ratio) / (1.0 - train_ratio)
    try:
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=test_ratio_in_temp,
            random_state=random_seed, stratify=y_temp,
        )
    except ValueError:
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=test_ratio_in_temp,
            random_state=random_seed,
        )
    return X_train, X_val, X_test, y_train, y_val, y_test


def _extract_all_features(image):
    """HOG+LBP+GLCM+边缘密度 拼接（notebook01 cell16）。底层调用 ga.extract_*。"""
    hog_feat = ga.extract_hog_features(image)
    lbp_feat = ga.extract_lbp_features(image)
    glcm_feat = ga.extract_glcm_features(image)
    edge_feat = np.array([ga.extract_edge_density(image)])
    return np.concatenate([hog_feat, lbp_feat, glcm_feat, edge_feat])


def compare_split_strategies(images, labels, max_samples=2000, random_seed=42):
    """划分策略汇总对比：留出法多比例 + 分层K折（notebook01 cell19，忠实搬运）。"""
    images_sub, labels_sub = ga._subsample_balanced(images, labels, max_samples, random_seed)
    print("    提取所有样本特征...")
    X_all = np.stack([_extract_all_features(img) for img in images_sub])
    y_all = labels_sub
    all_results = []

    for train_r in [0.5, 0.6, 0.7, 0.8, 0.9]:
        test_r = round(1.0 - train_r, 4)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_all, y_all, test_size=test_r, random_state=random_seed, stratify=y_all)
        model = RandomForestClassifier(n_estimators=100, max_depth=20,
                                       random_state=random_seed, n_jobs=-1)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        all_results.append({
            "策略": f"留出法 {train_r:.0%}/{test_r:.0%}", "类别": "留出法",
            "准确率": accuracy_score(y_te, y_pred),
            "F1分数": f1_score(y_te, y_pred, zero_division=0),
        })

    for k in [5, 10]:
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_seed)
        fold_acc, fold_f1 = [], []
        for tr_idx, te_idx in kf.split(X_all, y_all):
            model = RandomForestClassifier(n_estimators=100, max_depth=20,
                                           random_state=random_seed, n_jobs=-1)
            model.fit(X_all[tr_idx], y_all[tr_idx])
            y_pred = model.predict(X_all[te_idx])
            fold_acc.append(accuracy_score(y_all[te_idx], y_pred))
            fold_f1.append(f1_score(y_all[te_idx], y_pred, zero_division=0))
        all_results.append({
            "策略": f"分层{k}折CV", "类别": "K折交叉验证",
            "准确率": float(np.mean(fold_acc)), "F1分数": float(np.mean(fold_f1)),
        })
    return pd.DataFrame(all_results)


def compare_feature_subsets(images, labels, max_samples=2000, train_ratio=0.7, random_seed=42):
    """特征子集对比（notebook01 cell29，忠实搬运）。底层调用 ga.extract_features_separate。"""
    feature_groups = {
        "仅边缘密度": ["edge_density"], "仅HOG": ["hog"], "仅LBP": ["lbp"], "仅GLCM": ["glcm"],
        "HOG+LBP": ["hog", "lbp"], "HOG+LBP+GLCM": ["hog", "lbp", "glcm"],
        "全部特征": ["hog", "lbp", "glcm", "edge_density"],
    }
    images_sub, labels_sub = ga._subsample_balanced(images, labels, max_samples, random_seed)
    print("    预先提取所有独立特征...")
    all_feats = [ga.extract_features_separate(img) for img in images_sub]
    # 用样本索引做二路分层划分（70/30），避免 notebook 原 split_dataset 在 val_ratio=0 时
    # 退化成 test_size=1.0 的 bug；控制变量与 F1（划分对比）保持一致。
    all_idx = np.arange(len(images_sub))
    idx_tr, idx_te, y_tr, y_te = train_test_split(
        all_idx, labels_sub, test_size=1.0 - train_ratio,
        random_state=random_seed, stratify=labels_sub)
    idx_tr = idx_tr.astype(int)
    idx_te = idx_te.astype(int)
    results = []
    for name, keys in feature_groups.items():
        X_tr_f = np.concatenate(
            [np.concatenate([all_feats[i][k] for k in keys]) for i in idx_tr]
        ).reshape(len(idx_tr), -1)
        X_te_f = np.concatenate(
            [np.concatenate([all_feats[i][k] for k in keys]) for i in idx_te]
        ).reshape(len(idx_te), -1)
        model = RandomForestClassifier(n_estimators=100, max_depth=20,
                                       random_state=random_seed, n_jobs=-1)
        model.fit(X_tr_f, y_tr)
        y_pred = model.predict(X_te_f)
        results.append({
            "特征组合": name, "特征维度": X_tr_f.shape[1],
            "准确率": accuracy_score(y_te, y_pred),
            "F1分数": f1_score(y_te, y_pred, zero_division=0),
        })
        print(f"      {name}: 维度={X_tr_f.shape[1]}, F1={results[-1]['F1分数']:.4f}")
    return pd.DataFrame(results)


# ==================================================================
# 二、notebook 03 忠实搬运：CNN 数据加载 + 优化器/网格对比实验
# （来源：notebooks/03_深度学习对比.ipynb cell 3/13/15）
# ==================================================================
def build_cnn_loaders(max_samples=2000, batch_size=64):
    """复刻 notebook03 cell3 的数据加载与划分。底层用 ga.load_dataset/CrackDataset。"""
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    _d = os.getenv("CRACK_DATA_ROOT")
    data_root = Path(_d).expanduser().resolve() if _d else (PROJECT_ROOT / "data")
    images, labels = ga.load_dataset(data_root, max_samples=max_samples)
    train_idx, test_idx = train_test_split(
        np.arange(len(labels)), test_size=0.3, random_state=42, stratify=labels)
    train_idx2, val_idx = train_test_split(
        train_idx, test_size=0.15 / 0.7, random_state=42, stratify=labels[train_idx])
    default_preprocess = lambda img: ga.apply_median_filter(ga.apply_clahe(img))
    train_ds = ga.CrackDataset(images[train_idx2], labels[train_idx2], preprocess_fn=default_preprocess)
    val_ds = ga.CrackDataset(images[val_idx], labels[val_idx], preprocess_fn=default_preprocess)
    test_ds = ga.CrackDataset(images[test_idx], labels[test_idx], preprocess_fn=default_preprocess)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    print(f"    CNN 数据: 训练{len(train_idx2)} 验证{len(val_idx)} 测试{len(test_idx)}")
    return train_loader, val_loader, test_loader


def compare_optimizers(train_loader, val_loader, test_loader, device, epochs=15):
    """优化器对比（notebook03 cell13，忠实搬运）。"""
    import torch.optim as optim
    import torch.nn as nn
    optimizers_cfg = {
        "Adam (lr=1e-3)": (optim.Adam, {"lr": 1e-3, "weight_decay": 1e-4}),
        "Adam (lr=1e-4)": (optim.Adam, {"lr": 1e-4, "weight_decay": 1e-4}),
        "SGD+Momentum (lr=1e-2)": (optim.SGD, {"lr": 1e-2, "momentum": 0.9, "weight_decay": 1e-4}),
        "SGD+Momentum (lr=1e-3)": (optim.SGD, {"lr": 1e-3, "momentum": 0.9, "weight_decay": 1e-4}),
    }
    results = []
    for opt_name, (opt_cls, opt_kwargs) in optimizers_cfg.items():
        print(f"    优化器: {opt_name}")
        model = ga.CrackCNN(num_classes=2, input_channels=1, dropout_rate=0.5).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer_inst = opt_cls(model.parameters(), **opt_kwargs)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_inst, mode="min", factor=0.5, patience=5)
        best_acc = 0.0
        for epoch in range(1, epochs + 1):
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer_inst.zero_grad()
                loss = criterion(model(inputs), targets)
                loss.backward(); optimizer_inst.step()
            model.eval()
            va_c = sum(model(inputs.to(device)).argmax(1).eq(targets.to(device)).sum().item()
                       for inputs, targets in val_loader)
            va_acc = va_c / len(val_loader.dataset)
            scheduler.step(1 - va_acc)
            if va_acc > best_acc:
                best_acc = va_acc
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                preds.extend(model(inputs).argmax(1).cpu().numpy())
                trues.extend(targets.numpy())
        results.append({
            "优化器": opt_name,
            "最佳验证Acc": round(best_acc, 4),
            "测试准确率": round(accuracy_score(trues, preds), 4),
            "测试F1分数": round(f1_score(trues, preds, zero_division=0), 4),
        })
    return pd.DataFrame(results)


def grid_search_cnn(train_loader, val_loader, device,
                    lr_list=None, dropout_list=None, bs_list=None,
                    epochs_per_trial=15, patience=5):
    """三维超参数网格搜索（notebook03 cell15，忠实搬运）。"""
    import torch.optim as optim
    import torch.nn as nn
    if lr_list is None:
        lr_list = [1e-4, 5e-4, 1e-3]
    if dropout_list is None:
        dropout_list = [0.3, 0.5, 0.7]
    if bs_list is None:
        bs_list = [32, 64]
    results = []
    total = len(lr_list) * len(dropout_list) * len(bs_list)
    trial = 0
    for lr, dropout, bs in product(lr_list, dropout_list, bs_list):
        trial += 1
        print(f"    网格 {trial}/{total}: lr={lr}, dropout={dropout}, bs={bs}")
        train_data = train_loader.dataset
        tr_ld = DataLoader(train_data, batch_size=bs, shuffle=True)
        model = ga.CrackCNN(num_classes=2, input_channels=1, dropout_rate=dropout).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer_inst = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_inst, mode="min", factor=0.5, patience=patience)
        best_loss = float("inf"); best_epoch = 0; no_improve = 0
        best_state_local = None
        for epoch in range(1, epochs_per_trial + 1):
            model.train()
            for inputs, targets in tr_ld:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer_inst.zero_grad()
                loss = criterion(model(inputs), targets)
                loss.backward(); optimizer_inst.step()
            model.eval()
            va_loss = sum(criterion(model(inputs.to(device)), targets.to(device)).item() * inputs.size(0)
                          for inputs, targets in val_loader) / len(val_loader.dataset)
            scheduler.step(va_loss)
            if va_loss < best_loss:
                best_loss = va_loss; best_epoch = epoch
                best_state_local = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                break
        model.load_state_dict(best_state_local)
        model.eval()
        va_c = sum(model(inputs.to(device)).argmax(1).eq(targets.to(device)).sum().item()
                   for inputs, targets in val_loader)
        val_acc = va_c / len(val_loader.dataset)
        results.append({"lr": lr, "dropout": dropout, "batch_size": bs,
                        "best_val_loss": round(best_loss, 6),
                        "best_val_acc": round(val_acc, 4), "best_epoch": best_epoch})
    return pd.DataFrame(results).sort_values("best_val_loss").reset_index(drop=True)


# ==================================================================
# 三、画图辅助
# ==================================================================
def _save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [已保存] {name}")
    return path


def _bar_label(ax, bars, fmt="{:.4f}", offset=0.002, fontsize=9):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + offset,
                fmt.format(h), ha="center", va="bottom", fontsize=fontsize, fontweight="bold")


# ==================================================================
# 四、各图表绘制函数
# ==================================================================
def draw_F0():
    """数据集类别分布。"""
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    _d = os.getenv("CRACK_DATA_ROOT")
    root = Path(_d).expanduser().resolve() if _d else (PROJECT_ROOT / "data")
    n_pos = len(list((root / "Positive").glob("*.jpg")))
    n_neg = len(list((root / "Negative").glob("*.jpg")))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    cats = ["无裂纹 (Negative)", "有裂纹 (Positive)"]
    counts = [n_neg, n_pos]
    colors = ["#2ecc71", "#e74c3c"]
    bars = axes[0].bar(cats, counts, color=colors, edgecolor="white", linewidth=1.5)
    axes[0].set_ylabel("样本数量"); axes[0].set_title("数据集类别分布", fontweight="bold")
    for bar, c in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                     f"{c:,}\n({c / (n_pos + n_neg) * 100:.1f}%)", ha="center", fontsize=10, fontweight="bold")
    axes[1].pie(counts, labels=cats, colors=colors, autopct="%1.1f%%",
                explode=(0, 0.05), shadow=True, textprops={"fontsize": 11})
    axes[1].set_title("类别比例", fontweight="bold")
    plt.suptitle(f"数据集总览：共 {n_pos + n_neg:,} 张图像（完全均衡）", fontsize=14, fontweight="bold")
    return _save(fig, "F0_数据集类别分布.png")


def draw_F2():
    """预处理方式对比（来源：batch_verify results.csv，固定 RF+pretrained+holdout，按预处理聚合）。"""
    df = pd.read_csv(BATCH_DIR / "results.csv")
    df = df[(df["model"] == "random_forest") & (df["optimization"] == "pretrained")
            & (df["validation"] == "holdout")]
    order = ["none", "clahe", "gaussian", "median", "clahe+gaussian", "clahe+median"]
    label_map = {"none": "无预处理", "clahe": "CLAHE", "gaussian": "高斯滤波",
                 "median": "中值滤波", "clahe+gaussian": "CLAHE+高斯", "clahe+median": "CLAHE+中值"}
    agg = df.groupby("preprocessing")["f1"].mean()
    vals = [agg.get(p, np.nan) for p in order]
    names = [label_map[p] for p in order]
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["#bdc3c7", COLOR_TRAD, "#1abc9c", "#9b59b6", "#f39c12", "#e67e22"]
    bars = ax.bar(names, vals, color=colors, edgecolor="white", linewidth=1.5)
    _bar_label(ax, bars)
    ax.set_ylabel("F1 分数"); ax.set_ylim(0, max(vals) * 1.15)
    ax.set_title("预处理方式对比（控制变量：随机森林 + HOG/LBP/GLCM/边缘密度，分层 70/30）",
                 fontsize=13, fontweight="bold")
    plt.xticks(rotation=15)
    return _save(fig, "F2_预处理方式对比.png")


def draw_F4():
    """传统监督（SVM/朴素贝叶斯/逻辑回归）对比。"""
    df = pd.read_csv(RESULTS_DIR / "traditional_comparison.csv")
    sel = df[df["model"].isin(["svm", "naive_bayes", "logistic_regression"])].copy()
    name_map = {"svm": "SVM", "naive_bayes": "朴素贝叶斯", "logistic_regression": "逻辑回归"}
    sel["名称"] = sel["model"].map(name_map)
    x = np.arange(len(sel)); w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w / 2, sel["cv_f1"], w, label="交叉验证 F1", color=COLOR_TRAD, edgecolor="white")
    b2 = ax.bar(x + w / 2, sel["test_f1"], w, label="测试 F1", color="#1abc9c", edgecolor="white")
    _bar_label(ax, b1); _bar_label(ax, b2)
    ax.set_xticks(x); ax.set_xticklabels(sel["名称"])
    ax.set_ylabel("F1 分数"); ax.set_ylim(0, 1.05)
    ax.set_title("传统监督学习方法对比（SVM / 朴素贝叶斯 / 逻辑回归）", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    return _save(fig, "F4_传统监督学习对比.png")


def draw_F5():
    """树模型与集成对比 + 训练时间（双轴）。"""
    df = pd.read_csv(RESULTS_DIR / "traditional_comparison.csv")
    order = ["decision_tree", "random_forest", "xgboost", "lightgbm"]
    name_map = {"decision_tree": "决策树", "random_forest": "随机森林",
                "xgboost": "XGBoost", "lightgbm": "LightGBM"}
    sel = df[df["model"].isin(order)].set_index("model").loc[order].reset_index()
    sel["名称"] = sel["model"].map(name_map)
    x = np.arange(len(sel))
    fig, ax1 = plt.subplots(figsize=(11, 5))
    bars = ax1.bar(x, sel["test_f1"], 0.55, color=COLOR_TREE, edgecolor="white", linewidth=1.5, label="测试 F1")
    _bar_label(ax1, bars)
    ax1.set_ylabel("测试 F1 分数", color=COLOR_TREE)
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(x); ax1.set_xticklabels(sel["名称"])
    ax2 = ax1.twinx()
    ax2.plot(x, sel["time_s"], "o-", color="#e74c3c", linewidth=2.5, markersize=10, label="训练时间")
    for xi, t in zip(x, sel["time_s"]):
        ax2.annotate(f"{t:.1f}s", (xi, t), textcoords="offset points", xytext=(0, 12),
                     ha="center", color="#e74c3c", fontweight="bold", fontsize=10)
    ax2.set_ylabel("训练时间 (秒)", color="#e74c3c")
    ax2.set_ylim(0, max(sel["time_s"]) * 1.35)
    ax1.set_title("树模型与集成方法对比（准确率与训练时间）", fontsize=13, fontweight="bold")
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper left")
    return _save(fig, "F5_树模型与集成对比.png")


def _load_unsup():
    """读取 outputs/results/unsupervised_comparison.csv（官方聚类对比结果），统一方法名。
    DBSCAN 在本数据集无有效 2 簇参数（CLAUDE.md 已说明），已排除。
    注：该 CSV 含 silhouette/ari/nmi，内部一致；CH 指数仅在 combo_verify 的 unsup_results.json
    中存在且数值与本 CSV 不同源，故 F6 改用 nmi 替代 CH 以保证同源一致。"""
    df = pd.read_csv(RESULTS_DIR / "unsupervised_comparison.csv")
    name_map = {"K-Means": "K-Means", "GMM": "GMM",
                "Agglomerative(ward)": "层次聚类(ward)", "Spectral(rbf)": "谱聚类(rbf)"}
    df["method"] = df["method"].map(lambda x: name_map.get(str(x), str(x)))
    keep = ["K-Means", "GMM", "层次聚类(ward)", "谱聚类(rbf)"]
    df = df[df["method"].isin(keep)].copy()
    return df


def draw_F6():
    """无监督聚类对比（轮廓系数 / ARI / NMI，均来自 unsupervised_comparison.csv 同源数据）。"""
    df = _load_unsup()
    order = ["K-Means", "GMM", "层次聚类(ward)", "谱聚类(rbf)"]
    df = df.set_index("method").loc[order].reset_index()
    metrics = [("silhouette", "轮廓系数", COLOR_UNSUP),
               ("ari", "ARI (与真实标签一致性)", "#3498db"),
               ("nmi", "NMI (归一化互信息)", "#16a085")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, (col, title, color) in zip(axes, metrics):
        vals = df[col].astype(float).values
        bars = ax.bar(df["method"], vals, color=color, edgecolor="white", linewidth=1.5)
        _bar_label(ax, bars, fmt="{:.4f}", offset=max(vals.max(), 0.01) * 0.05)
        ax.set_title(title, fontweight="bold")
        ax.set_xticklabels(df["method"], rotation=20, ha="right")
        ax.set_ylim(0, max(vals.max(), 0.05) * 1.3)
    plt.suptitle("无监督聚类方法对比（DBSCAN 在本数据集无有效 2 簇参数，已排除）",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    return _save(fig, "F6_无监督聚类对比.png")


def draw_F7():
    """无监督 vs 监督：与真实标签一致性（ARI / F1）。"""
    df = _load_unsup()
    trad = pd.read_csv(RESULTS_DIR / "traditional_comparison.csv")
    cnn = pd.read_csv(RESULTS_DIR / "cnn_comparison.csv")
    best_trad_f1 = trad["test_f1"].max()
    best_cnn_acc = cnn["test_acc"].max()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    # 左：监督方法 F1/Acc
    sup_names = ["传统最佳\n(test F1)", "CNN最佳\n(test Acc)"]
    sup_vals = [best_trad_f1, best_cnn_acc]
    b1 = ax1.bar(sup_names, sup_vals, color=[COLOR_TRAD, COLOR_CNN], edgecolor="white", linewidth=1.5)
    _bar_label(ax1, b1)
    ax1.axhline(0.9, ls="--", color="gray", alpha=0.5)
    ax1.set_ylim(0, 1.08); ax1.set_ylabel("F1 / 准确率")
    ax1.set_title("监督学习（利用标签）", fontweight="bold")
    # 右：聚类 ARI
    order = ["K-Means", "GMM", "层次聚类(ward)", "谱聚类(rbf)"]
    df = df.set_index("method").loc[order].reset_index()
    vals = df["ari"].astype(float).values
    b2 = ax2.bar(df["method"], vals, color=COLOR_UNSUP, edgecolor="white", linewidth=1.5)
    _bar_label(ax2, b2, fmt="{:.4f}", offset=0.01)
    ax2.axhline(0.0, ls="--", color="red", alpha=0.6)
    ax2.text(3.4, 0.015, "ARI=0 ≈ 随机猜测", color="red", fontsize=9, ha="right")
    ax2.set_ylim(-0.02, max(0.25, max(vals) * 1.4)); ax2.set_ylabel("ARI (与真实标签一致性)")
    ax2.set_xticklabels(df["method"], rotation=20, ha="right")
    ax2.set_title("无监督聚类（不使用标签）", fontweight="bold")
    plt.suptitle("监督 vs 无监督：标签信息对裂纹识别至关重要", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return _save(fig, "F7_无监督vs监督对比.png")


def draw_F8():
    """CNN 损失函数对比（6种）。"""
    df = pd.read_csv(RESULTS_DIR / "cnn_comparison.csv")
    order = ["cross_entropy", "focal_gamma2", "focal_gamma3", "focal_balanced", "label_smoothing", "dice"]
    name_map = {"cross_entropy": "CrossEntropy", "focal_gamma2": "Focal(γ=2)",
                "focal_gamma3": "Focal(γ=3)", "focal_balanced": "Focal(α=0.5,γ=2)",
                "label_smoothing": "LabelSmoothing", "dice": "Dice"}
    df = df.set_index("loss").loc[order].reset_index()
    df["名称"] = df["loss"].map(name_map)
    x = np.arange(len(df)); w = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    b1 = ax.bar(x - w / 2, df["test_f1"], w, label="测试 F1", color=COLOR_CNN, edgecolor="white")
    b2 = ax.bar(x + w / 2, df["test_acc"], w, label="测试准确率", color="#f1c40f", edgecolor="white")
    _bar_label(ax, b1); _bar_label(ax, b2)
    ax.set_xticks(x); ax.set_xticklabels(df["名称"], rotation=15)
    ax.set_ylim(0.95, 1.001); ax.set_ylabel("F1 / 准确率")
    ax.set_title("CNN 损失函数对比（CrackCNN，CLAHE 预处理，分层 70/30，15 epoch）",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    return _save(fig, "F8_CNN损失函数对比.png")


def draw_F9():
    """CNN 损失函数训练曲线（来自 6 个 history.json）。"""
    order = ["cross_entropy", "focal_gamma2", "focal_gamma3", "focal_balanced", "label_smoothing", "dice"]
    name_map = {"cross_entropy": "CrossEntropy", "focal_gamma2": "Focal(γ=2)",
                "focal_gamma3": "Focal(γ=3)", "focal_balanced": "Focal(α=0.5,γ=2)",
                "label_smoothing": "LabelSmoothing", "dice": "Dice"}
    colors = plt.cm.tab10(np.linspace(0, 1, len(order)))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for loss, color in zip(order, colors):
        with open(MODELS_DIR / "cnn" / f"crackcnn_{loss}_history.json", encoding="utf-8") as f:
            h = json.load(f)
        ep = np.arange(1, len(h["train_loss"]) + 1)
        axes[0].plot(ep, h["train_loss"], label=name_map[loss], color=color, linewidth=2, marker="o", markersize=4)
        axes[1].plot(ep, h["val_f1"], label=name_map[loss], color=color, linewidth=2, marker="o", markersize=4)
    axes[0].set_xlabel("训练轮数 (epoch)"); axes[0].set_ylabel("训练损失")
    axes[0].set_title("各损失函数的训练损失曲线", fontweight="bold")
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("训练轮数 (epoch)"); axes[1].set_ylabel("验证 F1")
    axes[1].set_title("各损失函数的验证 F1 曲线", fontweight="bold")
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.05)
    plt.suptitle("CNN 损失函数训练过程对比", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return _save(fig, "F9_CNN损失函数训练曲线.png")


def draw_F12():
    """全方法效果总结：传统最佳 / CNN最佳 / 无监督最佳。"""
    trad = pd.read_csv(RESULTS_DIR / "traditional_comparison.csv")
    cnn = pd.read_csv(RESULTS_DIR / "cnn_comparison.csv")
    unsup_df = _load_unsup()
    best_trad = trad.loc[trad["test_f1"].idxmax()]
    best_cnn = cnn.loc[cnn["test_f1"].idxmax()]
    best_unsup_ari = unsup_df["ari"].astype(float).max()
    names = [f"传统最佳\n({best_trad['model']})", f"CNN最佳\n(Focal γ=2)", "无监督最佳\n(层次聚类)"]
    vals = [best_trad["test_f1"], best_cnn["test_f1"], best_unsup_ari]
    metrics_label = ["test F1", "test F1", "ARI"]
    colors = [COLOR_TREE, COLOR_CNN, COLOR_UNSUP]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(names, vals, color=colors, edgecolor="white", linewidth=1.5, width=0.55)
    for bar, v, ml in zip(bars, vals, metrics_label):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{v:.4f}\n({ml})", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.1); ax.set_ylabel("性能指标（注：无监督用 ARI）")
    ax.set_title("全方法效果总结：CNN 最优，传统集成次之，无监督受限于无标签",
                 fontsize=13, fontweight="bold")
    return _save(fig, "F12_全方法效果总结.png")


# ---- 需运行实验的图表 ----
def draw_F1(images, labels):
    df = compare_split_strategies(images, labels, max_samples=2000, random_seed=42)
    df.to_csv(OUT_DIR / "_data_F1.csv", index=False)
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = [COLOR_TRAD if c == "留出法" else COLOR_TREE for c in df["类别"]]
    x = np.arange(len(df))
    bars = ax.bar(x, df["F1分数"], color=colors, edgecolor="white", linewidth=1.5)
    _bar_label(ax, bars)
    ax.set_xticks(x); ax.set_xticklabels(df["策略"], rotation=25, ha="right")
    ax.set_ylim(0, 1.05); ax.set_ylabel("F1 分数")
    ax.set_title("数据划分方式对比（随机森林 + 全特征，控制变量）", fontsize=13, fontweight="bold")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=COLOR_TRAD, label="留出法"), Patch(color=COLOR_TREE, label="K折交叉验证")],
              loc="lower right")
    return _save(fig, "F1_数据划分方式对比.png")


def draw_F3(images, labels):
    df = compare_feature_subsets(images, labels, max_samples=2000, random_seed=42)
    df.to_csv(OUT_DIR / "_data_F3.csv", index=False)
    df = df.sort_values("F1分数", ascending=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = plt.cm.Greens(np.linspace(0.35, 0.9, len(df)))
    bars = ax.barh(df["特征组合"], df["F1分数"], color=colors, edgecolor="white")
    for bar, v, dim in zip(bars, df["F1分数"], df["特征维度"]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f} ({dim}维)", va="center", fontsize=9)
    ax.set_xlim(0, 1.02); ax.set_xlabel("F1 分数")
    ax.set_title("特征组合对比（随机森林 + CLAHE，控制变量）", fontsize=13, fontweight="bold")
    return _save(fig, "F3_特征组合对比.png")


def draw_F10():
    train_loader, val_loader, test_loader = build_cnn_loaders(max_samples=2000, batch_size=64)
    df = compare_optimizers(train_loader, val_loader, test_loader, DEVICE, epochs=15)
    df.to_csv(OUT_DIR / "_data_F10.csv", index=False)
    df = df.sort_values("测试F1分数", ascending=False)
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = [COLOR_CNN if "Adam" in o else "#c0392b" for o in df["优化器"]]
    x = np.arange(len(df))
    bars = ax.bar(x, df["测试F1分数"], 0.55, color=colors, edgecolor="white", linewidth=1.5)
    _bar_label(ax, bars)
    ax.set_xticks(x); ax.set_xticklabels(df["优化器"])
    ax.set_ylim(0, 1.08); ax.set_ylabel("测试 F1 分数")
    ax.set_title("CNN 优化器对比（Adam vs SGD+Momentum，CrackCNN + CLAHE，15 epoch）",
                 fontsize=13, fontweight="bold")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=COLOR_CNN, label="Adam"), Patch(color="#c0392b", label="SGD+Momentum")],
              loc="lower right")
    return _save(fig, "F10_CNN优化器对比.png")


def draw_F11():
    cache = OUT_DIR / "_data_F11.csv"
    if cache.exists():
        df = pd.read_csv(cache)
        print("    使用已缓存网格搜索结果 _data_F11.csv（不重训）")
    else:
        train_loader, val_loader, _ = build_cnn_loaders(max_samples=2000, batch_size=64)
        df = grid_search_cnn(train_loader, val_loader, DEVICE,
                             lr_list=[1e-4, 5e-4, 1e-3], dropout_list=[0.3, 0.5, 0.7],
                             bs_list=[32, 64], epochs_per_trial=15, patience=5)
        df.to_csv(cache, index=False)
    # 注：本数据集验证集较小，各组 best_val_acc 均为 1.0（无区分度），
    # 故热力图改用 best_val_loss（越低越优，0.001–0.01，有区分度）。
    vmin, vmax = df["best_val_loss"].min(), df["best_val_loss"].max()
    best = df.iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax_idx, bs in enumerate(sorted(df["batch_size"].unique())):
        subset = df[df["batch_size"] == bs]
        pivot = subset.pivot_table(index="lr", columns="dropout", values="best_val_loss")
        axes[ax_idx].imshow(pivot.values, cmap="RdYlGn_r", aspect="auto", vmin=vmin, vmax=vmax)
        axes[ax_idx].set_xticks(range(len(pivot.columns)))
        axes[ax_idx].set_xticklabels([f"{d:.1f}" for d in pivot.columns])
        axes[ax_idx].set_yticks(range(len(pivot.index)))
        axes[ax_idx].set_yticklabels([f"{lr:.0e}" for lr in pivot.index])
        axes[ax_idx].set_title(f"batch_size = {bs}", fontweight="bold")
        axes[ax_idx].set_xlabel("Dropout 比例"); axes[ax_idx].set_ylabel("学习率 (lr)")
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                v = pivot.iloc[i, j]
                axes[ax_idx].text(j, i, f"{v:.4f}", ha="center", va="center", fontweight="bold",
                                  color="white" if (v - vmin) > (vmax - vmin) * 0.55 else "black")
    plt.suptitle("CNN 超参数网格搜索（验证损失，↓越低越优；CrackCNN + CLAHE，lr × dropout × batch_size）\n"
                 f"最优组合：lr={best['lr']}, dropout={best['dropout']}, batch_size={int(best['batch_size'])} "
                 f"(val_loss={best['best_val_loss']:.4f})",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    return _save(fig, "F11_CNN超参数网格搜索.png")


# ==================================================================
# 五、主流程
# ==================================================================
def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    print(f"=== PPT 图表生成 | stage={stage} | 输出: {OUT_DIR} ===")
    print(f"=== 设备: {DEVICE} ===\n")

    # ---- 零训练组（仅读持久化数据）----
    print("[零训练组]")
    for fn in [draw_F0, draw_F2, draw_F4, draw_F5, draw_F6, draw_F7, draw_F8, draw_F9, draw_F12]:
        try:
            fn()
        except Exception as e:
            print(f"  [失败] {fn.__name__}: {e}")

    if stage in ("light", "all"):
        print("\n[轻量实验组 F1/F3 — notebook01 对比函数]")
        try:
            images, labels = ga.load_dataset(_data_root(), max_samples=1000)
            for fn in [draw_F1, draw_F3]:
                try:
                    fn(images, labels)
                except Exception as e:
                    print(f"  [失败] {fn.__name__}: {e}")
        except Exception as e:
            print(f"  [数据加载失败] {e}")

    if stage == "all":
        print("\n[CNN 实验组 F10/F11 — notebook03 对比函数，耗时较长]")
        for fn in [draw_F10, draw_F11]:
            try:
                fn()
            except Exception as e:
                print(f"  [失败] {fn.__name__}: {e}")

    print(f"\n=== 完成。输出目录: {OUT_DIR} ===")


def _data_root():
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    _d = os.getenv("CRACK_DATA_ROOT")
    return Path(_d).expanduser().resolve() if _d else (PROJECT_ROOT / "data")


if __name__ == "__main__":
    main()
