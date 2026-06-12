"""
超参数搜索模块
"""

from itertools import product
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import PROJECT_ROOT
from src.models.cnn import CrackCNN
from src.training.trainer import CNNTrainer


def grid_search_cnn(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    lr_list: List[float] | None = None,
    dropout_list: List[float] | None = None,
    batch_size_list: List[int] | None = None,
    epochs_per_trial: int = 15,
    patience: int = 5,
) -> pd.DataFrame:
    """对 CrackCNN 进行网格搜索。

    Parameters
    ----------
    train_loader : DataLoader, 训练集 DataLoader。
    val_loader : DataLoader, 验证集 DataLoader。
    device : torch.device, 计算设备。
    lr_list : list, 候选学习率。
    dropout_list : list, 候选 Dropout 比例。
    batch_size_list : list, 候选 batch size。
    epochs_per_trial : int, 每组参数训练轮数。
    patience : int, 早停轮数。
    """
    if lr_list is None:
        lr_list = [1e-4, 5e-4, 1e-3]
    if dropout_list is None:
        dropout_list = [0.3, 0.5, 0.7]
    if batch_size_list is None:
        batch_size_list = [32, 64]

    results: List[Dict] = []
    total = len(lr_list) * len(dropout_list) * len(batch_size_list)
    trial = 0

    best_global_loss = float("inf")
    best_global_dir: Path | None = None

    base_dir = PROJECT_ROOT / "outputs" / "grid_search"
    base_dir.mkdir(parents=True, exist_ok=True)

    for lr, dropout, batch_size in product(lr_list, dropout_list, batch_size_list):
        trial += 1
        trial_name = f"trial_{trial:03d}"
        trial_dir = base_dir / trial_name

        print(f"\n{'=' * 50}")
        print(
            f"实验 {trial}/{total}: lr={lr}, dropout={dropout}, batch_size={batch_size}"
        )
        print(f"{'=' * 50}")

        # 按 batch_size 重建 DataLoader
        train_data = train_loader.dataset
        new_train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

        model = CrackCNN(num_classes=2, input_channels=1, dropout_rate=dropout)
        trainer = CNNTrainer(
            model, new_train_loader, val_loader, device, save_dir=trial_dir
        )

        trainer.fit(
            epochs=epochs_per_trial,
            lr=lr,
            weight_decay=1e-4,
            patience=patience,
        )

        results.append(
            {
                "lr": lr,
                "dropout": dropout,
                "batch_size": batch_size,
                "best_val_loss": trainer.best_val_loss,
                "best_val_acc": trainer.val_accs[trainer.best_epoch - 1],
                "best_epoch": trainer.best_epoch,
            }
        )

        print(
            f"结果: best_val_loss={trainer.best_val_loss:.4f}, "
            f"best_val_acc={trainer.val_accs[trainer.best_epoch - 1]:.4f}"
        )

        # 保留全局最佳
        if trainer.best_val_loss < best_global_loss:
            best_global_loss = trainer.best_val_loss
            best_global_dir = trial_dir

    # 复制全局最佳权重到 output grid_search 根目录
    if best_global_dir is not None:
        import shutil

        src = best_global_dir / "best_model.pth"
        dst = base_dir / "best_model.pth"
        if src.exists():
            shutil.copy2(src, dst)
            print(f"\n全局最佳模型已保存到: {dst}")

    df = pd.DataFrame(results)
    df = df.sort_values("best_val_loss").reset_index(drop=True)
    return df
