"""
训练器模块：封装 CNN 训练、验证、早停和可视化。
"""

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class CNNTrainer:
    """CNN 训练器，封装训练循环、验证、早停和 TensorBoard 日志。"""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        save_dir: str | Path = "outputs",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.criterion: nn.Module | None = None
        self.optimizer: optim.Optimizer | None = None
        self.scheduler: optim.lr_scheduler.ReduceLROnPlateau | None = None
        self.writer: SummaryWriter | None = None

        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_accs: List[float] = []
        self.val_accs: List[float] = []
        self.best_val_loss = float("inf")
        self.best_epoch = 0

    def fit(
        self,
        epochs: int = 30,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 10,
        log_dir: str | None = None,
        criterion_name: str = "cross_entropy",
    ) -> Dict[str, List[float]]:
        """训练模型。

        Parameters
        ----------
        epochs : int, 最大训练轮数。
        lr : float, 初始学习率。
        weight_decay : float, L2 正则化系数。
        patience : int, 早停轮数。
        log_dir : str or None, TensorBoard 日志目录。
        criterion_name : str, "cross_entropy" 或 "focal"。
        """
        from src.training.losses import FocalLoss

        valid = {"cross_entropy", "focal"}
        if criterion_name not in valid:
            raise ValueError(f"不支持的损失函数: {criterion_name}，可选 {valid}")
        if criterion_name == "focal":
            self.criterion = FocalLoss(alpha=0.25, gamma=2.0)
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

        if log_dir is not None:
            self.writer = SummaryWriter(log_dir=log_dir)

        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        no_improve = 0
        best_state = None

        print(f"开始训练 (共 {epochs} 轮, 早停 patience={patience})")
        print("-" * 60)

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self._train_epoch()
            val_loss, val_acc = self._validate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)

            if self.writer is not None:
                self.writer.add_scalars(
                    "Loss", {"train": train_loss, "val": val_loss}, epoch
                )
                self.writer.add_scalars(
                    "Accuracy", {"train": train_acc, "val": val_acc}, epoch
                )
                for name, param in self.model.named_parameters():
                    self.writer.add_histogram(f"weights/{name}", param.data, epoch)

            self.scheduler.step(val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                best_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                self.save_checkpoint("best_model.pth")
                no_improve = 0
            else:
                no_improve += 1

            if epoch % 5 == 0 or epoch == 1:
                improved = "*" if no_improve == 0 else ""
                print(
                    f"Epoch {epoch:3d}/{epochs} | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} {improved}"
                )

            if no_improve >= patience:
                print(f"早停: 验证 loss 连续 {patience} 轮未改善, 在第 {epoch} 轮停止")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        print("-" * 60)
        print(
            f"训练完成, 最佳验证 Loss: {self.best_val_loss:.4f} (Epoch {self.best_epoch})"
        )

        if self.writer is not None:
            self.writer.close()

        return {
            "train_loss": self.train_losses,
            "val_loss": self.val_losses,
            "train_acc": self.train_accs,
            "val_acc": self.val_accs,
        }

    def _train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        return running_loss / total, correct / total

    def _validate(self) -> Tuple[float, float]:
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return running_loss / total, correct / total

    def evaluate(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """在给定数据集上评估，返回预测标签、概率和真实标签。"""
        self.model.eval()
        all_preds = []
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
                all_targets.extend(targets.numpy())

        return np.array(all_preds), np.array(all_probs), np.array(all_targets)

    def save_checkpoint(self, filename: str) -> None:
        torch.save(self.model.state_dict(), self.save_dir / filename)

    def load_checkpoint(self, filename: str) -> None:
        self.model.load_state_dict(
            torch.load(self.save_dir / filename, map_location=self.device)
        )

    def plot_history(self) -> plt.Figure:
        """绘制 Loss 和 Accuracy 曲线。"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(self.train_losses, label="训练 Loss", color="tomato")
        axes[0].plot(self.val_losses, label="验证 Loss", color="steelblue")
        axes[0].axvline(
            x=self.best_epoch - 1,
            color="green",
            linestyle="--",
            label=f"最佳 Epoch ({self.best_epoch})",
        )
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Loss 曲线")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(self.train_accs, label="训练准确率", color="tomato")
        axes[1].plot(self.val_accs, label="验证准确率", color="steelblue")
        axes[1].axvline(
            x=self.best_epoch - 1,
            color="green",
            linestyle="--",
            label=f"最佳 Epoch ({self.best_epoch})",
        )
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("准确率")
        axes[1].set_title("准确率曲线")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig
