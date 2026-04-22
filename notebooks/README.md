# Notebooks 说明

本目录存放研究性专题的全部 Jupyter Notebook，最终提交的**研究报告**也将整合于此。

## 文件命名规范

| 编号 | 文件名 | 用途 |
|------|--------|------|
| 01 | `01_数据探索与预处理.ipynb` | 数据读取、划分、预处理、可视化 |
| 02 | `02_传统机器学习.ipynb` | 决策树、SVM、K-Means 实验与对比 |
| 03 | `03_深度学习.ipynb` | CNN 模型设计、训练、调参 |
| 04 | `04_综合展示系统.ipynb` | 基于 ipywidgets 的交互式系统（最终演示+报告主体） |

## 提交规范

1. **清除输出**：非最终提交前，建议清除 Cell Output 后再提交到 Git（避免仓库膨胀）。
2. **微软雅黑**：在 Notebook 开头运行：
   ```python
   from src.plot_config import set_chinese_font
   set_chinese_font()
   ```
3. **代码可读性**：关键代码块保留在 `src/` 的 `.py` 模块中，Notebook 中只做调用和展示。

## 如何编辑

**主要方式：用 VSCode 直接打开 .ipynb 文件**

确保已激活 conda 环境 `bjtu_ml`，然后用 VSCode 直接打开 `notebooks/` 下的 `.ipynb` 文件即可。VSCode 内置 Jupyter 支持，无需启动额外服务。
