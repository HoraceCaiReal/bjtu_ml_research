# Notebooks 说明

本目录存放研究性专题的全部 Jupyter Notebook，最终提交的**研究报告**也将整合于此。

## 文件命名规范

| 编号 | 文件名 | 用途 | 对应模块 |
|------|--------|------|----------|
| 01 | `01_data_exploration.ipynb` | 数据探索与预处理实验 | CLAHE、高斯滤波、中值滤波效果对比 |
| 02 | `02_traditional_ml.ipynb` | 传统机器学习实验 | HOG+LBP+GLCM+边缘密度 → 决策树/SVM/KNN 对比 |
| 03 | `03_unsupervised.ipynb` | 无监督聚类实验 | K-Means/GMM/DBSCAN（核心）+ Agglomerative/谱聚类（补充） |
| 04 | `04_deep_learning.ipynb` | 深度学习实验 | 自建小型 CNN（CrackCNN）训练与调参 |
| 05 | `05_comparison.ipynb` | 综合对比分析 | 所有方法准确率/召回率/F1/耗时/可视化多维度对比（报告主体） |

## 提交规范

1. **清除输出**：非最终提交前，建议清除 Cell Output 后再提交到 Git（避免仓库膨胀）。
2. **中文字体**：在 Notebook 开头运行：
   ```python
   from src.plot_config import set_chinese_font
   set_chinese_font()
   ```
3. **代码可读性**：关键代码块保留在 `src/` 的 `.py` 模块中，Notebook 中只做调用和展示。

## 如何编辑

**主要方式：用 VSCode 直接打开 .ipynb 文件**

确保已激活 conda 环境 `bjtu_ml`，然后用 VSCode 直接打开 `notebooks/` 下的 `.ipynb` 文件即可。VSCode 内置 Jupyter 支持，无需启动额外服务。