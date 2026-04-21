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

## 如何启动

确保已激活 conda 环境 `bjtu_ml`：

```bash
conda activate bjtu_ml
jupyter lab
```

然后在浏览器中打开 `http://localhost:8888`，在 `notebooks/` 下编辑即可。
