# 裂纹图像识别系统 — QWEN.md

> 北京交通大学《机器学习与Python编程》课程研究性专题

## 项目概述

裂纹图像识别系统，用于将路面图像分类为有裂纹（Positive）或无裂纹（Negative）。项目采用 **Notebook 驱动** 的研究流程，结合传统机器学习、无监督聚类和自建 CNN 三种方法进行对比分析。

- **语言**：Python 3.10
- **环境管理**：Conda（环境名 `bjtu_ml`）
- **核心框架**：PyTorch 2.1.1 (CUDA 11.8 / CPU 自动回退)、scikit-learn 1.4.2、OpenCV 4.9.0
- **代码格式**：Ruff（替代 Black + Flake8 + isort）
- **协作**：Git 功能分支 + GitHub Issues 讨论

## 常用命令

```bash
# 创建并激活环境
conda env create -f environment.yml
conda activate bjtu_ml

# 运行测试
pytest tests/ -v

# 代码检查与格式化
ruff check src/ tests/
ruff format src/ tests/

# 验证数据集路径配置
python -c "from src.config import DATA_ROOT, check_data_exists; print(DATA_ROOT); check_data_exists()"

# 注册 Jupyter Kernel
python -m ipykernel install --user --name=bjtu_ml --display-name "Python (bjtu_ml)"

# 清除 Notebook 输出（首次搭建时执行一次）
nbstripout --install
```

## 项目结构

```
bjtu_ml_research/
├── notebooks/                  # Jupyter Notebook（研究报告主体）
│   ├── 01_data_exploration     # 数据探索与预处理
│   ├── 02_traditional_ml       # 传统机器学习（决策树/SVM/KNN）
│   ├── 03_unsupervised         # 无监督聚类（K-Means/GMM/DBSCAN）
│   ├── 04_deep_learning        # 深度学习（自建 CNN）
│   └── 05_comparison           # 综合对比与结论
├── src/                        # 可复用 Python 模块
│   ├── config.py               # 全局配置（.env 加载、路径、DEVICE）
│   ├── data_utils.py           # 图像预处理 + 特征提取（HOG/LBP/GLCM/边缘密度）
│   ├── plot_config.py          # Matplotlib 中文字体配置
│   ├── models/
│   │   ├── traditional.py      # 传统分类器（DecisionTree/SVM/KNN）
│   │   ├── unsupervised.py     # 无监督方法（K-Means/GMM/DBSCAN）
│   │   └── cnn.py              # 自建小型 CNN（CrackCNN）
│   ├── training/
│   │   ├── losses.py           # 损失函数
│   │   └── optimizers.py       # 超参数搜索 / LR 调度
│   └── evaluation/
│       └── metrics.py          # 评价指标（accuracy/precision/recall/F1）
├── tests/                      # 冒烟测试
├── data/                       # 数据集（本地，不上传 Git）
├── docs/                       # 文档、分工说明、讨论记录
├── reports/                    # 导出的 PDF/HTML 报告
├── outputs/                    # 实验输出（图片、日志）
├── videos/                     # 操作演示视频
├── .env.example                # 环境变量模板
├── environment.yml             # Conda 环境定义
├── pyproject.toml              # Ruff 配置
└── requirements.txt            # pip 依赖清单（备用）
```

## 数据流

```
.env (CRACK_DATA_ROOT)
  → src/config.py (DATA_ROOT, POSITIVE_DIR, NEGATIVE_DIR, DEVICE)
    → src/data_utils.py (load_dataset, 预处理, 特征提取)
      → src/models/* (训练与预测)
        → src/evaluation/metrics.py (评分)
```

## 开发约定

### 代码风格
- **Ruff** 为唯一的格式化和 lint 工具（line-length 88, target py310, LF 换行）
- `.gitattributes` 强制所有文本文件使用 `eol=lf`
- 函数 docstring 使用 Google 风格
- 注释和文档以中文为主

### Git 规范
- 分支命名：`feature/描述`、`fix/描述`、`docs/描述`
- 提交信息前缀：`feat:`、`fix:`、`docs:`、`refactor:`、`env:`
- `main` 分支只接受合并，不直接提交代码
- Notebook 输出由 `nbstripout` 在提交时自动清除
- `.env`、模型权重（`*.pth`、`*.pkl`）、数据集不上传 Git

### 环境管理
- 新增依赖：安装后手动编辑 `environment.yml`，**禁止**使用 `conda env export`（会破坏 PyTorch CUDA 版本后缀）
- 协作者同步环境：`conda env update -f environment.yml --prune`

### 重要注意事项
- `src/data_utils.py` 和 `src/models/` 中的函数目前多为 `raise NotImplementedError` 的桩代码，等待逐步实现
- 数据集路径通过 `.env` 文件的 `CRACK_DATA_ROOT` 变量配置，每人各自设置
- GPU 可用时自动使用 CUDA，否则回退 CPU；无 GPU 建议先用小数据集验证
- 画图前需调用 `from src.plot_config import set_chinese_font; set_chinese_font()` 配置中文字体
