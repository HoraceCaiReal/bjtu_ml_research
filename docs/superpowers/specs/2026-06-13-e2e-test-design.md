# 端到端测试方案设计

> 日期：2026-06-13
> 状态：已确认

## 目标

创建一个独立测试脚本 `scripts/e2e_test.py`，在 15 分钟内验证裂纹图像识别系统的端到端可用性，包括 Gradio 可视化界面的实际运行。

## 方案选择

- **选定方案**：独立测试脚本（方案A）
- **排除方案**：Notebook 串联（太慢）、扩展 `_smoke_test.py`（结构不匹配）

## 测试阶段

### Stage 0：环境检查（<30s）

- 验证 Python 版本 ≥ 3.10
- 检查关键依赖可导入：`torch`, `sklearn`, `cv2`, `xgboost`, `lightgbm`, `gradio`, `skimage`, `joblib`
- 报告 CUDA 可用性
- 通过标准：所有依赖导入成功

### Stage 1：数据路径验证（<30s）

- 读取 `.env` 中的 `CRACK_DATA_ROOT`
- 验证 `data/Positive/` 和 `data/Negative/` 目录存在且非空
- 抽样加载 10 张图片，验证可读
- 通过标准：目录存在 + 图片可加载

### Stage 2：特征提取验证（<30s）

- 对 1 张测试图片提取 HOG、LBP、GLCM、edge_density 特征
- 验证每个特征向量为 numpy array 且维度 > 0
- 通过标准：4 种特征全部提取成功

### Stage 3：传统模型推理验证（~1-2min）

加载 4 个代表性传统模型，对 Stage 2 的特征直接推理：

| 模型 | 文件 | 覆盖类型 |
|------|------|---------|
| Random Forest | `random_forest_best.joblib` | Bagging |
| XGBoost | `xgboost_best.joblib` | Boosting |
| SVM | `svm_best.joblib` | 核方法 |
| Logistic Regression | `logistic_regression_best.joblib` | 线性模型 |

- 通过标准：每个模型 `.predict()` 不报错，输出为 0 或 1

### Stage 4：CNN 推理验证（~1-2min）

- 从 `crackcnn_best_config.json` 读取配置（input_size=128）
- 在脚本内重建 `CrackCNN` 类定义（从 notebook 03 复制，保持自包含）
- 加载 `crackcnn_cross_entropy_best.pth` 权重
- 对 `data/real_test/` 或 `data/Positive/` 中的 3 张图片推理
- 通过标准：不报错 + 输出概率值在 [0, 1] 范围内

### Stage 5：Gradio 模块检查（<30s）

- `import src.gradio_app` 成功
- `create_interface()` 返回 `gr.Blocks` 对象
- `run_pipeline` 函数可调用
- 通过标准：以上三项均通过

### Stage 6：方法组合实际运行（~10-15min）

直接调用 `gradio_app.run_pipeline()`，用小样本（`max_samples=200`）模拟用户操作 3 个代表性组合：

#### 组合 1：Random Forest — 传统模型完整链路

```
模型: random_forest
数据处理: clahe+median, 特征: hog+lbp
验证: hold-out 80/20, grid_search, scoring=f1
通过标准: 运行无报错 + F1 > 0.5
```

#### 组合 2：CNN — 深度学习训练+推理链路

```
模型: cnn
数据处理: clahe+median
损失: cross_entropy, epochs=2, batch_size=32
验证: hold-out 80/20, scoring=f1
通过标准: 运行无报错 + 训练 loss 下降或输出有效指标
```

#### 组合 3：K-Means — 无监督聚类链路

```
模型: kmeans
数据处理: clahe+median, 特征: hog+lbp+glcm
n_clusters=2
验证: silhouette
通过标准: 运行无报错 + silhouette_score > 0
```

## 输出报告格式

```
====================================
  裂纹图像识别系统 — 端到端测试报告
====================================
Stage 0: 环境检查         ✅ PASS  (torch 2.x, CUDA: True)
Stage 1: 数据路径验证     ✅ PASS  (40000 images found)
Stage 2: 特征提取验证     ✅ PASS  (HOG:324 LBP:256 GLCM:48 ED:1)
Stage 3: 传统模型推理     ✅ PASS  (4/4 models OK)
Stage 4: CNN 推理验证     ✅ PASS  (probs: [0.987, 0.012, 0.956])
Stage 5: Gradio 模块检查  ✅ PASS  (interface created)
Stage 6: 方法组合运行     ✅ PASS  (3/3 combos OK)
  ├─ 组合1 RF:   F1=0.85 ✅
  ├─ 组合2 CNN:  trained ✅
  └─ 组合3 KM:   sil=0.42 ✅
------------------------------------
总计: 7/7 PASS  |  耗时: ~12 分钟
====================================
```

## 失败处理

- 任一阶段 FAIL 打印详细 traceback，继续跑后续阶段
- 最终 exit code：全 PASS → 0，任一 FAIL → 1
- 每个阶段记录耗时

## 运行方式

```bash
conda activate bjtu_ml
python scripts/e2e_test.py
```

## 技术决策

- **CNN 类内联重建**：从 notebook 03 复制 `CrackCNN` 类定义到测试脚本，保持测试自包含
- **`run_pipeline` 直接调用**：不通过 Gradio UI，而是直接调用 Python 函数，传入参数元组
- **小样本控制耗时**：`max_samples=200`，CNN `epochs=2`
- **宽松阈值**：F1 > 0.5（200 样本下的合理下限），silhouette > 0
