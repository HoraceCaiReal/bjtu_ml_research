# PPT 演讲图表生成 — 设计文档

- 日期：2026-06-14
- 目标：根据《裂纹图像识别系统_演讲大纲》生成 PPT 所需的全部数据图表，输出到 `reports/ppt图表输出/`
- 核心约束：**所有模型训练/评估结果数据必须来自项目已有流程**；新增代码仅用于"画图"及其必要的数据获取胶水，不得新增独立的训练/评估逻辑。

## 1. 背景与范围

演讲大纲按"数据处理 → 传统监督 → 树/集成 → 无监督 → CNN/总结"递进，每个板块的 PPT 页都要求配数据图表。本项目采用 notebook-only 架构，结果数据散落在 `outputs/results/*.csv`、`outputs/combo_verify/*.json`、`outputs/models/**` 以及 notebook 内联的对比实验函数中。

本任务要为大纲每个配图位置生成一张可直接贴入 PPT 的 PNG。共 **13 张图（F0–F12）**。

## 2. 数据来源映射（已逐项核实）

### 2.1 已持久化、可直接读取（零训练）

| 数据文件 | 内容 | 用于图表 |
|---|---|---|
| `outputs/batch_verify/20260613_043410/results.csv` | 全数据 combo 验证：模型×预处理(none/clahe/gaussian/median/组合)×criterion×optimization×validation(holdout/kfold) 的 acc/prec/recall/f1/auc | F2 预处理对比 |
| `outputs/results/traditional_comparison.csv` | 7 模型(DT/RF/XGB/LGBM/SVM/LR/NB) × {cv_f1, test_f1, test_auc, time_s} | F4 传统监督、F5 树/集成 |
| `outputs/results/unsupervised_comparison.csv` | 5 聚类 × {n_clusters, n_noise, silhouette, ari, nmi} | F6 聚类对比 |
| `outputs/combo_verify/unsup_results.json` | 5 聚类 × {silhouette, davies_bouldin, calinski_harabasz, ari, nmi}（更全） | F6 聚类对比 |
| `outputs/results/cnn_comparison.csv` | 6 损失配置(cross_entropy/focal_gamma2/gamma3/balanced/label_smoothing/dice) × {val_f1, test_f1, test_acc, train_time_s, epochs} | F8 损失对比、F12 总结 |
| `outputs/models/cnn/*_history.json` | 6 份，每份 {train_loss, train_acc, val_f1}（逐 epoch，15 轮） | F9 训练曲线 |
| `outputs/models/traditional/*_best_params.json` | 各模型 GridSearchCV 最优超参 | 参考 |

### 2.2 notebook 已有、但未持久化（需运行已有实验函数，用户已确认"全部运行"）

| 实验 | notebook 函数 | 底层依赖 | 用于图表 | 预估耗时 |
|---|---|---|---|---|
| 数据划分方式对比 | `01:compare_split_strategies` | `apply_*`/`extract_*`(gradio_app) + RF | F1 | ~3min |
| 特征组合对比 | `01:compare_feature_subsets` | `extract_features_separate`(gradio_app) + RF | F3 | ~3min |
| CNN 优化器对比 | `03:compare_optimizers` | `CrackCNN` + CE，Adam vs SGD+Momentum 各两档 | F10 | ~20min |
| CNN 超参数网格 | `03:grid_search_cnn` | `CrackCNN`，lr×dropout×bs = 3×3×2=18 组 | F11 | ~1–2hr |

> 说明：这 4 个函数是 notebook 既有代码的**忠实搬运**，其训练/评估全部调用 `src/gradio_app.py` 已有函数（`load_dataset`、`apply_clahe/gaussian/median`、`extract_hog/lbp/glcm/edge`、`extract_features_separate`、`_subsample_balanced`、`CrackCNN`、`_run_cnn`）。不新增任何训练/评估方法。

## 3. 图表清单（F0–F12）

| 编号 | 图表 | 对应大纲 | 数据来源 | 类型 |
|---|---|---|---|---|
| F0 | 数据集类别分布（20000 有裂/20000 无裂） | 一·开场 | `load_dataset` 统计 | 饼图+柱状 |
| F1 | 数据划分方式对比（留出法 50%~90% + K折 5/10） | 二·左 | `compare_split_strategies` | 柱状图 |
| F2 | 预处理方式对比（none/CLAHE/高斯/中值/组合） | 二·中 | `results.csv` 聚合 | 柱状图 |
| F3 | 特征组合对比（HOG/LBP/GLCM/边缘 单独+组合） | 二·右 | `compare_feature_subsets` | 柱状图 |
| F4 | 传统监督（SVM/朴素贝叶斯/逻辑回归）对比 | 三 | `traditional_comparison.csv` | 分组柱状 |
| F5 | 树/集成（DT/RF/XGB/LGBM）对比 + 训练时间 | 四 | `traditional_comparison.csv` | 双轴柱状 |
| F6 | 无监督聚类（5种）对比（轮廓/CH/ARI） | 五 | unsup CSV+json | 分组柱状 |
| F7 | 无监督 vs 监督 与真实标签一致性（ARI） | 五 | 综合 | 柱状+标注 |
| F8 | CNN 损失函数对比（6种，F1/acc） | 六 | `cnn_comparison.csv` | 分组柱状 |
| F9 | CNN 损失训练曲线（6条 F1/loss vs epoch） | 六 | `*_history.json` | 折线图 |
| F10 | CNN 优化器对比（Adam vs SGD+Momentum） | 六 | `compare_optimizers` | 柱状图 |
| F11 | CNN 超参数网格搜索热力图（lr×dropout，按 bs 分面） | 六 | `grid_search_cnn` | 热力图 |
| F12 | 全方法效果总结（传统最佳/CNN最佳/无监督最佳） | 六 | 3 CSV 综合 | 柱状图 |

## 4. 实现方式

### 4.1 脚本结构

新建 `scripts/generate_ppt_charts.py`，三层：

1. **画图层（全新代码）**：统一中文字体（Microsoft YaHei）、统一配色（传统=蓝、树/集成=绿、CNN=橙、无监督=紫）、保存 300dpi PNG 到 `reports/ppt图表输出/`。每个图一个函数 `draw_Fxx()`。
2. **数据加载层**：已持久化数据用 `pd.read_csv`/`json.load`；4 项实验调用从 notebook 搬运的对比函数。
3. **实验函数层（notebook 搬运，非新方法）**：`compare_split_strategies`/`compare_feature_subsets`/`compare_optimizers`/`grid_search_cnn`，逐字来自 notebook 01/03，仅去除 notebook 专属语句（`display()`/`plt.show()`/magic），改为返回 DataFrame。

### 4.2 运行

```bash
conda run -n bjtu_ml python scripts/generate_ppt_charts.py
```

bjtu_ml 环境已验证：torch 2.1.1+cu118（CUDA 可用）、sklearn 1.4.2、matplotlib 3.8.4。

### 4.3 输出命名

`reports/ppt图表输出/F{编号}_{中文名}.png`，如 `F1_数据划分方式对比.png`。

## 5. 关键设计决策

1. **预处理对比用 `results.csv`（全数据）而非 notebook 子采样**：更权威，且无需额外运行。**固定 `model=random_forest` + `optimization=pretrained` + `validation=holdout`**，仅对比 `preprocessing` 维度（none/clahe/gaussian/median/clahe+gaussian/clahe+median），避免 criterion/optimization/model 维度干扰（控制变量）。
2. **F1/F3 用 notebook 01 的控制变量实验**：固定 RF+全特征，仅变划分/特征，纯粹反映单一变量影响。
3. **F7 用 ARI 而非"准确率"**：聚类无真准确率；KMeans ARI≈0.10 接近随机。用"与真实标签一致性(ARI)"作统一标尺（监督≈1.0、聚类≈0.1），图上文字标注"无标签≈随机猜测"，传达大纲"标签至关重要"论点，且不造假数据。
4. **不修改 notebook 原文件**：对比函数搬运到生成脚本，notebook 保持不变。

## 6. 验证策略（用户重点要求）

实现完成后，按三层验证，并产出验证报告 `reports/ppt图表输出/_验证报告.md`。

### 6.1 数据有效性验证（已有数据是否 stale/可信）

针对直接读取的持久化数据：

- **合理性检查**：F1/AUC/acc ∈ [0,1]；训练时间为正、量级合理；模型排序符合常理（如集成 > 单树）。
- **交叉一致性**：`traditional_comparison.csv` 与 `combo_verify/all_results.json`（M 系列）的同模型指标是否同量级；`unsup_comparison.csv` 与 `unsup_results.json` 是否一致。
- **与项目文档对照**：`reports/组合验证评估报告.md`、`reports/项目汇报设计辅助文档.md` 中引用的数值是否与 CSV 吻合。
- **关键抽查（不重训，仅加载预训练模型预测）**：选 1–2 个传统模型，用 `src/gradio_app.py` 加载 `outputs/models/traditional/*.joblib` 在小子集上预测，确认指标与 CSV 同量级，证明 CSV 非 stale。

> 若发现 stale 或不一致，在验证报告中标注，并决定是否用当前代码重跑该项（重跑仍属"已有流程"）。

### 6.2 图表视觉验证（视觉 MCP）

对每张生成的 PNG，用视觉分析 MCP（`analyze_image`）检查：

- 中文标题/轴/图例正常显示（无方块/乱码/缺字）。
- 数值标签正确、无重叠/截断。
- 图表类型与大纲要求匹配。
- 配色/可读性达标（PPT 友好）。

### 6.3 大纲预期符合性验证

逐图对照大纲话术，核对图/数据是否支持其论断：

- 大纲："SVM 在 RBF 核下优于线性核" → 核对 SVM 数据。
- 大纲："XGBoost/LightGBM 最优，LightGBM 更快" → F5 是否体现。
- 大纲："无监督准确率远低于监督" → F7 是否体现。
- 大纲："CNN 最优，Focal Loss 最佳" → F8/F12 是否体现。
- 大纲："CLAHE 效果最显著" → F2 数据是否支持（**诚实呈现**：若 clahe 单独非最高，按实际数据画，并在报告中注明与大纲措辞的差异）。

发现不一致时：优先忠于数据，在验证报告中如实记录差异，供演讲者决定措辞。

## 7. 执行顺序

1. 写 `scripts/generate_ppt_charts.py`（画图层 + 数据加载层 + 4 个搬运的实验函数）。
2. 先跑"零训练"图表（F0/F2/F4–F9/F12），快速产出主体。
3. 跑轻量实验图表（F1/F3，~6min）。
4. 跑 CNN 实验图表（F10 ~20min、F11 ~1–2hr）。
5. 三层验证（6.1/6.2/6.3）+ 验证报告。
6. 修订（若有 stale/视觉/大纲不符）。

## 8. 不做的事（YAGNI）

- 不新增任何训练/评估/优化算法。
- 不修改 notebook 原文件、不改动 `src/gradio_app.py`。
- 不为单图写独立脚本；所有图在一个生成脚本内，按函数组织。
- 不生成大纲未要求的图表（如混淆矩阵、ROC 曲线——combo_verify 已有，不重复）。
