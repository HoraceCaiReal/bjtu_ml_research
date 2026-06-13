# Gradio 中文选项说明 & 交互音效 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Chinese `info` tooltips to every UI component, add Step overview Markdown descriptions, and implement interactive sound effects (hover/click/start/complete/error) with Battlefield 1 cash register sound for training completion.

**Architecture:** All changes confined to `src/gradio_app.py`. Chinese help text uses Gradio's native `info` parameter + Markdown blocks. Sound system injects JS via `gr.HTML`: a `SoundEngine` class using Web Audio API for synthetic tones + embedded `<audio>` element (base64 mp3) for the completion effect. Backend-to-frontend sound triggers use hidden HTML comment markers (`<!--SOUND:complete-->`) in the status markdown, detected by a MutationObserver.

**Tech Stack:** Python 3.10, Gradio (existing), standard library `base64`, vanilla JS (Web Audio API, MutationObserver), no new Python dependencies.

---

### Task 1: Add Step Overview Markdown Descriptions

**Files:** Modify `src/gradio_app.py`

- [ ] **Step 1: Add overview markdown after each Step title**

The 5 Step titles are at lines 1713, 1733, 1742, 1789, 1856. After each `gr.Markdown("### ...")` line, add a `gr.Markdown("> ...")` with the overview text.

**Edit 1 — After `gr.Markdown("### 📊 Step 1: 数据处理")` (line 1713):**

```python
                gr.Markdown("### 📊 Step 1: 数据处理")
                gr.Markdown(
                    "> 配置数据加载、预处理和特征提取方式。"
                    "预处理可降噪增强裂纹边缘；特征类型越多信息越丰富，但训练越慢。"
                )
```

**Edit 2 — After `gr.Markdown("### 🤖 Step 2: 模型选择")` (line 1733):**

```python
                gr.Markdown("### 🤖 Step 2: 模型选择")
                gr.Markdown(
                    "> 选择分类或聚类模型。传统方法训练快、可解释性强，"
                    "适合快速实验；CNN 学习能力更强但需要更多样本和训练时间；"
                    "无监督聚类无需标签即可发现数据模式。"
                )
```

**Edit 3 — After `gr.Markdown("### 🔧 Step 3: 模型超参数")` (line 1742):**

```python
                gr.Markdown("### 🔧 Step 3: 模型超参数")
                gr.Markdown(
                    "> 调整模型结构参数。默认值通常可行；"
                    "增大复杂度（深度/树数）可能提升拟合能力但增加过拟合风险，"
                    "建议从小值开始逐步尝试。"
                )
```

**Edit 4 — After `gr.Markdown("### 📉 Step 4: 损失函数 / 优化器")` (line 1789):**

```python
                gr.Markdown("### 📉 Step 4: 损失函数 / 优化器")
                gr.Markdown(
                    "> 配置损失函数和优化器。不同损失函数影响模型学习偏好；"
                    "预训练模式下此部分设置不生效（使用模型训练时的内置参数）。"
                )
```

**Edit 5 — After `gr.Markdown("### ⚡ Step 5: 参数优化 + 验证 + 指标")` (line 1856):**

```python
                gr.Markdown("### ⚡ Step 5: 参数优化 + 验证 + 指标")
                gr.Markdown(
                    "> 选择参数优化策略和验证方法。推荐先用 pretrained 快速评估模型效果，"
                    "确认方向后再用 grid_search 精细调参。"
                )
```

- [ ] **Step 2: Commit Step overviews**

```bash
git add src/gradio_app.py && git commit -m "feat: 添加Step概述中文说明"
```

---

### Task 2: Add `info` Parameters — Step 1 (Data Processing)

**Files:** Modify `src/gradio_app.py:1715-1730`

- [ ] **Step 1: Add `info` to all Step 1 components**

Replace lines 1715-1730 (the split_method through max_samples definitions) with:

```python
                split_method = gr.Dropdown(
                    choices=["holdout"], value="holdout", label="划分方法",
                    info="将数据随机分为训练集和测试集。目前仅支持留出法(holdout)")
                split_ratio = gr.Slider(0.5, 0.9, 0.7, step=0.05, label="训练集比例",
                                        visible=True,
                                        info="训练集占总数据的比例。0.7=70%训练，越高模型看到数据越多但测试评估越不稳定")
                use_stratify = gr.Checkbox(True, label="分层抽样",
                                           info="保持训练/测试集中正负样本比例一致，避免某类样本分布偏差导致评估失准")

                preprocessing = gr.Radio(
                    choices=["none", "clahe", "gaussian", "median",
                             "clahe+gaussian", "clahe+median"],
                    value="clahe+median", label="预处理方法",
                    info="图像预处理管线。clahe+median 增强裂纹对比度同时去噪，推荐默认；none 跳过预处理")

                features = gr.CheckboxGroup(
                    choices=["hog", "lbp", "glcm", "edge_density"],
                    value=["hog", "lbp", "glcm", "edge_density"], label="特征类型",
                    info="提取的特征类型。HOG 捕获边缘方向，LBP 描述局部纹理，GLCM 统计纹理共生矩阵，edge_density 量化边缘密度")
                max_samples = gr.Slider(200, 4000, 2000, step=200, label="样本数上限",
                                        info="使用的样本总数上限。越多越准确但训练越慢；2000 以上结果通常较稳定，内存不足时可降低")
```

- [ ] **Step 2: Commit Step 1 info**

```bash
git add src/gradio_app.py && git commit -m "feat: Step1组件添加中文info说明"
```

---

### Task 3: Add `info` Parameters — Step 2 & Step 3 (Model & Hyperparams)

**Files:** Modify `src/gradio_app.py:1735-1786`

- [ ] **Step 1: Add info to model_choice (line 1735-1738)**

```python
                model_choice = gr.Dropdown(
                    choices=MODEL_CHOICES,
                    value="随机森林 (Random Forest)", label="模型",
                    filterable=False,
                    info="选择分类/聚类模型。传统方法训练快、可解释；CNN 能力强但需更多资源；聚类无需标签")
```

- [ ] **Step 2: Add info to Step 3 hyperparameter components (lines 1744-1786)**

Replace the component definitions from line 1744 to 1786 with info-added versions:

```python
                # 决策树参数
                with gr.Group(visible=False) as dt_params:
                    dt_max_depth = gr.Slider(3, 50, 15, step=1, label="max_depth",
                        info="树的最大深度。越大越复杂、越容易过拟合。3-15 适合简单问题，>20 需谨慎")
                    dt_min_samples_split = gr.Slider(2, 20, 5, step=1, label="min_samples_split",
                        info="内部节点再划分所需最小样本数。越大越防过拟合，2-5 为常用范围")
                # SVM参数
                with gr.Group(visible=False) as svm_params:
                    svm_C = gr.Number(1.0, label="C (正则化)", precision=2,
                        info="正则化强度的倒数。越大拟合越强、易过拟合；越小边界越平滑。建议对数尺度调参（0.1, 1, 10）")
                # 朴素贝叶斯参数
                with gr.Group(visible=False) as nb_params:
                    nb_var_smoothing = gr.Number(1e-9, label="var_smoothing", precision=10,
                        info="方差平滑项，防止零方差导致数值问题。默认 1e-9 通常无需调整")
                # 随机森林参数（默认模型，初始可见）
                with gr.Group(visible=True) as rf_params:
                    rf_n_estimators = gr.Slider(50, 500, 100, step=10, label="n_estimators",
                        info="决策树数量。越多越稳定但收益递减，100-200 通常足够，更多训练变慢")
                    rf_max_depth = gr.Slider(3, 50, 20, step=1, label="max_depth",
                        info="单棵树最大深度。限制深度可防过拟合；20 左右为常用上限")
                    rf_min_samples_split = gr.Slider(2, 20, 5, step=1, label="min_samples_split",
                        info="内部节点再划分所需最小样本数。增大可防止学习噪声模式")
                # 逻辑回归参数
                with gr.Group(visible=False) as lr_params:
                    lr_C = gr.Number(1.0, label="C (正则化)", precision=2,
                        info="正则化强度的倒数。C 越大正则化越弱、越易过拟合。建议对数尺度调参")
                # XGBoost参数
                with gr.Group(visible=False) as xgb_params:
                    xgb_n_estimators = gr.Slider(50, 300, 100, step=10, label="n_estimators",
                        info="提升轮数（树的数量）。过多会过拟合，配合 learning_rate 使用；小学习率需更多轮数")
                    xgb_max_depth = gr.Slider(3, 12, 6, step=1, label="max_depth",
                        info="树的最大深度。XGBoost 通常用 3-8，较浅的树天然防过拟合")
                    xgb_subsample = gr.Slider(0.5, 1.0, 0.8, step=0.05, label="subsample",
                        info="每棵树随机采样的训练数据比例。0.8 是常用值，降低可增加随机性防过拟合")
                # LightGBM参数
                with gr.Group(visible=False) as lgbm_params:
                    lgbm_n_estimators = gr.Slider(50, 300, 100, step=10, label="n_estimators",
                        info="提升迭代次数。LightGBM 收敛快，100-200 通常足够；观察验证曲线判断是否过拟合")
                    lgbm_max_depth = gr.Slider(3, 12, 6, step=1, label="max_depth",
                        info="树深度。LightGBM 叶子生长策略下深度通常不大，-1=不限制")
                    lgbm_num_leaves = gr.Slider(15, 127, 31, step=4, label="num_leaves",
                        info="每棵树的叶子数。控制模型复杂度，通常设为 31-63；越大模型越复杂")
                # CNN参数
                with gr.Group(visible=False) as cnn_params:
                    cnn_dropout = gr.Slider(0.0, 0.9, 0.5, step=0.05, label="Dropout 比例",
                        info="随机失活比例。0.3-0.5 常用，训练时随机丢弃神经元防止过拟合。0=不使用 Dropout")
                    cnn_batch_size = gr.Slider(16, 256, 64, step=16, label="Batch Size",
                        info="每批样本数。小批量(32-64)训练快但梯度噪声大；大批量(128-256)梯度更稳定但需更多显存")
                    cnn_epochs = gr.Slider(5, 100, 30, step=5, label="最大 Epochs",
                        info="最大训练轮数。配合早停使用，设置较大值让早停机制自动选择最佳轮数")
                    cnn_early_stopping = gr.Slider(3, 30, 10, step=1, label="早停耐心值",
                        info="验证 loss 连续不下降的轮数后自动停止。越大容忍度越高，可能等到更好模型但也可能过拟合")
                    cnn_input_size = gr.Dropdown(
                        choices=[64, 128, 256], value=128, label="输入图像尺寸 (input_size)",
                        info="输入图像缩放尺寸。越大细节保留越多但训练显著变慢。128 为速度与精度平衡")
                    cnn_weight_decay = gr.Number(1e-4, label="Weight Decay (L2正则)", precision=5,
                        info="L2 正则化系数。限制权重大小防止过拟合，1e-4~1e-3 为常用范围。0=不使用")
                # 无监督参数
                with gr.Group(visible=False) as unsup_n_clusters:
                    unsup_n_clusters_val = gr.Slider(2, 10, 2, step=1, label="聚类数 (n_clusters)",
                        info="聚类簇数。对于裂纹检测设为 2（裂纹/非裂纹）；分析其他特征模式时可增大探索")
                with gr.Group(visible=False) as unsup_dbscan:
                    unsup_eps = gr.Slider(0.1, 2.0, 0.5, step=0.1, label="DBSCAN eps",
                        info="邻域半径。越大簇越少、噪声点越少；需根据数据密度调整，无经验时从默认值尝试")
                    unsup_min_samples = gr.Slider(2, 20, 5, step=1, label="DBSCAN min_samples",
                        info="核心点的最小邻域样本数。越大聚类越严格、越多点被标记为噪声")
```

- [ ] **Step 2: Commit Step 2 & 3 info**

```bash
git add src/gradio_app.py && git commit -m "feat: Step2-3组件添加中文info说明"
```

---

### Task 4: Add `info` Parameters — Step 4 (Loss / Optimizer)

**Files:** Modify `src/gradio_app.py:1798-1854`

- [ ] **Step 1: Add info to all Step 4 components**

Replace lines 1798-1854 with info-added versions:

```python
                # DT loss
                with gr.Group(visible=False) as dt_loss:
                    dt_criterion = gr.Dropdown(["gini", "entropy", "log_loss"], value="gini",
                        label="criterion (分裂准则)",
                        info="分裂质量度量。gini 计算快、默认首选；entropy 对不平衡数据略优；log_loss 为对数损失变体")
                # SVM loss
                with gr.Group(visible=False) as svm_loss:
                    svm_kernel = gr.Dropdown(["linear", "rbf", "poly"], value="rbf",
                        label="kernel (核函数)",
                        info="核函数。rbf 适合大多数非线性问题；linear 适合线性可分数据，训练更快且可解释")
                    svm_gamma = gr.Dropdown(["scale", "auto"], value="scale", label="gamma",
                        info="RBF 核的宽度参数。scale=1/(特征数×方差) 自动计算推荐默认；auto=1/特征数")
                # NB (无显式损失)
                with gr.Group(visible=False) as nb_loss_info:
                    gr.Markdown("*生成式模型，无显式损失函数；通过极大似然估计参数。*")
                # RF loss（默认模型，初始可见）
                with gr.Group(visible=True) as rf_loss:
                    rf_criterion = gr.Dropdown(["gini", "entropy", "log_loss"], value="gini",
                        label="criterion (分裂准则)",
                        info="同决策树。gini 为默认常用选择，大多数情况下与 entropy 效果接近")
                # LR loss
                with gr.Group(visible=False) as lr_loss:
                    lr_penalty = gr.Dropdown(["l1", "l2", "elasticnet"], value="l2",
                        label="penalty (正则化)",
                        info="正则化类型。l2 最常用；l1 产生稀疏解（自动特征选择）；elasticnet 混合两者")
                    lr_solver = gr.Dropdown(["lbfgs", "liblinear", "saga"], value="lbfgs",
                        label="solver (优化器)",
                        info="优化算法。lbfgs 适合大多数情况；liblinear 适合小数据集；saga 支持所有正则化类型")
                    lr_l1_ratio = gr.Slider(0.0, 1.0, 0.5, step=0.05,
                                            label="l1_ratio (elasticnet 混合比例)",
                                            visible=False,
                                            info="elasticnet 中 l1 的比例。0=纯 l2（平滑），1=纯 l1（稀疏）。仅在 penalty=elasticnet 时生效")
                # XGBoost loss
                with gr.Group(visible=False) as xgb_loss:
                    xgb_objective = gr.Dropdown(["binary:logistic", "binary:hinge"], value="binary:logistic",
                        label="objective (目标函数)",
                        info="目标函数。binary:logistic 输出概率（支持 ROC/PR）；binary:hinge 仅输出标签，概率曲线不可用")
                    xgb_learning_rate = gr.Slider(0.01, 0.5, 0.1, step=0.01, label="learning_rate (学习率)",
                        info="学习率/步长收缩。越小越稳健但需更多 n_estimators；常用 0.01-0.3")
                    xgb_hinge_warning = gr.Markdown(
                        "⚠️ **注意**：`binary:hinge` 仅输出硬标签 (0/1)，无法产生概率估计，"
                        "ROC-AUC 和 PR 曲线将不可用。",
                        visible=False,
                    )
                # LightGBM loss
                with gr.Group(visible=False) as lgbm_loss:
                    lgbm_objective = gr.Dropdown(["binary", "cross_entropy"], value="binary",
                        label="objective (目标函数)",
                        info="目标函数。binary 和 cross_entropy 均输出概率，效果相近；binary 为常用默认")
                    lgbm_learning_rate = gr.Slider(0.01, 0.5, 0.1, step=0.01, label="learning_rate (学习率)",
                        info="学习率。与 n_estimators 配合：小学习率+大迭代数通常泛化更好但训练更慢")
                # CNN loss
                with gr.Group(visible=False) as cnn_loss:
                    cnn_loss_fn = gr.Dropdown(["cross_entropy", "focal", "label_smoothing", "dice"],
                        value="cross_entropy", label="损失函数",
                        info="损失函数。cross_entropy 最通用；focal 自动关注难分样本；label_smoothing 软化标签防过拟合；dice 适合类别不平衡")
                    cnn_focal_alpha = gr.Dropdown(["None", "0.25", "0.5"], value="None",
                        label="Focal α (None=无类别权重)", visible=False,
                        info="Focal Loss 类别权重。None=无权重（适合平衡数据）；0.25=衰减易分类样本权重；0.5=更强衰减")
                    cnn_focal_gamma = gr.Slider(0.0, 5.0, 2.0, step=0.5, label="Focal γ", visible=False,
                        info="Focal Loss 聚焦参数。γ 越大越聚焦难分样本；0=退化为普通交叉熵；2-3 为论文推荐范围")
                    cnn_label_smoothing_epsilon = gr.Slider(0.0, 0.3, 0.1, step=0.05,
                        label="Label Smoothing ε", visible=False,
                        info="标签平滑强度。将硬标签 0/1 软化为 ε 和 1-ε。0.1=10% 平滑，减小过拟合风险")
                    cnn_optimizer = gr.Dropdown(["adam", "sgd"], value="adam", label="优化器",
                        info="优化器。adam 自适应学习率、收敛快推荐默认；sgd 需手动调学习率但泛化能力可能更好")
                    cnn_learning_rate = gr.Number(0.001, label="学习率", precision=5,
                        info="学习率。adam 通常用 1e-3~1e-4；sgd 通常用 1e-2~1e-3。太小收敛慢，太大不收敛")
                # KMeans loss
                with gr.Group(visible=False) as kmeans_loss:
                    kmeans_algorithm = gr.Dropdown(["lloyd", "elkan"], value="lloyd",
                        label="algorithm (优化算法)",
                        info="优化算法。lloyd 经典 EM 算法通用；elkan 用三角不等式加速，适合大数据集")
                # GMM loss
                with gr.Group(visible=False) as gmm_loss:
                    gmm_covariance_type = gr.Dropdown(["full", "tied", "diag", "spherical"],
                        value="full", label="covariance_type (协方差类型)",
                        info="协方差矩阵类型。full 最灵活但参数多、模型大；diag 对角协方差；spherical 各向同性最快")
                    gr.Markdown("⚠️ *`full` 协方差在高维特征下模型文件约 778MB，加载较慢。*")
                # DBSCAN (无显式损失)
                with gr.Group(visible=False) as dbscan_loss_info:
                    gr.Markdown("*基于密度的聚类，无显式损失函数；通过密度可达性定义簇。*")
                # Agglomerative loss
                with gr.Group(visible=False) as agg_loss:
                    agg_linkage = gr.Dropdown(["ward", "complete", "average", "single"],
                        value="ward", label="linkage (链接准则)",
                        info="簇间距离定义。ward 最小化簇内方差（需欧氏距离）；complete 最大距离；average 平均距离")
                # Spectral loss
                with gr.Group(visible=False) as spec_loss:
                    spec_affinity = gr.Dropdown(["rbf", "nearest_neighbors"], value="rbf",
                        label="affinity (相似度图)",
                        info="相似度图构建方式。rbf 用高斯核适合大多数情况；nearest_neighbors 用 KNN 图适合流形结构")
```

- [ ] **Step 2: Commit Step 4 info**

```bash
git add src/gradio_app.py && git commit -m "feat: Step4组件添加中文info说明"
```

---

### Task 5: Add `info` Parameters — Step 5 (Training Strategy)

**Files:** Modify `src/gradio_app.py:1858-1884`

- [ ] **Step 1: Add info to Step 5 components**

Replace lines 1858-1884 with info-added versions:

```python
                optimization_strategy = gr.Radio(
                    choices=["pretrained", "manual", "grid_search", "random_search"],
                    value="pretrained",
                    label="参数优化策略",
                    info="pretrained=加载预训练模型（最快）；manual=用当前手动参数训练；grid_search=穷举搜索最优组合；random_search=随机采样搜索",
                )
                with gr.Group(visible=False) as opt_search_params:
                    cv_folds_opt = gr.Slider(2, 10, 3, step=1, label="CV 折数",
                        info="交叉验证折数。折数越多评估越稳定但搜索耗时成倍增长；3-5 折为常用平衡范围")
                    n_iter = gr.Slider(10, 100, 30, step=10, label="随机搜索迭代次数 (仅RandomSearch)",
                        info="随机搜索的采样次数。越多越可能找到好参数组合；通常 30-50 次已足够覆盖主要参数空间")

                with gr.Group(visible=True) as supervised_val:
                    validation_method = gr.Radio(
                        choices=["holdout", "kfold"], value="holdout", label="验证方法",
                        info="holdout=单次划分验证（快、有方差）；kfold=K折交叉验证（更稳定可靠但慢K倍）")
                with gr.Group(visible=False) as unsup_val:
                    unsup_val_method = gr.Radio(
                        choices=["internal_external", "internal_only", "external_only"],
                        value="internal_external", label="评估指标范围",
                        info="internal=轮廓系数等内部指标；external=对比真实标签的外部指标；internal_external=两者都显示")

                with gr.Group(visible=False) as scoring_metric_grp:
                    scoring_metric = gr.Dropdown(
                        choices=["f1", "accuracy", "roc_auc", "precision", "recall"],
                        value="f1", label="优化目标指标 (GridSearch/RandomSearch时使用)",
                        info="参数搜索的优化目标。f1 平衡精确率和召回率；accuracy 适合平衡数据；roc_auc 评估排序质量")

                with gr.Group(visible=False) as unsup_opt_info:
                    gr.Markdown("*💡 聚类方法的 GridSearch/RandomSearch 使用 **轮廓系数 (Silhouette)** 作为搜索目标（非监督指标）。*")

                random_seed = gr.Number(42, label="随机种子", precision=0,
                    info="随机种子。固定可复现结果；改变可测试模型对数据划分的稳定性")
```

- [ ] **Step 2: Commit Step 5 info**

```bash
git add src/gradio_app.py && git commit -m "feat: Step5组件添加中文info说明"
```

---

### Task 6: Create the Sound Engine JS/HTML Injection

**Files:** Modify `src/gradio_app.py` (add after `create_interface()` definition, before `return app`)

- [ ] **Step 1: Add base64 encoding utility and sound engine HTML**

First, add a helper function near the top of `create_interface()` or right before the `gr.HTML` injection. The function reads and encodes the mp3 file:

Add this inside `create_interface()`, right after the `theme = gr.themes.Soft(...)` line (line 1702), before `with gr.Blocks(...)`:

```python
    # ---- 音效系统：加载训练完成音效为 base64 ----
    import base64 as _base64
    _mp3_path = Path(__file__).resolve().parent.parent / "sound" / "训练完成音效.mp3"
    _mp3_b64 = ""
    if _mp3_path.exists():
        _mp3_b64 = _base64.b64encode(_mp3_path.read_bytes()).decode()
```

Then, at the end of `create_interface()`, right before `return app` (after the `run_btn.click(...)` block, before `return app`), add the `gr.HTML` component:

```python
        # ---- 音效系统注入 ----
        _sound_html = f"""<div id="sound-engine-root" style="position:fixed;top:8px;right:16px;z-index:9999;display:flex;align-items:center;gap:8px;background:rgba(255,255,255,0.95);padding:4px 12px;border-radius:20px;box-shadow:0 2px 8px rgba(0,0,0,0.15);font-size:13px;font-family:sans-serif;">
<span id="sound-toggle-label">🔊</span>
<label style="cursor:pointer;display:flex;align-items:center;gap:4px;">
<input type="checkbox" id="sound-toggle" checked style="accent-color:#3b82f6;">
<span style="user-select:none;">音效</span>
</label>
</div>
<audio id="complete-audio" preload="auto" style="display:none;"
    src="data:audio/mp3;base64,{_mp3_b64 if _mp3_b64 else ''}"></audio>
<script>
(function() {{
    // SoundEngine: Web Audio API 音效引擎
    const SoundEngine = {{
        _ctx: null,
        _muted: false,
        _hoverThrottle: 0,

        _ensureCtx() {{
            if (!this._ctx) {{
                this._ctx = new (window.AudioContext || window.webkitAudioContext)();
            }}
            if (this._ctx.state === 'suspended') this._ctx.resume();
            return this._ctx;
        }},

        _tone(freq, duration, type, sweepTo) {{
            if (this._muted) return;
            try {{
                const ctx = this._ensureCtx();
                const now = ctx.currentTime;
                const osc = ctx.createOscillator();
                const gain = ctx.createGain();
                osc.type = type || 'sine';
                osc.frequency.setValueAtTime(freq, now);
                if (sweepTo) osc.frequency.linearRampToValueAtTime(sweepTo, now + duration);
                gain.gain.setValueAtTime(0.12, now);
                gain.gain.exponentialRampToValueAtTime(0.001, now + duration);
                osc.connect(gain);
                gain.connect(ctx.destination);
                osc.start(now);
                osc.stop(now + duration);
            }} catch(e) {{ /* 静默忽略音频错误 */ }}
        }},

        playHover() {{
            if (this._muted) return;
            const now = Date.now();
            if (now - this._hoverThrottle < 80) return;
            this._hoverThrottle = now;
            this._tone(800, 0.05, 'square');
        }},

        playClick() {{
            this._tone(1200, 0.10, 'sine');
        }},

        playStart() {{
            this._tone(400, 0.30, 'sine', 800);
        }},

        playComplete() {{
            if (this._muted) return;
            try {{
                const audio = document.getElementById('complete-audio');
                if (audio && audio.src && !audio.src.endsWith('base64,')) {{
                    audio.currentTime = 0;
                    audio.play().catch(() => {{}});
                }}
            }} catch(e) {{}}
        }},

        playError() {{
            this._tone(200, 0.20, 'sine');
            setTimeout(() => this._tone(150, 0.25, 'triangle'), 200);
        }},

        setMuted(m) {{
            this._muted = m;
            document.getElementById('sound-toggle-label').textContent = m ? '🔇' : '🔊';
        }},

        toggle() {{
            this.setMuted(!this._muted);
            try {{ localStorage.setItem('bjtu_ml_sound_muted', this._muted ? '1' : '0'); }} catch(e) {{}}
        }}
    }};

    // 从 localStorage 恢复静音状态
    try {{
        if (localStorage.getItem('bjtu_ml_sound_muted') === '1') {{
            SoundEngine.setMuted(true);
            document.getElementById('sound-toggle').checked = false;
        }}
    }} catch(e) {{}}

    // 静音开关事件
    document.getElementById('sound-toggle').addEventListener('change', function() {{
        SoundEngine.toggle();
    }});

    // 按钮 hover/click 音效：找到 Gradio 主按钮并绑定
    function bindButtonSounds() {{
        const btns = document.querySelectorAll('button');
        btns.forEach(function(btn) {{
            if (btn.textContent.includes('运行训练链路')) {{
                btn.addEventListener('mouseenter', function() {{ SoundEngine.playHover(); }});
                btn.addEventListener('click', function() {{ SoundEngine.playStart(); }});
            }}
        }});
    }}

    // 下拉框 hover 音效
    function bindDropdownSounds() {{
        const dropdowns = document.querySelectorAll('select');
        dropdowns.forEach(function(dd) {{
            dd.addEventListener('mouseenter', function() {{ SoundEngine.playHover(); }});
            dd.addEventListener('focus', function() {{ SoundEngine.playClick(); }});
        }});
    }}

    // 轮询绑定（Gradio 动态渲染，延迟尝试）
    let bindAttempts = 0;
    const bindInterval = setInterval(function() {{
        bindButtonSounds();
        bindDropdownSounds();
        bindAttempts++;
        if (bindAttempts > 20) clearInterval(bindInterval);
    }}, 500);

    // MutationObserver: 监听 status_output 区域检测训练完成/出错
    function setupStatusObserver() {{
        const checkAndTrigger = function() {{
            const statusEls = document.querySelectorAll('[data-testid="markdown"]');
            statusEls.forEach(function(el) {{
                const html = el.innerHTML || '';
                if (html.includes('<!--SOUND:complete-->')) {{
                    el.innerHTML = html.replace('<!--SOUND:complete-->', '');
                    SoundEngine.playComplete();
                }}
                if (html.includes('<!--SOUND:error-->')) {{
                    el.innerHTML = html.replace('<!--SOUND:error-->', '');
                    SoundEngine.playError();
                }}
            }});
        }};

        const observer = new MutationObserver(function() {{
            checkAndTrigger();
        }});

        // 观察整个 body 中 markdown 内容变化
        const targetNode = document.body;
        observer.observe(targetNode, {{ childList: true, subtree: true, characterData: true }});

        // 也定期检查（兜底）
        setInterval(checkAndTrigger, 800);
    }}

    // 页面加载完成后初始化
    if (document.readyState === 'loading') {{
        document.addEventListener('DOMContentLoaded', function() {{
            setupStatusObserver();
        }});
    }} else {{
        setupStatusObserver();
    }}
}})();
</script>"""
        gr.HTML(_sound_html, visible=True)
```

Note: The `gr.HTML` must be placed **inside** the `with gr.Blocks(...)` context (right before `return app`), but **outside** the `with gr.Row():` / sidebar layout, so the mute toggle appears in the top-right corner of the entire page.

- [ ] **Step 2: Commit sound engine**

```bash
git add src/gradio_app.py && git commit -m "feat: 添加音效引擎（Web Audio API + Battlefield1 mp3）"
```

---

### Task 7: Modify `run_pipeline()` to Embed Sound Markers

**Files:** Modify `src/gradio_app.py:1580-1599`

- [ ] **Step 1: Add sound markers to status returns**

In `run_pipeline()`, modify the success return (line 1581) to embed a completion marker, and the error return (line 1598-1599) to embed an error marker:

**Success path (replace line 1581):**

```python
        progress(0.95, desc="生成可视化图表...")
        total_elapsed = time.time() - t_total
        result["status"] = "\n".join(status_msgs) + f"\n\n⏱ 总耗时: {total_elapsed:.1f}s\n" + result["status"]
        # 嵌入音效触发标记
        result["status"] += "\n<!--SOUND:complete-->"

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
```

**Error path (replace lines 1597-1599):**

```python
    except Exception as e:
        import traceback
        err_msg = f"❌ 运行出错: {str(e)}\n\n```\n{traceback.format_exc()}\n```"
        # 嵌入音效触发标记
        err_msg += "\n<!--SOUND:error-->"
        return (err_msg, "", None, None, None, None, None, None, None)
```

- [ ] **Step 2: Commit sound markers**

```bash
git add src/gradio_app.py && git commit -m "feat: run_pipeline添加音效触发标记"
```

---

### Task 8: End-to-End Verification

**Files:** None (manual testing)

- [ ] **Step 1: Start the Gradio app**

```bash
cd c:\Users\xiaol\Desktop\BJTU_Python学习\研究性专题\bjtu_ml_research && python src/gradio_app.py
```

- [ ] **Step 2: Verify in browser at http://127.0.0.1:7860**

Checklist:
1. Each Step has an overview Markdown description (grey quote text under title) ✓
2. Hover over each slider/dropdown shows a grey `info` tooltip below the component ✓
3. Hover over "运行训练链路" button plays a subtle click sound ✓
4. Click "运行训练链路" plays a rising start sound ✓
5. Top-right corner shows 🔊 音效 toggle ✓
6. Toggle off → all sounds muted, icon changes to 🔇 ✓
7. Refresh page → mute state preserved ✓
8. Training completes → Battlefield 1 cash register sound plays ✓
9. Training errors → low buzz error sound plays ✓

- [ ] **Step 3: Kill the server (Ctrl+C)** after verification

- [ ] **Step 4: Commit verification notes**

```bash
git add -A && git commit -m "docs: 交互优化验证完成"
```

---

### Task 9: Final Cleanup & Lint

**Files:** `src/gradio_app.py`

- [ ] **Step 1: Run Ruff linter and formatter**

```bash
cd c:\Users\xiaol\Desktop\BJTU_Python学习\研究性专题\bjtu_ml_research && ruff check src/gradio_app.py && ruff format src/gradio_app.py
```

Expected: Ruff check passes with no errors (or only pre-existing ones), format applies cleanly.

- [ ] **Step 2: Commit if Ruff made changes**

```bash
git add src/gradio_app.py && git commit -m "style: ruff格式化" || echo "No formatting changes"
```
