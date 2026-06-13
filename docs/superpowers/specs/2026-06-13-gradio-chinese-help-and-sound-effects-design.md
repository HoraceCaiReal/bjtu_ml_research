# Gradio 可视化系统：中文选项说明 & 交互音效

**日期**：2026-06-13  
**状态**：已批准  
**目标文件**：`src/gradio_app.py`

---

## 概述

为裂纹图像识别系统的 Gradio 可视化界面添加两项 UX 增强：

1. **中文选项说明**：每个 UI 组件添加 `info` 属性（简短提示），每个 Step 区域顶部添加 Markdown 概述说明
2. **交互音效**：按钮 hover/click、训练开始/完成/出错时播放音效，训练完成使用《战地一》收银台音效

---

## 一、中文选项说明

### 1.1 Step 区域概述

在现有 `gr.Markdown("### ...")` 标题下新增说明块，用引用块 `>` 格式。

| Step | 概述说明 |
|------|---------|
| Step 1 | `> 配置数据加载、预处理和特征提取方式。预处理可降噪增强裂纹边缘；特征类型越多信息越丰富，但训练越慢。` |
| Step 2 | `> 选择分类模型。随机森林鲁棒且无需特征归一化，适合快速实验；CNN 需要更多样本和训练时间，但学习能力更强。无监督聚类无需标签即可发现数据模式。` |
| Step 3 | `> 调整模型结构参数。默认值通常可行；增大复杂度（深度/树数）可能提升拟合但增加过拟合风险。` |
| Step 4 | `> 配置损失函数和优化器。不同损失函数影响模型学习偏好；预训练模式下此部分设置不生效。` |
| Step 5 | `> 选择参数优化策略和验证方法。推荐先用 pretrained 快速评估，再用 grid_search 精细调参。` |

### 1.2 组件级 `info` 属性

为每个 Gradio 组件添加 `info` 参数。完整映射表如下：

#### Step 1: 数据处理

| 组件变量 | info |
|----------|------|
| `split_method` | `"划分方法：目前仅支持留出法(holdout)，将数据随机分为训练集和测试集"` |
| `split_ratio` | `"训练集占总数据的比例。0.7=70%训练，越高模型看到的数据越多但测试评估越不稳定"` |
| `use_stratify` | `"保持训练/测试集中正负样本比例一致，避免某类样本分布偏差导致评估失准"` |
| `preprocessing` | `"图像预处理管线。clahe+median 增强裂纹对比度同时去噪，推荐默认；none 跳过预处理"` |
| `features` | `"提取的特征类型。HOG 捕获边缘方向，LBP 描述局部纹理，GLCM 统计纹理共生矩阵，edge_density 量化边缘密度"` |
| `max_samples` | `"使用的样本总数上限。越多越准确但训练越慢；2000 以上结果通常较稳定，内存不足时可降低"` |

#### Step 2: 模型选择

| 组件变量 | info |
|----------|------|
| `model_choice` | `"选择分类/聚类模型。传统方法训练快、可解释；CNN 能力强但需要更多资源；聚类无需标签"` |

#### Step 3: 模型超参数

| 组件变量 | info |
|----------|------|
| `dt_max_depth` | `"树的最大深度。越大越复杂、越容易过拟合。3-15 适合简单问题，>20 需谨慎"` |
| `dt_min_samples_split` | `"内部节点再划分所需最小样本数。越大越防过拟合，2-5 为常用范围"` |
| `svm_C` | `"正则化强度的倒数。越大拟合越强、容易过拟合；越小边界越平滑。建议对数尺度调参（0.1, 1, 10）"` |
| `nb_var_smoothing` | `"方差平滑项，防止零方差导致数值问题。默认 1e-9 通常无需调整"` |
| `rf_n_estimators` | `"决策树数量。越多越稳定但收益递减，100-200 通常足够，更多训练变慢"` |
| `rf_max_depth` | `"单棵树最大深度。限制深度可防过拟合；None=不限制，20 左右为常用上限"` |
| `rf_min_samples_split` | `"内部节点再划分所需最小样本数。增大可防止学习噪声模式"` |
| `lr_C` | `"正则化强度的倒数。C 越大正则化越弱、越容易过拟合。建议对数尺度调参"` |
| `xgb_n_estimators` | `"提升轮数（树的数量）。过多会过拟合，配合 learning_rate 使用；小学习率需更多轮数"` |
| `xgb_max_depth` | `"树的最大深度。XGBoost 通常用 3-8，较浅的树天然防过拟合"` |
| `xgb_subsample` | `"每棵树随机采样的训练数据比例。0.8 是常用值，降低可增加随机性防过拟合"` |
| `lgbm_n_estimators` | `"提升迭代次数。LightGBM 收敛快，100-200 通常足够；观察验证曲线判断是否过拟合"` |
| `lgbm_max_depth` | `"树深度。LightGBM 叶子生长策略下深度通常不大，-1=不限制"` |
| `lgbm_num_leaves` | `"每棵树的叶子数。控制模型复杂度，通常设为 31-63；越大模型越复杂"` |
| `cnn_dropout` | `"随机失活比例。0.3-0.5 常用，训练时随机丢弃神经元防止过拟合。0=不使用 Dropout"` |
| `cnn_batch_size` | `"每批样本数。小批量(32-64)训练快但梯度噪声大；大批量(128-256)梯度更稳定但需更多显存"` |
| `cnn_epochs` | `"最大训练轮数。配合早停使用，设置较大值让早停机制自动选择最佳轮数"` |
| `cnn_early_stopping` | `"验证 loss 连续不下降的轮数后自动停止。越大容忍度越高，可能等到更好模型但也可能过拟合"` |
| `cnn_input_size` | `"输入图像缩放尺寸。越大细节保留越多但训练显著变慢、显存占用翻倍增长。128 为速度与精度平衡"` |
| `cnn_weight_decay` | `"L2 正则化系数。限制权重大小防止过拟合，1e-4~1e-3 为常用范围。0=不使用"` |
| `unsup_n_clusters_val` | `"聚类簇数。对于裂纹检测设为 2（裂纹/非裂纹）；分析其他特征模式时可增大探索"` |
| `unsup_eps` | `"邻域半径。越大簇越少、噪声点越少；需根据数据密度调整，无经验时从默认值尝试"` |
| `unsup_min_samples` | `"核心点的最小邻域样本数。越大聚类越严格、越多点被标记为噪声"` |

#### Step 4: 损失函数 / 优化器

| 组件变量 | info |
|----------|------|
| `dt_criterion` | `"分裂质量度量。gini 计算快、默认首选；entropy 对不平衡数据略优；log_loss 为对数损失变体"` |
| `svm_kernel` | `"核函数。rbf 适合大多数非线性问题；linear 适合线性可分数据，训练更快且可解释"` |
| `svm_gamma` | `"RBF 核的宽度参数。scale=1/(特征数×方差) 自动计算推荐默认；auto=1/特征数"` |
| `rf_criterion` | `"同决策树。gini 为默认常用选择，大多数情况下与 entropy 效果接近"` |
| `lr_penalty` | `"正则化类型。l2 最常用；l1 产生稀疏解（自动特征选择）；elasticnet 混合两者"` |
| `lr_solver` | `"优化算法。lbfgs 适合大多数情况；liblinear 适合小数据集；saga 支持所有正则化类型"` |
| `lr_l1_ratio` | `"elasticnet 中 l1 的比例。0=纯 l2（平滑），1=纯 l1（稀疏）。仅在 penalty=elasticnet 时生效"` |
| `xgb_objective` | `"目标函数。binary:logistic 输出概率（支持 ROC/PR）；binary:hinge 仅输出标签，概率曲线不可用"` |
| `xgb_learning_rate` | `"学习率/步长收缩。越小越稳健但需更多 n_estimators；常用 0.01-0.3，与 n_estimators 配合调整"` |
| `lgbm_objective` | `"目标函数。binary 和 cross_entropy 均输出概率，效果相近；binary 为常用默认"` |
| `lgbm_learning_rate` | `"学习率。与 n_estimators 配合：小学习率+大迭代数通常泛化更好但训练更慢"` |
| `cnn_loss_fn` | `"损失函数。cross_entropy 最通用；focal 自动关注难分样本；label_smoothing 软化标签防过拟合；dice 适合类别不平衡"` |
| `cnn_focal_alpha` | `"Focal Loss 类别权重。None=无权重（适合平衡数据）；0.25=衰减易分类样本权重；0.5=更强衰减"` |
| `cnn_focal_gamma` | `"Focal Loss 聚焦参数。γ 越大越聚焦难分样本；0=退化为普通交叉熵；2-3 为论文推荐范围"` |
| `cnn_label_smoothing_epsilon` | `"标签平滑强度。将硬标签 0/1 软化为 ε/(K-1) 和 1-ε。0.1=10% 平滑，减小过拟合风险"` |
| `cnn_optimizer` | `"优化器。adam 自适应学习率、收敛快推荐默认；sgd 需手动调学习率但泛化能力可能更好"` |
| `cnn_learning_rate` | `"学习率。adam 通常用 1e-3~1e-4；sgd 通常用 1e-2~1e-3。太小收敛慢，太大不收敛"` |
| `kmeans_algorithm` | `"优化算法。lloyd 经典 EM 算法通用；elkan 用三角不等式加速，适合大数据集"` |
| `gmm_covariance_type` | `"协方差矩阵类型。full 最灵活但参数多、模型大；diag 对角协方差；spherical 各向同性最快"` |
| `agg_linkage` | `"簇间距离定义。ward 最小化簇内方差（需欧氏距离）；complete 最大距离；average 平均距离"` |
| `spec_affinity` | `"相似度图构建方式。rbf 用高斯核适合大多数情况；nearest_neighbors 用 KNN 图适合流形结构"` |

#### Step 5: 参数优化 + 验证 + 指标

| 组件变量 | info |
|----------|------|
| `optimization_strategy` | `"pretrained=加载预训练模型（最快）；manual=用当前手动参数训练；grid_search=穷举搜索最优组合；random_search=随机采样搜索"` |
| `cv_folds_opt` | `"交叉验证折数。折数越多评估越稳定但搜索耗时成倍增长；3-5 折为常用平衡范围"` |
| `n_iter` | `"随机搜索的采样次数。越多越可能找到好参数组合；通常 30-50 次已足够覆盖主要参数空间"` |
| `validation_method` | `"holdout=单次划分验证（快、有方差）；kfold=K折交叉验证（更稳定可靠但慢K倍）"` |
| `unsup_val_method` | `"internal=轮廓系数等内部指标；external=对比真实标签的外部指标；internal_external=两者都显示"` |
| `scoring_metric` | `"参数搜索的优化目标。f1 平衡精确率和召回率；accuracy 适合平衡数据；roc_auc 评估排序质量"` |
| `random_seed` | `"随机种子。固定可复现结果；改变可测试模型对数据划分的稳定性"` |

---

## 二、音效系统

### 2.1 音效清单

| 触发时机 | 音效 | 实现 | 时长 |
|----------|------|------|------|
| 主按钮/下拉框 hover | 短促"咔嗒"声 | Web Audio API 方波，800Hz | ~50ms |
| 主按钮 click | 清脆"点击"声 | Web Audio API 正弦波，1200Hz | ~100ms |
| 训练开始（run_btn 点击） | 上升"启动"音 | Web Audio API 频率扫描 400→800Hz | ~300ms |
| 训练完成 | 《战地一》收银台音效 | 播放 `sound/训练完成音效.mp3` | 文件时长 |
| 运行出错 | 低沉"嗡嗡"声 | Web Audio API 正弦波，200Hz | ~200ms |

### 2.2 技术架构

```
gr.HTML (构建时注入)
├── <audio> 标签（训练完成 mp3，base64 内嵌）
├── SoundEngine JS 类
│   ├── init()                   — 创建 AudioContext（首次用户交互时）
│   ├── playTone(freq, dur, type, sweep) — 合成短音
│   ├── playHoverSound()
│   ├── playClickSound()
│   ├── playStartSound()
│   ├── playCompleteSound()      — 播放 <audio> 元素
│   └── playErrorSound()
├── CSS 静音开关样式
├── MutationObserver — 监听 status_output 区域
│   ├── 检测 <!--SOUND:complete--> → playCompleteSound()
│   └── 检测 <!--SOUND:error-->    → playErrorSound()
├── 全局事件绑定
│   ├── mouseenter → hover 音效（主按钮 + 模型下拉框）
│   ├── click      → click 音效（主按钮）
│   └── run_btn.click → playStartSound()
└── 静音开关
    ├── 右上角固定小按钮
    ├── localStorage 持久化状态
    └── 关闭时所有音效静音
```

### 2.3 后端配合

`run_pipeline()` 的 status 返回值中嵌入隐藏标记：

- 成功：状态消息末尾添加 `<!--SOUND:complete-->`
- 失败（except 块）：错误消息末尾添加 `<!--SOUND:error-->`

前端 `MutationObserver` 检测到标记后用 `element.innerHTML.replace()` 清除标记（防止重复触发），然后播放对应音效。

### 2.4 音效文件处理

`sound/训练完成音效.mp3`（~23KB）以 base64 内嵌到 HTML 中：

```python
import base64
with open("sound/训练完成音效.mp3", "rb") as f:
    mp3_b64 = base64.b64encode(f.read()).decode()
```

在注入的 JS 中动态创建 `<audio>` 元素并设置 `src=data:audio/mp3;base64,...`。

### 2.5 边界情况

- **AudioContext 限制**：浏览器要求用户交互后才能创建 AudioContext。在首次 click 事件时懒初始化。
- **mp3 加载失败**：降级到 Web Audio API 合成一段胜利旋律（上升三音 C-E-G）。
- **静音状态**：`localStorage` 读取失败时默认开启音效。
- **MutationObserver 误触发**：检测到标记后立即 `replace()` 清除，防止重复播放。
- **快速连续点击**：对 hover/click 音效做节流（throttle 100ms）。
- **移动端**：`mouseenter` 在触屏设备无效，hover 音效自动跳过。

---

## 三、修改范围

仅修改 `src/gradio_app.py` 一个文件，变更集中在 `create_interface()` 函数（约 1700-2110 行）。

### 具体变更点

1. **添加 Step 概述 Markdown**（5 处，在现有 `gr.Markdown("### Step N: ...")` 后各加一行）
2. **为所有组件添加 `info` 参数**（约 50 处）
3. **注入音效 HTML/JS**（在 `gr.Blocks` 末尾添加一个 `gr.HTML` 组件）
4. **修改 `run_pipeline` 返回值**（在 status 中嵌入 `<!--SOUND:complete-->` 或 `<!--SOUND:error-->`）
5. **给 `run_btn.click` 绑定加 `js` 参数**（播放开始音效）

---

## 四、不涉及

- 不修改任何模型训练逻辑
- 不修改 notebook 文件
- 不修改环境配置
- 不引入新的 Python 依赖（仅用标准库 `base64`）
- 不修改右侧结果面板
