# 系统改进设计：可视化系统四项体验优化

> 日期: 2026-06-13 | 来源: 组合验证评估报告 §9.3

## 改进清单

| # | 改进 | 方案 |
|:-:|------|------|
| 1 | pretrained 下 loss 参数禁用 | 灰色+提示文字 |
| 2 | 无监督质量警告 | Sil<0.1 且 ARI<0.15 时警告 |
| 3 | binary:hinge N/A 占位 | 新增 `_plot_not_available()` |
| 4 | KFold 摘要加标准差 | kfold 时摘要指标改为 mean±std |

## 改进1：Pretrained 模式下 loss 参数禁用

**文件**: `src/gradio_app.py`

**设计**:
- 扩展 `_on_opt_change()` 回调，增加 loss 参数组的 `interactive` 控制
- 当 `optimization_strategy == "pretrained"` 时，传统模型的 loss 组件设为 `interactive=False`（灰色不可编辑）
- 同时显示 `pretrained_loss_hint` 的 Markdown 提示条
- 新增提示文字："💡 预训练模式下使用模型内置参数，loss/核函数/目标函数设置不生效。切换为 manual/grid_search/random_search 模式可自定义。"

**影响组件**: `dt_criterion`, `svm_kernel`, `svm_gamma`, `rf_criterion`, `lr_penalty`, `lr_solver`, `xgb_objective`, `lgbm_objective`

## 改进2：无监督聚类质量警告

**文件**: `src/gradio_app.py` — `_run_unsupervised()` 函数

**设计**:
- 在 `_run_unsupervised()` return 之前检查聚类质量
- 条件：`sil < 0.1 AND ari < 0.15`（两者同时满足才警告）
- 追加 `status_msgs`："⚠️ 聚类质量极低（Silhouette={sil:.4f}, ARI={ari:.4f}），无监督方法可能不适合此任务。建议尝试监督学习方法。"

## 改进3：binary:hinge N/A 图表占位

**文件**: `src/gradio_app.py` — 新增 `_plot_not_available()` 函数

**设计**:
- 新增工具函数，生成居中显示 "N/A — 当前设置下不可用" 的空白图
- 在 `_run_traditional()` 中，当 `y_prob.max() == 0` 时，`roc_fig`/`pr_fig`/`prob_fig` 返回 `_plot_not_available()` 而非 `None`

## 改进4：KFold 摘要指标加标准差

**文件**: `src/gradio_app.py` — `_run_traditional()` 函数

**设计**:
- 在 kfold 模式下，已计算出每折均值±标准差
- 在 metrics_md 摘要表中，将 F1/Accuracy/Precision/Recall/ROC-AUC 的值格式从 `0.9632` 改为 `0.9632±0.0123`
- 保持非 kfold 模式的行为不变
