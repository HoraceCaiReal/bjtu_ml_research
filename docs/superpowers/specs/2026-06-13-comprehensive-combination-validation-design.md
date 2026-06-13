# 全组合验证方案设计

> 日期: 2026-06-13 | 状态: 已确认 → 执行中

## 目标

对 Gradio 可视化系统所有用户可选的方法组合进行批量运行验证，对照 PDF 指导书五环节要求（数据处理/模型选择/损失衡量/参数优化/模型验证），每个环节均覆盖多种对比方案。

## 组合矩阵 (298 runs)

| 模型 | 数据处理(6prep×pretrained) | 损失对比(Step4变体×baseline) | 参数优化(baseline×4策略) | 模型验证(kfold+多评分) | 组合数 |
|------|:--:|:--:|:--:|:--:|:--:|
| DT | 6prep×holdout | criterion×3 | pretrained/manual/grid/random | kfold, grid×3评分 | 27 |
| SVM | 6prep×holdout | kernel×3 | pretrained/manual/grid/random | kfold, grid×3评分 | 25 |
| NB | 6prep×holdout | 无 | pretrained/manual/grid/random | kfold, grid×3评分 | 11 |
| RF | 6prep×holdout | criterion×3 | pretrained/manual/grid/random | kfold, grid×3评分 | 27 |
| LR | 6prep×holdout | penalty×3 | pretrained/manual/grid/random | kfold, grid×3评分 | 27 |
| XGBoost | 6prep×holdout | objective×2 | pretrained/manual/grid/random | kfold, grid×3评分 | 18 |
| LightGBM | 6prep×holdout | objective×2 | pretrained/manual/grid/random | kfold, grid×3评分 | 18 |
| CNN | 6prep×holdout | loss×6+opt×2 | pretrained/manual/grid/random | random×3loss | 48 |
| KMeans | 6prep×内部+外部 | algorithm×2 | pretrained/manual/grid/random | 3种评估范围 | 11 |
| GMM | 6prep×内部+外部 | cov_type×4 | pretrained/manual/grid/random | 3种评估范围 | 29 |
| DBSCAN | 6prep×内部+外部 | 无 | pretrained/manual/grid/random | 3种评估范围 | 11 |
| Agg | 6prep×内部+外部 | linkage×4 | pretrained/manual/grid/random | 3种评估范围 | 29 |
| Spectral | 6prep×内部+外部 | affinity×2 | pretrained/manual/grid/random | 3种评估范围 | 17 |

基线固定参数: holdout 70/30 + 分层 + 全特征 + 2000样本 + seed=42

## 五环节覆盖

| PDF环节 | 对比维度 | 覆盖方案 |
|---------|---------|---------|
| ① 数据处理 | 6种预处理管线 | none/clahe/gaussian/median/clahe+gaussian/clahe+median |
| ② 模型选择 | 13种模型 | 7传统+1CNN+5无监督 |
| ③ 损失衡量 | 各模型Step4变体 | DT×3/SVM×3/RF×3/LR×3/XGB×2/LGBM×2/CNN×12/KM×2/GMM×4/AGG×4/SPEC×2 |
| ④ 参数优化 | 4策略×3评分 | pretrained/manual/grid/random, 评分f1/acc/auc |
| ⑤ 模型验证 | holdout/kfold/3种无监督 | 监督2种/无监督3种评估范围 |

## 输出

- `outputs/batch_verify/{timestamp}/results.csv` — 全量指标汇总
- `outputs/batch_verify/{timestamp}/plots/{组合}/` — 图表
- `reports/组合验证评估报告.md` — 分析报告
