"""修复所有 Notebook 中的非中文文本"""
import nbformat as nbf
from pathlib import Path

# 使用项目相对路径，避免硬编码本地用户目录
NB_DIR = Path(__file__).resolve().parent.parent / "notebooks"

def fix_notebook(filename, replacements):
    """在 notebook 的源代码中执行字符串替换。"""
    path = NB_DIR / filename
    nb = nbf.read(str(path), as_version=4)
    changed = 0
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        old_src = cell.source
        new_src = old_src
        for old, new in replacements:
            if old in new_src:
                new_src = new_src.replace(old, new)
                changed += 1
        if new_src != old_src:
            cell.source = new_src
    nbf.write(nb, str(path))
    print(f"  {filename}: {changed} 处修复")

# ============ 01 ============
fix_notebook("01_数据处理与特征工程.ipynb", [
    ("categories = ['Negative (无裂缝)', 'Positive (有裂缝)']",
     "categories = ['无裂缝', '有裂缝']"),
    ("axes[0, 0].set_ylabel('Positive (有裂缝)',",
     "axes[0, 0].set_ylabel('有裂缝',"),
    ("axes[1, 0].set_ylabel('Negative (无裂缝)',",
     "axes[1, 0].set_ylabel('无裂缝',"),
])

# ============ 02 ============
fix_notebook("02_传统监督学习对比.ipynb", [
    ("ax.set_xlabel('max_depth', fontsize=12)",
     "ax.set_xlabel('最大深度 (max_depth)', fontsize=12)"),
    ("ax.set_title('SVM：kernel 和 C 对性能的影响'",
     "ax.set_title('SVM：核函数和正则化参数对性能的影响'"),
    ("ax.set_xlabel('var_smoothing (对数尺度)'",
     "ax.set_xlabel('方差平滑参数 (对数尺度)'"),
])

# ============ 03 ============
fix_notebook("03_深度学习对比.ipynb", [
    # print statements
    ('print("损失函数定义完成: CrossEntropy, FocalLoss, LabelSmoothingCE, DiceLoss")',
     'print("损失函数定义完成: 交叉熵损失(CE), 焦点损失(Focal), 标签平滑损失(LabelSmooth), Dice损失(Dice)")'),
    # loss curves - axis labels
    ('axes[0].set_xlabel("Epoch")',
     'axes[0].set_xlabel("训练轮数")'),
    ('axes[0].set_ylabel("验证 Loss")',
     'axes[0].set_ylabel("验证损失")'),
    ('axes[0].set_title("不同损失函数的验证 Loss 曲线"',
     'axes[0].set_title("不同损失函数的验证损失曲线"'),
    ('axes[1].set_xlabel("Epoch")',
     'axes[1].set_xlabel("训练轮数")'),
    # grid search heatmap
    ('axes[ax_idx].set_xlabel("dropout")',
     'axes[ax_idx].set_xlabel("Dropout比例")'),
    ('axes[ax_idx].set_ylabel("learning_rate")',
     'axes[ax_idx].set_ylabel("学习率")'),
    # full training curves
    ("label='训练Loss'",
     "label='训练损失'"),
    ("label='验证Loss'",
     "label='验证损失'"),
    ("label='训练Acc'",
     "label='训练准确率'"),
    ("label='验证Acc'",
     "label='验证准确率'"),
    ("axes[0].set_xlabel('Epoch')",
     "axes[0].set_xlabel('训练轮数')"),
    ("axes[0].set_ylabel('Loss')",
     "axes[0].set_ylabel('损失')"),
    ("axes[0].set_title('Loss曲线'",
     "axes[0].set_title('损失曲线'"),
    ("axes[1].set_xlabel('Epoch')",
     "axes[1].set_xlabel('训练轮数')"),
    ("ax_roc.set_xlabel('FPR')",
     "ax_roc.set_xlabel('假阳性率 (FPR)')"),
    ("ax_roc.set_ylabel('TPR')",
     "ax_roc.set_ylabel('真阳性率 (TPR)')"),
    ("ax_roc.set_title('CNN ROC曲线'",
     "ax_roc.set_title('CNN 受试者工作特征曲线'"),
    # best epoch label
    ("label=f'最佳Epoch({best_epoch})'",
     "label=f'最佳轮次({best_epoch})'"),
])

# ============ 04 ============
fix_notebook("04_无监督学习对比.ipynb", [
    ("ax1.set_ylabel('Inertia (簇内平方距离)')",
     "ax1.set_ylabel('簇内平方距离 (Inertia)')"),
    ("ax1.set_title('K-Means 肘部法则'",
     "ax1.set_title('K均值 肘部法则'"),
    ("ax2.set_title('K-Means 轮廓系数 vs K'",
     "ax2.set_title('K均值 轮廓系数与K值关系'"),
])

print("\n所有笔记本修复完成！")
