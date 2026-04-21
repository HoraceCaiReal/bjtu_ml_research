"""
Matplotlib 字体配置模块

用法：在每个 Notebook 的最开头运行
    from src.plot_config import set_chinese_font
    set_chinese_font()

即可让 matplotlib 画图使用微软雅黑（或备选字体）。
"""

import matplotlib.pyplot as plt


def set_chinese_font():
    """
    设置 matplotlib 中文字体为微软雅黑。
    若微软雅黑不可用，依次尝试 WenQuanYi Micro Hei、SimHei。
    """
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "WenQuanYi Micro Hei",
        "SimHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示为方块的问题
