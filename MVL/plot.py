# 导入必要的库
import matplotlib.pyplot as plt
import numpy as np


def plot(log:dict=None):
    plt.rcParams['font.size'] = 7
    fig_size = (5.4, 2.1)  # 稍微增加高度以容纳图例
    y_label = r'$V(7)$' 
    settings = list(log.keys())
    curve_labels = list(log[settings[0]].keys())

    # 创建图形和子图，减少左右边距
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=fig_size, 
                            gridspec_kw={'width_ratios': [1, 1]})  # 确保两个子图宽度相等
    ax_left, ax_right = axes

    for curve_label in curve_labels:
        ax_left.plot(log[settings[0]][curve_label], label=curve_label, alpha=0.8, linewidth=1)
        ax_right.plot(log[settings[1]][curve_label], label=curve_label, alpha=0.8, linewidth=1)

    ax_left.set_title('(a)')
    ax_left.set_ylim(-1, 30)
    ax_right.set_title('(b)')
    ax_right.set_ylim(-1, 30)

    ax_left.set_ylabel(y_label)

    # # --- 添加网格线 (可选) ---
    # ax_left.grid(True, linestyle='--', alpha=0.6)
    # ax_right.grid(True, linestyle='--', alpha=0.6)

    # 获取图例句柄和标签
    handles, labels = ax_left.get_legend_handles_labels()
    
    # 将图例放在图形底部中央
    fig.legend(handles, labels, 
               loc='lower center', 
               bbox_to_anchor=(0.5, 0),  # 将图例定位在底部中央
               ncol=len(curve_labels),
               frameon=False)

    # 调整子图间距和边距
    # plt.subplots_adjust(
    #     left=0.1,      # 减小左边距
    #     right=0.95,    # 减小右边距
    #     bottom=0.18,   # 底部边距为图例留出空间
    #     top=0.9,      # 顶部边距
    #     wspace=0.2     # 减小子图之间的水平间距
    # )
    plt.subplots_adjust(
        left=0.1,      # 减小左边距
        right=0.95,    # 减小右边距
        bottom=0.2,   # 底部边距为图例留出空间
        top=0.9,      # 顶部边距
        wspace=0.2     # 减小子图之间的水平间距
    )


    plt.show()

    # 保存图像
    fig.savefig('MVL-EXP.png', dpi=300, bbox_inches='tight')
    fig.savefig('MVL-EXP.pdf', bbox_inches='tight')
