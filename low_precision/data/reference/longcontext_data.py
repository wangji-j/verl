import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from matplotlib import rcParams

# 读取CSV文件
df = pd.read_csv('csv/context_length_comparison.csv')

# 将OOM转换为NaN以便绘图，这是您原有的处理方式
df['veRL(吞吐)'] = pd.to_numeric(df['veRL(吞吐)'].replace('OOM', np.nan))
df['siiRL(吞吐)'] = pd.to_numeric(df['siiRL(吞吐)'].replace('OOM', np.nan))

# ==============================================================================
# ### 手动设置Y轴刻度 ###
manual_y_ticks = {
    '72b': [50, 100, 150, 200],
    '32b': [150, 350, 550, 750],
    '7b':  [0, 1000, 2000, 3000]
}
# ==============================================================================
# 定义绘图顺序
size_groups_order = ['7b', '32b', '72b']

# 创建图形 - 调整figsize以适应LaTeX文档宽度
fig, axs = plt.subplots(1, len(size_groups_order), figsize=(7.3, 1.5))

# --- 风格设置 ---
rcParams['font.weight'] = 549
plt.rc('font', size=7)
plt.rc('axes', titlesize=7, labelsize=7)
plt.rc('xtick', labelsize=7)
plt.rc('ytick', labelsize=7)
plt.rc('legend', fontsize=7)

# 设置颜色
color4 = (220/255, 20/255, 60/255)    # 红色
color5 = (70/255, 130/255, 180/255)   # 蓝色
verl_color = color4
siirl_color = color5

# 绘图循环
for i, size in enumerate(size_groups_order):
    ax = axs[i]
    data_for_size = df[df['size'] == size]

    ### 您原有的X轴动态确定逻辑 ###
    full_context_order = ['8k', '16k', '32k', '64k']
    valid_data = data_for_size.dropna(subset=['veRL(吞吐)', 'siiRL(吞吐)'], how='all').copy()
    valid_data['上下文'] = pd.Categorical(valid_data['上下文'], categories=full_context_order, ordered=True)
    valid_data.sort_values('上下文', inplace=True)
    local_context_order = valid_data['上下文'].tolist()
    local_x_pos = np.arange(len(local_context_order))
    verl_throughput = valid_data['veRL(吞吐)'].tolist()
    siirl_throughput = valid_data['siiRL(吞吐)'].tolist()

    # --- 绘制条形图 ---
    bar_width = 0.35
    ax.bar([x - bar_width/2 for x in local_x_pos], verl_throughput, bar_width,
           label='verl', color=verl_color, alpha=0.7)
    ax.bar([x + bar_width/2 for x in local_x_pos], siirl_throughput, bar_width,
           label='DistFlow', color=siirl_color, alpha=0.7)

    # ======================== 新增代码块：添加OOM标注 ========================
    # 根据您的要求，“不用加自动逻辑”，我们只针对32b模型的图表进行处理
    if size == '72b':
        # 检查'32k'是否存在于当前子图的x轴上
        if '32k' in local_context_order:
            idx_32k = local_context_order.index('32k')
            x_pos_32k = local_x_pos[idx_32k]
            # 调整y坐标以适应第三张图的刻度范围 (y轴范围是50-200)
            ax.text(x_pos_32k - bar_width/2-0.02, 53, 'OOM',
                    ha='center',
                    va='bottom',
                    color='black',
                    fontsize=5,
                    weight='medium')
    # ======================================================================

    # --- Y轴刻度设置 ---
    ticks_for_this_plot = manual_y_ticks.get(size)
    if ticks_for_this_plot and len(ticks_for_this_plot) == 4:
        ax.set_ylim(ticks_for_this_plot[0], ticks_for_this_plot[-1])
        ax.set_yticks(ticks_for_this_plot)
    else:
        print(f"警告：在 manual_y_ticks 中找不到模型 '{size}' 的设置，或刻度数量不为4。将使用自动刻度。")

    # --- X轴刻度设置 ---
    ax.set_xticks(local_x_pos)
    ax.set_xticklabels(local_context_order)

    # --- 底部标签设置 ---
    ax.set_title('')
    ax.set_xlabel('Context Length', fontweight='medium', fontsize=7, labelpad=2)
    prefix = f"({chr(ord('a') + i)})"
    model_label = f"Model={size[:-1]}B"
    full_model_label = f"{prefix} {model_label}"
    ax.text(x=0.5, y=-0.4, s=full_model_label,
            fontweight='medium',
            fontsize=7,
            ha='center',
            va='top',
            transform=ax.transAxes)

    # --- 其他样式 ---
    ax.grid(True, linestyle=':', alpha=0.6, linewidth=0.8)
    def format_func(value, tick_number):
        if value >= 1000:
            return f"{int(value/1000)}k"
        return str(int(value))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_func))

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(7)

    if i == 0:
        ax.legend(loc='upper right', framealpha=0, fontsize=7)

# 设置共享的Y轴标签
fig.text(0.001, 0.55, 'Throughput (token/s)', va='center', rotation='vertical', fontsize=7, fontweight='medium')

# 调整子图间距和边距
plt.subplots_adjust(wspace=0.25, left=0.08, right=0.98, top=0.85, bottom=0.25)

# 保存为PDF格式
plt.savefig('context_length_comparison_with_oom_label.pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)
print("图片已成功保存为 context_length_comparison_with_oom_label.pdf")
plt.savefig('context_length_comparison_with_oom_label.png', dpi=300, bbox_inches='tight', pad_inches=0.1)


# 显示图片尺寸信息
width_inches, height_inches = fig.get_size_inches()
print("\n" + "="*50)
print("图窗尺寸信息")
print(f"  - 图窗尺寸 (figsize): {width_inches:.2f} 英寸 (宽) x {height_inches:.2f} 英寸 (高)")
print("="*50)