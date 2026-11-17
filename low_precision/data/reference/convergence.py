import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from matplotlib import rcParams

# =============================================================================
# 1. 样式设置 (全局生效)
# =============================================================================
# 设置全局字体和线条样式
rcParams['font.weight'] = 549
plt.rc('font', size=9)
plt.rc('axes', titlesize=9, labelsize=9)
plt.rc('xtick', labelsize=9)
plt.rc('ytick', labelsize=9)
plt.rc('legend', fontsize=9)

# 设置颜色
color4 = (220/255, 20/255, 60/255)  # 红色 (veRL)
color5 = (70/255, 130/255, 180/255) # 蓝色 (sirl/DistFlow)
verl_color = color4
sirl_color = color5

# =============================================================================
# 2. 数据读取和处理
# =============================================================================
# 读取两个CSV文件
try:
    df_reward = pd.read_csv('csv/benchmark_verl_siirl_reward_mean.csv')
    df_entropy = pd.read_csv('csv/bench_mark_verl_siirl_entropy.csv') # 确保路径正确
except FileNotFoundError as e:
    print(f"错误：CSV文件未找到。请检查路径是否正确。错误信息: {e}")
    print("将使用虚拟数据进行演示。")
    steps = np.arange(0, 1001, 10)
    df_reward = pd.DataFrame({'step': steps, 'verl': 100 * np.log1p(steps / 100), 'siirl': 120 * np.log1p(steps / 120)})
    df_entropy = pd.DataFrame({'step': steps, 'verl': 1.5 - np.log1p(steps / 500), 'siirl': 1.4 - np.log1p(steps / 600)})

# =============================================================================
# 3. 绘图 (创建包含两个子图的Figure)
# =============================================================================
# **改动**: sharex=False，使两个子图的X轴独立。
# 稍微增加图形总高度以容纳新的X轴。
fig, axs = plt.subplots(2, 1, figsize=(3, 3), sharex=False)

# --- 绘制子图 (a): Reward Curve ---
axs[0].plot(df_reward['step'], df_reward['verl'], label='verl', color=verl_color, linewidth=1, alpha=0.9)
axs[0].plot(df_reward['step'], df_reward['siirl'], label='DistFlow', color=sirl_color, linewidth=1, alpha=0.8)

# --- 绘制子图 (b): Entropy Curve ---
axs[1].plot(df_entropy['step'], df_entropy['verl'], label='verl', color=verl_color, linewidth=1, alpha=0.9)
axs[1].plot(df_entropy['step'], df_entropy['siirl'], label='DistFlow', color=sirl_color, linewidth=1, alpha=0.8)

# =============================================================================
# 4. 图表格式化 (应用模板风格)
# =============================================================================

# --- 布局控制参数 (可在此处轻松调整) ---
# 控制两个子图之间的垂直间距 (因为新增了X轴，需要更大间距)
vertical_gap_between_plots = 0.25

# 控制(a), (b)标签距离对应X轴的垂直距离 (需要更大偏移以避开刻度)
subplot_label_y_offset = 0.2

# 为确保(a)和(b)都能显示，调整整个图表的底部边距
bottom_margin = 0.05
# --- 调整结束 ---

# 定义坐标轴刻度格式化函数
def format_func(value, tick_number):
    if value >= 1000:
        return f"{int(value/1000)}k"
    return str(int(value))

# --- 格式化子图 (a) ---
ax_reward = axs[0]
ax_reward.set_ylabel('Reward', fontweight='medium', fontsize=7)
# **改动**: 为图(a)添加自己的X轴标签
# ax_reward.set_xlabel('Step', fontweight='medium', fontsize=7, labelpad=3)
ax_reward.grid(True, linestyle=':', alpha=0.6, linewidth=0.8)
ax_reward.legend(loc='lower right', framealpha=0, fontsize=7)
# **改动**: 为图(a)的X轴和Y轴设置刻度格式
ax_reward.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))
for label in ax_reward.get_xticklabels() + ax_reward.get_yticklabels():
    label.set_fontsize(7)
# # 将子图标签 (a) 放置在图表正下方
# ax_reward.text(0.5, subplot_label_y_offset, '(a) Reward Curve', transform=ax_reward.transAxes,
#                ha='center', va='top', fontsize=7, fontweight='medium')

# --- 格式化子图 (b) ---
ax_entropy = axs[1]
ax_entropy.set_xlabel('Step', fontweight='medium', fontsize=7, labelpad=2)
ax_entropy.set_ylabel('Entropy', fontweight='medium', fontsize=7)
ax_entropy.grid(True, linestyle=':', alpha=0.6, linewidth=0.8)
for label in ax_entropy.get_xticklabels() + ax_entropy.get_yticklabels():
    label.set_fontsize(7)
# **改动**: 为图(b)的X轴设置刻度格式
ax_entropy.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))
# 将子图标签 (b) 放置在图表正下方
# ax_entropy.text(0.5, subplot_label_y_offset, '(b) Entropy Curve', transform=ax_entropy.transAxes,
#                 ha='center', va='top', fontsize=7, fontweight='medium')


# =============================================================================
# 5. 最终调整和保存
# =============================================================================
# 使用控制参数来调整布局
plt.subplots_adjust(hspace=vertical_gap_between_plots, bottom=bottom_margin, top=0.75)

fig.align_ylabels(axs)

# 保存为 PDF 和 PNG 格式
plt.savefig('reward_and_entropy_comparison_final.pdf', dpi=300, bbox_inches='tight')
plt.savefig('reward_and_entropy_comparison_final.png', dpi=300, bbox_inches='tight')

# 关闭图形以释放内存
plt.close(fig)

print("已根据您的最新要求完成图表生成，图(a)和图(b)均有独立的X轴。")