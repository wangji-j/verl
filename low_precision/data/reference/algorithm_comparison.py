

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from matplotlib import rcParams

# 读取CSV文件
df = pd.read_csv('csv/algorithm_comparison.csv')

# 关键修复：显式标记OOM情况并保留这些行
df['is_oom'] = df['verl-mean'] == 'OOM'

# 转换数据类型，但保留OOM标记
df['verl-mean-numeric'] = pd.to_numeric(df['verl-mean'].replace('OOM', '0'))
df['收益'] = pd.to_numeric(df['收益'], errors='coerce')

# 计算总吞吐量，使用数值版本
df['verl总吞吐'] = df['verl-mean-numeric'] * df['nGPU']
df['siirl总吞吐'] = df['sirl-mean'] * df['nGPU']

# 将数据按算法和模型大小分组
algos = df['algo'].unique()
sizes = df['size'].unique()

# 手动设置Y轴刻度
manual_y_ticks = {
    ('ppo', '7b'):  [0, 70000, 140000, 210000],
    ('ppo', '32b'): [0, 20000, 45000, 70000],
    ('ppo', '72b'): [0, 7000, 14000, 21000],
    ('grpo', '7b'):  [0, 120000, 240000, 360000],
    ('grpo', '32b'): [0, 30000, 60000, 90000],
    ('grpo', '72b'): [0, 10000, 20000, 30000],
}

# 设置字体和线条样式
rcParams['font.weight'] = 549
plt.rc('font', size=9)
plt.rc('axes', titlesize=9, labelsize=9)
plt.rc('xtick', labelsize=9)
plt.rc('ytick', labelsize=9)
plt.rc('legend', fontsize=9)

# 设置颜色
color4 = (220/255, 20/255, 60/255) # 红色
color5 = (70/255, 130/255, 180/255) # 蓝色
verl_color = color4
sirl_color = color5

# 为每个算法创建独立的图片
for algo_idx, algo in enumerate(algos):
    # 创建图形
    fig, axs = plt.subplots(1, len(sizes), figsize=(7.3, 1.5))
    
    # 遍历模型大小
    for i, size in enumerate(sizes):
        # 过滤数据
        data = df[(df['algo'] == algo) & (df['size'] == size)]
        
        if not data.empty:
            ax = axs[i]
            data = data.sort_values(by='nGPU')
            x = np.arange(len(data))
            width = 0.35
            
            # 绘制条形图 - 注意使用verl总吞吐，它已经将OOM点设为0
            verl_bars = ax.bar(x - width/2, data['verl总吞吐'], width, label='verl', color=verl_color, alpha=0.7)
            sirl_bars = ax.bar(x + width/2, data['siirl总吞吐'], width, label='DistFlow', color=sirl_color, alpha=0.7)
            
            # # 重要：在OOM数据点上添加标签
            # for j, (idx, row) in enumerate(data.iterrows()):
            #     if row['is_oom']:
            #         ax.annotate('OOM', 
            #                     xy=(x[j] - width/2, 5),  # 稍微高于0点
            #                     xytext=(x[j] - width/2, 1000),  # 向上偏移，根据图表调整
            #                     ha='center', 
            #                     va='center',
            #                     fontsize=6,
            #                     color=verl_color)
            # 重要：在OOM数据点上添加标签
            for j, (idx, row) in enumerate(data.iterrows()):
                if row['is_oom']:
                    ax.annotate('OOM', 
                                xy=(x[j] - width/2, -1),  # 位于柱状图位置的底部
                                xytext=(x[j] - width/2-0.02, 500),  # 向上偏移的文本位置，可根据需要调整
                                ha='center', 
                                va='bottom',  # 改为bottom使文本位于指定点上方
                                fontsize=5,   # 稍微调大字体
                                color='black',  # 将颜色改为黑色
                                weight='medium')  # 加粗文本以提高可读性
            
            # 手动设置Y轴刻度
            ticks_for_this_plot = manual_y_ticks.get((algo, size))
            if ticks_for_this_plot and len(ticks_for_this_plot) == 4:
                ax.set_ylim(ticks_for_this_plot[0], ticks_for_this_plot[-1])
                ax.set_yticks(ticks_for_this_plot)
            else:
                print(f"警告：在 manual_y_ticks 中找不到算法 '{algo}' 和模型 '{size}' 的设置，或刻度数量不为4。将使用自动刻度。")
                y_min = 0
                y_max = max(data['verl总吞吐'].max(), data['sirl总吞吐'].max()) * 1.15
                ax.set_ylim(y_min, y_max)
                tick_interval = y_max / 3
                y_ticks = [0, tick_interval, 2*tick_interval, y_max]
                ax.set_yticks(y_ticks)
            
            ax.set_xticks(x)
            ax.set_xticklabels(data['nGPU'])
            
            ax.set_title('')
            ax.set_xlabel('Number of GPUs', fontweight='medium', fontsize=7, labelpad=2)
            
            prefix = f"({chr(ord('a') + i)})"
            model_label = f"Model={size[:-1]}B"
            full_model_label = f"{prefix} {model_label}"
            
            ax.text(x=0.5, y=-0.4, s=full_model_label,
                    fontweight='medium',
                    fontsize=7,
                    ha='center',
                    va='top',
                    transform=ax.transAxes)
            
            ax.grid(True, linestyle=':', alpha=0.6, linewidth=0.8)
            
            # 格式化y轴标签
            def format_func(value, tick_number):
                if value == 0:
                    return '0'
                elif value >= 1000:
                    return f"{int(value/1000)}k"
                return str(int(value))
            
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_func))

            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(7)
            
            if i == 0:
                handles1, labels1 = ax.get_legend_handles_labels()
                ax.legend(handles1, labels1, loc='upper left', framealpha=0, fontsize=7)
    
    # 设置共享的Y轴标签
    fig.text(0.001, 0.55, 'Throughput (tokens/s)', va='center', rotation='vertical', fontsize=7, fontweight='medium')
    
    # 调整子图间距
    plt.subplots_adjust(wspace=0.25, left=0.08, right=0.98, top=0.85, bottom=0.25)
    
    # 保存图像 - 为每个算法单独保存
    plt.savefig(f'{algo}_performance_comparison.pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f'{algo}_performance_comparison.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    
    # 关闭图形以释放内存
    plt.close(fig)