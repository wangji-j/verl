# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# import matplotlib.ticker as ticker
# from matplotlib import rcParams

# # 读取CSV文件
# df = pd.read_csv('batch_size_comparison.csv')

# # 计算总吞吐量
# df['verl总吞吐'] = df['verl平均吞吐'] * df['卡数']
# df['siiRL总吞吐'] = df['siiRL平均吞吐'] * df['卡数']

# # 将数据按模型大小分组
# size_groups = df['size'].unique()

# # ==============================================================================
# # ### 手动设置Y轴刻度 ###
# manual_y_ticks = {
#     '7b':  [0, 100000, 200000, 300000],
#     '32b': [0, 30000, 60000, 90000],
#     '72b': [0, 20000, 40000, 60000]
# }
# # ==============================================================================

# # 创建图形
# fig, axs = plt.subplots(1, len(size_groups), figsize=(10, 3.2))

# # 设置字体和线条样式
# rcParams['font.weight'] = 549
# plt.rc('font', size=9)
# plt.rc('axes', titlesize=9, labelsize=9)
# plt.rc('xtick', labelsize=9)
# plt.rc('ytick', labelsize=9)
# plt.rc('legend', fontsize=9)

# # 设置颜色
# color4 = (220/255, 20/255, 60/255)    # 红色
# color5 = (70/255, 130/255, 180/255)   # 蓝色
# verl_color = color4
# siirl_color = color5

# # 遍历模型大小分组
# for i, size in enumerate(size_groups):
#     # 过滤数据
#     data = df[df['size'] == size]
    
#     if not data.empty:
#         ax = axs[i]
        
#         data = data.sort_values(by='卡数')
        
#         x = np.arange(len(data))
#         width = 0.35
        
#         verl_bars = ax.bar(x - width/2, data['verl总吞吐'], width, label='veRL', color=verl_color, alpha=0.7)
#         siirl_bars = ax.bar(x + width/2, data['siiRL总吞吐'], width, label='siiRL', color=siirl_color, alpha=0.7)
        
#         # --- 手动设置Y轴刻度 ---
#         ticks_for_this_plot = manual_y_ticks.get(size)
#         if ticks_for_this_plot and len(ticks_for_this_plot) == 4:
#             ax.set_ylim(ticks_for_this_plot[0], ticks_for_this_plot[-1])
#             ax.set_yticks(ticks_for_this_plot)
#         else:
#             print(f"警告：在 manual_y_ticks 中找不到模型 '{size}' 的设置，或刻度数量不为4。将使用自动刻度。")

        
#         ax.set_xticks(x)
#         ax.set_xticklabels(data['卡数'])
        
#         ### 标签改动开始 ###
#         # 1. 移除顶部的标题
#         ax.set_title('')  
        
#         # 2. 将 "Number of GPUs" 设置为X轴的主标签
#         ax.set_xlabel('# of GPUs', fontweight='medium', fontsize=10, labelpad=5)
        
#         # 3. 使用 ax.text() 单独添加第二行标签，以便精确控制其垂直位置
#         prefix = f"({chr(ord('a') + i)})"
#         model_label = f"Model={size[:-1]}B"
#         full_model_label = f"{prefix} {model_label}"
        
#         # 使用 ax.text 在指定坐标处添加文本
#         ax.text(x=0.5, y=-0.4, s=full_model_label,
#                 fontweight='medium',
#                 fontsize=9,
#                 ha='center', # 水平居中对齐
#                 va='top',    # 垂直顶部对齐
#                 transform=ax.transAxes)
#         ### 标签改动结束 ###
        
#         ax.grid(True, linestyle=':', alpha=0.6, linewidth=0.8)
        
#         # 格式化y轴标签
#         def format_func(value, tick_number):
#             if value >= 1000:
#                 return f"{int(value/1000)}k"
#             return str(int(value))
        
#         ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_func))
        
#         for label in ax.get_xticklabels() + ax.get_yticklabels():
#             label.set_fontsize(9)
        
#         if i == 0:
#             handles1, labels1 = ax.get_legend_handles_labels()
#             ax.legend(handles1, labels1, loc='upper left', framealpha=0)

# # 设置共享的Y轴标签
# fig.text(-0.01, 0.6, 'Throughput (tokens/s)', va='center', rotation='vertical', fontsize=9, fontweight='medium')

# # 自动调整布局，使其适应所有元素（包括我们手动添加的text）
# plt.tight_layout()

# # 保存图像
# plt.savefig('batch_size_total_throughput_final.png', dpi=400, bbox_inches='tight')

# ### 新增代码开始：输出图片尺寸 ###

# # 1. 从 figure 对象中获取 figsize (英寸)
# width_inches, height_inches = fig.get_size_inches()

# # 2. 获取你在 savefig 中设置的 dpi
# dpi = 400

# # 3. 计算最终的像素尺寸 (width = inches * dpi)
# width_pixels = int(width_inches * dpi)
# height_pixels = int(height_inches * dpi)

# # 4. 打印结果
# print("="*50)
# print("图片尺寸信息")
# print(f"  - 图窗尺寸 (figsize): {width_inches} 英寸 (宽) x {height_inches} 英寸 (高)")
# print(f"  - 分辨率 (DPI): {dpi} 点/英寸")
# print(f"  - 计算出的像素尺寸: {width_pixels} 像素 (宽) x {height_pixels} 像素 (高)")
# print(f"  - 注意: 由于保存时使用了 bbox_inches='tight'，\n    最终文件的实际像素尺寸可能会有微小的差异。")
# print("="*50)

# ### 新增代码结束 ###


# # # 显示图像
# # plt.show()

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from matplotlib import rcParams

# 读取CSV文件
df = pd.read_csv('csv/batch_size_comparison.csv')

# 计算总吞吐量
df['verl总吞吐'] = df['verl平均吞吐'] * df['卡数']
df['siiRL总吞吐'] = df['siiRL平均吞吐'] * df['卡数']

# 将数据按模型大小分组
size_groups = df['size'].unique()

# ==============================================================================
# ### 手动设置Y轴刻度 ###
manual_y_ticks = {
    '7b':  [0, 100000, 200000, 300000],
    '32b': [0, 30000, 60000, 90000],
    '72b': [0, 20000, 40000, 60000]
}
# ==============================================================================

# 创建图形 - 调整figsize以适应LaTeX文档宽度
fig, axs = plt.subplots(1, len(size_groups), figsize=(6.5, 1.5))

# 设置字体和线条样式
rcParams['font.weight'] = 549
plt.rc('font', size=9)
plt.rc('axes', titlesize=9, labelsize=9)
plt.rc('xtick', labelsize=9)
plt.rc('ytick', labelsize=9)
plt.rc('legend', fontsize=9)

# 设置颜色
color4 = (220/255, 20/255, 60/255)    # 红色
color5 = (70/255, 130/255, 180/255)   # 蓝色
verl_color = color4
siirl_color = color5

# 遍历模型大小分组
for i, size in enumerate(size_groups):
    # 过滤数据
    data = df[df['size'] == size]
    
    if not data.empty:
        ax = axs[i]
        
        data = data.sort_values(by='卡数')
        
        x = np.arange(len(data))
        width = 0.35
        
        verl_bars = ax.bar(x - width/2, data['verl总吞吐'], width, label='verl', color=verl_color, alpha=0.7)
        siirl_bars = ax.bar(x + width/2, data['siiRL总吞吐'], width, label='DistFlow', color=siirl_color, alpha=0.7)
        
        # --- 手动设置Y轴刻度 ---
        ticks_for_this_plot = manual_y_ticks.get(size)
        if ticks_for_this_plot and len(ticks_for_this_plot) == 4:
            ax.set_ylim(ticks_for_this_plot[0], ticks_for_this_plot[-1])
            ax.set_yticks(ticks_for_this_plot)
        else:
            print(f"警告：在 manual_y_ticks 中找不到模型 '{size}' 的设置，或刻度数量不为4。将使用自动刻度。")
        
        ax.set_xticks(x)
        ax.set_xticklabels(data['卡数'])
        
        ### 标签改动开始 ###
        # 1. 移除顶部的标题
        ax.set_title('')  
        
        # 2. 将 "# of GPUs" 设置为X轴的主标签 - 调整字体大小和间距
        ax.set_xlabel('Number of GPUs', fontweight='medium', fontsize=7, labelpad=2)
        
        # 3. 使用 ax.text() 单独添加第二行标签，调整位置和字体大小
        prefix = f"({chr(ord('a') + i)})"
        model_label = f"Model={size[:-1]}B"
        full_model_label = f"{prefix} {model_label}"
        
        # 使用 ax.text 在指定坐标处添加文本
        ax.text(x=0.5, y=-0.4, s=full_model_label,
                fontweight='medium',
                fontsize=7,
                ha='center', # 水平居中对齐
                va='top',    # 垂直顶部对齐
                transform=ax.transAxes)
        ### 标签改动结束 ###
        
        ax.grid(True, linestyle=':', alpha=0.6, linewidth=0.8)
        
        # 格式化y轴标签
        def format_func(value, tick_number):
            if value >= 1000:
                return f"{int(value/1000)}k"
            return str(int(value))
        
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_func))
        
        # 调整刻度标签字体大小
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(7)
        
        if i == 0:
            handles1, labels1 = ax.get_legend_handles_labels()
            ax.legend(handles1, labels1, loc='upper left', framealpha=0, fontsize=7)

# 设置共享的Y轴标签
fig.text(0.001, 0.55, 'Throughput (tokens/s)', va='center', rotation='vertical', fontsize=7, fontweight='medium')

# 调整子图间距和边距
plt.subplots_adjust(wspace=0.25, left=0.08, right=0.98, top=0.85, bottom=0.25)

# 保存图像 - 降低DPI以减小文件大小
plt.savefig('batch_size_total_throughput_final.pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.savefig('batch_size_total_throughput_final.png', dpi=300, bbox_inches='tight', pad_inches=0.1)

### 新增代码开始：输出图片尺寸 ###
# 1. 从 figure 对象中获取 figsize (英寸)
width_inches, height_inches = fig.get_size_inches()
# 2. 获取你在 savefig 中设置的 dpi
dpi = 300
# 3. 计算最终的像素尺寸 (width = inches * dpi)
width_pixels = int(width_inches * dpi)
height_pixels = int(height_inches * dpi)
# 4. 打印结果
print("="*50)
print("图片尺寸信息")
print(f"  - 图窗尺寸 (figsize): {width_inches} 英寸 (宽) x {height_inches} 英寸 (高)")
print(f"  - 分辨率 (DPI): {dpi} 点/英寸")
print(f"  - 计算出的像素尺寸: {width_pixels} 像素 (宽) x {height_pixels} 像素 (高)")
print(f"  - LaTeX建议: 使用 \\includegraphics[width=\\textwidth]{{batch_size_total_throughput_final.png}}")
print(f"  - 注意: 由于保存时使用了 bbox_inches='tight'，\n    最终文件的实际像素尺寸可能会有微小的差异。")
print("="*50)
### 新增代码结束 ###

# # 显示图像
# plt.show()