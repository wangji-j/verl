# # import matplotlib.pyplot as plt
# # import numpy as np
# # import matplotlib.ticker as ticker
# # from matplotlib import rcParams
# # import pandas as pd

# # # 从CSV文件导入数据
# # df = pd.read_csv('scaling_data.csv')

# # # 按模型分组数据
# # df_7b = df[df['Model'] == '7B']
# # df_32b = df[df['Model'] == '32B']
# # df_72b = df[df['Model'] == '72B']

# # # 提取每个模型的x和y数据
# # x_7b = df_7b['GPUs'].tolist()
# # y_7b = df_7b['Throughput'].tolist()

# # x_32b = df_32b['GPUs'].tolist()
# # y_32b = df_32b['Throughput'].tolist()

# # x_72b = df_72b['GPUs'].tolist()
# # y_72b = df_72b['Throughput'].tolist()

# # # 创建图形 - 只有一行三列，调整为更扁的比例
# # fig, axs = plt.subplots(1, 3, figsize=(12, 3))  # 从(15, 5)调整为(15, 4)
# # plt.subplots_adjust(wspace=0.3)

# # # 设置字体大小 - 使用您提供的参数
# # SMALL_SIZE = 12
# # MEDIUM_SIZE = 14
# # BIGGER_SIZE = 16

# # # 轻微加粗所有文字
# # rcParams['font.weight'] = 549  # 添加轻微加粗

# # # 使用您的字体大小设置
# # plt.rc('font', size=BIGGER_SIZE)
# # plt.rc('axes', titlesize=BIGGER_SIZE, labelsize=MEDIUM_SIZE)
# # plt.rc('xtick', labelsize=40)  # 保持您的大刻度标签大小
# # plt.rc('ytick', labelsize=40)  # 保持您的大刻度标签大小
# # plt.rc('legend', fontsize=MEDIUM_SIZE)
# # plt.rc('figure', titlesize=MEDIUM_SIZE)

# # # 绘制子图函数
# # def plot_subplot(ax, x, y_real, model_name, show_legend=False):
# #     # 创建理想线性扩展数据
# #     y_linear = [y_real[0] * (x_val / x[0]) for x_val in x]
    
# #     # 绘制实际数据
# #     ax.plot(x, y_real, 'o-', color='#1f77b4', label='siiRL', linewidth=2.5, markersize=7)
    
# #     # 绘制理想线性扩展
# #     ax.plot(x, y_linear, 'k:', label='linear', linewidth=2.5)
    
# #     # 设置标题格式
# #     ax.set_title(f"Model={model_name}")
# #     ax.set_xscale('log', base=2)
# #     ax.set_yscale('log', base=2)
    
# #     # 设置网格线
# #     ax.grid(True, linestyle=':', alpha=0.6, linewidth=0.8)
    
# #     # 设置x轴标签为实际值
# #     ax.set_xticks(x)
# #     ax.set_xticklabels([str(val) for val in x])
    
# #     # 格式化y轴标签
# #     def format_func(value, tick_number):
# #         if value >= 1000:
# #             return f"{int(value/1000)}k"
# #         return str(int(value))
    
# #     ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_func))
    
# #     # 增加y轴刻度密度 - 使用更多的刻度点
# #     y_min, y_max = ax.get_ylim()
# #     # 创建更多的刻度点
# #     ticks = np.logspace(np.log10(y_min), np.log10(y_max), 6)  # 6个刻度点
# #     ax.set_yticks(ticks)
    
# #     # 在plot_subplot函数中添加
# #     ax.tick_params(axis='both', which='major', labelsize=12)

# #     # 手动设置刻度标签的粗细
# #     for label in ax.get_xticklabels() + ax.get_yticklabels():
# #         label.set_fontweight(549)  # 使用semibold加粗

# #     # 只在指定的子图添加图例
# #     if show_legend:
# #         legend = ax.legend(loc='upper left', frameon=True, fancybox=True)
    
# #     return ax

# # # 绘制所有子图
# # plot_subplot(axs[0], x_7b, y_7b, "7B", show_legend=True)
# # plot_subplot(axs[1], x_32b, y_32b, "32B")
# # plot_subplot(axs[2], x_72b, y_72b, "72B")

# # # 添加共享的x轴和y轴标签 - 让它们更靠近图表
# # fig.text(0.5, 0.02, 'Number of GPUs', ha='center', fontsize=17)  # y从0.01改为0.02，更靠近图表
# # fig.text(0.015, 0.5, 'Throughput (token/s)', va='center', rotation='vertical', fontsize=17)  # x从0.02改为0.015，更靠近图表

# # # 调整布局 - 减小边距使坐标轴标题更靠近图表
# # plt.tight_layout(rect=[0.03, 0.06, 1, 0.95])  # 调整左边距和下边距

# # # 保存图像
# # plt.savefig('scaling_trend00.png', dpi=400, bbox_inches='tight')

# # # 显示图像
# # # plt.show()


# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib.ticker as ticker
# from matplotlib import rcParams
# import pandas as pd
# import io


# df = pd.read_csv('scaling_data.csv')
# # ==============================================================================


# # --- 1. 风格设置 (完全采纳“学习代码”的风格) ---
# rcParams['font.weight'] = 549
# plt.rc('font', size=7)
# plt.rc('axes', titlesize=7, labelsize=7)
# plt.rc('xtick', labelsize=7)
# plt.rc('ytick', labelsize=7)
# plt.rc('legend', fontsize=7)

# # --- 2. 创建图形 (采纳“学习代码”的尺寸) ---
# fig, axs = plt.subplots(1, 3, figsize=(6.5, 1.5))

# # 定义颜色和模型顺序
# siirl_color = (70/255, 130/255, 180/255) # 蓝色 (来自学习代码)
# linear_color = 'k'                     # 黑色
# model_params = ['7B', '32B', '72B']

# # --- 3. 绘图循环 (结构化，以应用统一格式) ---
# for i, model_name in enumerate(model_params):
#     ax = axs[i]
    
#     # 提取数据
#     model_df = df[df['Model'] == model_name]
#     x_data = model_df['GPUs'].tolist()
#     y_data = model_df['Throughput'].tolist()
    
#     # 创建理想线性扩展数据
#     y_linear = [y_data[0] * (x_val / x_data[0]) for x_val in x_data]

#     # --- 核心绘图内容 (保持折线图) ---
#     ax.plot(x_data, y_data, 'o-', color=siirl_color, label='siiRL', linewidth=1.5, markersize=4)
#     ax.plot(x_data, y_linear, ':', color=linear_color, label='linear', linewidth=1.5)

#     # --- 格式化 (采纳“学习代码”的风格) ---
    
#     # a. 设置对数坐标轴 (这是折线图的核心)
#     ax.set_xscale('log', base=2)
#     ax.set_yscale('log', base=2)

#     # b. 设置网格线
#     ax.grid(True, linestyle=':', alpha=0.6, linewidth=0.8)

#     # c. 设置X轴刻度和标签
#     ax.set_xticks(x_data)
#     ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter()) # 避免科学计数法
#     ax.set_xticklabels([str(val) for val in x_data])
    
#     # d. 设置Y轴刻度格式化
#     def format_func(value, tick_number):
#         if value >= 1000:
#             return f"{int(value/1000)}k"
#         return str(int(value))
#     ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(format_func))
#     # 自动调整Y轴刻度密度
#     ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4, integer=True, prune='both'))

#     # e. 设置坐标轴标签和子图标签 (关键改动)
#     ax.set_title('') # 清空原生标题
#     ax.set_xlabel('Number of GPUs', fontweight='medium', fontsize=7, labelpad=3)
    
#     prefix = f"({chr(ord('a') + i)})"
#     full_model_label = f"{prefix} Model={model_name}"
    
#     ax.text(x=0.5, y=-0.55, s=full_model_label,
#             fontweight='medium',
#             fontsize=7,
#             ha='center',
#             va='top',
#             transform=ax.transAxes)
    
#     # f. 调整刻度标签字体
#     for label in ax.get_xticklabels() + ax.get_yticklabels():
#         label.set_fontsize(7)

#     # g. 设置图例 (只在第一个子图显示，且无边框)
#     if i == 0:
#         ax.legend(loc='upper left', framealpha=0, fontsize=7)

# # --- 4. 设置共享的Y轴标签 (采纳“学习代码”的风格) ---
# fig.text(0.001, 0.55, 'Throughput (token/s)', va='center', rotation='vertical', fontsize=7, fontweight='medium')

# # --- 5. 调整布局 (采纳“学习代码”的风格) ---
# plt.subplots_adjust(wspace=0.3, left=0.08, right=0.98, top=0.85, bottom=0.25)

# # --- 6. 保存为PDF格式 (最终要求) ---
# plt.savefig('scaling_trend_new_style.pdf', bbox_inches='tight', pad_inches=0.05)

# print("图片已成功保存为: scaling_trend_new_style.pdf")

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from matplotlib import rcParams
import pandas as pd
import io

# ==============================================================================
# 注意：为了使代码可运行，此处使用了示例数据。
# 您在使用时，请删除或注释掉下面的 StringIO 部分，
# 并恢复您原来的 pd.read_csv('scaling_data.csv')。
# ==============================================================================
# csv_data = """Model,GPUs,Throughput
# 7B,8,950
# 7B,16,1850
# 7B,32,3600
# 7B,64,7000
# 32B,8,420
# 32B,16,820
# 32B,32,1600
# 32B,64,3100
# 72B,8,180
# 72B,16,350
# 72B,32,680
# 72B,64,1300
# """
# df = pd.read_csv(io.StringIO(csv_data))
df = pd.read_csv('csv/scaling_data.csv')
# ==============================================================================


# --- 1. 风格设置 (完全采纳“学习代码”的风格) ---
rcParams['font.weight'] = 549
plt.rc('font', size=7)
plt.rc('axes', titlesize=7, labelsize=7)
plt.rc('xtick', labelsize=7)
plt.rc('ytick', labelsize=7)
plt.rc('legend', fontsize=7)

# --- 2. 创建图形 (采纳“学习代码”的尺寸) ---
fig, axs = plt.subplots(1, 3, figsize=(7.3, 1.5))

# 定义颜色和模型顺序
siirl_color = (70/255, 130/255, 180/255) # 蓝色 (来自学习代码)
linear_color = 'k'                     # 黑色
model_params = ['7B', '32B', '72B']

# --- 3. 绘图循环 (结构化，以应用统一格式) ---
for i, model_name in enumerate(model_params):
    ax = axs[i]
    
    # 提取数据
    model_df = df[df['Model'] == model_name]
    x_data = model_df['GPUs'].tolist()
    y_data = model_df['Throughput'].tolist()
    
    # 创建理想线性扩展数据
    y_linear = [y_data[0] * (x_val / x_data[0]) for x_val in x_data]

    # --- 核心绘图内容 (保持折线图) ---
    ax.plot(x_data, y_data, 'o-', color=siirl_color, label='DistFlow', linewidth=1.5, markersize=4)
    ax.plot(x_data, y_linear, ':', color=linear_color, label='Linear', linewidth=1.5)

    # --- 格式化 (采纳“学习代码”的风格) ---
    
    # a. 设置对数坐标轴 (这是折线图的核心)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)

    # b. 设置网格线
    ax.grid(True, linestyle=':', alpha=0.6, linewidth=0.8)

    # c. 设置X轴刻度和标签
    ax.set_xticks(x_data)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter()) # 避免科学计数法
    ax.set_xticklabels([str(val) for val in x_data])
    
    # d. 设置Y轴刻度格式化 (关键修改部分)
    #    首先定义格式化函数
    def format_func(value, tick_number):
        if value >= 1000:
            return f"{int(value/1000)}k"
        return str(int(value))
    ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(format_func))
    
    # ========================== 代码修改开始 ==========================
    # 核心逻辑：在保持自动刻度划分风格的同时，确保第一个数据点的值显示在Y轴上
    # 1. 先让 matplotlib 根据数据范围和 nbins 自动计算一组“优雅”的刻度
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4, integer=True, prune='both'))
    
    # 2. 获取这组自动生成的刻度。(需要先调用 draw() 来确保刻度被最终计算)
    fig.canvas.draw() 
    current_yticks = ax.get_yticks()

    # 3. 将第一个数据点的Y坐标值添加到刻度列表中
    first_y_value = y_data[0]
    #    使用集合(set)合并、去重，然后排序
    new_yticks = sorted(list(set(current_yticks) | {first_y_value}))

    # 4. 设置最终的Y轴刻度
    ax.set_yticks(new_yticks)
    # ========================== 代码修改结束 ==========================

    # e. 设置坐标轴标签和子图标签 (关键改动)
    ax.set_title('') # 清空原生标题
    ax.set_xlabel('Number of GPUs', fontweight='medium', fontsize=7, labelpad=2)
    
    prefix = f"({chr(ord('a') + i)})"
    full_model_label = f"{prefix} Model={model_name}"
    
    ax.text(x=0.5, y=-0.4, s=full_model_label,
            fontweight='medium',
            fontsize=7,
            ha='center',
            va='top',
            transform=ax.transAxes)
    
    # f. 调整刻度标签字体
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(7)

    # g. 设置图例 (只在第一个子图显示，且无边框)
    if i == 0:
        ax.legend(loc='upper left', framealpha=0, fontsize=7)

# --- 4. 设置共享的Y轴标签 (采纳“学习代码”的风格) ---
fig.text(0.001, 0.55, 'Throughput (token/s)', va='center', rotation='vertical', fontsize=7, fontweight='medium')

# --- 5. 调整布局 (采纳“学习代码”的风格) ---
plt.subplots_adjust(wspace=0.3, left=0.08, right=0.98, top=0.85, bottom=0.25)

# --- 6. 保存为PDF格式 (最终要求) ---
plt.savefig('scaling_trend_new.pdf', dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.savefig('scaling_trend_new.png', dpi=300, bbox_inches='tight', pad_inches=0.05)


print("图片已成功保存为: scaling_trend_new_style.pdf")