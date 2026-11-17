import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from matplotlib import rcParams

# =============================================================================
# 1. 样式设置（对齐 reference/ 风格）
# =============================================================================
rcParams['font.weight'] = 549
plt.rc('font', size=9)
plt.rc('axes', titlesize=9, labelsize=9)
plt.rc('xtick', labelsize=9)
plt.rc('ytick', labelsize=9)
plt.rc('legend', fontsize=9)

# 配色：reference 用 crimson(主)+steel blue，扩展到 4 条
color_rdc   = (220/255,  20/255,  60/255)   # crimson —— 我们的方法(RDC+TIS) 作主色
color_r3    = ( 70/255, 130/255, 180/255)   # steel blue —— R3
color_tis   = ( 46/255, 139/255,  87/255)   # sea green —— TIS
color_base  = (128/255, 128/255, 128/255)   # gray —— baseline

# =============================================================================
# 2. 数据（AIME24.csv：4 个实验，各一条已合并的曲线）
# =============================================================================
df = pd.read_csv('AIME24.csv')
step = df.columns[0]

# (原始列名, 图例名, 颜色, zorder)
series = [
    ("async-top8RDC-TIS_20260817_204021 - val-core/AIME24/acc/mean@1", "RDC+TIS", color_rdc,  5),
    ("async-R3_20260815_001128 - val-core/AIME24/acc/mean@1",          "R3",      color_r3,   4),
    ("async-TIS_20260814_234915 - val-core/AIME24/acc/mean@1",         "TIS",     color_tis,  3),
    ("async-baseline_20260830_042118 - val-core/AIME24/acc/mean@1",    "Baseline",color_base, 2),
]

# =============================================================================
# 3. 绘图
# =============================================================================
fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.4))

for col, label, color, z in series:
    s = df[[step, col]].dropna()
    ax.plot(s[step], s[col], label=label, color=color,
            linewidth=1.2, alpha=0.9, marker='o', markersize=2.2,
            markeredgewidth=0, zorder=z)

# =============================================================================
# 4. 格式化（对齐模板）
# =============================================================================
def format_func(value, tick_number):
    if value >= 1000:
        return f"{int(value/1000)}k"
    return str(int(value))

ax.set_xlabel('Step', fontweight='medium', fontsize=7, labelpad=2)
ax.set_ylabel('AIME24', fontweight='medium', fontsize=7)
ax.grid(True, linestyle=':', alpha=0.6, linewidth=0.8)
ax.legend(loc='lower right', framealpha=0, fontsize=7)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))
ax.set_ylim(0, 0.8)
for lab in ax.get_xticklabels() + ax.get_yticklabels():
    lab.set_fontsize(7)

plt.subplots_adjust(left=0.13, right=0.97, top=0.95, bottom=0.15)

plt.savefig('aime24_score.pdf', dpi=300, bbox_inches='tight')
plt.savefig('aime24_score.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("saved: aime24_score.pdf / aime24_score.png")
