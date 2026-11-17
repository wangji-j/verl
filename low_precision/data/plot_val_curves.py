"""Generate all validation-set score curves (AIME24 / AIME25 / MATH-500 /
ZebraLogic) from a single code path so every figure shares identical size,
fonts, layout, and colors."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker
from matplotlib import rcParams

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / 'figures'
OUTPUT_DIR.mkdir(exist_ok=True)

# ---- 统一样式 ----
FIGSIZE = (3.4, 2.4)
AXIS_FONTSIZE = 7
TICK_FONTSIZE = 7
LEGEND_FONTSIZE = 7
# 单张图（AIME24/AIME25/MATH-500/ZebraLogic）放进正文会被缩放，
# 单独放大字号，使有效大小接近正文 10pt；不影响 2x2 合并图。
AXIS_FONTSIZE_SINGLE = 14
TICK_FONTSIZE_SINGLE = 13
# 11pt: as close to AXIS_FONTSIZE_SINGLE (14) as fits. At 14pt the four-entry
# legend spans x=[103,212] / y=[0.21,0.63] on AIME24 and all four curves run
# through it (framealpha=0, so there is no box to hide them). 11pt leaves only
# 2 GRPO vertices inside and clears reward.png entirely. Re-measure before
# raising this: the limit is the 3.4x2.4in canvas, not the font.
LEGEND_FONTSIZE_SINGLE = 11
# Type 42 (TrueType) instead of matplotlib's default Type 3: Type 3 fonts are
# rejected by most conference submission checkers and do not scale cleanly.
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['font.weight'] = 549
plt.rc('font', size=9)
plt.rc('axes', titlesize=9, labelsize=9)
plt.rc('xtick', labelsize=TICK_FONTSIZE)
plt.rc('ytick', labelsize=TICK_FONTSIZE)
plt.rc('legend', fontsize=LEGEND_FONTSIZE)

# 与 plot_training_curves.py 的 SERIES 配色对齐（matplotlib 默认色板）
color_base = '#1f77b4'   # blue      —— GRPO
color_rdc  = '#d62728'   # red       —— RDC
color_r3   = '#2ca02c'   # green     —— R3
color_tis  = '#ff7f0e'   # orange    —— TIS

# (实验前缀关键字, 图例名, 颜色, zorder)
SERIES = [
    ("baseline", "GRPO", color_base, 2),
    ("top8RDC-TIS", "RDC", color_rdc, 5),
    ("R3", "R3", color_r3, 4),
    ("async-TIS", "TIS", color_tis, 3),
]

# (csv 文件, y轴标签, 输出前缀)
JOBS = [
    ("AIME24.csv",     "AIME24",     "aime24_score"),
    ("AIME25.csv",     "AIME25",     "aime25_score"),
    ("math500.csv",    "MATH-500",   "math500_score"),
    ("zebralogic.csv", "ZebraLogic", "zebralogic_score"),
]


def format_func(value, _):
    return f"{int(value/1000)}k" if value >= 1000 else str(int(value))


def find_col(cols, key):
    # 主指标列（含 acc/mean@1、非 __MIN/__MAX），且实验前缀含 key
    for c in cols:
        if "acc/mean@1" in c and "__M" not in c and key in c:
            return c
    return None


def plot_series(ax, df, show_legend):
    step = df.columns[0]
    all_values = []
    for key, label, color, z in SERIES:
        col = find_col(df.columns, key)
        if col is None:
            continue
        s = df[[step, col]].dropna()
        ax.plot(s[step], s[col], label=label, color=color,
                linewidth=1.5, alpha=1.0, zorder=z)
        all_values.extend(s[col].tolist())
    ax.grid(True, linestyle='-', alpha=0.6, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))
    vals = pd.Series(all_values)
    lo, hi = float(vals.min()), float(vals.max())
    pad = max((hi - lo) * 0.08, 0.01)
    ax.set_ylim(lo - pad, hi + pad)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontsize(TICK_FONTSIZE)
    if show_legend:
        ax.legend(loc='lower right', framealpha=0, fontsize=LEGEND_FONTSIZE)


def make_grid():
    # 2x2: 第一行 AIME24 / AIME25，第二行 MATH-500 / ZebraLogic
    order = ['aime24_score', 'aime25_score', 'math500_score', 'zebralogic_score']
    lookup = {out: (csv, ylabel) for csv, ylabel, out in JOBS}
    fig, axes = plt.subplots(2, 2, figsize=(FIGSIZE[0] * 2, FIGSIZE[1] * 2))
    for idx, (ax, out) in enumerate(zip(axes.flat, order)):
        csv, ylabel = lookup[out]
        df = pd.read_csv(ROOT / csv)
        plot_series(ax, df, show_legend=(idx == 0))  # 仅左上角 AIME24 留图例
        ax.set_ylabel(ylabel, fontweight='medium', fontsize=AXIS_FONTSIZE)
        # 只有第二行标 X 轴标签，避免重复
        if idx >= 2:
            ax.set_xlabel('Step', fontweight='medium', fontsize=AXIS_FONTSIZE, labelpad=2)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.98, bottom=0.08,
                        wspace=0.18, hspace=0.2)
    for ext in ('pdf', 'png'):
        fig.savefig(OUTPUT_DIR / f'val_grid_2x2.{ext}', dpi=300)
    plt.close(fig)
    print("saved: figures/val_grid_2x2.pdf / .png  (2x2: AIME24 AIME25 / MATH-500 ZebraLogic)")


def main():
    for csv, ylabel, out in JOBS:
        df = pd.read_csv(ROOT / csv)
        step = df.columns[0]
        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
        all_values = []
        for key, label, color, z in SERIES:
            col = find_col(df.columns, key)
            if col is None:
                print(f"[warn] {csv}: 未找到 {key} 列")
                continue
            s = df[[step, col]].dropna()
            ax.plot(s[step], s[col], label=label, color=color,
                    linewidth=1.5, alpha=1.0, zorder=z)
            all_values.extend(s[col].tolist())
        ax.set_xlabel('Step', fontweight='medium', fontsize=AXIS_FONTSIZE_SINGLE, labelpad=2)
        ax.set_title(ylabel, fontweight='medium', fontsize=AXIS_FONTSIZE_SINGLE)
        ax.grid(True, linestyle='-', alpha=0.6, linewidth=0.8)
        ax.set_axisbelow(True)
        if out == 'aime24_score':
            ax.legend(loc='lower right', framealpha=0, fontsize=LEGEND_FONTSIZE_SINGLE)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))
        # 四张单图 y 轴统一两位小数（"0.XX"），保证裁剪后 bounding box 宽度一致
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v:.2f}'))
        vals = pd.Series(all_values)
        lo, hi = float(vals.min()), float(vals.max())
        pad = max((hi - lo) * 0.08, 0.01)
        ax.set_ylim(lo - pad, hi + pad)
        # MATH-500 纵轴范围窄、默认刻度过密，稀疏化 y 轴刻度（不改变 ylim）
        if out == 'math500_score':
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_fontsize(TICK_FONTSIZE_SINGLE)
        # 固定边距（不用 bbox_inches='tight'），保证四张图导出尺寸严格一致
        # 放大字号后适当增加左/下边距，避免标签被裁切
        fig.subplots_adjust(left=0.195, right=0.975, top=0.90, bottom=0.215)
        for ext in ('pdf', 'png'):
            fig.savefig(OUTPUT_DIR / f'{out}.{ext}', dpi=300)
        plt.close(fig)
        print(f"saved: figures/{out}.pdf / .png  (ylabel={ylabel}, ylim={lo:.3f}..{hi:.3f})")


if __name__ == '__main__':
    main()
    make_grid()
