"""Plot training diagnostics, focusing y limits on the non-baseline curves."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator, FuncFormatter

ROOT = Path(__file__).resolve().parent
SERIES = [
    ('async-baseline_', 'GRPO', '#1f77b4', '-'),
    ('async-top8RDC-TIS_', 'RDC', '#d62728', '-'),
    ('async-R3_', 'R3', '#2ca02c', '-'),
    ('async-TIS_', 'TIS', '#ff7f0e', '-'),
]
# (filename, ylabel, stem, include_baseline_in_ylim)
# include_baseline_in_ylim=True keeps the baseline curve fully in frame; only
# safe when the baseline stays on a comparable scale (e.g. entropy). For KL /
# reward the baseline diverges and would flatten the other curves, so leave it
# False and let the y limits focus on the corrected runs.
JOBS = [
    ('async-response-length-mean.csv', 'Response Length', 'response_length', False),
    ('async-reward-mean.csv', 'Reward', 'reward', False),
    ('async-rollout-train-kl.csv', 'Training-Inference KL', 'rollout_train_kl', False),
    ('async-rollout-train-ppl-ratio.csv', 'Perplexity Ratio', 'perplexity_ratio', False),
    ('training-rollout-probs-diff-mean.csv', 'Probability Difference', 'probability_difference', False),
    ('async-grad-norm.csv', 'Grad Norm', 'gradient_norm', False),
    ('async-entropy.csv', 'Entropy', 'entropy', True),
]


def limits(values):
    lo, hi = values.min(), values.max()
    pad = max((hi - lo) * .08, abs(hi) * .02, 1e-6)
    return lo - pad, hi + pad


def ema_smooth(series, weight):
    """wandb/TensorBoard 风格指数移动平均（带 debias 修正）。
    NaN 原样保留、不参与累积。weight 越大越平滑（0 无平滑，1 全平）。"""
    smoothed = []
    last = 0.0
    num_acc = 0
    for v in series:
        if pd.isna(v):
            smoothed.append(v)
            continue
        last = last * weight + (1 - weight) * v
        num_acc += 1
        debias = 1 - weight ** num_acc
        smoothed.append(last / debias)
    return pd.Series(smoothed, index=series.index)


SMOOTH_WEIGHT = 0.5


# 第三章 analysis 用到的四张图放大字号（与验证集单图口径一致：
# 坐标轴标签 14pt / 刻度 13pt / 图例 9pt），其余图保持原 7pt。
ENLARGED_STEMS = {'reward', 'response_length', 'gradient_norm', 'rollout_train_kl'}


def main():
    plt.rcParams.update({'font.size': 7, 'axes.labelsize': 7, 'legend.fontsize': 7,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    for filename, ylabel, stem, include_baseline in JOBS:
        enlarged = stem in ENLARGED_STEMS
        axis_fs = 14 if enlarged else 7
        tick_fs = 13 if enlarged else 7
        legend_fs = 11 if enlarged else 7  # 14pt would sit on the curves; see plot_val_curves.py
        data = pd.read_csv(ROOT / filename).sort_values('Step')
        curves = {}
        fig, ax = plt.subplots(figsize=(3.4, 2.4))
        for prefix, label, color, style in SERIES:
            columns = [c for c in data if c.startswith(prefix) and not c.endswith(('__MIN', '__MAX'))]
            if len(columns) != 1:
                raise ValueError(f'{filename}: expected one column for {prefix}, got {columns}')
            curves[label] = pd.to_numeric(data[columns[0]], errors='raise')
        # analysis 四图对训练曲线做 EMA 平滑（原始数据仍写入 *_plot_data.csv）
        plotted = {label: (ema_smooth(v, SMOOTH_WEIGHT) if enlarged else v)
                   for label, v in curves.items()}
        for prefix, label, color, style in SERIES:
            lw = 1.5 if enlarged else 1.0
            ax.plot(data['Step'], plotted[label], label=label, color=color,
                    linestyle=style, linewidth=lw, alpha=1.0)
        scoped = [v.dropna() for label, v in plotted.items()
                  if include_baseline or label != 'GRPO']
        ymin, ymax = limits(pd.concat(scoped))
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel('Step', fontweight='medium', labelpad=2, fontsize=axis_fs)
        if enlarged:
            # 纵轴标签放到绘图区上方（居中），避免与左上角科学计数法偏移标重叠
            ax.set_title(ylabel, fontweight='medium', fontsize=axis_fs, pad=6)
            ax.set_ylabel('')
        else:
            ax.set_ylabel(ylabel, fontweight='medium', fontsize=axis_fs)
        ax.grid(True, linestyle='-', alpha=.6, linewidth=.8)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        if stem == 'rollout_train_kl':
            # KL 值量级小，用普通小数（0.0015 等）而非科学计数法，
            # 去掉顶部 ×10^-3 偏移标，使标题位置与其它无偏移标的图对齐
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.3f}'))
        elif stem == 'response_length':
            # 直接显示整数（8000 等），去掉左上角 ×10^4 偏移标，避免与标题挤在一起
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{int(v)}'))
        else:
            ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 4), useMathText=True)
        ax.tick_params(labelsize=tick_fs)
        ax.yaxis.get_offset_text().set_size(tick_fs)
        ax.margins(x=.02)
        if stem == 'reward':
            ax.legend(loc='lower right', ncol=1, framealpha=0, fontsize=legend_fs)
        if enlarged:
            # Fixed canvas + fixed axes rectangle so the grid area is identical
            # across the four analysis figures. The reserved top margin holds the
            # scientific-notation offset label (present on response length / KL)
            # without shrinking the plotting area, and we skip the 'tight' bbox
            # so every figure keeps the same size.
            fig.subplots_adjust(left=.195, right=.975, top=.90, bottom=.215)
            for ext in ('png', 'pdf'):
                fig.savefig(ROOT / 'figures' / f'{stem}.{ext}', dpi=300)
        else:
            fig.tight_layout(pad=.6)
            for ext in ('png', 'pdf'):
                fig.savefig(ROOT / 'figures' / f'{stem}.{ext}', dpi=300, bbox_inches='tight', pad_inches=.05)
        pd.DataFrame({'Step': data['Step'], **curves}).to_csv(ROOT / f'{stem}_plot_data.csv', index=False)
        plt.close(fig)


if __name__ == '__main__':
    main()
