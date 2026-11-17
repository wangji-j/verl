"""Generate the AIME24 validation figure from the canonical export."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / 'figures'
# 与 plot_training_curves.py / plot_val_curves.py 的配色对齐（matplotlib 默认色板）
SERIES = [
    ('Baseline', 'baseline', '#1f77b4'),
    ('RDC+TIS', 'top8RDC-TIS', '#d62728'),
    ('R3', 'async-R3', '#2ca02c'),
    ('TIS', 'async-TIS', '#ff7f0e'),
]


def main():
    data = pd.read_csv(ROOT / 'AIME24.csv')
    step = data.columns[0]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    values = []
    for label, key, color in SERIES:
        column = next(c for c in data.columns if key in c and 'acc/mean@1' in c and '__' not in c)
        series = data[[step, column]].dropna()
        values.extend(series[column])
        ax.plot(series[step], series[column], label=label, color=color, linewidth=1.0, alpha=1.0)
    lo, hi = min(values), max(values)
    pad = max((hi - lo) * .08, .01)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel('Step', fontsize=7)
    ax.set_ylabel('AIME24', fontsize=7)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x / 1000:g}k' if x >= 1000 else f'{x:g}'))
    ax.grid(True, linestyle='-', alpha=.6)
    ax.legend(loc='lower right', frameon=False, fontsize=6.5)
    fig.tight_layout(pad=.6)
    for extension in ('pdf', 'png'):
        fig.savefig(OUTPUT_DIR / f'aime24_score.{extension}', dpi=300, bbox_inches='tight', pad_inches=.05)
    plt.close(fig)


if __name__ == '__main__':
    main()
