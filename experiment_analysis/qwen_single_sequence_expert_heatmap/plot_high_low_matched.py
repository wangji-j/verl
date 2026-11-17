#!/usr/bin/env python3
"""CPU-only, length-matched high/low drift heatmaps; preserve earlier artifacts."""
import csv
import importlib.util
import json
from pathlib import Path
import sys

import plot_expert_heatmap as base
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_hex
import numpy as np
import torch


def main():
    source = base.DEFAULT_SOURCE.with_name('probe_lag0128.pt')
    record = torch.load(source, weights_only=True, map_location='cpu')
    assert int(record['lag']) == 128
    comp = record['mu_vs_cur']
    # Cast unsigned counts before any arithmetic, including differences.
    arrays = [comp[k].to(torch.int64).numpy() for k in ('ref_counts', 'cur_counts')]
    assert arrays[0].shape == arrays[1].shape and arrays[0].shape[1:] == (48, 128)
    counts = np.stack([a[:, 46] for a in arrays], axis=1)
    ids, lengths = [record[k].numpy() for k in ('probe_ids', 'lengths')]
    valid = comp['valid_tokens'].numpy().astype(float)
    assert np.isfinite(counts).all() and (counts >= 0).all()
    assert np.isfinite(lengths).all() and np.isfinite(valid).all()
    assert ((valid >= 0) & (valid <= lengths)).all()
    totals = counts.sum(-1)
    assert (totals == valid[:, None] * 8).all()
    eligible = (valid > 0) & (totals > 0).all(-1) & (lengths >= 12288)
    shares = np.divide(counts, totals[..., None], out=np.zeros_like(counts, dtype=float), where=totals[..., None] > 0)
    tv = abs(shares[:, 1] - shares[:, 0]).sum(-1) / 2
    assert np.isfinite(shares).all() and np.isfinite(tv).all()
    high_rows = np.flatnonzero(ids == 245)
    assert len(high_rows) == 1
    high = int(high_rows[0])
    assert eligible[high]
    pool = eligible & (ids != 245)
    tiers = [
        ('exact_response_length', pool & (lengths == lengths[high])),
        ('within_10_percent_response_length', pool & (abs(lengths - lengths[high]) <= .1 * lengths[high])),
        ('response_length_at_least_12288', pool),
    ]
    tier, mask = next((name, m) for name, m in tiers if m.any())
    candidates = np.flatnonzero(mask)
    low = int(sorted(candidates, key=lambda i: (tv[i], ids[i], i))[0])
    selected = [high, low]
    pct = shares[selected] * 100
    delta = pct[:, 1] - pct[:, 0]
    np.testing.assert_allclose(pct.sum(-1), 100)
    np.testing.assert_allclose(abs(delta).sum(-1) / 200, tv[selected])
    assert tv[low] == tv[candidates].min() and tv[high] > tv[low]

    cmap = LinearSegmentedColormap.from_list('shared_blue', base.RAMP, N=256)
    poles = ['#2a78d6', '#f0efec', '#e34948']
    dcmap = LinearSegmentedColormap.from_list('signed_change', poles, N=257)
    spec = importlib.util.spec_from_file_location('validator', base.DEFAULT_VALIDATOR)
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    lightness = lambda cm, n: np.array([validator.lin2oklab(*validator.lin(to_hex(cm(x))))[0] for x in np.linspace(0, 1, n)])
    assert (np.diff(lightness(cmap, 256)) <= 0).all()
    arm = lightness(dcmap, 257)
    assert (np.diff(arm[:129]) >= 0).all() and (np.diff(arm[128:]) <= 0).all()
    pole_checks, ok = validator.validate([poles[0], poles[-1]], 'light', base.SURFACE)
    assert ok, pole_checks
    contrasts = {c: validator.contrast(c, base.SURFACE) for c in (base.INK, base.SECONDARY)}
    assert min(contrasts.values()) >= 4.5
    vmax, dmax = float(np.ceil(pct.max())), float(abs(delta).max())
    prefix = Path(__file__).with_name(f'qwen_high_low_layer47_lag128_probe245_probe{int(ids[low])}_matched')
    for extension in ('.png', '.pdf', '.csv', '.json'):
        if prefix.with_suffix(extension).exists():
            raise FileExistsError(f'Preserve existing artifact: {prefix.with_suffix(extension)}')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'text.color': base.INK,
                         'axes.labelcolor': base.SECONDARY, 'xtick.color': base.SECONDARY,
                         'ytick.color': base.INK, 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(19, 8.5), facecolor=base.SURFACE)
    fig.text(.085, .945, 'Qwen3-30B-A3B | Length-matched high and low expert-usage drift', fontsize=21, weight='semibold')
    fig.text(.085, .902, 'Layer 47 / 48 (index 46)  |  Parameter-update lag 128  |  All 128 experts, original ID order', fontsize=13, color=base.SECONDARY)
    for j, (row, label, bottom) in enumerate(zip(selected, ['High drift', 'Low drift'], [.64, .36])):
        fig.text(.085, bottom + .18, f'{label}  |  Probe {int(ids[row])}  |  TV = {tv[row]:.6f}  |  {int(valid[row]):,} aligned / {int(lengths[row]):,} response tokens', fontsize=13, weight='semibold')
        ax = fig.add_axes([.085, bottom, .415, .145], facecolor=base.SURFACE)
        dax = fig.add_axes([.60, bottom, .355, .145], facecolor=base.SURFACE)
        image = ax.imshow(pct[j], cmap=cmap, norm=Normalize(0, vmax), aspect='auto', interpolation='nearest')
        ax.axhline(.5, color=base.SURFACE, lw=2)
        ax.set_yticks([0, 1], ['Rollout', 'Training'])
        dim = dax.imshow(delta[j][None], cmap=dcmap, norm=Normalize(-dmax, dmax), aspect='auto', interpolation='nearest')
        dax.set_yticks([0], ['Training\n− Rollout'])
        for axis in (ax, dax):
            axis.set_xticks(list(range(0, 128, 16)) + [127])
            if j == 1:
                axis.set_xlabel('Expert ID (zero-based)', labelpad=7)
            axis.tick_params(length=0, pad=6, labelsize=10)
            for spine in axis.spines.values():
                spine.set_visible(False)
    for im, rect, ticks, label in [
        (image, [.085, .225, .415, .022], np.linspace(0, vmax, 5), 'Expert selection share (%) | one scale for all four rows'),
        (dim, [.60, .225, .355, .022], np.linspace(-dmax, dmax, 5), 'Training − rollout (pp) | one symmetric scale for both sequences')]:
        cb = fig.colorbar(im, cax=fig.add_axes(rect), orientation='horizontal', ticks=ticks)
        cb.set_label(label, fontsize=10, labelpad=6)
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=0, labelsize=9)
    fig.text(.085, .112, f'High: fixed probe 245 from the previous figure. Low: minimum layer-47 TV among {len(candidates)} other exact-length candidates.' if tier == 'exact_response_length' else f'Low: minimum layer-47 TV in {tier} ({len(candidates)} candidates).', fontsize=11, color=base.SECONDARY)
    fig.text(.085, .065, 'Share = expert count / sum of counts on each side; TV = ½ sum of absolute share differences. Counts use the common aligned response mask.\nAggregated expert usage, not token-level routes or activation amplitude. These selected examples do not establish a population or causal effect.', fontsize=10, color=base.SECONDARY, linespacing=1.65)
    fig.savefig(prefix.with_suffix('.png'), dpi=200, facecolor=base.SURFACE)
    fig.savefig(prefix.with_suffix('.pdf'), facecolor=base.SURFACE)
    plt.close(fig)
    with prefix.with_suffix('.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['drift_label', 'probe_id', 'response_length', 'aligned_valid_tokens', 'layer_index', 'expert_id', 'rollout_count', 'training_count', 'rollout_share_pct', 'training_share_pct', 'training_minus_rollout_pp'])
        for j, (row, label) in enumerate(zip(selected, ['high', 'low'])):
            writer.writerows([label, int(ids[row]), int(lengths[row]), int(valid[row]), 46, e, int(counts[row, 0, e]), int(counts[row, 1, e]), float(pct[j, 0, e]), float(pct[j, 1, e]), float(delta[j, e])] for e in range(128))
    describe = lambda i: {'probe_id': int(ids[i]), 'row_index': int(i), 'response_length': int(lengths[i]), 'aligned_valid_tokens': int(valid[i]), 'raw_normalized_tv': float(tv[i]), 'count_totals': totals[i].tolist(), 'expected_valid_tokens_times_8': int(valid[i] * 8)}
    metadata = {
        'source': str(source), 'source_sha256': base.sha256(source),
        'script': str(Path(__file__).resolve()), 'script_sha256': base.sha256(__file__),
        'reused_script': str(Path(base.__file__).resolve()), 'reused_script_sha256': base.sha256(base.__file__),
        'command': [sys.executable, str(Path(__file__).resolve())],
        'record': {k: record[k] for k in ['lag', 'step', 'param_version', 'local_trigger_step', 'v0', 'alignment']},
        'layer_index_zero_based': 46, 'layer_one_based': 47, 'high': describe(high), 'low': describe(low),
        'selection': {'high': 'Fixed probe 245 from previous maximum-TV plot; no re-selection.',
                      'low_rule': 'Among other eligible sequences with response length >=12288: prefer exact high response length; if empty prefer within 10%; if empty use full >=12288 bucket. Minimize raw normalized layer-46 TV within the first nonempty tier. Ties: probe ID, row index ascending.',
                      'selected_tier': tier, 'candidate_count': len(candidates),
                      'tier_counts': {name: int(m.sum()) for name, m in tiers},
                      'candidates': [describe(int(i)) for i in sorted(candidates, key=lambda i: (tv[i], ids[i], i))]},
        'semantics': {'rollout': 'mu_vs_cur.ref_counts, cached rollout routes', 'training': 'mu_vs_cur.cur_counts, current-weight training-side recomputation on fixed response', 'normalization': 'count / side count sum', 'tv': '0.5 * sum(abs(normalized_training-normalized_rollout))', 'difference': 'training minus rollout, percentage points', 'alignment': 'Common response and aligned mask; aligned tokens are one fewer than response tokens here', 'lag': 'Relative parameter-update lag, not global training step'},
        'validation': {'load': 'weights_only=True, map_location=cpu', 'counts_cast_before_arithmetic': 'int64', 'all_layer47_totals_equal_valid_tokens_times_8': True, 'finite_nonnegative_counts_and_shares': True, 'selected_share_sums_pct': pct.sum(-1).tolist(), 'minimum_tv_in_selected_tier': True},
        'palette': {'sequential': base.RAMP, 'range_pct': [0, vmax], 'difference': poles, 'range_pp': [-dmax, dmax], 'expert_order': list(range(128)), 'sequential_monotone_OKLab_L': True, 'diverging_monotone_arms': True, 'neutral_midpoint': True, 'pole_checks': pole_checks, 'text_contrast': contrasts, 'validator': str(base.DEFAULT_VALIDATOR), 'validator_sha256': base.sha256(base.DEFAULT_VALIDATOR), 'relief': 'Numeric shared colorbars, directly labeled rows, CSV table'},
        'limitations': ['Relative high/low comparison among deliberately selected examples, not population inference.', 'Low is the minimum in the exact-length tier, not necessarily the minimum in the broader >=12288 bucket.', 'Counts aggregate tokens and cannot reveal token-level rerouting, activation amplitude, gate probabilities, outputs, or gradients.', 'No actual rejection or acceptance membership is inferred.'],
        'versions': {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__, 'matplotlib': base.matplotlib.__version__},
        'outputs': {ext: str(prefix.with_suffix('.' + ext)) for ext in ('png', 'pdf', 'csv', 'json')},
    }
    prefix.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({'high': metadata['high'], 'low': metadata['low'], 'selection_tier': tier, 'candidate_count': len(candidates), 'outputs': metadata['outputs']}, indent=2))


if __name__ == '__main__':
    main()
