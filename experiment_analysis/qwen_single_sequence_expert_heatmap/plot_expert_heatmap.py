#!/usr/bin/env python3
"""Reproduce a CPU-only, single-sequence, single-layer expert-usage heatmap.
Run: python plot_expert_heatmap.py
No token-level routes or gate/activation magnitudes are inferred from counts.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_hex
import numpy as np
import torch

DEFAULT_SOURCE = Path('/inspire/hdd3/project/qianghuaxuexi/hujiarui-25046/ckpts/router drift control/async-routerlag-probe-from-randomdrop240_20261110_045428/router_lag_probe/probe_lag0016.pt')
DEFAULT_VALIDATOR = Path('/tmp/claude-0/bundled-skills/2.1.217/ded6c7bbef1fe729d48d7b108cfacd9c/dataviz/scripts/validate_palette.py')
RAMP = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']
SURFACE, INK, SECONDARY = '#fcfcfb', '#0b0b0b', '#52514e'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--palette-validator', type=Path, default=DEFAULT_VALIDATOR)
    parser.add_argument('--selection', choices=['median', 'max'], default='median',
                        help='Maximum TV adds a difference row and writes separate output files.')
    args = parser.parse_args()
    # Fail closed if weights_only is unsupported or the input is incompatible.
    record = torch.load(args.source, weights_only=True, map_location='cpu')
    comp = record['mu_vs_cur']
    layer = 23
    ref = comp['ref_counts'].to(torch.int64).numpy()
    cur = comp['cur_counts'].to(torch.int64).numpy()
    assert ref.shape == cur.shape and ref.ndim == 3 and ref.shape[1:] == (48, 128)
    assert np.all(ref >= 0) and np.all(cur >= 0)
    ids, lengths = record['probe_ids'].numpy(), record['lengths'].numpy()
    valid = comp['valid_tokens'].numpy().astype(np.float64)
    assert ids.shape == lengths.shape == valid.shape == (ref.shape[0],)
    assert np.isfinite(valid).all() and np.all((valid >= 0) & (valid <= lengths))
    counts = np.stack([ref[:, layer], cur[:, layer]], axis=1)
    sums = counts.sum(axis=2)
    eligible = (valid > 0) & (lengths > 0) & (sums > 0).all(axis=1)
    candidates = np.flatnonzero(eligible)
    assert len(candidates)
    shares = counts[candidates].astype(np.float64) / sums[candidates, :, None]
    tv = np.abs(shares[:, 0] - shares[:, 1]).sum(axis=1) / 2
    median = float(np.median(tv))
    # Deterministic ties: smallest probe ID, then original row index.
    selection_score = np.abs(tv - median) if args.selection == 'median' else -tv
    order = np.lexsort((candidates, ids[candidates], selection_score))
    chosen = int(order[0])
    index = int(candidates[chosen])
    pct = shares[chosen] * 100
    assert np.isfinite(pct).all() and np.all(pct >= 0)
    np.testing.assert_allclose(pct.sum(axis=1), 100, atol=1e-10)
    assert selection_score[chosen] == selection_score.min()
    delta = pct[1] - pct[0]
    percentile = float(100 * np.mean(tv <= tv[chosen]))
    warnings = []
    expected = valid[:, None] * 8
    count_bad = np.any(sums != expected, axis=1)
    if count_bad.any():
        warnings.append(f'{int(count_bad.sum())} candidate rows have count totals unequal to valid_tokens * 8.')
    length_diff = lengths - valid
    if np.any(length_diff):
        warnings.append(f'{int(np.count_nonzero(length_diff))} rows have fewer aligned valid tokens than response length; counts use the aligned common mask.')
    selected_bad = bool(count_bad[index])
    if selected_bad:
        warnings.append('Selected sequence has anomalous count totals; shares normalize each observed total separately.')
    cmap = LinearSegmentedColormap.from_list('documented_blue_100_700', RAMP, N=256)
    # Use the skill validator's color math for the applicable sequential/text checks.
    spec = importlib.util.spec_from_file_location('palette_validator', args.palette_validator)
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    colors = [to_hex(cmap(x)) for x in np.linspace(0, 1, 256)]
    lightness = np.array([validator.lin2oklab(*validator.lin(c))[0] for c in colors])
    assert np.all(np.diff(lightness) <= 0), 'Sequential lightness must decrease monotonically'
    text_contrast = {c: validator.contrast(c, SURFACE) for c in [INK, SECONDARY]}
    assert min(text_contrast.values()) >= 4.5
    palette_check = {
        'validator': str(args.palette_validator.resolve()),
        'validator_sha256': sha256(args.palette_validator),
        'sequential_monotone_OKLab_L': True,
        'sampled_steps': 256,
        'text_contrast': text_contrast,
        'scope': 'Continuous sequential magnitude: check monotone lightness, not categorical pair separation or ordinal minimum-step contrast. Low-value pale cells are supported by a shared numeric colorbar and CSV.'}
    diverging_colors = ['#2a78d6', '#f0efec', '#e34948']
    delta_cmap = LinearSegmentedColormap.from_list('blue_neutral_red', diverging_colors, N=257)
    if args.selection == 'max':
        pole_checks, poles_ok = validator.validate([diverging_colors[0], diverging_colors[2]], 'light', SURFACE)
        assert poles_ok, pole_checks
        arm_l = np.array([validator.lin2oklab(*validator.lin(to_hex(delta_cmap(x))))[0]
                          for x in np.linspace(0, 1, 257)])
        assert np.all(np.diff(arm_l[:129]) >= 0) and np.all(np.diff(arm_l[128:]) <= 0)
        palette_check['diverging'] = {'colors': diverging_colors, 'pole_checks': pole_checks,
                                     'arms_monotone_OKLab_L': True, 'neutral_midpoint': True}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = '_max_tv_delta' if args.selection == 'max' else ''
    prefix = args.output_dir / f'qwen_single_sequence_layer24_lag{int(record["lag"])}{suffix}'
    with prefix.with_suffix('.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['expert_id_zero_based', 'rollout_count', 'training_count', 'rollout_selection_share_pct', 'training_selection_share_pct', 'training_minus_rollout_pp'])
        for e in range(128):
            writer.writerow([e, int(counts[index, 0, e]), int(counts[index, 1, e]), pct[0, e], pct[1, e], pct[1, e] - pct[0, e]])
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'text.color': INK, 'axes.labelcolor': SECONDARY, 'xtick.color': SECONDARY, 'ytick.color': INK, 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(16, 4.4), facecolor=SURFACE)
    ax = fig.add_axes([0.105, 0.40, 0.86, 0.28], facecolor=SURFACE)
    vmax = float(np.ceil(pct.max()))
    image = ax.imshow(pct, cmap=cmap, norm=Normalize(0, vmax), interpolation='nearest', aspect='auto')
    ax.axhline(0.5, color=SURFACE, linewidth=2)
    ax.set_yticks([0, 1], ['Rollout', 'Training'])
    ax.set_xticks(list(range(0, 128, 16)) + [127])
    ax.set_xlabel('Expert ID (zero-based)', labelpad=8)
    ax.tick_params(length=0, pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(0.105, 0.90, 'Qwen3-30B-A3B | Single-sequence expert selection', fontsize=19, weight='semibold')
    fig.text(0.105, 0.80, f'Probe {int(ids[index])}  |  Layer 24 / 48 (index 23)  |  Parameter-update lag {int(record["lag"])}  |  TV = {tv[chosen]:.6f}', fontsize=12, color=SECONDARY)
    cax = fig.add_axes([0.60, 0.17, 0.365, 0.035])
    colorbar = fig.colorbar(image, cax=cax, orientation='horizontal', ticks=np.linspace(0, vmax, 5))
    colorbar.set_label('Expert selection share (%) — shared scale', fontsize=10, labelpad=5)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(length=0, labelsize=9)
    fig.text(0.105, 0.22, f'{int(valid[index]):,} aligned response tokens; top-8 routing\nSelection totals: rollout {int(sums[index, 0]):,} / training {int(sums[index, 1]):,}', fontsize=10, color=SECONDARY, linespacing=1.6)
    note = 'Median-TV sample at this layer. Share = count / side total; not activation magnitude. Lag is relative parameter updates, not a global training step.'
    if selected_bad or length_diff[index] != 0:
        note += '\nMask/count warning: see provenance JSON for aligned versus response lengths and observed totals.'
    fig.text(0.105, 0.035, note, fontsize=9, color=SECONDARY)
    if args.selection == 'max':
        fig.clear()
        fig.set_size_inches(16, 6.2)
        ax = fig.add_axes([0.105, 0.53, 0.86, 0.23], facecolor=SURFACE)
        dax = fig.add_axes([0.105, 0.36, 0.86, 0.105], facecolor=SURFACE)
        image = ax.imshow(pct, cmap=cmap, norm=Normalize(0, vmax), interpolation='nearest', aspect='auto')
        ax.axhline(0.5, color=SURFACE, linewidth=2)
        ax.set_yticks([0, 1], ['Rollout', 'Training'])
        ax.set_xticks([])
        delta_limit = float(np.max(np.abs(delta))) or 1.0
        delta_image = dax.imshow(delta[None], cmap=delta_cmap, norm=Normalize(-delta_limit, delta_limit),
                                 interpolation='nearest', aspect='auto')
        dax.set_yticks([0], ['Training\n− Rollout'])
        dax.set_xticks(list(range(0, 128, 16)) + [127])
        dax.set_xlabel('Expert ID (zero-based)', labelpad=8)
        for axis in [ax, dax]:
            axis.tick_params(length=0, pad=8)
            for spine in axis.spines.values():
                spine.set_visible(False)
        fig.text(0.105, 0.93, 'Qwen3-30B-A3B | Maximum-TV sequence at layer 24', fontsize=19, weight='semibold')
        fig.text(0.105, 0.865, f'Probe {int(ids[index])}  |  Layer 24 / 48 (index 23)  |  Parameter-update lag {int(record["lag"])}  |  TV = {tv[chosen]:.6f}', fontsize=12, color=SECONDARY)
        fig.text(0.105, 0.81, f'{int(valid[index]):,} aligned / {int(lengths[index]):,} response tokens  |  Layer median TV {median:.6f}  |  Percentile {percentile:.0f} (≤ selected)', fontsize=11, color=SECONDARY)
        for im, rect, ticks, label in [
            (image, [0.105, 0.19, 0.365, 0.027], np.linspace(0, vmax, 5), 'Expert selection share (%) — shared top-row scale'),
            (delta_image, [0.60, 0.19, 0.365, 0.027], np.linspace(-delta_limit, delta_limit, 5), 'Training − rollout (percentage points) — symmetric scale')]:
            cb = fig.colorbar(im, cax=fig.add_axes(rect), orientation='horizontal', ticks=ticks)
            cb.set_label(label, fontsize=10, labelpad=5)
            cb.outline.set_visible(False)
            cb.ax.tick_params(length=0, labelsize=9)
        fig.text(0.105, 0.055, f'Maximum TV among all {len(candidates)} valid sequences; no minimum-length filter. Share = count / side total.\nCounts aggregate over the aligned response; this does not measure token-level rerouting. Lag counts relative parameter updates.', fontsize=10, color=SECONDARY, linespacing=1.6)
    fig.savefig(prefix.with_suffix('.png'), dpi=200, facecolor=SURFACE)
    fig.savefig(prefix.with_suffix('.pdf'), facecolor=SURFACE)
    plt.close(fig)
    repo = Path(__file__).resolve().parents[2]
    sources = [repo / 'verl/experimental/fully_async_policy/fully_async_router_lag_probe_main.py', repo / 'verl/experimental/fully_async_policy/router_lag_probe.py', repo / 'verl/utils/router_mismatch_metrics.py']
    metadata = {
        'source': str(args.source.resolve()), 'source_sha256': sha256(args.source),
        'script': str(Path(__file__).resolve()), 'script_sha256': sha256(__file__),
        'command': [sys.executable, str(Path(__file__).resolve()), '--source', str(args.source.resolve()), '--output-dir', str(args.output_dir.resolve()), '--palette-validator', str(args.palette_validator.resolve())],
        'model': 'Qwen3-30B-A3B (experiment identity supplied by caller)',
        'tensor_shape': list(ref.shape), 'layer_index_zero_based': layer, 'layer_one_based': layer + 1,
        'probe_id': int(ids[index]), 'row_index': index,
        'record': {k: record[k] for k in ['lag', 'step', 'param_version', 'local_trigger_step', 'v0', 'alignment']},
        'lag_definition': 'Relative parameter-update lag from cached reference; lag = internal probe step - 1. Not global training step or parameter-version difference.',
        'semantics': {'rollout': 'mu_vs_cur.ref_counts: cached rollout mu routes', 'training': 'mu_vs_cur.cur_counts: current-weight plain training-side recompute on the same fixed response', 'alignment': 'mu +1 shifts training route positions; counts use response_mask AND aligned_mask on both sides'},
        'selection': {'rule': 'Fixed lag 16 and layer index 23. Eligible: positive length, valid_tokens and both count sums. Minimize abs(single-layer raw normalized TV - eligible median); ties by smallest probe ID then row index.', 'eligible_count': len(candidates), 'total_count': len(ids), 'median_tv': median, 'selected_tv': float(tv[chosen]), 'distance_to_median': float(abs(tv[chosen] - median)), 'eligible_probe_ids': ids[candidates].tolist(), 'eligible_tv': tv.tolist()},
        'response_length': int(lengths[index]), 'aligned_valid_tokens': float(valid[index]),
        'selected_count_totals': sums[index].tolist(), 'expected_length_times_8': int(lengths[index] * 8), 'expected_valid_tokens_times_8': float(valid[index] * 8),
        'selected_share_sums_pct': pct.sum(axis=1).tolist(),
        'validation': {'finite_nonnegative': True, 'share_sums_100': True, 'closest_to_median': True, 'rows_with_count_total_anomaly': np.flatnonzero(count_bad).tolist(), 'rows_with_mask_length_difference': np.flatnonzero(length_diff).tolist(), 'mask_length_difference_distribution': {str(float(v)): int((length_diff == v).sum()) for v in np.unique(length_diff)}},
        'palette': {'ramp': RAMP, 'surface': SURFACE, 'vmin_pct': 0, 'vmax_pct': vmax, 'checks': palette_check},
        'warnings': warnings,
        'limitations': ['Counts aggregate across response tokens and do not expose token-level rerouting.', 'Selection share is not gate probability, activation amplitude, output difference, or gradient difference.', 'One median-TV sample at one layer and lag is descriptive, not a population or causal conclusion.', 'Training row is current-weight recomputation; no kept/filtered membership is available.'],
        'source_code': {str(p): sha256(p) for p in sources if p.exists()},
        'versions': {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__, 'matplotlib': matplotlib.__version__},
    }
    metadata['command'].extend(['--selection', args.selection])
    metadata['selection'].update({
        'mode': args.selection,
        'rule': f'Fixed source lag {int(record["lag"])} and layer index 23. Eligible: positive length, valid_tokens and both count sums; no minimum-length filter. ' + ('Maximize normalized layer TV' if args.selection == 'max' else 'Minimize distance to median TV') + '; ties by smallest probe ID then row index.',
        'max_tv': float(tv.max()), 'percentile_le_selected': percentile,
        'percentile_definition': '100 * count(eligible TV <= selected TV) / eligible count',
        'eligible_response_lengths': lengths[candidates].tolist(),
        'eligible_valid_tokens': valid[candidates].tolist()})
    metadata['validation']['closest_to_median'] = bool(abs(tv[chosen] - median) == np.min(abs(tv - median)))
    metadata['validation']['maximum_tv'] = bool(tv[chosen] == tv.max())
    metadata['difference'] = {'max_abs_pp': float(np.abs(delta).max()), 'l1_pp': float(np.abs(delta).sum()),
                              'changed_expert_count': int(np.count_nonzero(delta)),
                              'positive_mass_pp': float(delta[delta > 0].sum()),
                              'negative_mass_pp': float(-delta[delta < 0].sum())}
    if args.selection == 'max':
        metadata['palette']['difference'] = {'colors': diverging_colors, 'vmin_pp': -delta_limit, 'vmax_pp': delta_limit}
        metadata['limitations'][2] = 'Maximum-TV sequence deliberately selects an extreme at one layer and lag, not a representative or causal result.'
    prefix.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({'outputs': str(prefix) + '.*', 'probe_id': int(ids[index]), 'tv': float(tv[chosen]), 'median_tv': median, 'warnings': warnings, 'palette_checks': palette_check}, indent=2))


if __name__ == '__main__':
    main()
