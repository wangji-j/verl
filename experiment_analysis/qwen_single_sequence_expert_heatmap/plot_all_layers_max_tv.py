#!/usr/bin/env python3
"""CPU-only maximum over all valid sequence/layer pairs at lag 16 and 128.
Run: python plot_all_layers_max_tv.py
Reuses the existing heatmap's data location, style, and provenance helpers.
"""
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


def summary(values):
    return dict(zip(['min', 'p25', 'median', 'p75', 'max'],
                    np.percentile(values, [0, 25, 50, 75, 100]).tolist()))


def main():
    candidates, snapshots, selected_data = [], [], {}
    for lag in (16, 128):
        path = base.DEFAULT_SOURCE.with_name(f'probe_lag{lag:04d}.pt')
        record = torch.load(path, weights_only=True, map_location='cpu')
        assert int(record['lag']) == lag
        comp = record['mu_vs_cur']
        raw = [comp[k].numpy() for k in ['ref_counts', 'cur_counts']]
        for array in raw:
            assert np.isfinite(array).all() and (array >= 0).all()
            assert np.equal(array, np.floor(array)).all()
        ref, cur = [a.astype(np.int64) for a in raw]
        assert ref.shape == cur.shape and ref.shape[1:] == (48, 128)
        ids, lengths = record['probe_ids'].numpy(), record['lengths'].numpy()
        valid = comp['valid_tokens'].numpy()
        assert ids.shape == lengths.shape == valid.shape == (len(ref),)
        assert np.isfinite(lengths).all() and np.isfinite(valid).all()
        assert ((valid >= 0) & (valid <= lengths)).all()
        totals = np.stack([ref.sum(-1), cur.sum(-1)], axis=-1)
        eligible = (valid[:, None] > 0) & (lengths[:, None] > 0) & (totals > 0).all(-1)
        shares = [np.divide(a, a.sum(-1, keepdims=True), out=np.zeros_like(a, dtype=float),
                            where=a.sum(-1, keepdims=True) > 0) for a in (ref, cur)]
        tv = np.abs(shares[1] - shares[0]).sum(-1) / 2
        bad = (totals != valid[:, None, None] * 8).any(-1)
        for row, layer in np.argwhere(eligible):
            candidates.append({'lag': lag, 'probe_id': int(ids[row]), 'row_index': int(row),
                               'layer_one_based': int(layer + 1), 'layer_index_zero_based': int(layer),
                               'tv': float(tv[row, layer]), 'response_length': int(lengths[row]),
                               'valid_tokens': int(valid[row])})
        rows = eligible.any(-1)
        snapshots.append({'lag': lag, 'source': str(path), 'sha256': base.sha256(path),
                          'tensor_shape': list(ref.shape), 'eligible_pairs': int(eligible.sum()),
                          'excluded_pairs': int(eligible.size - eligible.sum()),
                          'eligible_sequences': int(rows.sum()),
                          'response_length_summary': summary(lengths[rows]),
                          'valid_tokens_summary': summary(valid[rows]),
                          'eligible_sequences_below_32_valid_tokens': int(((valid < 32) & rows).sum()),
                          'count_total_anomalies_all_pairs': int(bad.sum()),
                          'count_total_anomalies_eligible_pairs': int((bad & eligible).sum()),
                          'length_minus_valid_distribution': {str(int(v)): int(((lengths-valid) == v).sum()) for v in np.unique(lengths-valid)}})
        selected_data[lag] = (ref, cur, totals)
    candidates.sort(key=lambda x: (-x['tv'], x['lag'], x['probe_id'], x['row_index'], x['layer_index_zero_based']))
    chosen = candidates[0]
    lag, row, layer = chosen['lag'], chosen['row_index'], chosen['layer_index_zero_based']
    ref, cur, totals = selected_data[lag]
    counts = np.stack([ref[row, layer], cur[row, layer]])
    pct = counts / counts.sum(-1, keepdims=True) * 100
    delta = pct[1] - pct[0]
    np.testing.assert_allclose(pct.sum(-1), 100)
    np.testing.assert_allclose(abs(delta).sum()/200, chosen['tv'])
    top = np.lexsort((np.arange(128), -abs(delta)))[:12]
    layer24 = [next(c for c in candidates if c['lag'] == lag_value and c['layer_one_based'] == 24) for lag_value in (16, 128)]
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
    checks, ok = validator.validate([poles[0], poles[-1]], 'light', base.SURFACE)
    assert ok, checks
    contrasts = {c: validator.contrast(c, base.SURFACE) for c in (base.INK, base.SECONDARY)}
    assert min(contrasts.values()) >= 4.5
    vmax, dmax = float(np.ceil(pct.max())), float(abs(delta).max())
    prefix = Path(__file__).with_name(f'qwen_all_layers_lag16_128_max_tv_layer{layer+1}_lag{lag}_probe{chosen["probe_id"]}')
    with prefix.with_suffix('.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['expert_id_zero_based', 'rollout_count', 'training_count', 'rollout_share_pct', 'training_share_pct', 'training_minus_rollout_pp'])
        writer.writerows([e, int(counts[0,e]), int(counts[1,e]), pct[0,e], pct[1,e], delta[e]] for e in range(128))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'text.color': base.INK,
                         'axes.labelcolor': base.SECONDARY, 'xtick.color': base.SECONDARY,
                         'ytick.color': base.INK, 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(18, 11), facecolor=base.SURFACE)
    fig.text(.10, .955, 'Qwen3-30B-A3B | Largest expert-usage drift in two snapshots', fontsize=22, weight='semibold')
    fig.text(.10, .915, f'Probe {chosen["probe_id"]}  |  Layer {layer+1} / 48 (index {layer})  |  Lag {lag}  |  TV = {chosen["tv"]:.6f}', fontsize=15)
    fig.text(.10, .88, f'{chosen["valid_tokens"]:,} aligned / {chosen["response_length"]:,} response tokens  |  {len(candidates):,} valid sequence-layer pairs  |  No minimum-length filter', color=base.SECONDARY)
    ax = fig.add_axes([.10, .73, .85, .115], facecolor=base.SURFACE)
    dax = fig.add_axes([.10, .65, .85, .052], facecolor=base.SURFACE)
    image = ax.imshow(pct, cmap=cmap, norm=Normalize(0, vmax), aspect='auto', interpolation='nearest')
    ax.axhline(.5, color=base.SURFACE, lw=2)
    ax.set_yticks([0,1], ['Rollout', 'Training'])
    ax.set_xticks([])
    dim = dax.imshow(delta[None], cmap=dcmap, norm=Normalize(-dmax, dmax), aspect='auto', interpolation='nearest')
    dax.set_yticks([0], ['Training\n− Rollout'])
    dax.set_xticks(list(range(0,128,8)) + [127])
    dax.set_xlabel('Expert ID (zero-based); all 128 experts in original order', labelpad=8)
    for axis in (ax, dax):
        axis.tick_params(length=0, pad=8)
        for spine in axis.spines.values():
            spine.set_visible(False)
    for im, rect, ticks, label in [
        (image, [.10,.555,.37,.018], np.linspace(0,vmax,5), 'Selection share (%) | shared rollout / training scale'),
        (dim, [.58,.555,.37,.018], np.linspace(-dmax,dmax,5), 'Training − rollout (pp) | symmetric around zero')]:
        cb = fig.colorbar(im, cax=fig.add_axes(rect), orientation='horizontal', ticks=ticks)
        cb.set_label(label, fontsize=11)
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=0, labelsize=10)
    fig.text(.10,.465,'12 experts with the largest absolute share changes', fontsize=16, weight='semibold')
    fig.text(.10,.435,'Same expert order in both panels; labels show rollout → training counts and signed share change.', fontsize=11, color=base.SECONDARY)
    left = fig.add_axes([.10,.12,.37,.285], facecolor=base.SURFACE)
    right = fig.add_axes([.58,.12,.23,.285], facecolor=base.SURFACE)
    y = np.arange(12)
    for i,e in enumerate(top):
        left.plot(pct[:,e], [i,i], color='#c9c8c4', lw=2, zorder=1)
    left.scatter(pct[0,top],y, s=60, marker='o', facecolors=base.SURFACE, edgecolors=base.RAMP[-2], linewidths=1.8, label='Rollout', zorder=3)
    left.scatter(pct[1,top],y, s=65, marker='s', facecolors=base.RAMP[-2], edgecolors=base.SURFACE, linewidths=1, label='Training', zorder=2)
    left.set_xlim(0, vmax * 1.03)
    left.set_xlabel('Selection share (%)')
    left.legend(frameon=False, loc='lower right', fontsize=10)
    right.barh(y, delta[top], color=[dcmap(Normalize(-dmax,dmax)(v)) for v in delta[top]], height=.48)
    right.axvline(0, color='#c9c8c4', lw=1)
    right.set_xlim(-dmax*1.08, dmax*1.08)
    for i,e in enumerate(top):
        right.text(1.04,i,f'{delta[e]:+.3f} pp   ({counts[0,e]:,} → {counts[1,e]:,})',
                   transform=right.get_yaxis_transform(), va='center', fontsize=10, clip_on=False)
    right.set_xlabel('Training − rollout (percentage points)')
    for axis in (left,right):
        axis.set_yticks(y, [f'E{e}' for e in top])
        axis.set_ylim(11.65,-.65)
        axis.tick_params(length=0, pad=5)
        for spine in axis.spines.values():
            spine.set_visible(False)
    fig.text(.10,.035,'Scope: lag 16 and 128 only, all 48 layers. Each side is normalized by its own total; no separate color scales.\nCounts summarize aligned response tokens, not token-level rerouting, gate amplitudes, outputs, or gradients. This is a selected extreme.', fontsize=10, color=base.SECONDARY, linespacing=1.5)
    fig.savefig(prefix.with_suffix('.png'),dpi=200,facecolor=base.SURFACE)
    fig.savefig(prefix.with_suffix('.pdf'),facecolor=base.SURFACE)
    plt.close(fig)
    metadata = {'script': str(Path(__file__).resolve()), 'script_sha256': base.sha256(__file__),
                'reused_script': str(Path(base.__file__).resolve()), 'reused_script_sha256': base.sha256(base.__file__),
                'command': [sys.executable, str(Path(__file__).resolve())],
                'selection_rule': 'Maximize 0.5*sum(abs(ref_counts/ref_total-cur_counts/cur_total)) across lag 16 and 128, every row and all 48 layers. Eligible: positive response length, valid tokens, and both layer totals. No minimum-length filtering. Ties: lag, probe ID, row, layer index ascending.',
                'scope': {'lags': [16,128], 'layers_one_based': [1,48], 'eligible_pairs':len(candidates), 'not_scanned': 'All other lags; this is not a maximum over all 129 snapshots.'},
                'snapshots': snapshots, 'selected': chosen, 'top_20_candidates': candidates[:20],
                'layer24_maxima': layer24, 'maximum_minus_best_layer24_tv': chosen['tv']-max(c['tv'] for c in layer24),
                'selected_counts': {'rollout':counts[0].tolist(), 'training':counts[1].tolist(), 'totals':counts.sum(-1).tolist(), 'expected_valid_tokens_times_8':chosen['valid_tokens']*8},
                'validation': {'finite_nonnegative_integer_counts': True, 'share_sums_pct':pct.sum(-1).tolist(), 'tv_recomputed':float(abs(delta).sum()/200), 'selected_totals_equal_valid_times_8':bool((counts.sum(-1)==chosen['valid_tokens']*8).all())},
                'top_12_experts': [{'expert_id':int(e), 'rollout_count':int(counts[0,e]), 'training_count':int(counts[1,e]), 'rollout_share_pct':float(pct[0,e]), 'training_share_pct':float(pct[1,e]), 'delta_pp':float(delta[e])} for e in top],
                'palette': {'sequential':base.RAMP,'sequential_range_pct':[0,vmax], 'difference':poles,'difference_range_pp':[-dmax,dmax], 'sequential_monotone_OKLab_L':True,'diverging_monotone_arms':True,'neutral_midpoint':True,'pole_checks':checks,'text_contrast':contrasts,'validator':str(base.DEFAULT_VALIDATOR), 'relief':'CSV, signed direct labels, and distinct rollout/training marker shapes'},
                'limitations': ['Selected extreme over two snapshots only; no population or causal claim.', 'Aggregated expert counts cannot identify token-level rerouting or output/gradient differences.', 'Training means current-weight recomputation on the fixed response. Lag means relative parameter updates.'],
                'versions':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'matplotlib':base.matplotlib.__version__}}
    prefix.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({'outputs':str(prefix)+'.*','selected':chosen,'snapshots':snapshots,'layer24_maxima':layer24,'top_12_experts':metadata['top_12_experts'],'top_20_candidates':candidates[:20]},indent=2))


if __name__ == '__main__':
    main()
