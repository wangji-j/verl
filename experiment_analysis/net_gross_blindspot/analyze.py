"""Stream router-lag tensors on CPU and summarize net/gross blind spots."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def stats(rows):
    if not rows:
        return {'n': 0}
    out = {'n': len(rows)}
    for key in ['raw_net', 'net', 'gross', 'overlap', 'logprob']:
        v = np.array([r[key] for r in rows])
        out.update({key + '_mean': float(v.mean()), key + '_median': float(np.median(v)),
                    key + '_p90': float(np.quantile(v, .9))})
    out['overlap_gt_0.1_rate'] = float(np.mean([r['overlap'] > .1 for r in rows]))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, default=Path('/inspire/hdd3/project/qianghuaxuexi/hujiarui-25046/ckpts/router drift control/async-routerlag-probe-from-randomdrop240_20261110_045428/router_lag_probe'))
    p.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    rows, lagrows, checks = [], [], {}
    files = sorted(args.input.glob('probe_lag*.pt'))
    for file in files:
        x = torch.load(file, weights_only=True, map_location='cpu')
        lag = int(x['lag'])
        for arm in ['mu', 'pi0']:
            d = x[arm + '_vs_cur']
            ref, cur = [d[k].to(torch.int64) for k in ['ref_counts', 'cur_counts']]
            valid = d['valid_tokens'].double()
            diff = (ref - cur).abs()
            layerzero = (diff.sum(-1) == 0)
            allzero = layerzero.all(-1)
            raw = (diff.double().sum(-1) / (2 * valid[:, None] * 8)).mean(-1)
            check = checks.setdefault(arm, {'countsum_bad_ref': 0, 'countsum_bad_cur': 0,
                                           'raw_recompute_max_abs_error': 0., 'raw_recompute_bad_atol_1e-6': 0,
                                           'nonpositive_valid_tokens': 0, 'ranges': {}})
            check['nonpositive_valid_tokens'] += int((valid <= 0).sum())
            for key, counts in [('ref', ref), ('cur', cur)]:
                check['countsum_bad_' + key] += int((counts.sum(-1) != valid[:, None] * 8).sum())
            err = (raw - d['d_route_raw'].double()).abs()
            check['raw_recompute_max_abs_error'] = max(check['raw_recompute_max_abs_error'], float(err.max()))
            check['raw_recompute_bad_atol_1e-6'] += int((err > 1e-6).sum())
            fields = {k: d[k] for k in ['d_route', 'd_route_raw', 'flip_rate', 'overlap_mismatch', 'valid_tokens']}
            fields.update(logprob=x['logprob_absdiff_' + arm], lengths=x['lengths'], ref_counts=ref, cur_counts=cur)
            for key, tensor in fields.items():
                a = tensor.double()
                c = check['ranges'].setdefault(key, {'min': float('inf'), 'max': float('-inf'), 'nonfinite': 0})
                finite = torch.isfinite(a)
                c['nonfinite'] += int((~finite).sum())
                if finite.any():
                    c['min'] = min(c['min'], float(a[finite].min()))
                    c['max'] = max(c['max'], float(a[finite].max()))
            batch = []
            for i, pid in enumerate(x['probe_ids'].tolist()):
                r = dict(arm=arm, lag=lag, probe_id=int(pid), length=int(x['lengths'][i]), valid_tokens=int(valid[i]),
                         raw_net=float(d['d_route_raw'][i]), net=float(d['d_route'][i]), gross=float(d['flip_rate'][i]),
                         overlap=float(d['overlap_mismatch'][i]), logprob=float(x['logprob_absdiff_' + arm][i]),
                         all_layer_counts_equal=bool(allzero[i]), zero_layers=int(layerzero[i].sum()), layers=ref.shape[1])
                rows.append(r)
                batch.append(r)
            n = len(batch)
            lagrows.append(dict(arm=arm, lag=lag, n=n, unique_probes=len({r['probe_id'] for r in batch}),
                                exact_zero_count=sum(r['all_layer_counts_equal'] for r in batch),
                                exact_zero_rate=sum(r['all_layer_counts_equal'] for r in batch)/n,
                                exact_zero_positive_gross_count=sum(r['all_layer_counts_equal'] and r['gross'] > 0 for r in batch),
                                exact_zero_positive_gross_rate=sum(r['all_layer_counts_equal'] and r['gross'] > 0 for r in batch)/n,
                                zero_layer_rate=float(layerzero.double().mean()),
                                gross_mean=np.mean([r['gross'] for r in batch]), overlap_mean=np.mean([r['overlap'] for r in batch])))
        del x
    summary = {'files': len(files), 'definition': {'observation': 'one probe at one lag for one reference arm',
               'gross': 'flip_rate: fraction of token-layer top-k sets that changed',
               'overlap': 'overlap_mismatch: replaced-slot fraction', 'raw_net': 'mean layer TV of expert counts',
               'net': 'd_route (tau adjusted)', 'main_excludes_lag0': True,
               'bottom_decile': 'raw_net <= pooled empirical 10th percentile, including ties',
               'caveat': 'Zero layer counts do not identify layer rerouting. Repeated probes/lags are dependent. Gross drift is not evidence of training harm.'},
               'checks': checks, 'arms': {}}
    group_rows, examples = [], []
    for arm in ['mu', 'pi0']:
        rr = [r for r in rows if r['arm'] == arm and r['lag'] != 0]
        threshold = float(np.quantile([r['raw_net'] for r in rr], .1))
        a = dict(observations=len(rr), unique_probes=len({r['probe_id'] for r in rr}),
                 unique_lags=len({r['lag'] for r in rr}), exact_zero_count=sum(r['all_layer_counts_equal'] for r in rr),
                 exact_zero_positive_gross_count=sum(r['all_layer_counts_equal'] and r['gross'] > 0 for r in rr),
                 zero_layers=sum(r['zero_layers'] for r in rr), total_layers=sum(r['layers'] for r in rr),
                 bottom_decile_raw_net_threshold=threshold)
        a['exact_zero_rate'] = a['exact_zero_count']/len(rr)
        a['exact_zero_positive_gross_rate'] = a['exact_zero_positive_gross_count']/len(rr)
        a['zero_layer_rate'] = a['zero_layers']/a['total_layers']
        groups = {'overall': rr}
        for t in [.001, .005, .01]:
            groups[f'raw_net<={t}'] = [r for r in rr if r['raw_net'] <= t]
        groups['bottom_decile'] = [r for r in rr if r['raw_net'] <= threshold]
        a['groups'] = {k: stats(v) for k, v in groups.items()}
        for key, s in a['groups'].items():
            group_rows.append(dict(arm=arm, group=key, **s))
        a['correlations'] = {}
        for net in ['raw_net', 'net']:
            for target in ['gross', 'overlap', 'logprob']:
                v, w = [[r[k] for r in rr] for k in [net, target]]
                a['correlations'][net + '__' + target] = dict(pearson=float(pearsonr(v, w).statistic), spearman=float(spearmanr(v, w).statistic))
        for group in ['raw_net<=0.01', 'bottom_decile']:
            for r in sorted(groups[group], key=lambda r: -r['gross'])[:10]:
                examples.append(dict(selection=group, **r))
        summary['arms'][arm] = a
    write_csv(args.output/'observations.csv', rows)
    write_csv(args.output/'by_lag.csv', lagrows)
    # Empty groups have only n; normalize CSV keys.
    keys = list(dict.fromkeys(k for r in group_rows for k in r))
    write_csv(args.output/'groups.csv', [{k: r.get(k, '') for k in keys} for r in group_rows])
    write_csv(args.output/'examples.csv', examples)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
