"""Observed-node extrema from daily forecasts; calendar completeness is explicit."""
from if_release.runtime import expand as _release_expand
from pathlib import Path
from datetime import datetime, timedelta
import json, csv
import numpy as np
ROOT = Path(_release_expand('${WORKSPACE}'))
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'outputs' / HERE.name
models = ['S0', 'Mamba', 'PCFNet', 'ModernTCN', 'VanillaLSTM', 'DLinear', 'TimeXer', 'PatchTST', 'iTransformer', 'TimesNet']
records = []
coverage = []
truth = np.load(OUT / 'dataset/target_physical.npy')[110:]
for model in models:
    for seed in range(2024, 2029):
        z = np.load(OUT / f'{model}_seed{seed}/test_predictions.npz')
        origin = z['origin']
        mean = z['mean']
        std = z['std']
        mask = z['mask'][:, :, 1:]
        predicted = z['prediction'].astype(np.float64) * std + mean
        observed = truth
        dates = [datetime(2022, 1, 1) + timedelta(minutes=int(o)) for o in origin]
        groups = {'daily': {d.strftime('%Y-%m-%d'): [i] for i, d in enumerate(dates)}, 'monthly': {month: [i for i, d in enumerate(dates) if d.strftime('%Y-%m') == month] for month in sorted({d.strftime('%Y-%m') for d in dates})}, 'quarterly': {quarter: [i for i, d in enumerate(dates) if f'{d.year}-Q{(d.month - 1) // 3 + 1}' == quarter] for quarter in sorted({f'{d.year}-Q{(d.month - 1) // 3 + 1}' for d in dates})}, 'annual': {str(year): [i for i, d in enumerate(dates) if d.year == year] for year in sorted({d.year for d in dates})}}
        for scale, periods in groups.items():
            for period, days in periods.items():
                d = dates[days[0]]
                if scale == 'daily':
                    expected = 1
                elif scale == 'monthly':
                    nextmonth = datetime(d.year + (d.month == 12), d.month % 12 + 1, 1)
                    expected = (nextmonth - datetime(d.year, d.month, 1)).days
                elif scale == 'quarterly':
                    month = (d.month - 1) // 3 * 3 + 1
                    qstart = datetime(d.year, month, 1)
                    qend = datetime(d.year + int(month == 10), (month + 2) % 12 + 1, 1)
                    expected = (qend - qstart).days
                else:
                    expected = (datetime(d.year + 1, 1, 1) - datetime(d.year, 1, 1)).days
                complete = len(days) == expected
                for c in range(4):
                    valid = mask[days, :, c]
                    yy = observed[days, :, c + 1][valid]
                    pp = predicted[days, :, c + 1][valid]
                    absolute_minutes = (origin[days, None] + np.arange(1440)[None, :])[valid]
                    for extreme, op in [('maximum', np.argmax), ('minimum', np.argmin)]:
                        iy = int(op(yy))
                        ip = int(op(pp))
                        records.append(dict(model=model, seed=seed, scale=scale, period=period, pier=c + 1, complete_calendar=complete, covered_days=len(days), expected_days=expected, observed_nodes=int(valid.sum()), expected_nodes=expected * 1440, observed_coverage=float(valid.sum() / (expected * 1440)), extreme=extreme, observed_extreme_mm=float(yy[iy]), predicted_extreme_mm=float(pp[ip]), signed_error_mm=float(pp[ip] - yy[iy]), absolute_error_mm=float(abs(pp[ip] - yy[iy])), occurrence_time_error_hours=float(abs(int(absolute_minutes[ip]) - int(absolute_minutes[iy])) / 60)))
                if model == 'S0' and seed == 2024:
                    coverage.append(dict(scale=scale, period=period, covered_days=len(days), expected_days=expected, complete_calendar=complete, measured_nodes=int(mask[days].sum()), measured_nodes_by_pier=mask[days].sum(axis=(0, 1)).astype(int).tolist(), note='Extrema use only common retained measured nodes, so missing observations may conceal the true full-period extremum.'))
with (OUT / 'test_extrema_details.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(records[0]))
    w.writeheader()
    w.writerows(records)
(OUT / 'test_extrema_coverage.json').write_text(json.dumps(coverage, indent=2))
print('Extrema rows', len(records), 'calendar complete periods', [(r['scale'], r['period']) for r in coverage if r['complete_calendar'] and r['scale'] != 'daily'])
