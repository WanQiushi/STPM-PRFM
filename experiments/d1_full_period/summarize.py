"""Summarize completed full219 inference, preserving per-seed and cohort results."""
from if_release.runtime import expand as _release_expand, loads as _release_json_loads
from pathlib import Path
import json, csv, statistics
ROOT = Path(_release_expand('${WORKSPACE}'))
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'outputs' / HERE.name
plan = _release_json_loads((HERE / 'plan.json').read_text())
labels = {'S0': 'Proposed', 'S1': 'No temperature ODE', 'S2': 'No temperature residual', 'S3': 'No trend supervision', 'S4': 'No thermal lag', 'S5': 'No BW physical rate', 'S6': 'No forecast-temperature adaptation', 'ModernTCN': 'TCN', 'VanillaLSTM': 'LSTM'}
rows = []
raw = []
locations = []
hours = []
fields = ['temperature_rmse_c', 'temperature_mae_c', 'response_rmse_mm', 'response_mae_mm', 'response_mse_mm2']
for job in plan['jobs']:
    result = _release_json_loads((OUT / job['key'] / 'metrics.json').read_text())
    raw.append(result)
    for split, cohorts in result['metrics'].items():
        for cohort, metrics in cohorts.items():
            rows.append(dict(model=job['model'], label=labels.get(job['model'], job['model']), family=job['family'], seed=job['seed'], split=split, cohort=cohort, observed_temperature_nodes=metrics['measured_counts'][0], observed_response_nodes=sum(metrics['measured_counts'][1:]), **{k: metrics[k] for k in ['windows'] + fields}))
            for c in range(4):
                locations.append(dict(model=job['model'], seed=job['seed'], split=split, cohort=cohort, pier=c + 1, rmse_mm=metrics['channel_rmse_mm'][c], mae_mm=metrics['channel_mae_mm'][c], observed_nodes=metrics['measured_counts'][c + 1]))
            for hour, value in enumerate(metrics['response_hourly_rmse_mm'], 1):
                hours.append(dict(model=job['model'], seed=job['seed'], split=split, cohort=cohort, forecast_hour=hour, response_rmse_mm=value))
with (OUT / 'per_seed_metrics.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
for name, records in [('per_location_metrics.csv', locations), ('per_horizon_metrics.csv', hours)]:
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
summary = {}
for model in dict.fromkeys((j['model'] for j in plan['jobs'])):
    summary[model] = {}
    for split in ['validation', 'test']:
        summary[model][split] = {}
        for cohort in ['all_days', 'original_causal_history', 'reconstructed_history', 'legacy_frozen_cohort']:
            group = [r for r in rows if r['model'] == model and r['split'] == split and (r['cohort'] == cohort)]
            summary[model][split][cohort] = {k: dict(mean=statistics.mean((r[k] for r in group)), sample_std=statistics.stdev((r[k] for r in group))) for k in fields}
result = dict(protocol='d1_full219_asof_linear_history_v1', training_performed=False, weights_reselected=False, seeds=list(range(2024, 2029)), per_seed=rows, statistics=summary, provenance=raw)
(OUT / 'comparison.json').write_text(json.dumps(result, indent=2))
main = ['S0', 'Mamba', 'PCFNet', 'ModernTCN', 'VanillaLSTM', 'DLinear', 'TimeXer', 'PatchTST', 'iTransformer', 'TimesNet']
ablation = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6']

def fmt(model, split, metric):
    v = summary[model][split]['all_days'][metric]
    return f"{v['mean']:.3f} +/- {v['sample_std']:.3f}"
lines = ['# D1 full110validation/109test days: frozen-weight evaluation', '', 'Protocol d1_full219_asof_linear_history_v1. Fixed original checkpoints, no training/reselection, five-seed mean and sampleSD.', 'Historical entries unavailable at forecast origin are reconstructed using only past available anchors. Original measured labels only; no ensemble.', 'This is a new evaluation protocol; original26/39 results are preserved and reported separately in comparison.json.', '']
for title, models in [('Main comparison', main), ('Core ablations', ablation)]:
    lines += ['## ' + title, '', '|Method|Validation response RMSE/mm|Test response RMSE/mm|Validation response MAE/mm|Test response MAE/mm|Validation temperature RMSE/C|Test temperature RMSE/C|', '|---|---:|---:|---:|---:|---:|---:|']
    for model in models:
        vals = [fmt(model, split, metric) for metric, split in [('response_rmse_mm', 'validation'), ('response_rmse_mm', 'test'), ('response_mae_mm', 'validation'), ('response_mae_mm', 'test'), ('temperature_rmse_c', 'validation'), ('temperature_rmse_c', 'test')]]
        lines.append('|' + labels.get(model, model) + '|' + '|'.join(vals) + '|')
    lines.append('')
(OUT / 'results_table.md').write_text('\n'.join(lines))
flat = []
for model in main + ablation[1:]:
    for split in ['validation', 'test']:
        row = dict(method=labels.get(model, model), model=model, split=split, days=110 if split == 'validation' else 109)
        for metric in fields:
            row[metric + '_mean'] = summary[model][split]['all_days'][metric]['mean']
            row[metric + '_sample_sd'] = summary[model][split]['all_days'][metric]['sample_std']
        flat.append(row)
with (OUT / 'summary_metrics.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(flat[0]))
    w.writeheader()
    w.writerows(flat)
print('\n'.join(lines))
