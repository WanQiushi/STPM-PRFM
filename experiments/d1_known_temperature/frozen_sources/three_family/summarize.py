"""Aggregate independently scored seeds; D1 native1min, D21 matched15min."""
from if_release.runtime import expand as _release_expand, loads as _release_json_loads
from pathlib import Path
import json, statistics
ROOT = Path(_release_expand('${WORKSPACE}'))
OUT = ROOT / 'outputs' / Path(__file__).resolve().parent.name
plan = _release_json_loads((OUT / 'plan.json').read_text())
rows = []
for job in plan['jobs']:
    folder = Path(job['output'])
    if job['dataset'] == 'D1':
        scores = _release_json_loads((folder / 'suite_results.json').read_text())['results']['accuracy']
        source = folder / 'suite_results.json'
        for split in ['validation', 'test']:
            m = scores[split]
            rows.append(dict(dataset='D1', model=job['model'], seed=job['seed'], split=split, grid_minutes=1, temperature_rmse_c=m['temperature_rmse_c'], response_rmse_mm=m['pooled_rmse_mm'], source=str(source)))
    else:
        scores = _release_json_loads((folder / 'grid_scores.json').read_text())
        source = folder / 'grid_scores.json'
        for split in ['validation', 'test']:
            m = scores[split]['15']
            rows.append(dict(dataset='D21', model=job['model'], seed=job['seed'], split=split, grid_minutes=15, temperature_rmse_c=m['temperature_rmse_c'], response_rmse_mm=m['pooled_rmse_mm'], source=str(source)))
stats = {}
for dataset in ['D1', 'D21']:
    stats[dataset] = {}
    for model in ['VanillaLSTM', 'Mamba', 'PCFNet']:
        stats[dataset][model] = {}
        for split in ['validation', 'test']:
            selected = [r for r in rows if r['dataset'] == dataset and r['model'] == model and (r['split'] == split)]
            stats[dataset][model][split] = {key: dict(mean=statistics.mean((r[key] for r in selected)), sample_std=statistics.stdev((r[key] for r in selected))) for key in ['temperature_rmse_c', 'response_rmse_mm']}
(OUT / 'comparison.json').write_text(json.dumps(dict(per_seed=rows, statistics=stats, aggregation='Five independent seeds2024-2028 mean/sample SD; no ensemble', plan=str(OUT / 'plan.json')), indent=2), encoding='utf-8')
lines = ['# Raw LSTM, original Mamba1 and PCFNet: five-seed results', '']
for dataset in ['D1', 'D21']:
    lines.extend([dataset + (':26val/39test one-minute support-displacement scores' if dataset == 'D1' else ':60val/60test15-minute deflection scores'), '', '|Model|Val temperature RMSE/C|Val response RMSE/mm|Test temperature RMSE/C|Test response RMSE/mm|', '|---|---:|---:|---:|---:|'])
    for model in ['VanillaLSTM', 'Mamba', 'PCFNet']:
        s = stats[dataset][model]
        v, t = (s['validation'], s['test'])
        values = [v['temperature_rmse_c'], v['response_rmse_mm'], t['temperature_rmse_c'], t['response_rmse_mm']]
        lines.append('|' + model + '|' + '|'.join((f"{x['mean']:.6f} +/- {x['sample_std']:.6f}" for x in values)) + '|')
    lines.append('')
lines.extend(['PCFNet uses the author backbone with a numerically equivalent parallel-prefix EMA; no calendar. Original Mamba1 and standard LSTM have explicitly documented direct forecasting heads. Existing baselines and A2 are not retrained here.'])
(OUT / 'results_table.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
