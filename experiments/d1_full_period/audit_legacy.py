"""Compare completed inference on the preserved original evaluation cohorts."""
from if_release.runtime import expand as _release_expand, loads as _release_json_loads
from pathlib import Path
import json, statistics
ROOT = Path(_release_expand('${WORKSPACE}'))
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'outputs' / HERE.name
plan = _release_json_loads((HERE / 'plan.json').read_text())
reference = _release_json_loads((HERE / 'frozen_reference_comparison.json').read_text())['statistics']['D1']
abla = _release_json_loads((HERE / 'frozen_reference_ablations.json').read_text())['rows']
details = []
for job in plan['jobs']:
    result = _release_json_loads((OUT / job['key'] / 'metrics.json').read_text())
    if job['family'] == 'framework':
        old = next((r for r in abla if r['arm'] == job['model'] and r['seed'] == job['seed']))
        new = result['metrics']['validation']['legacy_frozen_cohort']['response_rmse_mm']
        details.append(dict(kind='ablation_validation_per_seed', model=job['model'], seed=job['seed'], reference_rmse_mm=old['response_rmse_mm'], current_rmse_mm=new, difference_mm=new - old['response_rmse_mm']))
for model in reference:
    current_model = 'S0' if model == 'A2' else model
    for split in ['validation', 'test']:
        new = statistics.mean((_release_json_loads((OUT / f'{current_model}_seed{s}/metrics.json').read_text())['metrics'][split]['legacy_frozen_cohort']['response_rmse_mm'] for s in range(2024, 2029)))
        old = reference[model][split]['response_rmse_mm']['mean']
        details.append(dict(kind='main_' + split + '_five_seed_mean', model=model, reference_rmse_mm=old, current_rmse_mm=new, difference_mm=new - old))
summary = dict(details=details, maximum_absolute_difference_mm=max((abs(r['difference_mm']) for r in details)), interpretation='Same weights and original cohort. Small floating-point differences can occur under a different GPU and inference batching; no weights or checkpoints were selected on these comparisons.')
(OUT / 'legacy_reproduction_audit.json').write_text(json.dumps(summary, indent=2))
print('Maximum legacy RMSE difference/mm:', summary['maximum_absolute_difference_mm'])
