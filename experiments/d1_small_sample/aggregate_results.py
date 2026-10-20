"""Summarize completed jobs in physical units; this entry point never trains."""
from if_release.runtime import loads as _release_json_loads
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
PROJECT = Path(__file__).resolve().parent
SCALARS = ['temperature_rmse_c', 'temperature_mae_c', 'pooled_rmse_mm', 'pooled_mae_mm', 'joint_standardized_rmse', 'parameters', 'elapsed_seconds']

def flatten(summary, split, run_directory):
    metrics = summary['results'][split]
    if summary['model'] == 'A2':
        response, temperature = (metrics['displacement'], metrics['temperature'])
        protocol = _release_json_loads((run_directory / 'protocol.json').read_text(encoding='utf-8'))
        std = protocol['std']
        counts = temperature['measured_counts'] + response['measured_counts']
        rmse = [temperature['rmse']] + response['channel_rmse']
        normalized_sse = sum(((r / s) ** 2 * n for r, s, n in zip(rmse, std, counts)))
        result = {'temperature_rmse_c': temperature['rmse'], 'temperature_mae_c': temperature['mae'], 'pooled_rmse_mm': response['rmse'], 'pooled_mae_mm': response['mae'], 'joint_standardized_rmse': math.sqrt(normalized_sse / sum(counts)), 'rmse_mm': response['channel_rmse'], 'mae_mm': response['channel_mae'], 'measured_counts': counts, 'windows': response['windows']}
    else:
        result = dict(metrics)
    result.update(parameters=summary['parameters'], elapsed_seconds=summary['elapsed_seconds'])
    return result

def aggregate(output, models=None):
    index = _release_json_loads((PROJECT / 'job_index.json').read_text(encoding='utf-8'))
    jobs = [job for job in index['jobs'] if models is None or job['model'] in models]
    completed, statuses = ([], [])
    for job in jobs:
        path = PROJECT / job['output'] / 'summary.json'
        if not path.exists():
            statuses.append({**job, 'status': 'no_saved_summary'})
            continue
        summary = _release_json_loads(path.read_text(encoding='utf-8'))
        statuses.append({**job, 'status': summary['status']})
        if summary['status'] != 'completed':
            continue
        for split in ['validation', 'test']:
            completed.append({'budget': job['budget'], 'month': job['month'], 'model': job['model'], 'seed': job['seed'], 'split': split, **flatten(summary, split, path.parent)})
    groups = []
    for month in index['months']:
        for model in index['models']:
            if models is not None and model not in models:
                continue
            for split in ['validation', 'test']:
                rows = [row for row in completed if row['month'] == month and row['model'] == model and (row['split'] == split)]
                if not rows:
                    continue
                group = {'budget': f'{month}m', 'month': month, 'model': model, 'split': split, 'n_seeds': len(rows), 'complete_five_seed_group': len(rows) == 5, 'seeds': [row['seed'] for row in rows]}
                for metric in SCALARS:
                    values = [row[metric] for row in rows]
                    group[metric + '_mean'] = statistics.mean(values)
                    group[metric + '_std'] = statistics.stdev(values) if len(values) > 1 else None
                for channel in range(4):
                    for metric in ['rmse_mm', 'mae_mm']:
                        values = [row[metric][channel] for row in rows]
                        group[f'{metric}_channel{channel + 1}_mean'] = statistics.mean(values)
                        group[f'{metric}_channel{channel + 1}_std'] = statistics.stdev(values) if len(values) > 1 else None
                groups.append(group)
    for group in groups:
        reference = next((row for row in groups if row['month'] == 9 and row['model'] == group['model'] and (row['split'] == group['split'])), None)
        if reference is not None:
            group['rmse_increase_vs_9m_percent'] = 100 * (group['pooled_rmse_mm_mean'] / reference['pooled_rmse_mm_mean'] - 1)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'results.json').write_text(json.dumps({'seed_std_ddof': 1, 'planned_jobs': len(jobs), 'selected_models': models, 'statuses': statuses, 'per_seed': completed, 'groups': groups, 'cost_policy': 'A2 elapsed time includes all fresh OOF/producer/response stages and evaluation; baseline elapsed time is its complete run.'}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for name, rows in [('per_seed', completed), ('five_seed_groups', groups)]:
        if rows:
            fields = list(dict.fromkeys((key for row in rows for key in row)))
            with (output / (name + '.csv')).open('w', encoding='utf-8-sig', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
    print(f'Saved {len(completed) // 2} completed runs and {len(groups)} groups to {output}.')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=PROJECT / 'summary')
    parser.add_argument('--models', nargs='+')
    args = parser.parse_args()
    aggregate(args.output, args.models)
if __name__ == '__main__':
    main()
