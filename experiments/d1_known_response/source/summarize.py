"""Summaries and predeclared calendar-block paired uncertainty after all45runs."""
from if_release.runtime import loads as _release_json_loads
from pathlib import Path
import csv
import json
import math
import numpy as np
OUT = Path(__file__).resolve().parent.parent
ARMS = [f'O{i}' for i in range(9)]
SEEDS = list(range(2024, 2029))

def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')

def main():
    scores = []
    rows = []
    arrays = []
    parameters = {}
    for arm in ARMS:
        arm_arrays = []
        arm_scores = []
        for seed in SEEDS:
            folder = OUT / f'{arm}_seed{seed}'
            result = _release_json_loads((folder / 'summary.json').read_text())
            metrics = result['metrics']['test_1min']
            rows.append(dict(arm=arm, seed=seed, **{k: metrics[k] for k in ['rmse_mm', 'mae_mm', 'mse_mm2']}, selected_epoch=result['selected_epoch'], elapsed_seconds=result['elapsed_seconds']))
            arm_scores.append(metrics['rmse_mm'])
            a = np.load(folder / 'test_1min_daily_errors.npz')
            day = a['origin'] // 1440 - 621
            daily = np.zeros((109, 3))
            daily[day] = np.stack([a['sse'].sum(1), a['sae'].sum(1), a['n'].sum(1)], axis=-1)
            arm_arrays.append(daily)
            parameters[f'{arm}_seed{seed}'] = _release_json_loads((folder / 'selected_parameters.json').read_text())
        arrays.append(np.stack(arm_arrays))
        scores.append(dict(arm=arm, rmse_mm_mean=float(np.mean(arm_scores)), rmse_mm_sample_sd=float(np.std(arm_scores, ddof=1))))
    with (OUT / 'per_seed_metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    dump(OUT / 'selected_parameter_comparison.json', parameters)
    a = np.stack(arrays)
    paired = {}
    all_bootstrap = {}
    measured = a[0, 0, :, 2] > 0
    daily_mse = np.divide(a[:, :, :, 0], a[:, :, :, 2], out=np.zeros_like(a[:, :, :, 0]), where=a[:, :, :, 2] > 0)
    mean_daily_effect = (daily_mse - daily_mse[0:1]).mean(1)
    for block in [7, 3, 14]:
        rng = np.random.default_rng(20261020)
        starts = rng.integers(0, 109 - block + 1, size=(10000, math.ceil(109 / block)))
        indices = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(10000, -1)[:, :109]
        bootstrap = []
        for arm_index in range(9):
            data = a[arm_index]
            totals = data[:, indices].sum(2)
            rmse = np.sqrt(totals[:, :, 0] / totals[:, :, 2]).mean(0)
            bootstrap.append(rmse)
        bootstrap = np.stack(bootstrap)
        for i, arm in enumerate(ARMS[1:], start=1):
            difference = bootstrap[i] - bootstrap[0]
            eligible = measured[indices]
            effects = mean_daily_effect[i, indices]
            day_mse_difference = (effects * eligible).sum(1) / eligible.sum(1)
            win_fraction = ((effects > 0) * eligible).sum(1) / eligible.sum(1)
            all_bootstrap.setdefault(arm, {})[str(block)] = dict(delta_pooled_rmse_mm_ci95=np.quantile(difference, [0.025, 0.975]).tolist(), delta_mean_daily_mse_mm2_ci95=np.quantile(day_mse_difference, [0.025, 0.975]).tolist(), full_better_day_fraction_ci95=np.quantile(win_fraction, [0.025, 0.975]).tolist())
    valid = a[0, 0, :, 2] > 0
    for i, arm in enumerate(ARMS[1:], start=1):
        full_mse = a[0, :, valid, 0] / a[0, :, valid, 2]
        variant_mse = a[i, :, valid, 0] / a[i, :, valid, 2]
        delta = variant_mse - full_mse
        day_delta = delta.mean(1)
        paired[arm] = dict(mean_daily_mse_difference_mm2=float(day_delta.mean()), full_better_days=int((day_delta > 0).sum()), variant_better_days=int((day_delta < 0).sum()), tied_days=int((day_delta == 0).sum()), positive_mean_effect_seed_count=int((delta.mean(0) > 0).sum()), calendar_block_intervals=all_bootstrap[arm], description='Positive difference favors O0. Mean daily MSE effects and pooled RMSE are distinct estimands; 109calendar positions retain missing days.')
    dump(OUT / 'comparison.json', dict(status='completed', independent_response_seeds=SEEDS, main_grid_minutes=1, selection_grid_minutes=15, summary=scores, paired=paired))
    lines = ['# D1 observed-temperature response comparison', '', 'Independent response seeds2024–2028; 52validation/60test days; 15min training and selection,1min final scoring.', 'Positive RMSE increase favors complete O0. This is conditional response evaluation with measured future temperature.', '', '| Arm | Test RMSE (mm), mean ± sample SD | Increase over O0 (%) |', '|---|---:|---:|']
    for item in scores:
        increase = 100 * (item['rmse_mm_mean'] / scores[0]['rmse_mm_mean'] - 1)
        lines.append(f"| {item['arm']} | {item['rmse_mm_mean']:.6f} ± {item['rmse_mm_sample_sd']:.6f} | {increase:.3f} |")
    lines.extend(['', 'Paired calendar-block uncertainty:comparison.json. Independent parameter distributions:selected_parameter_comparison.json.', 'Every run contains selected and last checkpoints, training curve, predictions, field rates, RK4 component increments and daily SSE/SAE/N.', 'Within-seed comparisons use the same software environment. Execution time is not a controlled hardware benchmark.', 'No synthetic dataset, STPM training, joint end-to-end adaptation, or changes to original frozen source/weights.'])
    (OUT / 'results_table.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (OUT / 'completed_experiment_record.md').write_text('# D1 observed-temperature45-run completion\n\n- Project:IFpaper_v1.\n- Purpose:observed-temperature response-structure ablations O0–O8, five independent response seeds.\n- Dataset and operations:dataset/README.md, dataset/manifest.json, protocol.json, source/.\n- Result:45training/validation/test runs completed; exact metrics:results_table.md and comparison.json.\n- Products:per-seed checkpoints/predictions/daily errors, parameter comparison and paired intervals.\n', encoding='utf-8')
    print('SUMMARY_COMPLETED', json.dumps(scores), flush=True)
if __name__ == '__main__':
    main()
