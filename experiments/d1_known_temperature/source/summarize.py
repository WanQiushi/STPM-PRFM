"""Final server aggregation: five reused H1 runs plus forty completed baselines."""
from pathlib import Path
from datetime import datetime, timedelta
import csv
import math
import numpy as np
from common import ROOT, read, dump, now

def main():
    plan = read(ROOT / 'plan.json')
    models = plan['models']
    seeds = plan['seeds']
    summaries = []
    all_sse = []
    all_sae = []
    all_n = []
    per_seed = []
    for model in models:
        model_summaries = []
        sse = []
        sae = []
        n = []
        for seed in seeds:
            folder = ROOT / ('h1_reference' if model == 'H1' else 'runs') / f'{model}_seed{seed}'
            summary = read(folder / 'summary.json')
            model_summaries.append(summary)
            with np.load(folder / 'test_daily_errors.npz') as daily:
                sse.append(daily['sse'])
                sae.append(daily['sae'])
                n.append(daily['n'])
                origins = daily['origin']
            per_seed.append(dict(model=model, seed=seed, **summary['metrics']['test'], selected_epoch=summary['selected_epoch'], epochs_ran=summary['epochs_ran'], reused_h1=model == 'H1'))
        summaries.append(model_summaries)
        all_sse.append(sse)
        all_sae.append(sae)
        all_n.append(n)
    sse = np.array(all_sse)
    sae = np.array(all_sae)
    n = np.array(all_n)
    rmse = np.sqrt(sse.sum((2, 3)) / n.sum((2, 3)))
    mae = sae.sum((2, 3)) / n.sum((2, 3))
    daily_mse = sse.sum(-1) / n.sum(-1)
    calendar = np.zeros((9, 5, 109, 2))
    day_index = origins // 1440 - 621
    calendar[:, :, day_index, 0] = sse.sum(-1)
    calendar[:, :, day_index, 1] = n.sum(-1)
    boot_effects = {}
    for block in [7, 3, 14]:
        rng = np.random.default_rng(20261020)
        starts = rng.integers(0, 109 - block + 1, size=(10000, math.ceil(109 / block)))
        ix = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(10000, -1)[:, :109]
        boot = []
        for i in range(9):
            totals = calendar[i][:, ix].sum(2)
            boot.append(np.sqrt(totals[:, :, 0] / totals[:, :, 1]).mean(0))
        boot = np.stack(boot)
        boot_effects[str(block)] = boot[1:] - boot[:1]
    dates = [(datetime(2022, 1, 1) + timedelta(minutes=int(o))).date().isoformat() for o in origins]
    comparison = []
    paired_daily = []
    labels = {'H1': 'H1 (reused full architecture)', 'VanillaLSTM': 'LSTM', 'TimesNet': 'TimesNet-LR'}
    for i, model in enumerate(models):
        row = dict(model=model, label=labels.get(model, model), n_seeds=5, rmse_seed=rmse[i].tolist(), rmse_mean=float(rmse[i].mean()), rmse_sample_sd=float(rmse[i].std(ddof=1)), mae_mean=float(mae[i].mean()), mae_sample_sd=float(mae[i].std(ddof=1)), relative_rmse_change_pct=float(100 * (rmse[i].mean() / rmse[0].mean() - 1)), selected_epoch=[x['selected_epoch'] for x in summaries[i]])
        if i:
            delta = daily_mse[i] - daily_mse[0]
            row.update(h1_better_days=int((delta.mean(0) > 0).sum()), h1_better_days_each_seed=(delta > 0).sum(1).tolist(), h1_better_pooled_seed_count=int((rmse[i] > rmse[0]).sum()), delta_rmse_mean_mm=float((rmse[i] - rmse[0]).mean()), calendar_block_ci95={str(b): np.quantile(boot_effects[str(b)][i - 1], [0.025, 0.975]).tolist() for b in [7, 3, 14]}, eight_comparison_interval=np.quantile(boot_effects['7'][i - 1], [0.05 / 16, 1 - 0.05 / 16]).tolist())
            for j, seed in enumerate(seeds):
                for d, date in enumerate(dates):
                    paired_daily.append(dict(model=model, seed=seed, date=date, mse_difference_mm2=float(delta[j, d]), rmse_difference_mm=float(np.sqrt(daily_mse[i, j, d]) - np.sqrt(daily_mse[0, j, d]))))
        comparison.append(row)
    result = ROOT / 'comparison'
    result.mkdir(exist_ok=True)
    dump(result / 'comparison.json', dict(status='completed', created_beijing=now(), sample_minutes=15, new_baseline_runs=40, reused_h1_runs=5, summary=comparison, uncertainty='109calendar positions,7-day blocks10000rep,3/14-day sensitivity;conditional on five fitted seeds', h1_scope=plan['h1_reuse']['scope'], future_response_input=False))
    dump(result / 'per_seed_metrics.json', per_seed)
    dump(result / 'daily_paired_errors.json', paired_daily)
    headers = ['model', 'rmse_mean', 'rmse_sample_sd', 'mae_mean', 'mae_sample_sd', 'relative_rmse_change_pct']
    with (result / 'comparison.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, headers, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(comparison)
    lines = ['# D1 known-temperature H1 versus eight manuscript baselines', '', 'Five seeds2024-2028;60 common observed-temperature test days;15-minute response grid;mean±sampleSD,mm.', 'H1:reuse five completed full-architecture O0 runs,no retraining;forty newly trained conditional baselines.', 'All methods receive the same30day history and measured next24h temperature;future displacement labels are not inputs.', '', '| Model | RMSE (mm) | MAE (mm) | Change versus H1 | H1 better days |', '| --- | --- | --- | --- | --- |']
    for row in comparison:
        win = f"{row['h1_better_days']}/60" if row['model'] != 'H1' else '—'
        lines.append(f"| {row['label']} | {row['rmse_mean']:.4f} ± {row['rmse_sample_sd']:.4f} | {row['mae_mean']:.4f} ± {row['mae_sample_sd']:.4f} | {row['relative_rmse_change_pct']:+.2f}% | {win} |")
    lines += ['', 'Positive differences favor H1. Calendar-block intervals and allseed/day values are inJSON.', 'This experiment tests whole-H1 conditional prediction performance;it does not identify a unique physical hysteresis mechanism.', 'The baselines use the declared common30-parameter future-covariate projection;these are conditional adaptations of the manuscript backbones.', 'Learning rate:H1 already trained at0.001,baselines retain0.0001. H1 retains its existing0.001 neural-rate regularization;baselines have no corresponding field.', '']
    (result / 'results_table.md').write_text('\n'.join(lines), encoding='utf-8')
    (result / 'experiment_record.md').write_text(f'# Known-temperature baseline comparison complete\n\n- Date:{now()};IFpaper_v1.\n- Dataset,normalization and model/config versions:../dataset/README.md,../plan.json,../source_provenance.json.\n- Operation:aggregate40 baseline runs and5 reused H1 runs,all60test days,15min;paired calendar blocks.\n- Result:H1 meanRMSE{rmse[0].mean():.9f}mm;all model results inresults_table.md/comparison.json.\n- Products:this comparison folder;original frozen weights/manuscript unchanged.\n', encoding='utf-8')
    print(result / 'results_table.md', flush=True)
if __name__ == '__main__':
    main()
