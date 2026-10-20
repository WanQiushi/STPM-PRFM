"""Save actual completed metrics and comparison line plots, without evaluating test data."""
from if_release.runtime import loads as _release_json_loads
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'temperature_prediction/outputs' / HERE.name
sys.path.insert(0, str(HERE / 'figure_tools'))
from setup_style import setup_style
from visual_qa import audit_layout, render_preview
from profile_data import profile_data, render_report

def main():
    figures = OUT / 'figures'
    figures.mkdir()
    data, records, daily = ({}, [], [])
    for seed in [2024, 2025]:
        run = OUT / ('shape_ode_residual_seed' + str(seed))
        with np.load(run / 'validation_predictions.npz') as z:
            data[seed] = {key: z[key] for key in ['origin', 'prediction', 'base_prediction', 'target', 'mask', 'ode_shape', 'residual_prediction']}
        base = _release_json_loads((run / 'base_best_validation.json').read_text())
        final = _release_json_loads((run / 'best_validation.json').read_text())
        for label, metrics in [('base', base), ('base_plus_residual', final)]:
            records.append({'seed': seed, 'model': label, **{key: metrics[key] for key in ['rmse_celsius', 'mae_celsius', 'last_12h_rmse_celsius']}})
        for key, label in [('base_prediction', 'base'), ('prediction', 'base_plus_residual')]:
            d = data[seed]
            error = (d[key] - d['target'])[..., 0]
            mask = d['mask'][..., 0]
            rmse = np.sqrt((error ** 2 * mask).sum(1) / mask.sum(1))
            for i, origin in enumerate(d['origin'].reshape(-1)):
                daily.append({'seed': seed, 'model': label, 'origin': int(origin), 'rmse_celsius': float(rmse[i])})
    pd.DataFrame(records).to_csv(OUT / 'comparison_metrics.csv', index=False)
    daily_frame = pd.DataFrame(daily)
    daily_frame.to_csv(OUT / 'daily_metrics.csv', index=False)
    (figures / 'data_profile.md').write_text(render_report(profile_data(daily_frame, group_cols=['seed', 'model'])), encoding='utf-8')
    setup_style(journal='general', lang='en', use_sciplots=False)
    plt.rcParams.update({'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9, 'legend.fontsize': 8, 'figure.constrained_layout.use': False})
    plots = []
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.5), sharex=True, sharey='row')
    fig.subplots_adjust(left=0.09, right=0.98, top=0.86, bottom=0.17, hspace=0.45, wspace=0.15)
    for row, (origin, date) in enumerate([(874080, '2023-08-31'), (869760, '2023-08-28')]):
        for col, seed in enumerate([2024, 2025]):
            d = data[seed]
            i = int(np.flatnonzero(d['origin'].reshape(-1) == origin)[0])
            ax = axes[row, col]
            t = np.arange(96) / 4
            ax.plot(t, np.where(d['mask'][i, :, 0], d['target'][i, :, 0], np.nan), color='#111111', lw=1.7, label='Measured')
            ax.plot(t, d['base_prediction'][i, :, 0], color='#D55E00', ls='--', lw=1.4, label='ODE shape + level/scale')
            ax.plot(t, d['prediction'][i, :, 0], color='#0072B2', ls='-', marker='o', markevery=12, ms=3, lw=1.5, label='Base + residual')
            ax.set_title(f'{date} / seed {seed}')
            ax.set_xlim(0, 24)
            ax.set_xticks(np.arange(0, 25, 4))
            ax.grid(alpha=0.18)
            if col == 0:
                ax.set_ylabel('Temperature (degC)')
    fig.suptitle('Shape ODE + small residual: fixed validation examples', y=0.98)
    fig.text(0.5, 0.105, 'Hours after forecast origin', ha='center')
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(0.5, 0.015), ncol=3, frameon=False)
    plots.append((fig, '01_prediction_examples'))
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.9), sharey=True)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.1, hspace=0.42, wspace=0.15)
    max_rmse = 0.0
    for row, seed in enumerate([2024, 2025]):
        run = OUT / ('shape_ode_residual_seed' + str(seed))
        for col, stage in enumerate(['base', 'residual']):
            history = pd.read_csv(run / (stage + '_history.csv'))
            axes[row, col].plot(history['epoch'], history['rmse_celsius'], color='#0072B2', lw=1.5, marker='o', markevery=8, ms=3)
            best = history.loc[history['rmse_celsius'].idxmin()]
            axes[row, col].plot(best['epoch'], best['rmse_celsius'], marker='o', mfc='white', mec='#0072B2', ms=6)
            axes[row, col].set_title(f'seed {seed}: {stage}')
            axes[row, col].set_xlabel('Epoch')
            axes[row, col].grid(alpha=0.18)
            max_rmse = max(max_rmse, float(history['rmse_celsius'].max()))
        axes[row, 0].set_ylabel('Validation RMSE (degC)')
    axes[0, 0].set_ylim(0, max_rmse * 1.08)
    fig.suptitle('Two-stage learning curves: observed validation nodes', y=0.98)
    plots.append((fig, '02_validation_learning'))
    qa = {}
    for fig, name in plots:
        render_preview(fig, str(figures / (name + '_preview.png')), dpi=300)
        qa[name] = audit_layout(fig)
        plt.close(fig)
    (figures / 'layout_qa.json').write_text(json.dumps(qa, indent=2))
    lines = ['# Shape ODE + residual completion', '', '- Date: 2026-10-02 (Asia/Shanghai).', '- Two completed seeds; stage A trains normalized ODE shape and historical level/amplitude; stage B freezes it and learns full-scale residual.', '- Inputs/targets: existing v9 causal contract; 66 validation origins, 6204 measured nodes; no test forecast.', '- Data/code: each run config.json, frozen_source and data_report.json; stage-specific experiment records retained.', '- Shapes are dimensionless; mean and amplitude are predicted from history; future measured components are training labels only.']
    for row in records:
        lines.append(f"- seed{row['seed']} {row['model']}: RMSE {row['rmse_celsius']:.6f}, MAE {row['mae_celsius']:.6f}, last12h {row['last_12h_rmse_celsius']:.6f} degC.")
    lines += ['- Products: comparison_metrics.csv, daily_metrics.csv, figures and each run best.pt/validation_predictions.npz.', '- Figures are PNG previews with layout diagnostics. Human/AI visual review and vector exports remain pending.', '- Residual is not restricted to high-frequency detail. Validation was used for checkpoint selection; no claim of significant improvement is automatic.']
    (OUT / 'completion_report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'metrics': records, 'layout_qa': qa}))
if __name__ == '__main__':
    main()
