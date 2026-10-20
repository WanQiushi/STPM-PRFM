"""Portrait component panels for the selected D2 day, with fixed print fonts."""
from if_release.runtime import loads as _release_json_loads
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from PIL import Image
from plot_fullperiod import ROOT, POSITIONS, COLORS, MEASURED_COLOR, WIDTH_IN, apply_style, style_axes, limits, DATA_ROOT, OUTPUT
Z = np.load(DATA_ROOT / 'd2/plot_data_fig6_seed2024.npz')
NAMES = Z['models'].tolist()
STYLES = ['-', '--', '--', '--', '--']
METRICS = DATA_ROOT / 'd2/extracted_metrics.json'

def select_day():
    data = _release_json_loads(METRICS.read_text(encoding='utf-8'))
    models = list(dict.fromkeys((r['model'] for r in data['runs'])))
    seed_values = {name: np.stack([r['per_day_rmse_mm'] for r in data['runs'] if r['model'] == name]) for name in models}
    means = {name: a.mean(axis=0) for name, a in seed_values.items()}
    baselines = [n for n in models if n != 'Proposed']
    matrix = np.stack([means[n] for n in baselines])
    best_idx = matrix.argmin(axis=0)
    best_value = matrix.min(axis=0)
    gains = (best_value - means['Proposed']) / best_value * 100
    day = int(gains.argmax())
    ranking = []
    for rank, d in enumerate(np.argsort(-gains), start=1):
        ranking.append(dict(rank=rank, test_index0=int(d), simulation_day=int(data['origins'][d] // 1440 + 1), strongest_baseline=baselines[best_idx[d]], proposed_rmse_mean_mm=float(means['Proposed'][d]), baseline_rmse_mean_mm=float(best_value[d]), relative_reduction_percent=float(gains[d])))
    selected = ranking[0]
    selected.update(definition='Compute seven-location daily RMSE per seed on 96 common 15-minute nodes; average seeds 2024--2028; maximize relative reduction against the best of all eight main-table baselines on each day.', source=str(METRICS), display_seed=2024, proposed_sample_sd_mm=float(seed_values['Proposed'][:, day].std(ddof=1)), baseline_sample_sd_mm=float(seed_values[selected['strongest_baseline']][:, day].std(ddof=1)), interpretation='Post-hoc strongest-advantage example; not typical performance and not a training selection.')
    (ROOT / 'selected_day_recomputed.json').write_text(json.dumps(selected, indent=2), encoding='utf-8')
    pd.DataFrame(ranking).to_csv(ROOT / 'daily_advantage_recomputed.csv', index=False)
    print(json.dumps(selected, indent=2))
    return (day, selected)

def figure(day, selected):
    apply_style()
    fig, axes = plt.subplots(7, 4, sharex='col', figsize=(WIDTH_IN, 7.0))
    fig.subplots_adjust(left=0.16, right=0.975, top=0.945, bottom=0.14, wspace=0.65, hspace=0.55)
    fig.text(0.012, 0.53, 'Deflection (mm)', rotation=90, ha='center', va='center', fontsize=8)
    for c in range(7):
        p = axes[c, 0].get_position()
        fig.text(0.06, (p.y0 + p.y1) / 2, POSITIONS[c], ha='center', va='center', fontsize=8.5, fontweight='bold')
    markers = [None, 'o', '^', 's', 'v']
    hours = Z['lead_hours']
    for r, label in enumerate(['LF relation', 'LF prediction', 'MF hysteresis', 'MF prediction']):
        p = axes[0, r].get_position()
        fig.text((p.x0 + p.x1) / 2, 0.982, label, ha='center', va='center', fontsize=8.5, fontweight='bold', color='#26323C')
        band = 'lf' if r < 2 else 'mf'
        for c in range(7):
            ax = axes[c, r]
            truth = Z[f'response_{band}_reference_mm'][:, c]
            pred = Z[f'{band}_prediction_mm'][:, :, c]
            if r in [0, 2]:
                tx = Z[f'observed_temperature_{band}_c'][:, 0]
                px = Z[f'predicted_temperature_{band}_c'][:, 0]
                bx = Z[f'baseline_temperature_{band}_c'][:, :, 0]
                for k in [1, 2, 3, 4]:
                    ax.plot(bx[k - 1], pred[k], color=COLORS[NAMES[k]], ls=STYLES[k], lw=0.5, alpha=0.82, marker=markers[k], markevery=24, ms=1.8, mfc='white', mew=0.4, zorder=3)
                ax.plot(tx, truth, color=MEASURED_COLOR, lw=0.65, zorder=5)
                ax.plot(px, pred[0], color=COLORS['Proposed'], lw=0.95, zorder=6)
                ax.set_xlim(limits(tx, px, bx))
                ax.set_xlabel(f'{band.upper()} temperature (°C)', labelpad=1.8)
                ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
            else:
                for k in [1, 2, 3, 4]:
                    ax.plot(hours, pred[k], color=COLORS[NAMES[k]], ls=STYLES[k], lw=0.5, alpha=0.82, marker=markers[k], markevery=24, ms=1.8, mfc='white', mew=0.4, zorder=3)
                ax.plot(hours, truth, color=MEASURED_COLOR, lw=0.65, zorder=5)
                ax.plot(hours, pred[0], color=COLORS['Proposed'], lw=0.95, zorder=6)
                if r == 3:
                    reversible = Z['native_reversible_prediction_mm'][:, c]
                    memory = Z['native_memory_prediction_mm'][:, c]
                    ax.plot(hours, reversible, color='#7A8793', ls='-.', lw=0.55, zorder=4)
                    ax.plot(hours, memory, color='#45414F', ls=':', lw=0.65, zorder=4)
                    ax.set_ylim(limits(truth, pred, reversible, memory))
                ax.set_xlim(0, 24)
                ax.set_xticks([0, 12, 24])
                ax.set_xlabel('Time (h)', labelpad=1.8)
            if r != 3:
                ax.set_ylim(limits(truth, pred))
            style_axes(ax)
            ax.tick_params(labelsize=7, labelbottom=c == 6)
            if c < 6:
                ax.set_xlabel('')
            ax.yaxis.set_major_locator(MaxNLocator(nbins=2))
            index = c * 4 + r
            panel = chr(97 + index) if index < 26 else 'a' + chr(97 + index - 26)
            ax.set_title(f'({panel})', loc='left', fontweight='bold', pad=2)
    hh = [Line2D([], [], color=MEASURED_COLOR, lw=0.7, label='Decomposition reference')]
    hh += [Line2D([], [], color=COLORS[name], ls=STYLES[k], lw=0.95 if k == 0 else 0.6, marker=markers[k], ms=2.4, mfc='white', label={'TCN': 'ModernTCN', 'TimesNet': 'TimesNet-LR'}.get(name, name)) for k, name in enumerate(NAMES)]
    hh += [Line2D([], [], color='#7A8793', ls='-.', lw=0.6, label='Reversible $kq$'), Line2D([], [], color='#45414F', ls=':', lw=0.65, label='Memory $sz$')]
    fig.legend(handles=hh, ncol=4, loc='lower center', bbox_to_anchor=(0.54, 0.005), handlelength=1.8, columnspacing=0.65, handletextpad=0.45, labelspacing=0.6)
    return fig
if __name__ == '__main__':
    selected = _release_json_loads((DATA_ROOT / 'd2/selected_day_recomputed.json').read_text(encoding='utf-8'))
    fig = figure(selected['test_index0'], selected)
    for ext in ['pdf', 'svg']:
        fig.savefig(OUTPUT / ('fig09_d2_components.' + ext))
    fig.savefig(OUTPUT / 'fig09_d2_components.png', dpi=600)
    plt.close(fig)
