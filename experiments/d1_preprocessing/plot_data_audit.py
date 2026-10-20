"""Diagnostic figures. Gaps remain gaps; daily envelopes are not confidence bands."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
root = Path(__file__).resolve().parents[1]
out = root / 'reports/20260922_data_audit'
figdir = out / 'figures'
figdir.mkdir(exist_ok=True)
z = np.load(root / 'data/processed/minute_observed.npz', allow_pickle=False)
t = pd.DatetimeIndex(z['timestamps'])
channels = z['channels'].tolist()
df = pd.DataFrame(z['values'], index=t, columns=channels)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.18, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
figures = {}
daily = df.resample('D')
med = daily.median()
low = daily.min()
high = daily.max()
fig, axes = plt.subplots(7, 1, figsize=(10, 12), sharex=True, layout='constrained')
for j, (ax, k) in enumerate(zip(axes, channels)):
    ax.fill_between(med.index, low[k], high[k], color='#56B4E9', alpha=0.35, label='Daily min-max')
    ax.plot(med.index, med[k], color='#0072B2', lw=1, label='Daily median')
    ax.set_ylabel('degC' if j == 0 else 'mm')
    ax.set_title(k, loc='left', fontsize=10)
axes[0].legend(loc='upper right', ncol=2, frameon=False)
axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
fig.suptitle('Two-year observed values: daily median and full min-max range', fontsize=13)
figures['01_overview'] = fig
coverage = pd.DataFrame(np.isfinite(z['values']), index=t, columns=channels).resample('D').mean() * 100
coverage['All seven'] = pd.Series(np.isfinite(z['values']).all(axis=1), index=t).resample('D').mean() * 100
fig, ax = plt.subplots(figsize=(10, 4), layout='constrained')
im = ax.imshow(coverage.to_numpy().T, aspect='auto', interpolation='nearest', cmap='viridis', vmin=0, vmax=100)
ticks = np.arange(0, len(coverage), 90)
ax.set_xticks(ticks, coverage.index[ticks].strftime('%Y-%m'))
ax.set_yticks(np.arange(8), list(coverage.columns))
ax.grid(False)
fig.colorbar(im, ax=ax, label='Observed minutes per day (%)', shrink=0.85)
ax.set_title('Daily availability; bottom row requires all seven channels')
figures['02_availability'] = fig

def event_figure(name, start, end, selected, title):
    view = df.loc[start:end, selected]
    fig, axes = plt.subplots(len(selected), 1, figsize=(10, 2 * len(selected) + 0.5), sharex=True, layout='constrained')
    for ax, k in zip(axes, selected):
        ax.plot(view.index, view[k], color='#0072B2', lw=0.85, marker='.', markersize=2.2)
        ax.set_ylabel('degC' if k.startswith('d1') else 'mm')
        ax.set_title(k, loc='left', fontsize=10)
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    locator.intervald[mdates.MINUTELY] = [1, 2, 5, 10, 15, 20, 30]
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))
    fig.suptitle(title, fontsize=12)
    figures[name] = fig
event_figure('03_january_episode', '2022-01-04', '2022-01-08 03:00', ['Temperature', 'Pier4', 'Pier3', 'Pier2'], 'January 2022 episode: minute observations, separate physical axes')
event_figure('04_extreme_zoom', '2022-01-08 00:00', '2022-01-08 01:15', ['Temperature', 'Pier4', 'Pier3', 'Auxiliary2'], '2022-01-08: extreme Pier4 readings and neighboring channels')
event_figure('05_gap_recovery', '2022-11-23 14:30', '2022-11-23 18:00', ['Temperature', 'Pier3', 'Pier2', 'Pier4'], '2022-11-23: missing interval and recovery observations')
event_figure('06_approach_excursion', '2023-02-03 08:45', '2023-02-03 10:00', ['Temperature', 'Auxiliary1', 'Pier1'], '2023-02-03: short approach-bridge displacement excursion')
event_figure('07_september_spike', '2023-09-01 00:20', '2023-09-01 01:10', ['Temperature', 'Pier4', 'Pier3'], '2023-09-01: isolated Pier4 excursion')
fig, axes = plt.subplots(2, 1, figsize=(10, 6), layout='constrained')
for ax, day, title in zip(axes, ['2022-08-13', '2023-12-22'], ['Temperature maximum day', 'Temperature minimum day']):
    v = df.loc[day, 'Temperature']
    ax.plot(v.index, v, color='#0072B2', lw=1)
    ax.set_ylabel('degC')
    ax.set_title(title + ' | ' + day, loc='left')
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
fig.suptitle('Temperature extrema in their day-long context', fontsize=12)
figures['08_temperature_extrema'] = fig
event_figure('09_temperature_local_excursion', '2022-04-23 01:30', '2022-04-23 03:00', ['Temperature', 'Pier2', 'Pier3'], '2022-04-23: temperature local excursion candidate')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vector', action='store_true')
    args = parser.parse_args()
    for name, fig in figures.items():
        fig.savefig(figdir / (name + '.png'), dpi=160)
        if args.vector:
            fig.savefig(figdir / (name + '.pdf'))
    print('Saved', len(figures), 'diagnostic figures')
