"""Native-width full-period figures; no selected-day panels or smoothing."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from PIL import Image
ROOT = Path(__file__).resolve().parent
FIGURES = ROOT.parent
sys.path.insert(0, str(FIGURES))
from manuscript_plot_style import apply_style, style_axes, WIDTH_IN, COLORS, MEASURED_COLOR
import os
DATA_ROOT = Path(os.environ['IF_FIGURE_DATA'])
OUTPUT = Path(os.environ['IF_FIGURE_OUTPUT'])
D1_SOURCE = DATA_ROOT / 'd1/full109_plot_data.npz'
STYLES = ['-', '--', (0, (4, 2, 1, 2)), ':']

def segmented(x, y):
    return (np.column_stack((x, np.full(len(x), np.nan))).ravel(), np.column_stack((y, np.full(len(y), np.nan))).ravel())

def limits(*arrays):
    v = np.concatenate([np.asarray(a).ravel() for a in arrays])
    lo, hi = (np.nanmin(v), np.nanmax(v))
    pad = max((hi - lo) * 0.07, 0.05)
    return (lo - pad, hi + pad)

def handles(names, reference='Observed'):
    result = [Line2D([], [], color=MEASURED_COLOR, lw=0.65, label=reference)]
    result += [Line2D([], [], color=COLORS[name], ls=STYLES[k], lw=0.8 if k == 0 else 0.55, label={'TCN': 'ModernTCN', 'TimesNet': 'TimesNet-LR'}.get(name, name)) for k, name in enumerate(names)]
    return result

def d1_figure():
    apply_style()
    z = np.load(D1_SOURCE)
    idx = [z['model'].tolist().index(n) for n in ['S0', 'PCFNet', 'VanillaLSTM', 'ModernTCN']]
    pred = z['prediction'][idx]
    names = ['Proposed', 'PCFNet', 'LSTM', 'TCN']
    dates = np.datetime64('2022-01-01T00:00') + z['origin'][:, None].astype('timedelta64[m]') + np.arange(1440)[None, :].astype('timedelta64[m]')
    x = mdates.date2num(dates.ravel())
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(WIDTH_IN, 4.6))
    fig.subplots_adjust(left=0.12, right=0.985, top=0.95, bottom=0.17, hspace=0.47)
    fig.supylabel('Displacement (mm)', x=0.012, y=0.56, fontsize=8)
    for ch, ax in enumerate(axes, start=1):
        actual = np.where(z['mask'][:, :, ch], z['measured'][:, :, ch], np.nan).ravel()
        for k in [1, 2, 3, 0]:
            ax.plot(x, pred[k, :, :, ch].ravel(), color=COLORS[names[k]], ls=STYLES[k], lw=0.65 if k == 0 else 0.38, alpha=1 if k == 0 else 0.78, zorder=5 if k == 0 else 2)
        ax.plot(x, actual, color=MEASURED_COLOR, lw=0.35, alpha=0.8, zorder=3)
        ax.set_ylim(limits(pred[:, :, :, ch], actual))
        ax.set_xlim(x[0], x[-1])
        ax.set_title(f'({chr(96 + ch)}) Pier {ch}', loc='left', fontweight='bold', pad=3)
        ax.set_xticks(mdates.date2num(np.array(['2023-09-14', '2023-10-20', '2023-11-25', '2023-12-31'], dtype='datetime64[D]')))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        style_axes(ax)
    axes[-1].set_xlabel('Forecast date', labelpad=4)
    labels = axes[-1].get_xticklabels()
    labels[0].set_ha('left')
    labels[-1].set_ha('right')
    fig.legend(handles=handles(names, 'Measured'), ncol=3, loc='lower center', bbox_to_anchor=(0.54, 0.027), handlelength=1.9, columnspacing=1, handletextpad=0.4)
    return fig
if __name__ == '__main__':
    fig = d1_figure()
    for ext in ['pdf', 'svg']:
        fig.savefig(OUTPUT / ('fig05_d1_full_test.' + ext))
    fig.savefig(OUTPUT / 'fig05_d1_full_test.png', dpi=600)
    plt.close(fig)
