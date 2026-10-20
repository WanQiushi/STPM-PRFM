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
Z = np.load(DATA_ROOT / 'd2/plot_data_full60_seed2024.npz')
NAMES = Z['models'].tolist()
STYLES = ['-', '--', (0, (4, 2, 1, 2)), ':']
POSITIONS = ['$L/8$', '$L/4$', '$3L/8$', '$L/2$', '$5L/8$', '$3L/4$', '$7L/8$']
TIMES = Z['origin'][:, None] / 1440 + Z['lead_hours'][None, :] / 24

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

def d2_full_figure():
    apply_style()
    fig, axes = plt.subplots(7, 1, sharex=True, figsize=(WIDTH_IN, 6.7))
    fig.subplots_adjust(left=0.12, right=0.985, top=0.965, bottom=0.12, hspace=0.43)
    fig.supylabel('Deflection (mm)', x=0.012, y=0.55, fontsize=8)
    for ch, ax in enumerate(axes):
        for k in [1, 2, 3, 0]:
            xx, yy = segmented(TIMES, Z['prediction_mm'][k, :, :, ch])
            ax.plot(xx, yy, color=COLORS[NAMES[k]], ls=STYLES[k], lw=0.65 if k == 0 else 0.4, alpha=1 if k == 0 else 0.8, zorder=5 if k == 0 else 2)
        observed = np.where(Z['target_mask'][:, :, ch], Z['target_mm'][:, :, ch], np.nan)
        ax.plot(TIMES.ravel(), observed.ravel(), color=MEASURED_COLOR, lw=0.4, zorder=4, alpha=0.85)
        ax.set_title(f'({chr(97 + ch)}) {POSITIONS[ch]}', loc='left', fontweight='bold', pad=3)
        ax.set_ylim(limits(Z['prediction_mm'][:, :, :, ch], observed))
        ax.set_xlim(340, 400)
        ax.set_xticks([340, 360, 380, 400])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        style_axes(ax)
    axes[-1].set_xlabel('Simulation day', labelpad=4)
    fig.legend(handles=handles(NAMES), ncol=3, loc='lower center', bbox_to_anchor=(0.54, 0.018), handlelength=1.9, columnspacing=1, handletextpad=0.4)
    return fig
if __name__ == '__main__':
    fig = d2_full_figure()
    for ext in ['pdf', 'svg']:
        fig.savefig(OUTPUT / ('fig08_d2_full_test.' + ext))
    fig.savefig(OUTPUT / 'fig08_d2_full_test.png', dpi=600)
    plt.close(fig)
