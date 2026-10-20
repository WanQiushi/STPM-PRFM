"""Sixteen native-size manuscript panels: four response views by four piers."""
from pathlib import Path
import argparse, json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from PIL import Image
sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from manuscript_plot_style import apply_style, style_axes, COLORS, WIDTH_IN, LABEL_SIZE, TICK_SIZE, LEGEND_SIZE, TITLE_SIZE, MEASURED_COLOR
apply_style()
plt.rcParams.update({'savefig.bbox': None, 'path.simplify': True, 'path.simplify_threshold': 0.05})
import os
DATA_ROOT = Path(os.environ['IF_FIGURE_DATA'])
OUTPUT = Path(os.environ['IF_FIGURE_OUTPUT'])
z = np.load(DATA_ROOT / 'd1/component_plot_data.npz')
day = int(z['selected_day'])
hours = np.arange(1440) / 60
names = ['Proposed'] + z['models'].tolist()[1:]
markers = ['D', 'o', '^', 's', 'v']
colors = [COLORS[name] for name in names]
BLACK = MEASURED_COLOR
TERM1 = '#7A8793'
TERM2 = '#45414F'

def bounds(values):
    a = np.concatenate([np.asarray(v).ravel() for v in values])
    low = np.nanmin(a)
    high = np.nanmax(a)
    pad = max((high - low) * 0.08, 0.3)
    return (low - pad, high + pad)

def style(ax):
    style_axes(ax)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))

def time_axis(ax):
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 12, 24])
    ax.set_xlabel('Time (h)', labelpad=1.8)

def compare(ax, x, ours, truth, baseline_x, baseline_y):
    for k in range(4):
        ax.plot(baseline_x[k], baseline_y[k], color=colors[k + 1], ls='--', lw=0.5, alpha=0.82, marker=markers[k + 1], markevery=360, ms=1.8, mfc='white', mew=0.4, zorder=3)
    ax.plot(x, truth, color=BLACK, lw=0.65, zorder=5)
    ax.plot(x, ours, color=colors[0], ls='-', lw=0.95, zorder=6)
fig = plt.figure(figsize=(WIDTH_IN, 6.05))
grid = fig.add_gridspec(4, 4, left=0.16, right=0.955, top=0.9, bottom=0.16, wspace=0.43, hspace=0.83)
axes = np.array([[fig.add_subplot(grid[r, c]) for c in range(4)] for r in range(4)])
handles = [Line2D([], [], color=BLACK, lw=0.7, label='Decomposition reference')]
handles += [Line2D([], [], color=color, ls='-' if k == 0 else '--', lw=0.95 if k == 0 else 0.6, marker=None if k == 0 else markers[k], mfc='white', ms=2.4, label={'TCN': 'ModernTCN', 'TimesNet': 'TimesNet-LR'}.get(name, name)) for k, (name, color) in enumerate(zip(names, colors))]
handles += [Line2D([], [], color=TERM1, ls='-.', lw=0.6, label='Reversible $kq$'), Line2D([], [], color=TERM2, ls=':', lw=0.65, label='Memory $sz$')]
fig.legend(handles=handles, ncol=4, loc='lower center', bbox_to_anchor=(0.52, 0.015), frameon=False, handlelength=1.8, columnspacing=0.65, handletextpad=0.45, labelspacing=0.6)
fig.text(0.012, 0.53, 'Displacement (mm)', rotation=90, ha='center', va='center', fontsize=LABEL_SIZE)
for c in range(4):
    p = axes[0, c].get_position()
    fig.text((p.x0 + p.x1) / 2, 0.947, f'Pier {c + 1}', ha='center', va='center', fontsize=TITLE_SIZE, fontweight='bold')
for c in range(4):
    ch = c + 1
    m = z['mask'][day, :, ch]
    lf_ref = np.where(m, z['lf_reference'][:, ch], np.nan)
    mf_ref = np.where(m, z['mf_reference'][:, ch], np.nan)
    actual = np.where(m, z['measured'][day, :, ch], np.nan)
    ax = axes[1, c]
    compare(ax, hours, z['lf_prediction'][:, c], lf_ref, [hours] * 4, z['baseline_lf'][:, :, ch])
    ax.set_ylim(bounds([lf_ref, z['lf_prediction'][:, c], z['baseline_lf'][:, :, ch]]))
    time_axis(ax)
    ax = axes[0, c]
    valid = m & z['mask'][day, :, 0]
    for k in range(4):
        ax.plot(z['baseline_lf'][k, :, 0], z['baseline_lf'][k, :, ch], color=colors[k + 1], ls='--', lw=0.5, alpha=0.82, marker=markers[k + 1], markevery=360, ms=1.8, mfc='white', mew=0.4, zorder=3)
    ax.plot(np.where(valid, z['lf_reference'][:, 0], np.nan), np.where(valid, z['lf_reference'][:, ch], np.nan), color=BLACK, lw=0.65, zorder=5)
    ax.plot(z['lf_temperature'], z['lf_prediction'][:, c], color=colors[0], lw=0.95, zorder=6)
    ax.set_xlim(bounds([z['lf_temperature'], z['baseline_lf'][:, :, 0], np.where(valid, z['lf_reference'][:, 0], np.nan)]))
    ax.set_ylim(bounds([z['lf_prediction'][:, c], z['baseline_lf'][:, :, ch], lf_ref]))
    ax.set_xlabel('LF temperature (°C)', labelpad=1.8)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax = axes[3, c]
    compare(ax, hours, z['mf_prediction'][:, c], mf_ref, [hours] * 4, z['baseline_mf'][:, :, ch])
    ax.plot(hours, z['reversible'][:, c], color=TERM1, ls='-.', lw=0.55, zorder=4)
    ax.plot(hours, z['memory'][:, c], color=TERM2, ls=':', lw=0.65, zorder=4)
    ax.set_ylim(bounds([mf_ref, z['mf_prediction'][:, c], z['baseline_mf'][:, :, ch], z['reversible'][:, c], z['memory'][:, c]]))
    time_axis(ax)
    ax = axes[2, c]
    valid = m & z['mask'][day, :, 0]
    for k in range(4):
        ax.plot(z['baseline_mf'][k, :, 0], z['baseline_mf'][k, :, ch], color=colors[k + 1], ls='--', lw=0.5, alpha=0.82, marker=markers[k + 1], markevery=360, ms=1.8, mfc='white', mew=0.4, zorder=3)
    ax.plot(np.where(valid, z['mf_reference'][:, 0], np.nan), np.where(valid, z['mf_reference'][:, ch], np.nan), color=BLACK, lw=0.65, zorder=5)
    ax.plot(z['mf_temperature'], z['mf_prediction'][:, c], color=colors[0], lw=0.95, zorder=6)
    ax.set_xlim(bounds([z['mf_temperature'], z['baseline_mf'][:, :, 0], np.where(valid, z['mf_reference'][:, 0], np.nan)]))
    ax.set_ylim(bounds([z['mf_prediction'][:, c], z['baseline_mf'][:, :, ch], mf_ref]))
    ax.set_xlabel('MF temperature (°C)', labelpad=1.8)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
for r, label in enumerate(['LF relation', 'LF prediction', 'MF hysteresis', 'MF prediction']):
    p = axes[r, 0].get_position()
    fig.text(0.055, (p.y0 + p.y1) / 2, label, rotation=90, ha='center', va='center', fontsize=LABEL_SIZE, fontweight='bold', color='#26323C')
    for c in range(4):
        style(axes[r, c])
        axes[r, c].set_title(f'({chr(97 + r * 4 + c)})', loc='left', fontweight='bold', pad=2)
for ext in ['pdf', 'svg']:
    fig.savefig(OUTPUT / ('fig06_d1_components.' + ext))
fig.savefig(OUTPUT / 'fig06_d1_components.png', dpi=600)
plt.close(fig)
