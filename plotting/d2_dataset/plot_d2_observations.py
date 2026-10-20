"""D2 observations with the manuscript's shared figure typography and style.

The data, eight panels, axis ranges and 280/60/60-day split are unchanged.
"""
from pathlib import Path
import json
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from manuscript_plot_style import apply_style, style_axes, WIDTH_IN, TITLE_SIZE, LEGEND_SIZE
import os
DATA_ROOT = Path(os.environ['IF_FIGURE_DATA'])
OUTPUT = Path(os.environ['IF_FIGURE_OUTPUT'])
SOURCE = DATA_ROOT / 'd2_dataset/minute_synthetic.npz'
NAME = 'd2_400day_observations'
TITLES = ['Temperature', 'Deflection at $L/8$', 'Deflection at $L/4$', 'Deflection at $3L/8$', 'Deflection at $L/2$', 'Deflection at $5L/8$', 'Deflection at $3L/4$', 'Deflection at $7L/8$']
SPLITS = [(0, 280, '#EDF3F8', 'Training: 280 days'), (280, 340, '#F2F2F2', 'Validation: 60 days'), (340, 400, '#F9EBDD', 'Test: 60 days')]
apply_style()
plt.rcParams.update({'path.simplify': True, 'path.simplify_threshold': 0.1})
with np.load(SOURCE, allow_pickle=False) as data:
    values = data['values']
x = np.arange(len(values), dtype=np.float64) / 1440
fig, axes = plt.subplots(4, 2, figsize=(WIDTH_IN, 5.36), sharex=True)
fig.subplots_adjust(left=0.128, right=0.975, bottom=0.15, top=0.95, hspace=0.4, wspace=0.43)
for j, ax in enumerate(axes.flat):
    for lo, hi, color, _ in SPLITS:
        ax.axvspan(lo, hi, color=color, zorder=0, linewidth=0)
    for boundary in [280, 340]:
        ax.axvline(boundary, color='#777777', lw=0.5, ls=(0, (4, 3)), zorder=2)
    color = '#D55E00' if j == 0 else '#0072B2'
    ax.plot(x, values[:, j], color=color, lw=0.55, solid_capstyle='round', zorder=3)
    ax.set_title(f'({chr(97 + j)}) {TITLES[j]}', loc='left', pad=4, fontweight='bold', fontsize=TITLE_SIZE)
    ax.set_ylabel('Temperature (°C)' if j == 0 else 'Deflection (mm)', labelpad=3)
    style_axes(ax, grid_axis='y')
    ax.minorticks_off()
    ax.set_xlim(0, 400)
    ax.set_xticks([0, 100, 200, 300, 400])
    if j == 0:
        ax.set_ylim(7, 50)
        ax.set_yticks([10, 20, 30, 40, 50])
    else:
        ax.set_ylim(-720, 760)
        ax.set_yticks([-600, -300, 0, 300, 600])
for ax in axes[-1]:
    ax.set_xlabel('Simulation day', labelpad=3)
fig.legend(handles=[Patch(facecolor=c, edgecolor='#AAB0B5', linewidth=0.5, label=label) for _, _, c, label in SPLITS], ncol=3, loc='lower center', bbox_to_anchor=(0.54, 0.015), frameon=False, handlelength=1.3, columnspacing=1.1, handletextpad=0.4, fontsize=LEGEND_SIZE)
for ext in ['pdf', 'svg']:
    fig.savefig(OUTPUT / ('fig07_d2_observations.' + ext))
fig.savefig(OUTPUT / 'fig07_d2_observations.png', dpi=600)
plt.close(fig)
