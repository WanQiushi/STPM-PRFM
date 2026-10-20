from if_release.runtime import expand as _release_expand
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
sys.path.insert(0, _release_expand('${TOOLS}/scipilot-figure-skill/scripts'))
from profile_data import profile_data, render_report
from visual_qa import audit_layout, render_preview
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
z = np.load(ROOT / 'data/minute_synthetic.npz')
values = z['values']
clean = z['clean_values']
states = z['thermal_states_c']
channels = z['channels'].tolist()
frame = pd.DataFrame(values, columns=channels)
info = profile_data(frame)
(OUT / 'data_profile.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
(OUT / 'data_profile.txt').write_text(render_report(info), encoding='utf-8')
corr = np.corrcoef(values[:, 1:].T)
centered = clean[:, 1:].astype(float)
centered -= centered.mean(axis=0)
eig = np.linalg.eigvalsh(centered.T @ centered)
energy = eig[::-1] / eig.sum()
(OUT / 'response_structure.json').write_text(json.dumps(dict(correlation=corr.tolist(), clean_covariance_energy_fraction=energy.tolist(), interpretation='The uniform thermal fields lead to highly correlated deflections and predominantly one spatial response pattern. This dataset does not establish the benefit of sensor fusion or independent BW memory.'), indent=2), encoding='utf-8')
plt.rcParams.update({'font.family': 'Arial', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.5), layout='constrained')
start = 280 * 1440
end = 283 * 1440
sl = slice(start, end, 5)
x = np.arange(start, end, 5) / 1440.0
axes[0].plot(x, values[sl, 0], color='#E69F00', lw=0.8, label='Simulated monitored T')
axes[0].plot(x, states[sl, 0], color='#0072B2', ls='--', lw=1.2, label='Girder state (2 h)')
axes[0].plot(x, states[sl, 1], color='#009E73', ls='-.', lw=1.2, label='Cable state (4 h)')
axes[0].set_ylabel('Temperature (degC)')
axes[0].set_xlabel('Simulation day')
axes[0].set_title('(a) Thermal lag: first three validation days', loc='left', fontsize=10)
axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=8)
axes[0].margins(y=0.3)
colors = ['#333333', '#56B4E9', '#0072B2', '#CC79A7']
styles = ['-', '--', '-.', ':']
for j in range(4):
    axes[1].plot(x, values[sl, j + 1], color=colors[j], ls=styles[j], lw=1.1, label='D%02d (%g span)' % (j + 1, (j + 1) / 8))
axes[1].set_ylabel('Deflection (mm)')
axes[1].set_xlabel('Simulation day')
axes[1].set_title('(b) Spatial response: four of seven positions', loc='left', fontsize=10)
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False, fontsize=8)
axes[1].margins(y=0.3)
lo = start
hi = start + 1440
xx = clean[lo:hi:5, 0]
yy = clean[lo:hi:5, 4]
axes[2].plot(xx, yy, color=colors[3], lw=1.3, label='Clean midspan trajectory')
axes[2].scatter([xx[0]], [yy[0]], s=24, color='black', marker='o', label='Day start', zorder=3)
for k in [45, 120, 220]:
    axes[2].annotate('', xy=(xx[k + 8], yy[k + 8]), xytext=(xx[k], yy[k]), arrowprops=dict(arrowstyle='->', color=colors[3], lw=1.2))
axes[2].set_xlabel('Clean monitored temperature (degC)')
axes[2].set_ylabel('Midspan deflection (mm)')
axes[2].set_title('(c) Temperature-deflection path: day 280', loc='left', fontsize=10)
axes[2].legend(loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, fontsize=8)
axes[2].margins(y=0.3)
for ax in axes:
    ax.grid(alpha=0.15)
    ax.tick_params(labelsize=8)
issues = audit_layout(fig)
(OUT / 'layout_audit.json').write_text(json.dumps(issues, ensure_ascii=False, indent=2), encoding='utf-8')
render_preview(fig, str(OUT / 'dataset_preview.png'), dpi=150)
if '--export' in sys.argv:
    fig.savefig(OUT / 'dataset_preview_300dpi.png', dpi=300)
    fig.savefig(OUT / 'dataset_preview.pdf')
    fig.savefig(OUT / 'dataset_preview.svg')
from matplotlib.lines import Line2D
for line in fig.findobj(match=Line2D):
    rgb = to_rgb(line.get_color())
    luma = sum((a * b for a, b in zip(rgb, [0.2126, 0.7152, 0.0722])))
    line.set_color((luma, luma, luma))
for annotation in axes[2].texts:
    arrow = annotation.arrow_patch
    rgb = to_rgb(arrow.get_edgecolor())
    luma = sum((a * b for a, b in zip(rgb, [0.2126, 0.7152, 0.0722])))
    arrow.set_edgecolor((luma, luma, luma))
    arrow.set_facecolor((luma, luma, luma))
fig.savefig(OUT / 'dataset_preview_grayscale.png', dpi=150)
print(json.dumps(dict(layout_issues=issues, first_covariance_energy=float(energy[0]), min_offdiagonal_corr=float(corr[np.triu_indices(7, 1)].min()))))
