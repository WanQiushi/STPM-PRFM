"""Shared publication style for manuscript data figures, at final print size."""
import matplotlib as mpl
WIDTH_IN = 390 / 72.27
LABEL_SIZE = 8
TICK_SIZE = 7
LEGEND_SIZE = 7.5
TITLE_SIZE = 8.5
GRID_COLOR = '#E4E8EC'
MEASURED_COLOR = '#16191D'
COLORS = {'Proposed': '#0072B2', 'PCFNet': '#CC6677', 'LSTM': '#117733', 'TCN': '#B09020', 'DLinear': '#AA4499', 'TimeXer': '#56B4E9', 'PatchTST': '#D55E00', 'iTransformer': '#7F7F7F', 'TimesNet': '#496B8A'}
ALIASES = {'S0': 'Proposed', 'A2': 'Proposed', 'VanillaLSTM': 'LSTM', 'ModernTCN': 'TCN'}

def model_name(name):
    return ALIASES.get(name, name)

def apply_style():
    mpl.rcParams.update({'font.family': 'Arial', 'font.size': LABEL_SIZE, 'axes.labelsize': LABEL_SIZE, 'axes.titlesize': TITLE_SIZE, 'xtick.labelsize': TICK_SIZE, 'ytick.labelsize': TICK_SIZE, 'legend.fontsize': LEGEND_SIZE, 'legend.frameon': False, 'axes.linewidth': 0.5, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.axisbelow': True, 'xtick.direction': 'out', 'ytick.direction': 'out', 'xtick.major.size': 2, 'ytick.major.size': 2, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.pad': 2, 'ytick.major.pad': 2, 'grid.color': GRID_COLOR, 'grid.linewidth': 0.35, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold', 'mathtext.cal': 'Arial', 'mathtext.sf': 'Arial', 'mathtext.tt': 'Arial', 'mathtext.fallback': None, 'savefig.facecolor': 'white', 'figure.facecolor': 'white', 'axes.unicode_minus': True})

def style_axes(ax, grid_axis='both'):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_linewidth(0.5)
    ax.tick_params(labelsize=TICK_SIZE, direction='out', length=2, width=0.5, pad=2)
    ax.set_axisbelow(True)
    ax.grid(True, axis=grid_axis, color=GRID_COLOR, linewidth=0.35)
