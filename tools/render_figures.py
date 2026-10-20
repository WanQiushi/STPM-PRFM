"""Render manuscript data figures from the explicitly supplied plot caches."""
from pathlib import Path
import argparse
import os
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = {'5': 'd1/plot_fullperiod.py', '6': 'd1/plot_components.py', '7': 'd2_dataset/plot_d2_observations.py', '8': 'd2/plot_fullperiod.py', '9': 'd2/plot_selected_components.py'}
parser = argparse.ArgumentParser()
parser.add_argument('--figure', choices=['all', *SCRIPTS], default='all')
parser.add_argument('--data-dir', type=Path, default=ROOT / 'local_assets/figure_data')
parser.add_argument('--output', type=Path, default=ROOT / 'outputs/figures')
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, IF_FIGURE_DATA=str(args.data_dir.resolve()), IF_FIGURE_OUTPUT=str(args.output.resolve()), MPLBACKEND='Agg')
selected = SCRIPTS if args.figure == 'all' else {args.figure: SCRIPTS[args.figure]}
for number, script in selected.items():
    subprocess.run([sys.executable, str(ROOT / 'plotting' / script)], cwd=ROOT, env=env, check=True)
    print(f'Rendered Figure {number}', flush=True)
