"""Summarize five paired runs; sample standard deviation uses ddof=1."""
from if_release.runtime import loads as _release_json_loads
import argparse
import json
from pathlib import Path
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
records = [_release_json_loads((args.output / 'evaluation' / f'S0_seed{s}' / 'metrics.json').read_text()) for s in range(2024, 2029)]
summary = {'seeds': list(range(2024, 2029)), 'std_ddof': 1, 'scope': 'independent PRFM pretraining and adaptation; matching existing STPM seeds; fixed OOF cache'}
for split in ['validation', 'test']:
    summary[split] = {}
    for metric in ['response_rmse_mm', 'response_mae_mm', 'response_mse_mm2', 'channel_rmse_mm']:
        values = np.asarray([r['metrics'][split]['all_days'][metric] for r in records])
        summary[split][metric] = {'per_seed': values.tolist(), 'mean': values.mean(axis=0).tolist(), 'std': values.std(axis=0, ddof=1).tolist()}
(args.output / 'five_seed_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))
