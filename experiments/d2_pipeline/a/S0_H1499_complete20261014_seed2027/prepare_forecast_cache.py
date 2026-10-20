from if_release.runtime import loads as _release_json_loads
from pathlib import Path
import json
import numpy as np
HERE = Path(__file__).resolve().parent
cfg = _release_json_loads((HERE / 'config.json').read_text())
out = HERE / cfg['output']
cache = out / 'cache'
cache.mkdir()
normal = _release_json_loads((out / 'normalization.json').read_text())
mean = np.asarray(normal['mean'])
std = np.asarray(normal['std'])
with np.load(HERE / cfg['data']) as z:
    values = ((z['values'].astype(float) - mean) / std).astype('float32')
    mask = z['valid_observed'].copy()
ho = np.arange(-30 * 1440, 0, 15)
fo = np.arange(0, 1440, 15)
jobs = _release_json_loads((HERE / 'pipeline.json').read_text())['jobs']
temperature_root = HERE / cfg['temperature_reference'] / 'output/temperature'
origins = []
predictions = []
for job in jobs[:-1]:
    with np.load(temperature_root / (job['name'] + '_A2') / 'train_forecasts.npz') as z:
        origins.append(z['origin'])
        predictions.append(z['forecast_temperature_celsius'])
splits = {'train': (np.concatenate(origins), np.concatenate(predictions))}
with np.load(temperature_root / 'final_A2/val_forecasts.npz') as z:
    splits['validation'] = (z['origin'].copy(), z['forecast_temperature_celsius'].copy())
for part, (origins, predictions) in splits.items():
    np.savez_compressed(cache / (part + '.npz'), origin=origins, history=np.stack([values[o + ho] for o in origins]), forecast_temperature_celsius=predictions[:, ::15].astype('float32'), target=np.stack([values[o + fo, 1:] for o in origins]), mask=np.stack([mask[o + fo, 1:] for o in origins]))
report = dict(counts={k: len(v[0]) for k, v in splits.items()}, sample_minutes=15, train_temperature='Strictly chronological A2 OOF; normalization/checkpoint selection before each predicted block', validation_temperature='Final A2 trained on train and selected on val', test_in_adaptation=False)
(cache / 'README.md').write_text('# Synthetic A2 response adaptation cache\n\n' + json.dumps(report, indent=2), encoding='utf-8')
(out / 'forecast_cache_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print('CACHE_PREPARED ' + json.dumps(report), flush=True)
