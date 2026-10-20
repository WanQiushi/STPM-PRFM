"""Construct all219 daily as-of-origin histories without changing source data."""
from if_release.runtime import expand as _release_expand, loads as _release_json_loads
from pathlib import Path
from datetime import datetime, timedelta
import json
import numpy as np
ROOT = Path(_release_expand('${WORKSPACE}'))
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'outputs' / HERE.name
DATA = OUT / 'dataset'
DATA.mkdir(parents=True)
raw = np.load(ROOT / 'data/mainbridge_temperature_v2/minute_complete.npz')
values = raw['values'].astype(np.float64)
observed = raw['valid_observed'].astype(bool)
available = raw['input_available_at']
origins = np.arange(511 * 1440, 730 * 1440, 1440)
history = np.lib.format.open_memmap(DATA / 'history_physical.npy', mode='w+', dtype=np.float64, shape=(219, 43200, 5))
history_mask = np.lib.format.open_memmap(DATA / 'history_mask.npy', mode='w+', dtype=bool, shape=(219, 43200, 5))
reports = []
for i, origin in enumerate(origins):
    h = values[origin - 43200:origin].copy()
    bad = available[origin - 43200:origin] >= origin
    for c in range(5):
        positions = np.flatnonzero(~bad[:, c])
        missing = np.flatnonzero(bad[:, c])
        h[missing, c] = np.interp(missing, positions, h[positions, c])
    history[i] = h
    history_mask[i] = observed[origin - 43200:origin]
    reports.append(dict(origin=int(origin), date=(datetime(2022, 1, 1) + timedelta(minutes=int(origin))).date().isoformat(), split='validation' if i < 110 else 'test', historical_entries_reconstructed=bad.sum(0).astype(int).tolist(), original_history_causal=bool(not bad.any()), measured_targets=observed[origin:origin + 1440].sum(0).astype(int).tolist()))
history.flush()
history_mask.flush()
np.save(DATA / 'origins.npy', origins)
np.save(DATA / 'target_physical.npy', np.stack([values[o:o + 1440] for o in origins]))
np.save(DATA / 'target_mask.npy', np.stack([observed[o:o + 1440] for o in origins]))
np.save(DATA / 'temperature_values.npy', values[:, :1])
np.save(DATA / 'temperature_observed.npy', observed[:, :1])
legacy = _release_json_loads((ROOT / 'experiments/minute_joint5_forecast_20261007/dataset/protocol.json').read_text())
policy = _release_json_loads((HERE / 'plan.json').read_text())
manifest = dict(data=str(ROOT / 'data/mainbridge_temperature_v2/minute_complete.npz'), source_version='mainbridge_temperature_v2', protocol='d1_full219_asof_linear_history_v1', input_policy=policy['input_policy'], score_policy=policy['score_policy'], history_shape=[219, 43200, 5], target_shape=[219, 1440, 5], units=['degC', 'mm', 'mm', 'mm', 'mm'], anonymous_labels=['Temperature', 'Pier1', 'Pier2', 'Pier3', 'Pier4'], origin_epoch='2022-01-01T00:00:00; original monitoring local time', origins=origins.tolist(), legacy_origins=legacy['origins'], days=reports)
(DATA / 'manifest.json').write_text(json.dumps(manifest, indent=2))
(DATA / 'README.md').write_text(_release_expand("# D1 full219 frozen-weight evaluation data\n\nSource: ${WORKSPACE}/data/mainbridge_temperature_v2/minute_complete.npz.\nThis is a new evaluation protocol; original data and frozen evaluation lists remain unchanged.\nHistory is physical-unit float64 [219,43200,5], with one forecast origin per midnight.\nValidation: 2023-05-27 through2023-09-13,110days. Test:2023-09-14 through2023-12-31,109days.\nTarget is [219,1440,5]; target_mask marks retained measured values, used for scoring only.\nIndices count minutes since2022-01-01. Channels are temperature and four anonymized bearing displacements.\nOnly historical entries whose original input_available_at>=origin are reconstructed.\nLinear interpolation uses available anchors in the same historical30day window; nearest historical anchor is held at window edges.\nEvery anchor precedes origin. All original available entries are unchanged; no future labels enter model inputs.\nhistory_mask preserves original measured-source indicators; it does not mark reconstruction as measured.\nNormalization remains checkpoint-specific, unchanged from training. No fitting or checkpoint selection occurs here.\nRead with np.load('history_physical.npy',mmap_mode='r'); other arrays follow the filenames.\nBuild code: experiments/d1_full219_frozen_inference_20261017/prepare.py. Complete origin/coverage details:manifest.json.\n"))
summary = {s: dict(days=sum((r['split'] == s for r in reports)), original_causal_days=sum((r['split'] == s and r['original_history_causal'] for r in reports))) for s in ['validation', 'test']}
print(json.dumps(summary), flush=True)
