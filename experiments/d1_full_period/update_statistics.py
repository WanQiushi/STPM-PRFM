from if_release.runtime import expand as _release_expand, loads as _release_json_loads
from pathlib import Path
import json
import numpy as np
ROOT = Path(__file__).resolve().parent
EVALUATION = Path(_release_expand('${RESULTS}\\d1_prfm_full219_paired_20261020_2042\\completed_20261020_2045\\evaluation'))
TRUTH = Path(_release_expand('${RESULTS}\\d1_bestday_overlay_baseline8_20261017\\predictions_seed2024_full109.npz'))
ztruth = np.load(TRUTH)
truth = ztruth['measured'][:, :, 1:]
rows = []
for seed in range(2024, 2029):
    z = np.load(EVALUATION / f'S0_seed{seed}/test_predictions.npz')
    prediction = (z['prediction'].astype(np.float64) * z['std'] + z['mean'])[:, :, 1:]
    mask = z['mask'][:, :, 1:]
    row = {'seed': seed}
    for name, operation in [('maximum', np.argmax), ('minimum', np.argmin)]:
        errors, times = ([], [])
        for day in range(109):
            for channel in range(4):
                valid = mask[day, :, channel]
                minutes = np.flatnonzero(valid)
                observed = truth[day, valid, channel]
                predicted = prediction[day, valid, channel]
                iy, ip = (int(operation(observed)), int(operation(predicted)))
                errors.append(abs(predicted[ip] - observed[iy]))
                times.append(abs(int(minutes[ip]) - int(minutes[iy])) / 60)
        row[name + '_mae_mm'] = float(np.mean(errors))
        row[name + '_time_mae_h'] = float(np.mean(times))
    rows.append(row)
summary = {key: {'mean': float(np.mean([r[key] for r in rows])), 'sample_std': float(np.std([r[key] for r in rows], ddof=1))} for key in rows[0] if key != 'seed'}
result = {'windows': 109, 'locations': 4, 'pairs_per_seed': 436, 'seed_metrics': rows, 'summary': summary, 'truth_source': str(TRUTH)}
(ROOT / 'daily_extrema_updated.json').write_text(json.dumps(result, indent=2))
comparison = _release_json_loads((EVALUATION.parent / 'comparison.json').read_text())
old = Path(_release_expand('${RESULTS}\\d1_full219_frozen_inference_20261017'))
baseline = np.mean([_release_json_loads((old / f'PCFNet_seed{s}/metrics.json').read_text())['metrics']['test']['all_days']['channel_rmse_mm'] for s in range(2024, 2029)], axis=0)
proposed = np.array(comparison['test']['channel_rmse_mm']['new']['mean'])
gains = (1 - proposed / baseline) * 100
(ROOT / 'location_gains.json').write_text(json.dumps(gains.tolist(), indent=2))
print(json.dumps(summary, indent=2))
print('LOCATION_GAINS', gains.tolist())
