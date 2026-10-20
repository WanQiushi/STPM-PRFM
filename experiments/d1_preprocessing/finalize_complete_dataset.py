"""User-requested boundary fill: first-minute four values from next minute."""
from pathlib import Path
import json
import numpy as np
root = Path(__file__).resolve().parents[1]
source = root / 'data/imputation_latc28_20260923/minute_imputed.npz'
out = root / 'data/complete_minute_v1'
out.mkdir(parents=True, exist_ok=True)
z = np.load(source, allow_pickle=False)
payload = {k: z[k] for k in z.files}
x = payload['values'].copy()
before = x.copy()
method = payload['imputation_method'].copy()
channels = payload['channels'].tolist()
changed = []
for name in ['Temperature', 'Pier3', 'Pier4', 'Auxiliary2']:
    j = channels.index(name)
    x[0, j] = x[1, j]
    method[0, j] = 4
    changed.append({'sensor_id': name, 'timestamp': str(payload['timestamps'][0]), 'source_timestamp': str(payload['timestamps'][1]), 'value': float(x[0, j])})
payload.update(values=x, imputation_method=method, available=np.isfinite(x))
np.savez_compressed(out / 'minute_complete.npz', **payload)
summary = {'source': str(source), 'output': str(out / 'minute_complete.npz'), 'shape': list(x.shape), 'remaining_nonfinite': int((~np.isfinite(x)).sum()), 'complete_minute_rows': int(np.isfinite(x).all(axis=1).sum()), 'unchanged_previous_finite_values': bool(np.array_equal(x[np.isfinite(before)], before[np.isfinite(before)])), 'boundary_fills': changed, 'method_counts': {str(k): int((method == k).sum()) for k in range(5)}}
(out / 'completion_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
