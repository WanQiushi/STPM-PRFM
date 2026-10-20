from if_release.runtime import expand as _release_expand
from pathlib import Path
import json
import numpy as np
root = Path(_release_expand('${WORKSPACE}'))
dest = root / 'data/mainbridge_minute_v1'
dest.mkdir(exist_ok=True)
keep = [0, 2, 3, 4, 5]
summary = {}
for source, name in [('data/processed/minute_observed.npz', 'minute_observed.npz'), ('data/complete_minute_v1/minute_complete.npz', 'minute_complete.npz')]:
    z = np.load(root / source)
    out = {}
    for key in z.files:
        a = z[key]
        if key == 'input_indices':
            out[key] = np.arange(5)
        elif key == 'output_indices':
            out[key] = np.arange(1, 5)
        elif key in ['channels', 'units']:
            out[key] = a[keep]
        elif a.ndim == 2:
            out[key] = a[:, keep]
        else:
            out[key] = a
    np.savez_compressed(dest / name, **out)
    summary[name] = {'source': source, 'shape': list(out['values'].shape), 'channels': out['channels'].tolist(), 'nonfinite_values': int((~np.isfinite(out['values'])).sum()), 'input_indices': out['input_indices'].tolist(), 'output_indices': out['output_indices'].tolist()}
(dest / 'build_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(json.dumps(summary, ensure_ascii=False))
