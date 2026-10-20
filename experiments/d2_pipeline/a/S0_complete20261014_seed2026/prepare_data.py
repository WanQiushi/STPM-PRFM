from if_release.runtime import loads as _release_json_loads
from pathlib import Path
import json
import numpy as np
HERE = Path(__file__).resolve().parent
cfg = _release_json_loads((HERE / 'config.json').read_text(encoding='utf-8'))
out = HERE / cfg['output']
out.mkdir(parents=True)
with np.load(HERE / cfg['data']) as z:
    values = z['values']
    mask = z['valid_observed']
    channels = z['channels'].tolist()
end = cfg['split_days'][0] * 1440
measured = np.where(mask[:end], values[:end].astype(float), np.nan)
mean = np.nanmean(measured, axis=0)
std = np.nanstd(measured, axis=0)
normal = dict(mean=mean.tolist(), std=std.tolist(), channels=channels, fitted_on='Noisy synthetic observations within the training interval only, including temperature')
(out / 'normalization.json').write_text(json.dumps(normal, indent=2), encoding='utf-8')
p = _release_json_loads((HERE.parents[1] / 'data/h1_window_origins_199.json').read_text())
origins = {'train': p['origins']['train'], 'val': p['origins']['validation'], 'test': p['origins']['test']}
protocol = dict(config=cfg, channels=channels, origins=origins, counts={k: len(v) for k, v in origins.items()}, mean=mean.tolist(), std=std.tolist(), H1_identification_uses_known_future_temperature=True, final_forecast_uses_known_future_temperature=False, latent_truth_in_forward=False)
(HERE / 'h1_pretraining_protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
(out / 'preparation_report.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
print('PREPARED ' + json.dumps({'counts': protocol['counts'], 'data': cfg['data'], 'mean': mean.tolist(), 'std': std.tolist()}), flush=True)
