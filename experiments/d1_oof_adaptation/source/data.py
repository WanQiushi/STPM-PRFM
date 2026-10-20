"""Original-scale inputs and measured-only metrics for the fixed-driver experiment."""
from if_release.runtime import loads as _release_json_loads
import json
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset

def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def build_cache(root, out, cfg, split):
    normal = _release_json_loads((out / 'normalization.json').read_text(encoding='utf-8'))
    with np.load(root / cfg['data']) as data:
        values = data['values'].astype(np.float64)
        observed = data['valid_observed'].astype(bool)
        available = data['input_available_at']
        channels = data['channels'].tolist()
    mean = np.asarray(normal['h1_mean'])
    std = np.asarray(normal['h1_std'])
    with np.load(out / 'temperature_cache' / (split + '.npz')) as forecast:
        origins = forecast['origin'].copy()
        predicted = forecast['prediction_celsius'].astype('float32')
        offsets = forecast['query_offsets_minutes'].copy()
        metadata = {key: forecast[key].copy() for key in forecast.files if key not in ('origin', 'prediction_celsius', 'query_offsets_minutes')}
    history_offsets = np.arange(-cfg['history_days'] * 1440, 0, cfg['sample_minutes'])
    history_indices = origins[:, None] + history_offsets[None]
    future_indices = origins[:, None] + offsets[None]
    arrays = dict(history=((values[history_indices] - mean) / std).astype('float32'), forecast_temperature_celsius=predicted, target=((values[future_indices, 1:] - mean[1:]) / std[1:]).astype('float32'), mask=observed[future_indices, 1:], origin=origins)
    np.savez_compressed(out / 'cache' / (split + '.npz'), **arrays)
    temperature_truth = values[future_indices, :1]
    temperature_mask = observed[future_indices, :1]
    temperature_error = predicted.astype(np.float64) - temperature_truth
    diagnostic = dict(temperature_truth_celsius=temperature_truth, temperature_mask=temperature_mask, origin=origins, query_offsets_minutes=offsets, **metadata)
    np.savez_compressed(out / 'cache' / (split + '_temperature_diagnostics.npz'), **diagnostic)
    counts = arrays['mask'].sum((0, 1))
    tcount = temperature_mask.sum()
    report = dict(windows=len(origins), fields={key: list(value.shape) for key, value in arrays.items()}, measured_displacement_nodes=counts.tolist(), measured_temperature_nodes=int(tcount), forecast_rmse_celsius=float(np.sqrt((temperature_error ** 2 * temperature_mask).sum() / tcount)), forecast_bias_celsius=float((temperature_error * temperature_mask).sum() / tcount), history_available_before_origin=int(np.sum(available[history_indices].max((1, 2)) < origins)), first_origin_minute=int(origins.min()), last_origin_minute=int(origins.max()), channels=channels, data_shape=list(values.shape))
    if split == 'train':
        report['folds'] = {}
        for day in np.unique(metadata['fold_start_day']):
            selected = metadata['fold_start_day'] == day
            error, mask = (temperature_error[selected], temperature_mask[selected])
            report['folds'][str(int(day))] = dict(windows=int(selected.sum()), forecast_rmse_celsius=float(np.sqrt((error ** 2 * mask).sum() / mask.sum())), forecast_bias_celsius=float((error * mask).sum() / mask.sum()))
    return report

class CachedWindows(Dataset):

    def __init__(self, path):
        with np.load(path) as data:
            self.arrays = {key: data[key].copy() for key in ('history', 'forecast_temperature_celsius', 'target', 'mask', 'origin')}

    def __len__(self):
        return len(self.arrays['origin'])

    def __getitem__(self, index):
        return {key: value[index] for key, value in self.arrays.items()}

def response_metrics(arrays, normal):
    error = (arrays['prediction'].astype(np.float64) - arrays['target']) * np.asarray(normal['h1_std'][1:])
    mask = arrays['mask']
    counts = mask.sum((0, 1))
    sse = (error ** 2 * mask).sum((0, 1))
    return dict(pooled_rmse_mm=float(np.sqrt(sse.sum() / counts.sum())), rmse_mm=np.sqrt(sse / counts).tolist(), mae_mm=((np.abs(error) * mask).sum((0, 1)) / counts).tolist(), bias_mm=((error * mask).sum((0, 1)) / counts).tolist(), measured_counts=counts.tolist(), windows=len(arrays['origin']), tail_2h_rmse_mm=np.sqrt((error[:, -8:] ** 2 * mask[:, -8:]).sum((0, 1)) / mask[:, -8:].sum((0, 1))).tolist())
