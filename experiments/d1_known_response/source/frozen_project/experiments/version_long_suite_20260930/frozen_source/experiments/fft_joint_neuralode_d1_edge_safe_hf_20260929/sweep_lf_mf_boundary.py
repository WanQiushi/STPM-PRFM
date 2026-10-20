from __future__ import annotations
from if_release.runtime import loads as _release_json_loads
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(__file__).with_name('model.py')
model_spec = importlib.util.spec_from_file_location('edge_safe_model', MODEL_PATH)
model_module = importlib.util.module_from_spec(model_spec)
model_spec.loader.exec_module(model_module)
EdgeSafeFFT = model_module.EdgeSafeFFT

def load_windows(cfg):
    archive = np.load(ROOT / cfg['data'], allow_pickle=False)
    values = archive['values'].astype(np.float32)
    observed = archive['valid_observed'].astype(bool)
    available = archive['input_available_at']
    fixed = _release_json_loads((ROOT / cfg['fixed_origins']).read_text(encoding='utf-8'))
    step = int(cfg['sample_minutes'])
    train_end = int(cfg['split_days'][0]) * 1440
    validation_end = int(sum(cfg['split_days'][:2])) * 1440
    fixed_validation = set(map(int, fixed['val']))
    history_offsets = np.arange(-int(cfg['history_days']) * 1440, 0, step)
    future_offsets = np.arange(0, 1440, step)
    candidates = np.arange(train_end, validation_end - 1440 + 1, 1440)
    windows = []
    selected = []
    for origin in candidates:
        if int(origin) not in fixed_validation:
            continue
        history_indices = origin + history_offsets
        future_indices = origin + future_offsets
        if available[history_indices].max() >= origin:
            continue
        if not observed[future_indices, 0].all():
            continue
        indices = np.arange(origin - int(cfg['history_days']) * 1440 + 1440, origin + 1440, step, dtype=np.int64)
        window = values[indices]
        if not np.isfinite(window).all():
            continue
        windows.append(window.T)
        selected.append(int(origin))
    return (torch.from_numpy(np.stack(windows)), np.asarray(selected))

def lag_temperature(temperature, tau_hours, dt_hours):
    batch, length = temperature.shape
    tau = np.asarray(tau_hours, dtype=np.float64)
    alpha = 1.0 - np.exp(-dt_hours / tau)
    state = np.repeat(temperature[:, :1], len(tau), axis=1).astype(np.float64)
    sequence = []
    for index in range(length):
        state = state + alpha[None] * (temperature[:, index:index + 1] - state)
        sequence.append(state.copy())
    return np.stack(sequence, axis=1)

def closed_area(x, y):
    return 0.5 * np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1)

def nonlinear_residual_ratio(x, y):
    x_centered = x - x.mean(1, keepdims=True)
    y_centered = y - y.mean(1, keepdims=True)
    slope = np.sum(x_centered * y_centered, axis=1) / np.maximum(np.sum(x_centered * x_centered, axis=1), 1e-12)
    fit = y.mean(1, keepdims=True) + slope[:, None] * x_centered
    return np.sqrt(np.mean((y - fit) ** 2, axis=1)) / np.maximum(np.ptp(y, axis=1), 1e-12)

def bw_ols_r2(x, y, dt_hours):
    values = []
    for window in range(len(x)):
        dx = np.diff(x[window]) / dt_hours
        dy = np.diff(y[window]) / dt_hours
        y_mid = 0.5 * (y[window, :-1] + y[window, 1:])
        design = np.stack((dx, -np.abs(dx) * y_mid, -dx * np.abs(y_mid)), axis=1)
        params = np.linalg.lstsq(design, dy, rcond=None)[0]
        prediction = design @ params
        denominator = np.sum((dy - dy.mean()) ** 2)
        values.append(1.0 - np.sum((dy - prediction) ** 2) / max(denominator, 1e-12))
    return np.asarray(values)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cfg = _release_json_loads(Path(args.config).read_text(encoding='utf-8'))
    windows, origins = load_windows(cfg)
    periods = [24.0, 30.0, 36.0, 43.312984466552734, 48.0, 60.0, 72.0, 96.0]
    tau = [1.298506736755371, 1.236011028289795, 1.290512204170227, 1.2954124212265015]
    results = {}
    representative = {}
    for period in periods:
        local_cfg = dict(cfg)
        local_cfg['init_static_hours'] = period
        decomposer = EdgeSafeFFT(windows.shape[-1], float(cfg['sample_minutes']) / 60.0, local_cfg)
        with torch.no_grad():
            parts = decomposer(windows)
        static = parts['static'].cpu().numpy()
        dynamic = parts['dynamic'].cpu().numpy()
        temperature_mf = dynamic[:, 0]
        displacement_mf = dynamic[:, 1:]
        effective = lag_temperature(temperature_mf, tau, float(cfg['sample_minutes']) / 60.0)
        q = effective[:, -96:]
        y = displacement_mf[:, :, -96:].transpose(0, 2, 1)
        area = np.abs(closed_area(q, y))
        nonlinear = np.stack([nonlinear_residual_ratio(q[:, :, j], y[:, :, j]) for j in range(4)], axis=1)
        bw_r2 = np.stack([bw_ols_r2(q[:, :, j], y[:, :, j], 0.25) for j in range(4)], axis=1)
        low, mid, _ = decomposer.masks()
        frequency = decomposer.frequency.cpu().numpy()
        index_24 = int(np.argmin(np.abs(frequency - 1.0 / 24.0)))
        index_48 = int(np.argmin(np.abs(frequency - 1.0 / 48.0)))
        key = f'{period:g}h'
        results[key] = {'static_period_hours': period, 'mid_mask_at_24h': float(mid[index_24]), 'mid_mask_at_48h': float(mid[index_48]), 'mf_range_mm_median': np.median(np.ptp(y, axis=1), axis=0).tolist(), 'loop_area_median': np.median(area, axis=0).tolist(), 'nonlinear_residual_ratio_median': np.median(nonlinear, axis=0).tolist(), 'bw_n1_ols_r2_median': np.median(bw_r2, axis=0).tolist()}
        representative[key] = {'mf_range_mm': np.ptp(y[10], axis=0).tolist(), 'loop_area': area[10].tolist(), 'nonlinear_residual_ratio': nonlinear[10].tolist(), 'bw_n1_ols_r2': bw_r2[10].tolist()}
    payload = {'windows': int(len(windows)), 'origins': origins.tolist(), 'validation_medians': results, 'representative_index_10': representative}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'lf_mf_boundary_sweep.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload, indent=2))
if __name__ == '__main__':
    main()
