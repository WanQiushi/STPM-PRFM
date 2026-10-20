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
BASE_MODEL = ROOT / 'experiments' / 'fft_joint_neuralode_d1_20260929' / 'model.py'
base_spec = importlib.util.spec_from_file_location('periodic_fft_model', BASE_MODEL)
base_module = importlib.util.module_from_spec(base_spec)
base_spec.loader.exec_module(base_module)
LearnableFFT = base_module.LearnableFFT

def load_windows(cfg):
    archive = np.load(ROOT / cfg['data'], allow_pickle=False)
    values = archive['values'].astype(np.float32)
    observed = archive['valid_observed'].astype(bool)
    available = archive['input_available_at']
    fixed = _release_json_loads((ROOT / cfg['fixed_origins']).read_text(encoding='utf-8'))
    step = int(cfg['sample_minutes'])
    points_per_day = 1440 // step
    history_points = int(cfg['history_days']) * points_per_day
    train_end = int(cfg['split_days'][0]) * 1440
    validation_end = int(sum(cfg['split_days'][:2])) * 1440
    fixed_validation = set(map(int, fixed['val']))
    candidates = np.arange(train_end, validation_end - 1440 + 1, 1440)
    windows = []
    selected = []
    history_offsets = np.arange(-int(cfg['history_days']) * 1440, 0, step)
    future_offsets = np.arange(0, 1440, step)
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
        if len(indices) != history_points:
            continue
        window = values[indices, 1:]
        if not np.isfinite(window).all():
            continue
        windows.append(window.T)
        selected.append(int(origin))
    return (torch.from_numpy(np.stack(windows)), np.asarray(selected), points_per_day)

def decompose(module, windows):
    with torch.no_grad():
        parts = module(windows)
    return {key: value.cpu().numpy() for key, value in parts.items() if key in {'static', 'dynamic', 'noise'}}

def summarize(parts, points_per_day):
    hf = parts['noise'][:, :, -points_per_day:]
    mf = parts['dynamic'][:, :, -points_per_day:]
    tail = hf[:, :, -8:]
    channel_corr = []
    for item in hf:
        corr = np.corrcoef(item)
        channel_corr.append(np.mean(np.abs(corr[np.triu_indices(4, 1)])))
    return {'hf_rms_mm': np.sqrt(np.mean(hf ** 2, axis=(0, 2))).tolist(), 'hf_tail_2h_rms_mm': np.sqrt(np.mean(tail ** 2, axis=(0, 2))).tolist(), 'hf_absmax_mm': np.max(np.abs(hf), axis=(0, 2)).tolist(), 'hf_endpoint_abs_mean_mm': np.mean(np.abs(hf[:, :, -1]), axis=0).tolist(), 'hf_common_mode_abs_correlation_mean': float(np.mean(channel_corr)), 'mf_16h_to_end_change_mm_mean': np.mean(mf[:, :, -1] - mf[:, :, -32], axis=0).tolist()}

def summarize_representative(parts, index):
    static = parts['static'][index]
    dynamic = parts['dynamic'][index]
    noise = parts['noise'][index]
    complete = static + dynamic + noise
    return {'full_16h_to_end_change_mm': (complete[:, -1] - complete[:, -32]).tolist(), 'lf_16h_to_end_change_mm': (static[:, -1] - static[:, -32]).tolist(), 'mf_16h_to_end_change_mm': (dynamic[:, -1] - dynamic[:, -32]).tolist(), 'hf_16h_to_end_change_mm': (noise[:, -1] - noise[:, -32]).tolist(), 'hf_absmax_24h_mm': np.max(np.abs(noise[:, -96:]), axis=1).tolist()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cfg = _release_json_loads(Path(args.config).read_text(encoding='utf-8'))
    windows, origins, points_per_day = load_windows(cfg)
    periodic_cfg = dict(cfg)
    periodic_cfg['noise_hours_range'] = [0.5, 8.0]
    periodic_cfg['init_noise_hours'] = 7.376662731170654
    periodic = LearnableFFT(windows.shape[-1], cfg['sample_minutes'] / 60.0, periodic_cfg)
    fixed1_cfg = dict(cfg)
    fixed1_cfg['init_noise_hours'] = 1.0
    fixed1 = EdgeSafeFFT(windows.shape[-1], cfg['sample_minutes'] / 60.0, fixed1_cfg)
    fixed2_cfg = dict(cfg)
    fixed2_cfg['init_noise_hours'] = 1.999
    fixed2 = EdgeSafeFFT(windows.shape[-1], cfg['sample_minutes'] / 60.0, fixed2_cfg)
    variants = {'periodic_7p376h': decompose(periodic, windows), 'edge_safe_1h': decompose(fixed1, windows), 'edge_safe_2h': decompose(fixed2, windows)}
    metrics = {name: summarize(parts, points_per_day) for name, parts in variants.items()}
    metrics['representative'] = {name: summarize_representative(parts, 10) for name, parts in variants.items()}
    metrics['windows'] = int(len(windows))
    metrics['representative_index'] = 10
    metrics['representative_origin'] = int(origins[10])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'decomposition_metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    payload = {'origins': origins}
    for name, parts in variants.items():
        for component, values in parts.items():
            payload[f'{name}_{component}'] = values
    np.savez_compressed(output / 'decomposition_components.npz', **payload)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
