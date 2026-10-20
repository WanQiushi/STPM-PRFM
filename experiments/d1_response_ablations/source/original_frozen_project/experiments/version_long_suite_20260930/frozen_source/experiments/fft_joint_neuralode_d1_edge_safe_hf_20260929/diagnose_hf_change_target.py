from __future__ import annotations
from if_release.runtime import loads as _release_json_loads
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
from sweep_lf_mf_boundary import EdgeSafeFFT, bw_ols_r2, closed_area, lag_temperature, load_windows, nonlinear_residual_ratio
ROOT = Path(__file__).resolve().parents[2]
BASE_MODEL = ROOT / 'experiments' / 'fft_joint_neuralode_d1_20260929' / 'model.py'
spec = importlib.util.spec_from_file_location('circular_fft_model', BASE_MODEL)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
LearnableFFT = module.LearnableFFT

def endpoint_gap_ratio(values):
    return np.abs(values[:, -1] - values[:, 0]) / np.maximum(np.ptp(values, axis=1), 1e-12)

def branch_gap(x, y, bins=12):
    result = np.full(len(x), np.nan)
    for window in range(len(x)):
        dx = np.gradient(x[window])
        edges = np.linspace(x[window].min(), x[window].max(), bins + 1)
        gaps = []
        for low, high in zip(edges[:-1], edges[1:]):
            rising = (x[window] >= low) & (x[window] < high) & (dx > 0)
            falling = (x[window] >= low) & (x[window] < high) & (dx < 0)
            if rising.sum() >= 2 and falling.sum() >= 2:
                gaps.append(abs(np.median(y[window, rising]) - np.median(y[window, falling])))
        if gaps:
            result[window] = np.median(gaps) / max(np.ptp(y[window]), 1e-12)
    return result

def summarize(decomposer, windows, tau, dt):
    with torch.no_grad():
        dynamic = decomposer(windows)['dynamic'].cpu().numpy()
    effective = lag_temperature(dynamic[:, 0], tau, dt)
    x = effective[:, -96:]
    y = dynamic[:, 1:, -96:].transpose(0, 2, 1)
    area = np.abs(closed_area(x, y))
    nonlinear = np.stack([nonlinear_residual_ratio(x[:, :, j], y[:, :, j]) for j in range(4)], axis=1)
    bw_r2 = np.stack([bw_ols_r2(x[:, :, j], y[:, :, j], dt) for j in range(4)], axis=1)
    branch = np.stack([branch_gap(x[:, :, j], y[:, :, j]) for j in range(4)], axis=1)
    x_gap = endpoint_gap_ratio(x)
    y_gap = endpoint_gap_ratio(y)
    representative = 10
    return {'validation_median': {'target_area_C_mm': np.median(area, axis=0).tolist(), 'branch_gap_over_y_range': np.nanmedian(branch, axis=0).tolist(), 'nonlinear_residual_over_y_range': np.median(nonlinear, axis=0).tolist(), 'bw_n1_ols_r2': np.median(bw_r2, axis=0).tolist(), 'x_endpoint_gap_over_range': np.median(x_gap, axis=0).tolist(), 'y_endpoint_gap_over_range': np.median(y_gap, axis=0).tolist(), 'mf_range_mm': np.median(np.ptp(y, axis=1), axis=0).tolist()}, 'representative_index_10': {'target_area_C_mm': area[representative].tolist(), 'branch_gap_over_y_range': branch[representative].tolist(), 'nonlinear_residual_over_y_range': nonlinear[representative].tolist(), 'bw_n1_ols_r2': bw_r2[representative].tolist(), 'x_endpoint_gap_over_range': x_gap[representative].tolist(), 'y_endpoint_gap_over_range': y_gap[representative].tolist(), 'mf_range_mm': np.ptp(y[representative], axis=0).tolist()}}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cfg = _release_json_loads(Path(args.config).read_text(encoding='utf-8'))
    windows, origins = load_windows(cfg)
    dt = float(cfg['sample_minutes']) / 60.0
    tau = [1.298506736755371, 1.236011028289795, 1.290512204170227, 1.2954124212265015]
    methods = {}
    for edge_safe in (False, True):
        for noise_hours in (7.3767, 1.0):
            local = dict(cfg)
            local['static_hours_range'] = [24.0, 168.0]
            local['noise_hours_range'] = [0.5, 8.0]
            local['init_static_hours'] = 43.312984466552734
            local['init_noise_hours'] = noise_hours
            cls = EdgeSafeFFT if edge_safe else LearnableFFT
            decomposer = cls(windows.shape[-1], dt, local)
            key = f"{('edge_safe' if edge_safe else 'circular')}_hf{noise_hours:g}h"
            methods[key] = summarize(decomposer, windows, tau, dt)
    payload = {'windows': int(len(windows)), 'origins': origins.tolist(), 'design': '2x2 decomposition-only ablation; no model training', 'methods': methods}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload, indent=2))
if __name__ == '__main__':
    main()
