"""Audit the actual frozen full H1 under predicted-temperature forcing."""
from if_release.runtime import expand as _release_expand, loads as _release_json_loads
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import torch
from torch.nn import functional as F
from h1_ablation import H1Ablation, H1_ORIGINAL, core
HERE = Path(__file__).resolve().parent
ROOT = Path(_release_expand('${WORKSPACE}'))
OUT = ROOT / 'outputs' / HERE.name / 'state_diagnostics'
OLD = ROOT / 'outputs/d1_full219_frozen_inference_20261017'

def main():
    OUT.mkdir()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    device = torch.device('cuda:0')
    original = torch.load(H1_ORIGINAL / 'h1_original.pt', map_location='cpu', weights_only=False)
    adapted = torch.load(H1_ORIGINAL / 'best_va1.pt', map_location='cpu', weights_only=False)
    cfg = dict(original['config'], sample_minutes=1, ablation='full')
    model = H1Ablation(cfg).to(device).eval().requires_grad_(False)
    state = {k[3:]: v for k, v in adapted['model'].items() if k.startswith('h1.')}
    state['decomposer.frequency'] = model.decomposer.frequency
    state['decomposer.padded_frequency'] = model.decomposer.padded_frequency
    model.load_state_dict(state)
    norm = _release_json_loads((H1_ORIGINAL / 'normalization.json').read_text())
    mean = np.asarray(norm['h1_mean'])
    std = np.asarray(norm['h1_std'])
    with np.load(OLD / 'S0_seed2024/test_predictions.npz') as saved:
        temp = saved['prediction'][:, :, :1] * saved['std'][:1] + saved['mean'][:1]
        origin = saved['origin'].copy()
        reference = saved['prediction'][:, :, 1:].copy()
    with np.load(OLD / 'S0_seed2024/validation_predictions.npz') as saved:
        vt = saved['prediction'][:, :, 0] * saved['std'][0] + saved['mean'][0]
    val_ranges = np.ptp(vt, axis=1)
    test_ranges = np.ptp(temp[:, :, 0], axis=1)
    hourly = np.diff(temp[:, ::60, 0], axis=1)
    cyclic = (hourly > 0).any(1) & (hourly < 0).any(1)
    candidates = np.flatnonzero(cyclic)
    cyclic_day = int(candidates[np.argmin(np.abs(test_ranges[candidates] - np.median(val_ranges)))])
    flat_day = int(np.argmin(np.abs(test_ranges - np.quantile(val_ranges, 0.25))))
    selection = dict(temperature_seed=2024, cycle_day_index=cyclic_day, flat_day_index=flat_day, origins=[int(origin[cyclic_day]), int(origin[flat_day])], rule='predicted temperature range nearest validation median with both hourly directions, and nearest validation q25; chronological ties', response_error_used_for_selection=False)
    (OUT / 'case_selection.json').write_text(json.dumps(selection, indent=2))
    history = np.load(OLD / 'dataset/history_physical.npy', mmap_mode='r')
    outputs = {}
    fields = ['prediction', 'lf_prediction', 'reversible_prediction', 'memory_prediction', 'temperature_mf', 'effective_temperature_mf', 'effective_temperature_rate', 'bw_latent', 'pure_bw_rate', 'raw_neural_rate']
    with torch.inference_mode():
        for i in range(0, len(origin), 16):
            j = min(i + 16, len(origin))
            result = model(torch.as_tensor(((history[110 + i:110 + j] - mean) / std).astype('float32'), device=device), torch.as_tensor(((temp[i:j] - mean[:1]) / std[:1]).astype('float32'), device=device), torch.as_tensor(origin[i:j], device=device))
            for k in fields:
                outputs.setdefault(k, []).append(result[k].cpu().numpy())
    outputs = {k: np.concatenate(v) for k, v in outputs.items()}
    np.savez_compressed(OUT / 'full_predicted_temperature_states.npz', **outputs, origin=origin, temperature_celsius=temp, mean=mean, std=std)
    beta = F.softplus(model.beta_raw)
    calendar = model.season_features(torch.as_tensor(origin, device=device), 1440)
    a = 1 + cfg['seasonal_a_scale'] * torch.tanh(calendar @ model.a_season_coeff.T)
    params = dict(tau_hours=model.thermal_lag_hours().tolist(), k=model.k.tolist(), s=model.s.tolist(), beta=beta.tolist(), gamma=(0.9 * beta * torch.tanh(model.gamma_raw)).tolist(), a_test_min=a.amin((0, 1)).tolist(), a_test_max=a.amax((0, 1)).tolist(), response_models=1, temperature_seed=2024, checkpoint=str(H1_ORIGINAL / 'best_va1.pt'))
    scale = std[1:] * model.s.cpu().numpy()
    p = outputs['pure_bw_rate'] * scale
    r = outputs['raw_neural_rate'] * scale
    rms = lambda x: np.sqrt(np.mean(x.astype('float64') ** 2, axis=(0, 1)))
    rate = dict(physics_rms_mm_per_h=rms(p).tolist(), neural_rms_mm_per_h=rms(r).tolist(), reversible_rms_mm_per_h=rms(outputs['effective_temperature_rate'] * (std[1:] * model.k.cpu().numpy())).tolist(), amplitude_fraction_not_causal=(np.abs(r).sum((0, 1)) / (np.abs(p).sum((0, 1)) + np.abs(r).sum((0, 1)))).tolist(), physics_neural_correlation=[float(np.corrcoef(p[:, :, m].ravel(), r[:, :, m].ravel())[0, 1]) for m in range(4)], max_prediction_difference_from_original_cache_mm=float(np.max(np.abs(outputs['prediction'] - reference) * std[1:])))
    rate['output_rms_mm'] = {k: rms(outputs[k] * std[1:]).tolist() for k in ['lf_prediction', 'reversible_prediction', 'memory_prediction']}
    rate['physics_abs_integral_mm_per_day'] = (np.abs(p).sum(1) / 60).mean(0).tolist()
    rate['neural_abs_integral_mm_per_day'] = (np.abs(r).sum(1) / 60).mean(0).tolist()
    dq = outputs['effective_temperature_rate']
    rate['by_direction'] = {}
    for direction, mask in [('heating', dq > 0), ('cooling', dq < 0)]:
        rate['by_direction'][direction] = {'physics_rms_mm_per_h': np.sqrt(np.where(mask, p ** 2, 0).sum((0, 1)) / mask.sum((0, 1))).tolist(), 'neural_rms_mm_per_h': np.sqrt(np.where(mask, r ** 2, 0).sum((0, 1)) / mask.sum((0, 1))).tolist()}
    (OUT / 'parameters.json').write_text(json.dumps(params, indent=2))
    (OUT / 'rate_amplitudes.json').write_text(json.dumps(rate, indent=2))
    probes = []
    with torch.inference_mode():
        for day in [cyclic_day, flat_day]:
            z0 = torch.as_tensor(outputs['bw_latent'][day:day + 1, 720], device=device)
            q0 = torch.as_tensor(outputs['effective_temperature_mf'][day:day + 1, 720], device=device)
            u = torch.as_tensor(outputs['temperature_mf'][day:day + 1, 720], device=device)
            season = calendar[day:day + 1, 720]
            for mode in ['fixed_q', 'fixed_u', 'fixed_q_without_R']:
                z = z0.clone()
                q = q0.clone()
                for step in range(1, 1441):
                    nq = model.lag_step(q, u) if mode == 'fixed_u' else q
                    dq = (nq - q) / model.step_hours
                    field = lambda zz: model.bw_rate(zz, dq, nq, season)[1 if mode == 'fixed_q_without_R' else 0]
                    z = core.rk4(z, field, model.step_hours)
                    q = nq
                    if step in [60, 360, 1440]:
                        probes.append(dict(day_index=day, origin=int(origin[day]), mode=mode, hours=step / 60, state_change=(z - z0).cpu().numpy().tolist(), memory_change_mm=((z - z0) * torch.as_tensor(scale, device=device)).cpu().numpy().tolist()))
    (OUT / 'constant_driver_probes.json').write_text(json.dumps(probes, indent=2))
    (OUT / 'experiment_record.md').write_text(f"# Frozen full-model state audit\n\n- Date: {datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()}; IFpaper_v1.\n- Driver: full STPM seed2024 forecasts on 109 original test days; original shared H1, no training.\n- Products: case_selection.json, full_predicted_temperature_states.npz, parameters.json, rate_amplitudes.json, constant_driver_probes.json.\n- Rates distinguish reversible, pure BW and neural terms; amplitude fractions are not causal contribution.\n- Original-cache prediction difference: {rate['max_prediction_difference_from_original_cache_mm']} mm.\n")
    print('DIAGNOSTICS_COMPLETED ' + str(OUT), flush=True)
if __name__ == '__main__':
    main()
