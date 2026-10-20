"""Fixed adapted-H1 evaluation on each native grid and shared30min grid; no test."""
from if_release.runtime import loads as _release_json_loads
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import torch
from temperature_metrics import decomposition
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'temperature_prediction/outputs' / HERE.name
SOURCE = ROOT / 'outputs/va_1_20261004'
PREVIOUS = ROOT / 'temperature_prediction/outputs/va_2_2_five_seed_20261008'

def score(prediction, target, mask):
    error = prediction.astype(np.float64) - target.astype(np.float64)
    count = mask.sum((0, 1))
    sse = (error ** 2 * mask).sum((0, 1))
    return {'pooled_rmse': float(np.sqrt(sse.sum() / count.sum())), 'channel_rmse': np.sqrt(sse / count).tolist(), 'channel_mae': ((np.abs(error) * mask).sum((0, 1)) / count).tolist(), 'channel_bias': ((error * mask).sum((0, 1)) / count).tolist(), 'measured_nodes': int(count.sum()), 'measured_nodes_per_channel': count.astype(int).tolist(), 'windows': len(prediction)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-minutes', type=int, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    dt = args.sample_minutes
    torch.set_num_threads(4)
    device = torch.device(args.device)
    directory = OUT / f'evaluation_{dt}min'
    directory.mkdir()
    with np.load(OUT / f'datasets/h1_{dt}min.npz') as raw:
        data = {key: raw[key].copy() for key in raw.files}
    ytruth = data['target'].astype(np.float64) * data['std'][1:] + data['mean'][1:]
    ystep = 30 // dt
    model = None
    if dt != 15:
        settings = _release_json_loads((SOURCE / 'settings.json').read_text())
        sys.path.insert(0, str(SOURCE / 'frozen_project' / settings['h1_source']))
        from hybrid_model import HybridKqsz
        original = torch.load(SOURCE / 'h1_original.pt', map_location='cpu', weights_only=False)
        adapted = torch.load(SOURCE / 'best_va1.pt', map_location='cpu', weights_only=False)
        cfg = {**original['config'], 'sample_minutes': dt}
        model = HybridKqsz(cfg)
        state = {key[len('h1.'):]: value for key, value in adapted['model'].items() if key.startswith('h1.')}
        state['decomposer.frequency'] = model.decomposer.frequency
        state['decomposer.padded_frequency'] = model.decomposer.padded_frequency
        model.load_state_dict(state)
        model = model.to(device).eval()
        model.requires_grad_(False)
    results = []
    daily = []
    for seed in range(2024, 2029):
        name = f'{dt}min_seed{seed}'
        tdir = PREVIOUS / f'trend_seed{seed}' if dt == 15 else OUT / name
        with np.load(tdir / 'validation_predictions.npz') as raw:
            t = {key: raw[key].copy() for key in ['prediction', 'target', 'mask', 'origin']}
        index = {int(o): i for i, o in enumerate(t['origin'])}
        rows = np.array([index[int(o)] for o in data['origin']])
        prediction_t = t['prediction'][rows]
        stored = {key: [] for key in ['prediction', 'prediction_mm', 'lf_prediction', 'mf_prediction']}
        if dt == 15:
            with np.load(PREVIOUS / f'fixed_h1_validation/va1_best23__trend_seed{seed}.npz') as raw:
                yindex = {int(o): i for i, o in enumerate(raw['origin'])}
                yrows = np.array([yindex[int(o)] for o in data['origin']])
                arrays = {key: raw[key][yrows].copy() for key in stored}
        else:
            with torch.no_grad():
                for start in range(0, len(rows), 16):
                    sl = slice(start, start + 16)
                    future_t = (prediction_t[sl] - data['mean'][:1]) / data['std'][:1]
                    result = model(torch.as_tensor(data['history'][sl], dtype=torch.float32, device=device), torch.as_tensor(future_t, dtype=torch.float32, device=device), torch.as_tensor(data['origin'][sl], dtype=torch.long, device=device))
                    result['prediction_mm'] = result['prediction'] * torch.as_tensor(data['std'][1:], dtype=torch.float32, device=device) + torch.as_tensor(data['mean'][1:], dtype=torch.float32, device=device)
                    for key in stored:
                        stored[key].append(result[key].cpu().numpy())
            arrays = {key: np.concatenate(value) for key, value in stored.items()}
        arrays.update(target=data['target'], mask=data['mask'], origin=data['origin'], mean=data['mean'], std=data['std'], input_temperature_celsius=prediction_t, query_offsets_minutes=data['query_offsets_minutes'])
        np.savez_compressed(directory / f'{name}__adapted_h1.npz', **arrays)
        temperature = score(prediction_t, data['temperature_truth_celsius'], data['temperature_mask'])
        common_t = score(prediction_t[:, ::ystep], data['temperature_truth_celsius'][:, ::ystep], data['temperature_mask'][:, ::ystep])
        native_y = score(arrays['prediction_mm'], ytruth, data['mask'])
        common_y = score(arrays['prediction_mm'][:, ::ystep], ytruth[:, ::ystep], data['mask'][:, ::ystep])
        full_t = score(t['prediction'], t['target'], t['mask'])
        full_common_t = score(t['prediction'][:, ::ystep], t['target'][:, ::ystep], t['mask'][:, ::ystep])
        diagnostics, _ = decomposition(t)
        entry = {'sample_minutes': dt, 'seed': seed, 'native_full_temperature_rmse_celsius': full_t['pooled_rmse'], 'common30_full_temperature_rmse_celsius': full_common_t['pooled_rmse'], 'native_common_window_temperature_rmse_celsius': temperature['pooled_rmse'], 'common30_common_window_temperature_rmse_celsius': common_t['pooled_rmse'], 'native_displacement_rmse_mm': native_y['pooled_rmse'], 'common30_displacement_rmse_mm': common_y['pooled_rmse'], 'native_temperature': temperature, 'common30_temperature': common_t, 'native_displacement': native_y, 'common30_displacement': common_y, 'full_temperature_diagnostics': diagnostics}
        results.append(entry)
        err = arrays['prediction_mm'].astype(np.float64) - ytruth
        for i, o in enumerate(data['origin']):
            daily.append({'sample_minutes': dt, 'seed': seed, 'origin': int(o), 'date': str(pd.Timestamp('2022-01-01') + pd.to_timedelta(o, unit='m')), 'native_rmse_mm': float(np.sqrt((err[i] ** 2 * data['mask'][i]).sum() / data['mask'][i].sum())), 'common30_rmse_mm': float(np.sqrt((err[i, ::ystep] ** 2 * data['mask'][i, ::ystep]).sum() / data['mask'][i, ::ystep].sum()))})
        print(json.dumps(entry), flush=True)
    metrics = [key for key in results[0] if key.endswith('_mm') or key.endswith('_celsius')]
    stats = [{'sample_minutes': dt, 'metric': key, 'seeds': 5, 'mean': float(np.mean([r[key] for r in results])), 'sample_standard_deviation': float(np.std([r[key] for r in results], ddof=1))} for key in metrics]
    report = {'version': 'va-2.2', 'sample_minutes': dt, 'status': 'completed', 'finished_beijing': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), 'temperature_seed_count': 5, 'h1_checkpoint_count': 1, 'h1_parameters_updated': False, 'h1_transfer': 'same15-minute-trained learned parameters; native sample_minutes, FFT buffers, padding and ODE dt regenerated', 'common_grid_minutes': 30, 'common_origins': data['origin'].tolist(), 'h1_checkpoint': str(SOURCE / 'best_va1.pt'), 'dataset': str(OUT / f'datasets/h1_{dt}min.npz'), 'test_evaluated': False, 'results': results, 'five_seed_statistics': stats}
    (directory / 'comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, dict)} for r in results]).to_csv(directory / 'per_seed.csv', index=False)
    pd.DataFrame(stats).to_csv(directory / 'five_seed_summary.csv', index=False)
    pd.DataFrame(daily).to_csv(directory / 'response_daily.csv', index=False)
    (directory / 'experiment_record.md').write_text(f'# va-2.2 {dt}min evaluation completed\n\n- Date: ' + report['finished_beijing'] + '; IFpaper_v1.\n- Purpose: evaluate five temperature seeds through one fixed adapted H1 on native and common30min nodes.\n- H1 learned parameters unchanged; native-grid migration changes history FFT and numerical stepping.\n- Dataset/origins: ' + report['dataset'] + '; common windows ' + str(len(data['origin'])) + '.\n- Result mean/sampleSD: ' + json.dumps(stats) + '.\n- Products: comparison.json, per_seed.csv, five_seed_summary.csv, response_daily.csv and five predicted H1 NPZ.\n- Future labels only scored; no test evaluation, no plot, no ensemble or confidence interval.\n', encoding='utf-8')
    print('EVALUATION COMPLETED ' + json.dumps(stats), flush=True)
if __name__ == '__main__':
    main()
