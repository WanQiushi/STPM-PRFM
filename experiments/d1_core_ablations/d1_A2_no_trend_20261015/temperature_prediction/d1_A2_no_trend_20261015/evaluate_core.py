"""Native1min common-node evaluation for all remaining core ablations; CPU4."""
from if_release.runtime import loads as _release_json_loads
import argparse
import json
import statistics
from pathlib import Path
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import torch
from h1_ablation import H1Ablation, HybridKqsz, H1_ORIGINAL
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'temperature_prediction/outputs' / HERE.name
REFERENCE = ROOT / 'temperature_prediction/outputs/va_2_2_sampling_20261008'

def score(prediction, target, mask):
    error = prediction.astype(np.float64) - target.astype(np.float64)
    count = mask.sum((0, 1))
    sse = (error ** 2 * mask).sum((0, 1))
    sae = (np.abs(error) * mask).sum((0, 1))
    return {'rmse': float(np.sqrt(sse.sum() / count.sum())), 'mae': float(sae.sum() / count.sum()), 'mse': float(sse.sum() / count.sum()), 'measured_nodes': int(count.sum()), 'channel_rmse': np.sqrt(sse / count).tolist(), 'channel_mae': (sae / count).tolist(), 'channel_bias': ((error * mask).sum((0, 1)) / count).tolist()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', required=True)
    parser.add_argument('--seeds', required=True)
    args = parser.parse_args()
    group = args.group
    seeds = [int(value) for value in args.seeds.split(',')]
    torch.set_num_threads(4)
    device = torch.device('cpu')
    original = torch.load(H1_ORIGINAL / 'h1_original.pt', map_location='cpu', weights_only=False)
    if group in ['A5', 'A6']:
        trained = torch.load(OUT / (group + '_h1/adapted_best.pt'), map_location='cpu', weights_only=False)
        cfg = {**trained['config'], 'sample_minutes': 1}
        model = H1Ablation(cfg)
        state = trained['model']
    elif group == 'A7':
        model = HybridKqsz({**original['config'], 'sample_minutes': 1})
        state = original['model']
    else:
        adapted = torch.load(H1_ORIGINAL / 'best_va1.pt', map_location='cpu', weights_only=False)
        model = HybridKqsz({**original['config'], 'sample_minutes': 1})
        state = {key[len('h1.'):]: value for key, value in adapted['model'].items() if key.startswith('h1.')}
    state['decomposer.frequency'] = model.decomposer.frequency
    state['decomposer.padded_frequency'] = model.decomposer.padded_frequency
    model.load_state_dict(state)
    model = model.to(device).eval()
    model.requires_grad_(False)
    with np.load(REFERENCE / 'datasets/h1_1min.npz') as raw:
        data = {key: raw[key].copy() for key in raw.files}
    ytruth = data['target'].astype(np.float64) * data['std'][1:] + data['mean'][1:]
    for seed in seeds:
        folder = OUT / f'evaluation_{group}/seed{seed}'
        folder.mkdir(parents=True)
        start = time.time()
        source = OUT / f'A2_lambda0_seed{seed}' if group in ['A1', 'A2', 'A4'] else REFERENCE / f'1min_seed{seed}'
        with np.load(source / 'validation_predictions.npz') as raw:
            temperature = {key: raw[key].copy() for key in ['prediction', 'target', 'mask', 'origin']}
        index = {int(origin): i for i, origin in enumerate(temperature['origin'])}
        trows = np.array([index[int(origin)] for origin in data['origin']])
        input_t = temperature['prediction'][trows]
        saved = {key: [] for key in ['prediction', 'lf_prediction', 'mf_prediction', 'effective_temperature_mf', 'reversible_prediction', 'memory_prediction', 'bw_latent']}
        with torch.no_grad():
            for start_index in range(0, len(trows), 16):
                sl = slice(start_index, start_index + 16)
                result = model(torch.as_tensor(data['history'][sl], dtype=torch.float32, device=device), torch.as_tensor((input_t[sl] - data['mean'][:1]) / data['std'][:1], dtype=torch.float32, device=device), torch.as_tensor(data['origin'][sl], dtype=torch.long, device=device))
                for key in saved:
                    saved[key].append(result[key].cpu().numpy())
        arrays = {key: np.concatenate(value) for key, value in saved.items()}
        prediction_mm = arrays['prediction'].astype(np.float64) * data['std'][1:] + data['mean'][1:]
        reference_path = ROOT / 'temperature_prediction/outputs/va_core_ablation_all_20261008' / f'evaluation_A2/seed{seed}/metrics.json'
        reference_metrics = _release_json_loads(reference_path.read_text())
        y = score(prediction_mm, ytruth, data['mask'])
        full_y = dict(rmse=reference_metrics['displacement_rmse_mm'], mae=reference_metrics['displacement_mae_mm'])
        t = score(input_t, data['temperature_truth_celsius'], data['temperature_mask'])
        full_t = score(temperature['prediction'], temperature['target'], temperature['mask'])
        if group in ['A1', 'A2', 'A4']:
            params_t = _release_json_loads((source / 'summary.json').read_text())['parameters']['total']
        else:
            params_t = 78523
        row = {'group': group, 'seed': seed, 'sample_minutes': 1, 'windows': len(trows), 'displacement_rmse_mm': y['rmse'], 'displacement_mae_mm': y['mae'], 'displacement_mse_mm2': y['mse'], 'temperature_rmse_celsius': t['rmse'], 'temperature_mae_celsius': t['mae'], 'full66_temperature_rmse_celsius': full_t['rmse'], 'full66_temperature_mae_celsius': full_t['mae'], 'tail2h_displacement_rmse_mm': score(prediction_mm[:, -120:], ytruth[:, -120:], data['mask'][:, -120:])['rmse'], 'reference_A2_displacement_rmse_mm': full_y['rmse'], 'reference_A2_displacement_mae_mm': full_y['mae'], 'delta_rmse_mm': y['rmse'] - full_y['rmse'], 'delta_mae_mm': y['mae'] - full_y['mae'], 'channel_rmse_mm': y['channel_rmse'], 'channel_mae_mm': y['channel_mae'], 'channel_bias_mm': y['channel_bias'], 'observed_y_nodes': y['measured_nodes'], 'observed_t_nodes': t['measured_nodes'], 'parameters': params_t + sum((p.numel() for p in model.parameters())), 'elapsed_evaluation_seconds': time.time() - start, 'test_evaluated': False, 'device': 'cpu', 'trend_weight': 0.0, 'reference_metrics': str(reference_path), 'h1_checkpoint_count': 1, 'temperature_source': str(source), 'finished_beijing': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()}
        (folder / 'metrics.json').write_text(json.dumps(row, indent=2), encoding='utf-8')
        np.savez_compressed(folder / 'predictions.npz', **arrays, prediction_mm=prediction_mm, target=data['target'], mask=data['mask'], origin=data['origin'], mean=data['mean'], std=data['std'], input_temperature_celsius=input_t, query_offsets_minutes=data['query_offsets_minutes'])
        (folder / 'experiment_record.md').write_text('# va core ablation ' + group + '\n\n- Date: ' + row['finished_beijing'] + '; IFpaper_v1.\n- Operation: fixed native1min26-window comparison, CPU4 inference after training or reuse; no test/future true T input.\n- Result: ' + json.dumps(row) + '.\n- Products: metrics.json, predictions.npz; source/config archived under experiment source and training outputs.\n', encoding='utf-8')
        print(json.dumps(row), flush=True)
if __name__ == '__main__':
    main()
