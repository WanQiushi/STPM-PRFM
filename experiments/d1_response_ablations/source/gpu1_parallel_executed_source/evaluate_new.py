"""Score new H1 variants using the unchanged full-model STPM forecast caches."""
from if_release.runtime import expand as _release_expand
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from h1_ablation import H1Ablation
ROOT = Path(_release_expand('${WORKSPACE}'))
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'outputs' / HERE.name
OLD = ROOT / 'outputs/d1_full219_frozen_inference_20261017'

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', required=True)
    args = p.parse_args()
    started = time.time()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    device = torch.device('cuda:0')
    trained = torch.load(OUT / args.arm / 'adapted_best.pt', map_location='cpu', weights_only=False)
    cfg = dict(trained['config'], sample_minutes=1)
    model = H1Ablation(cfg).to(device).eval().requires_grad_(False)
    state = trained['model']
    state['decomposer.frequency'] = model.decomposer.frequency
    state['decomposer.padded_frequency'] = model.decomposer.padded_frequency
    model.load_state_dict(state)
    mean = np.asarray(trained['mean'])
    std = np.asarray(trained['std'])
    history = np.load(OLD / 'dataset/history_physical.npy', mmap_mode='r')
    reports = []
    for seed in range(2024, 2029):
        folder = OUT / (args.arm + '_seed' + str(seed))
        folder.mkdir()
        metrics = {}
        for split, first, last in [('validation', 0, 110), ('test', 110, 219)]:
            with np.load(OLD / f'S0_seed{seed}' / (split + '_predictions.npz')) as saved:
                target = saved['target'].copy()
                mask = saved['mask'].copy()
                origin = saved['origin'].copy()
                temperature = saved['prediction'][:, :, :1] * saved['std'][:1] + saved['mean'][:1]
            future = ((temperature - mean[:1]) / std[:1]).astype('float32')
            prediction = []
            with torch.inference_mode():
                for i in range(0, len(origin), 16):
                    j = min(i + 16, len(origin))
                    result = model(torch.as_tensor(((history[first + i:first + j] - mean) / std).astype('float32'), device=device), torch.as_tensor(future[i:j], device=device), torch.as_tensor(origin[i:j], device=device))
                    prediction.append(result['prediction'].cpu().numpy())
            response = np.concatenate(prediction)
            prediction = np.concatenate([future, response], axis=2)
            error = (prediction.astype('float64') - target) * std
            sse = np.where(mask, error ** 2, 0).sum(1)
            sae = np.where(mask, np.abs(error), 0).sum(1)
            n = mask.sum(1)
            metric = dict(response_rmse_mm=float(np.sqrt(sse[:, 1:].sum() / n[:, 1:].sum())), response_mae_mm=float(sae[:, 1:].sum() / n[:, 1:].sum()), per_location_rmse_mm=np.sqrt(sse[:, 1:].sum(0) / n[:, 1:].sum(0)).tolist(), measured_counts=n.sum(0).tolist(), windows=len(origin))
            metrics[split] = metric
            np.savez_compressed(folder / (split + '_predictions.npz'), prediction=prediction, target=target, mask=mask, origin=origin, mean=mean, std=std)
            np.savez_compressed(folder / (split + '_daily_errors.npz'), sse=sse[:, 1:], sae=sae[:, 1:], n=n[:, 1:], origin=origin)
        record = dict(arm=args.arm, temperature_seed=seed, response_seed=2024, metrics=metrics, checkpoint=str(OUT / args.arm / 'adapted_best.pt'), forecast_source=str(OLD / f'S0_seed{seed}'), normalization='original H1 normalization', test_used_for_selection=False)
        (folder / 'metrics.json').write_text(json.dumps(record, indent=2))
        (folder / 'experiment_record.md').write_text(f"# Predicted-temperature ablation evaluation\n\n- Project/date: IFpaper_v1, 2026-10-20.\n- Arm: {args.arm}; temperature seed {seed}; shared H1 seed2024.\n- Driver: unchanged full STPM prediction cache; data protocol: {OLD}/dataset/manifest.json.\n- Result: test RMSE {metrics['test']['response_rmse_mm']} mm.\n- Products: {folder}.\n")
        reports.append(record)
        print(json.dumps(record), flush=True)
    (OUT / (args.arm + '_evaluation_summary.json')).write_text(json.dumps(dict(rows=reports, seconds=time.time() - started), indent=2))
if __name__ == '__main__':
    main()
