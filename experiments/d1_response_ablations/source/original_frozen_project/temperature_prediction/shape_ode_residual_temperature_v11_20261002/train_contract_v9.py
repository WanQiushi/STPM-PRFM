"""Train causal 24-hour d1 temperature forecasts; keep the test split sealed."""
from if_release.runtime import loads as _release_json_loads
import argparse
import json
import random
import shutil
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from data_contract_v8 import ROOT, Windows, dump, load, on_device
from models import build_model

class ForecastWindows(Windows):

    def __getitem__(self, index):
        batch = super().__getitem__(index)
        origin = batch['origin']
        batch['history_mask'] = self.mask[origin - 10081:origin].copy()
        batch['target'] = batch['target'][::15].copy()
        batch['mask'] = batch['mask'][::15].copy()
        batch['previous_day'] = batch['previous_day'][::15].copy()
        return batch

def objective(result, batch):
    error = result['prediction'] - batch['target']
    return (error.square() * batch['mask']).sum() / batch['mask'].sum()

def evaluate(model, loader, device, mean, std, collect=False):
    model.eval()
    sse = sae = previous_sse = count = 0.0
    block_sse, block_count = (np.zeros(4), np.zeros(4))
    hourly_sse, hourly_count = (np.zeros(24), np.zeros(24))
    stored = {key: [] for key in ('origin', 'prediction', 'target', 'mask')}
    with torch.no_grad():
        for batch in loader:
            current = on_device(batch, device)
            prediction = model(current)['prediction']
            error = prediction - current['target']
            mask = current['mask']
            sse += (error.square() * mask).sum().item()
            sae += (error.abs() * mask).sum().item()
            previous_sse += ((current['previous_day'] - current['target']).square() * mask).sum().item()
            count += mask.sum().item()
            for block in range(4):
                section = slice(block * 24, (block + 1) * 24)
                block_sse[block] += (error[:, section].square() * mask[:, section]).sum().item()
                block_count[block] += mask[:, section].sum().item()
            for hour in range(24):
                section = slice(hour * 4, (hour + 1) * 4)
                hourly_sse[hour] += (error[:, section].square() * mask[:, section]).sum().item()
                hourly_count[hour] += mask[:, section].sum().item()
            if collect:
                stored['origin'].append(batch['origin'].numpy())
                stored['prediction'].append(prediction.cpu().numpy() * std + mean)
                stored['target'].append(batch['target'].numpy() * std + mean)
                stored['mask'].append(batch['mask'].numpy())
    metrics = {'temperature_mse_normalized': sse / count, 'rmse_celsius': float(np.sqrt(sse / count) * std[0]), 'mae_celsius': float(sae / count * std[0]), 'previous_day_rmse_celsius': float(np.sqrt(previous_sse / count) * std[0]), 'six_hour_rmse_celsius': (np.sqrt(block_sse / block_count) * std[0]).tolist(), 'first_12h_rmse_celsius': float(np.sqrt(block_sse[:2].sum() / block_count[:2].sum()) * std[0]), 'last_12h_rmse_celsius': float(np.sqrt(block_sse[2:].sum() / block_count[2:].sum()) * std[0]), 'hourly_rmse_celsius': (np.sqrt(hourly_sse / hourly_count) * std[0]).tolist(), 'measured_query_points': int(count)}
    if collect:
        return (metrics, {key: np.concatenate(value, axis=0) for key, value in stored.items()})
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    cfg = _release_json_loads(Path(args.config).read_text(encoding='utf-8'))
    output = ROOT / 'temperature_prediction' / 'outputs' / 'latent_ode_temperature_v9' / cfg['run_name']
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(Path(__file__).resolve().parent, output / 'frozen_source', ignore=shutil.ignore_patterns('__pycache__'))
    dump(output / 'config.json', cfg)
    values, mask, sums, counts, parts, mean, std, report = load(cfg)
    dump(output / 'data_report.json', {'counts': report, 'measured_query_points': {name: int(sum((mask[o:o + 1440:15, 0].sum() for o in origins))) for name, origins in parts.items()}, 'mean_celsius': float(mean[0]), 'std_celsius': float(std[0]), 'data': cfg['data'], 'channel': 'Temperature', 'sensor_type': 'unconfirmed', 'history_policy': 'seven prior days aggregated to 672 quarter-hour means and measured coverage; 30 prior measured daily means and coverage; known calendar; same inputs in both models', 'query_offsets_minutes': list(range(0, 1440, 15)), 'anchor': 'last causal historical value at origin-1 minute', 'normalization': 'training-period measured values only', 'availability_filter': 'all 10081 historical input_available_at values strictly less than origin; inherited v8 contract', 'checkpoint_selection': 'validation observed-node temperature MSE', 'test_evaluated': False})
    dump(output / 'window_origins.json', parts)
    print('DATA ' + json.dumps(report), flush=True)
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.set_num_threads(4)
    device = torch.device(args.device)
    model = build_model(cfg).to(device)
    print('MODEL ' + json.dumps({'model': cfg['model'], 'parameters': sum((p.numel() for p in model.parameters())), 'device': str(device)}), flush=True)
    loaders = {name: DataLoader(ForecastWindows(values, mask, sums, counts, origins), batch_size=cfg['batch_size'], shuffle=name == 'train', num_workers=0) for name, origins in parts.items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    history, best, bad = ([], float('inf'), 0)
    started = time.time()
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        total, batches = (0.0, 0)
        for batch in loaders['train']:
            current = on_device(batch, device)
            result = model(current)
            loss = objective(result, current)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            batches += 1
        metrics = evaluate(model, loaders['val'], device, mean, std)
        row = {'epoch': epoch, 'loss': total / batches, 'seconds': time.time() - started, **metrics}
        history.append(row)
        pd.DataFrame(history).to_csv(output / 'history.csv', index=False)
        print(json.dumps(row), flush=True)
        if metrics['temperature_mse_normalized'] < best:
            best, bad = (metrics['temperature_mse_normalized'], 0)
            torch.save({'model': model.state_dict(), 'config': cfg, 'mean': mean, 'std': std, 'epoch': epoch}, output / 'best.pt')
            dump(output / 'best_validation.json', metrics)
        else:
            bad += 1
        if bad >= cfg['patience']:
            break
    state = torch.load(output / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(state['model'])
    metrics, arrays = evaluate(model, loaders['val'], device, mean, std, collect=True)
    dump(output / 'best_validation.json', metrics)
    np.savez_compressed(output / 'validation_predictions.npz', **arrays, query_offsets_minutes=np.arange(0, 1440, 15), units='celsius', mean=mean, std=std)
    summary = {'status': 'completed', 'best_epoch': state['epoch'], 'parameters_trainable': sum((p.numel() for p in model.parameters())), 'seconds': time.time() - started, 'gpu_max_allocated_bytes': torch.cuda.max_memory_allocated(device), 'test_evaluated': False, 'prediction_units': 'celsius'}
    dump(output / 'summary.json', summary)
    print('COMPLETED ' + json.dumps(summary), flush=True)
if __name__ == '__main__':
    main()
