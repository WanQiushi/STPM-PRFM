"""Train the compact structured GRU for Temperature."""
from if_release.runtime import loads as _release_json_loads
import argparse
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from models import build_model
ROOT = Path(__file__).resolve().parents[2]

def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')

def load(cfg):
    z = np.load(ROOT / cfg['data'], allow_pickle=False)
    values = z['values'][:, :1].astype(np.float64)
    mask = z['valid_observed'][:, :1]
    available_at = z['input_available_at'][:, 0]
    train_end = cfg['split_days'][0] * 1440
    val_end = sum(cfg['split_days'][:2]) * 1440
    mean = np.nanmean(np.where(mask[:train_end], values[:train_end], np.nan), axis=0)
    std = np.nanstd(np.where(mask[:train_end], values[:train_end], np.nan), axis=0)
    values = ((values - mean) / std).astype('float32')
    cumulative = np.vstack((np.zeros((1, 1)), np.cumsum(np.where(mask, values, 0), axis=0, dtype='float64')))
    counts = np.vstack((np.zeros((1, 1)), np.cumsum(mask, axis=0, dtype='float64')))
    parts, report = ({}, {})
    for name, start, end, stride in [('train', 180 * 1440, train_end, cfg['train_stride']), ('val', train_end, val_end, 1440), ('test', val_end, len(values), 1440)]:
        candidates = np.arange(start, end - 1440 + 1, stride)
        selected = [int(origin) for origin in candidates if available_at[origin - 10081:origin].max() < origin]
        parts[name] = selected
        report[name] = {'candidates': len(candidates), 'selected': len(selected), 'measured_target_points': int(sum((mask[o:o + 1440, 0].sum() for o in selected)))}
    return (values, mask, cumulative, counts, parts, mean, std, report)

class Windows(Dataset):

    def __init__(self, values, mask, sums, counts, origins):
        self.values, self.mask = (values, mask)
        self.sums, self.counts, self.origins = (sums, counts, origins)

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, index):
        origin = self.origins[index]
        edges = origin + np.arange(-180, 1) * 1440
        count = self.counts[edges[1:]] - self.counts[edges[:-1]]
        summed = self.sums[edges[1:]] - self.sums[edges[:-1]]
        daily = np.divide(summed, count, out=np.zeros_like(summed), where=count > 0)
        return {'daily': daily.astype('float32'), 'coverage': (count / 1440).astype('float32'), 'recent': self.values[origin - 10081:origin].copy(), 'phase': np.array([(origin - 1) / 1440 * 2 * math.pi, (origin - 1) / 1440 / 365.25 * 2 * math.pi], dtype='float32'), 'target': self.values[origin:origin + 1440].copy(), 'mask': self.mask[origin:origin + 1440].copy(), 'previous_day': self.values[origin - 1440:origin].copy(), 'origin': origin}

def on_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}

def observed_huber(prediction, target, mask):
    loss = nn.functional.smooth_l1_loss(prediction, target, reduction='none')
    return (loss * mask).sum() / mask.sum()

def target_components(target, mask):
    count = mask.sum(1).clamp_min(1)
    level = (target * mask).sum(1) / count
    centered = target - level.unsqueeze(1)
    amplitude = ((centered.square() * mask).sum(1) / count).sqrt().clamp_min(0.02)
    blocks = target.view(-1, 96, 15, 1)
    block_mask = mask.view(-1, 96, 15, 1)
    block_count = block_mask.sum(2).clamp_min(1)
    coarse = (blocks * block_mask).sum(2) / block_count
    shape = (coarse[:, :, 0] - level) / amplitude
    valid_blocks = block_mask.sum(2)[:, :, 0] >= 12
    return (level, amplitude, shape, valid_blocks)

def objective(result, batch, cfg):
    target_level, target_amplitude, target_shape, valid_blocks = target_components(batch['target'], batch['mask'])
    temperature = observed_huber(result['prediction'], batch['target'], batch['mask'])
    level = nn.functional.smooth_l1_loss(result['level'], target_level)
    amplitude = nn.functional.smooth_l1_loss(result['amplitude'], target_amplitude)
    shape_loss = nn.functional.smooth_l1_loss(result['shape'], target_shape, reduction='none')
    shape = (shape_loss * valid_blocks).sum() / valid_blocks.sum()
    loss = temperature + cfg['level_weight'] * level + cfg['amplitude_weight'] * amplitude + cfg['shape_weight'] * shape
    return (loss, {'temperature_huber': temperature.item(), 'level_huber': level.item(), 'amplitude_huber': amplitude.item(), 'shape_huber': shape.item()})

def block_detail(sequence, mask):
    blocks = sequence.view(-1, 96, 15, 1)
    block_mask = mask.view(-1, 96, 15, 1)
    count = block_mask.sum(2, keepdim=True).clamp_min(1)
    mean = (blocks * block_mask).sum(2, keepdim=True) / count
    return (blocks - mean).view(-1, 1440, 1)

def evaluate(model, loader, device, std, collect=False):
    model.eval()
    sse = sae = previous_sse = count = 0.0
    detail_sse = detail_count = 0.0
    level_ae = amplitude_ae = days = 0.0
    block_sse, block_count = (np.zeros(4), np.zeros(4))
    stored = {key: [] for key in ('origin', 'prediction', 'target', 'mask', 'level', 'amplitude', 'shape')}
    with torch.no_grad():
        for batch in loader:
            current = on_device(batch, device)
            result = model(current)
            error = result['prediction'] - current['target']
            measured = current['mask']
            sse += (error.square() * measured).sum().item()
            sae += (error.abs() * measured).sum().item()
            previous_sse += ((current['previous_day'] - current['target']).square() * measured).sum().item()
            count += measured.sum().item()
            predicted_detail = block_detail(result['prediction'], measured)
            target_detail = block_detail(current['target'], measured)
            detail_sse += ((predicted_detail - target_detail).square() * measured).sum().item()
            detail_count += measured.sum().item()
            target_level, target_amplitude, _, _ = target_components(current['target'], measured)
            level_ae += (result['level'] - target_level).abs().sum().item()
            amplitude_ae += (result['amplitude'] - target_amplitude).abs().sum().item()
            days += len(target_level)
            for block in range(4):
                section = slice(block * 360, (block + 1) * 360)
                block_sse[block] += (error[:, section].square() * measured[:, section]).sum().item()
                block_count[block] += measured[:, section].sum().item()
            if collect:
                stored['origin'].append(batch['origin'].numpy())
                stored['prediction'].append(result['prediction'].cpu().numpy())
                stored['target'].append(batch['target'].numpy())
                stored['mask'].append(batch['mask'].numpy())
                stored['level'].append(result['level'].cpu().numpy())
                stored['amplitude'].append(result['amplitude'].cpu().numpy())
                stored['shape'].append(result['shape'].cpu().numpy())
    metrics = {'temperature_mse': sse / count, 'rmse_celsius': float(np.sqrt(sse / count) * std[0]), 'mae_celsius': float(sae / count * std[0]), 'previous_day_rmse_celsius': float(np.sqrt(previous_sse / count) * std[0]), 'detail_rmse_celsius': float(np.sqrt(detail_sse / detail_count) * std[0]), 'level_mae_celsius': float(level_ae / days * std[0]), 'amplitude_mae_celsius': float(amplitude_ae / days * std[0]), 'six_hour_rmse_celsius': (np.sqrt(block_sse / block_count) * std[0]).tolist(), 'measured_points': int(count)}
    if collect:
        return (metrics, {key: np.concatenate(values, axis=0) for key, values in stored.items()})
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    cfg = _release_json_loads(Path(args.config).read_text(encoding='utf-8'))
    output = ROOT / 'temperature_prediction' / 'outputs' / 'structured_gru_temperature_v8' / cfg['run_name']
    output.mkdir(parents=True, exist_ok=True)
    dump(output / 'config.json', cfg)
    values, mask, sums, counts, parts, mean, std, report = load(cfg)
    dump(output / 'data_report.json', {'counts': report, 'mean_celsius': float(mean[0]), 'std_celsius': float(std[0]), 'history_policy': 'seven prior days at one-minute input, internally averaged to 15-minute GRU steps; 30 prior daily means and coverage; no ODE or future observations', 'recent_minutes': 10081, 'forecast_minutes': 1440, 'input_channel': 'Temperature', 'target_channel': 'Temperature'})
    dump(output / 'window_origins.json', parts)
    print('DATA ' + json.dumps(report), flush=True)
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.set_num_threads(4)
    device = torch.device(args.device)
    model = build_model(cfg).to(device)
    loaders = {name: DataLoader(Windows(values, mask, sums, counts, origins), batch_size=cfg['batch_size'], shuffle=name == 'train', num_workers=0) for name, origins in parts.items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    history, best, bad = ([], float('inf'), 0)
    started = time.time()
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        total, terms, batches = (0.0, {}, 0)
        for batch in loaders['train']:
            current = on_device(batch, device)
            result = model(current)
            loss, current_terms = objective(result, current, cfg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            batches += 1
            for key, value in current_terms.items():
                terms[key] = terms.get(key, 0.0) + value
        metrics = evaluate(model, loaders['val'], device, std)
        row = {'epoch': epoch, 'loss': total / batches, 'seconds': time.time() - started, **metrics, **{key: value / batches for key, value in terms.items()}}
        history.append(row)
        pd.DataFrame(history).to_csv(output / 'history.csv', index=False)
        print(json.dumps(row), flush=True)
        if metrics['temperature_mse'] < best:
            best, bad = (metrics['temperature_mse'], 0)
            torch.save({'model': model.state_dict(), 'config': cfg, 'mean': mean, 'std': std, 'epoch': epoch}, output / 'best.pt')
            dump(output / 'best_validation.json', metrics)
        else:
            bad += 1
        if bad >= cfg['patience']:
            break
    state = torch.load(output / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(state['model'])
    metrics, arrays = evaluate(model, loaders['val'], device, std, collect=True)
    dump(output / 'best_validation.json', metrics)
    np.savez_compressed(output / 'validation_predictions.npz', **arrays, mean=mean, std=std)
    summary = {'status': 'completed', 'best_epoch': state['epoch'], 'parameters_trainable': sum((parameter.numel() for parameter in model.parameters())), 'seconds': time.time() - started, 'gpu_max_allocated_bytes': torch.cuda.max_memory_allocated(device), 'test_evaluated': False}
    dump(output / 'summary.json', summary)
    print('COMPLETED ' + json.dumps(summary), flush=True)
if __name__ == '__main__':
    main()
