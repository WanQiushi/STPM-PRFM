"""Fresh v9 initialization, fixed A2, and original 15-minute v11 OOF producer."""
from if_release.runtime import loads as _release_json_loads
import csv
import json
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from data import CachedWindows, TemperatureWindows, on_device
PACKAGE = Path(__file__).resolve().parents[1]
SOURCE = PACKAGE / 'vendor/temperature_a2'
sys.path.insert(0, str(SOURCE))
from ode_core_v9 import SmallLatentODE
from temperature_ablation import AblatedTemperature
from models import ShapeODEResidual, CachedResidual
from fixed_trend import final_temperature_objective

def dump(path, content):
    Path(path).write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding='utf-8')

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def point_loss(result, batch):
    return ((result['prediction'] - batch['target']).square() * batch['mask']).sum() / batch['mask'].sum()

def component_loss(result, batch, cfg):
    count = batch['mask'].sum(1).clamp_min(1)
    level = (batch['target'] * batch['mask']).sum(1) / count
    centered = batch['target'] - level.unsqueeze(1)
    amplitude = ((centered.square() * batch['mask']).sum(1) / count).sqrt().clamp_min(0.02)
    shape = centered / amplitude.unsqueeze(1)
    return final_temperature_objective(result, batch, cfg['trend_weight']) + cfg['level_weight'] * (result['level'] - level).square().mean() + cfg['amplitude_weight'] * (result['amplitude'] - amplitude).square().mean() + cfg['shape_weight'] * ((result['shape'] - shape).square() * batch['mask']).sum() / batch['mask'].sum()

def evaluate(model, loader, device, data):
    model.eval()
    saved = {name: [] for name in ['origin', 'prediction', 'target', 'mask']}
    sse = sae = count = 0.0
    with torch.no_grad():
        for batch in loader:
            current = on_device(batch, device)
            prediction = model(current)['prediction']
            error = prediction - current['target']
            mask = current['mask']
            sse += (error.square() * mask).sum().item()
            sae += (error.abs() * mask).sum().item()
            count += mask.sum().item()
            saved['prediction'].append(prediction.cpu().numpy() * data.std[:1] + data.mean[:1])
            saved['target'].append(batch['target'].numpy() * data.std[:1] + data.mean[:1])
            saved['mask'].append(batch['mask'].numpy())
            saved['origin'].append(batch['origin'].numpy())
    metrics = {'temperature_mse_normalized': sse / count, 'rmse_celsius': float(np.sqrt(sse / count) * data.std[0]), 'mae_celsius': float(sae / count * data.std[0]), 'measured_count': int(count)}
    return (metrics, {name: np.concatenate(values) for name, values in saved.items()})

def write_history(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def fit_stage(model, loaders, data, cfg, stage, output, device):
    lr = cfg['lr'] if stage == 'v9' else cfg['lr_' + stage]
    cap = cfg['epochs'] if stage == 'v9' else cfg['epochs_' + stage]
    optimizer = torch.optim.Adam([parameter for parameter in model.parameters() if parameter.requires_grad], lr=lr)
    best, bad, rows = (float('inf'), 0, [])
    started = time.time()
    for epoch in range(1, cap + 1):
        model.train()
        total = 0.0
        for batch in loaders['train']:
            current = on_device(batch, device)
            result = model(current)
            loss = point_loss(result, current) if stage == 'v9' else component_loss(result, current, cfg) if stage == 'base' else final_temperature_objective(result, current, cfg['trend_weight'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        metrics, arrays = evaluate(model, loaders['val'], device, data)
        row = {'epoch': epoch, 'stage': stage, 'loss': total / len(loaders['train']), 'seconds': time.time() - started, **metrics}
        rows.append(row)
        write_history(output / (stage + '_history.csv'), rows)
        print(json.dumps(row), flush=True)
        if metrics['temperature_mse_normalized'] < best:
            best, bad = (metrics['temperature_mse_normalized'], 0)
            torch.save({'model': model.state_dict(), 'config': cfg, 'epoch': epoch, 'mean': data.mean[:1], 'std': data.std[:1]}, output / (stage + '_best.pt'))
            dump(output / (stage + '_best_validation.json'), metrics)
            np.savez_compressed(output / (stage + '_validation.npz'), **arrays, units='celsius')
        else:
            bad += 1
        if bad >= cfg['patience']:
            break
    checkpoint = torch.load(output / (stage + '_best.pt'), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    return {'best_epoch': checkpoint['epoch'], 'epochs': epoch, 'seconds': time.time() - started, 'validation': _release_json_loads((output / (stage + '_best_validation.json')).read_text())}

def cache_base(model, loader, device):
    saved = {name: [] for name in ['features', 'base', 'target', 'mask', 'origin']}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            result = model(on_device(batch, device))
            for name in ['features', 'base']:
                saved[name].append(result[name].cpu().numpy())
            for name in ['target', 'mask', 'origin']:
                saved[name].append(batch[name].numpy())
    return {name: np.concatenate(values) for name, values in saved.items()}

def train_temperature(data, train_origins, validation_origins, seed, output, device, recipe, kind='A2'):
    """Every call starts fresh; v11 uses its original scalar heads and no trend loss."""
    output.mkdir(parents=True)
    cfg = dict(recipe)
    cfg.update(seed=seed, sample_minutes=1 if kind == 'A2' else 15, ablation='A2' if kind == 'A2' else 'original_v11')
    if kind == 'v11':
        cfg.update(level_weight=0.5, amplitude_weight=0.25, shape_weight=0.25, trend_weight=0.0)
    dump(output / 'config.json', cfg)
    dump(output / 'normalization.json', {'mean': data.mean.tolist(), 'std': data.std.tolist()})
    dump(output / 'window_origins.json', {'train': list(train_origins), 'val': list(validation_origins)})
    datasets = {name: TemperatureWindows(data, origins, cfg['sample_minutes']) for name, origins in [('train', train_origins), ('val', validation_origins)]}
    seed_all(seed)
    warm = SmallLatentODE(cfg).to(device)
    warm_loaders = {name: DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=name == 'train', num_workers=0) for name, dataset in datasets.items()}
    warm_report = fit_stage(warm, warm_loaders, data, cfg, 'v9', output, device)
    seed_all(seed)
    model = (AblatedTemperature(cfg) if kind == 'A2' else ShapeODEResidual(cfg)).to(device)
    model.ode.load_state_dict(warm.state_dict())
    del warm, warm_loaders
    for parameter in model.residual.parameters():
        parameter.requires_grad_(False)
    generator = torch.Generator().manual_seed(seed)
    loaders = {name: DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=name == 'train', num_workers=0, generator=generator if name == 'train' else None) for name, dataset in datasets.items()}
    base_report = fit_stage(model, loaders, data, cfg, 'base', output, device)
    cached = {name: cache_base(model, DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=False), device) for name, dataset in datasets.items()}
    for name, arrays in cached.items():
        np.savez_compressed(output / ('frozen_base_cache_' + name + '.npz'), **arrays)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.residual.parameters():
        parameter.requires_grad_(True)
    residual_generator = torch.Generator().manual_seed(seed)
    cached_loaders = {name: DataLoader(CachedWindows(arrays), batch_size=cfg['batch_size'], shuffle=name == 'train', generator=residual_generator if name == 'train' else None) for name, arrays in cached.items()}
    residual = CachedResidual(model.residual)
    residual_report = fit_stage(residual, cached_loaders, data, cfg, 'residual', output, device)
    torch.save({'model': model.state_dict(), 'config': cfg, 'mean': data.mean[:1], 'std': data.std[:1], 'base_epoch': base_report['best_epoch'], 'residual_epoch': residual_report['best_epoch']}, output / 'best.pt')
    model.eval()
    summary = {'kind': kind, 'status': 'completed', 'seed': seed, 'sample_minutes': cfg['sample_minutes'], 'parameters': sum((parameter.numel() for parameter in model.parameters())), 'v9': warm_report, 'base': base_report, 'residual': residual_report, 'train_windows': len(train_origins), 'validation_windows': len(validation_origins), 'initialization': 'fresh v9 fitted within this budget/fold', 'full_data_weights_used': False}
    dump(output / 'summary.json', summary)
    return (model, summary)

def predict(model, data, origins, sample_minutes, batch_size, device, path):
    loader = DataLoader(TemperatureWindows(data, origins, sample_minutes), batch_size=batch_size, shuffle=False)
    metrics, arrays = evaluate(model, loader, device, data)
    np.savez_compressed(path, **arrays, forecast_temperature_celsius=arrays['prediction'], units='celsius', sample_minutes=sample_minutes)
    return (arrays['prediction'], metrics)
