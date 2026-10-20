"""va-2.2 temperature: original v11 stages, with optional final-temperature linear-trend supervision."""
from if_release.runtime import loads as _release_json_loads
import argparse
import json
import random
import shutil
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from data_contract_v8 import ROOT, dump, load, on_device
from train_contract_v9 import ForecastWindows, evaluate, objective
from models import build_model, CachedResidual
from temperature_metrics import decomposition
from fixed_trend import trend_objective, final_temperature_objective
HERE = Path(__file__).resolve().parent

def components(target, mask):
    count = mask.sum(1).clamp_min(1)
    level = (target * mask).sum(1) / count
    centered = target - level.unsqueeze(1)
    amplitude = ((centered.square() * mask).sum(1) / count).sqrt().clamp_min(0.02)
    return (level, amplitude, centered / amplitude.unsqueeze(1))

def base_objective(result, batch, cfg):
    level, amplitude, shape = components(batch['target'], batch['mask'])
    temperature_loss = final_temperature_objective(result, batch, cfg['trend_weight'])
    level_loss = (result['level'] - level).square().mean()
    amplitude_loss = (result['amplitude'] - amplitude).square().mean()
    shape_loss = ((result['shape'] - shape).square() * batch['mask']).sum() / batch['mask'].sum()
    return temperature_loss + cfg['level_weight'] * level_loss + cfg['amplitude_weight'] * amplitude_loss + cfg['shape_weight'] * shape_loss

def train_stage(model, loaders, cfg, stage, output, device, mean, std):
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg['lr_' + stage])
    rows, best, bad = ([], float('inf'), 0)
    started = time.time()
    for epoch in range(1, cfg['epochs_' + stage] + 1):
        model.train()
        total, batches = (0.0, 0)
        curve_sse = trend_sse = observed_nodes = 0.0
        for batch in loaders['train']:
            current = on_device(batch, device)
            result = model(current)
            loss = base_objective(result, current, cfg) if stage == 'base' else final_temperature_objective(result, current, cfg['trend_weight'])
            nodes = current['mask'].sum().item()
            curve_sse += objective(result, current).item() * nodes
            trend_sse += trend_objective(result, current).item() * nodes
            observed_nodes += nodes
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            batches += 1
        metrics, arrays = evaluate(model, loaders['val'], device, mean, std, collect=True)
        diagnostics, daily = decomposition(arrays)
        metrics.update(diagnostics)
        row = {'stage': stage, 'epoch': epoch, 'loss': total / batches, 'train_curve_mse_normalized': curve_sse / observed_nodes, 'train_trend_mse_normalized': trend_sse / observed_nodes, 'seconds': time.time() - started, **metrics}
        rows.append(row)
        pd.DataFrame(rows).to_csv(output / (stage + '_history.csv'), index=False)
        print(json.dumps(row), flush=True)
        if metrics['temperature_mse_normalized'] < best:
            best, bad = (metrics['temperature_mse_normalized'], 0)
            torch.save({'model': model.state_dict(), 'config': cfg, 'epoch': epoch, 'mean': mean, 'std': std}, output / (stage + '_best.pt'))
            dump(output / (stage + '_best_validation.json'), metrics)
            np.savez_compressed(output / (stage + '_best_predictions.npz'), **arrays, mean=mean, std=std, query_offsets_minutes=np.arange(0, 1440, cfg['sample_minutes']), units='celsius')
            daily.to_csv(output / (stage + '_best_daily.csv'), index=False)
        else:
            bad += 1
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'config': cfg, 'epoch': epoch, 'stage': stage, 'bad_epochs': bad, 'best_mse': best, 'mean': mean, 'std': std, 'shuffle_generator_state': loaders['train'].generator.get_state(), 'torch_rng_state': torch.get_rng_state(), 'cuda_rng_state': torch.cuda.get_rng_state(device)}, output / (stage + '_last.pt'))
        dump(output / 'progress.json', {'status': 'running', 'stage': stage, 'epoch': epoch, 'bad_epochs': bad, 'trend_weight': cfg['trend_weight']})
        if bad >= cfg['patience']:
            break
    checkpoint = torch.load(output / (stage + '_best.pt'), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    metrics, arrays = evaluate(model, loaders['val'], device, mean, std, collect=True)
    diagnostics, daily = decomposition(arrays)
    metrics.update(diagnostics)
    dump(output / (stage + '_best_validation.json'), metrics)
    duration = time.time() - started
    (output / (stage + '_experiment_record.md')).write_text(f"# va-2.2 temperature / {cfg['run_name']}: {stage}\n\n- Date: {datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()}; project IFpaper_v1.\n- Purpose: {('ODE shape plus historical level/amplitude' if stage == 'base' else 'small full-scale residual with frozen base')}.\n- Data/interface/configuration: README.md and config.json; synthetic windows recorded in window_origins.json, {1440 // cfg['sample_minutes']} nodes at {cfg['sample_minutes']} minutes.\n- Supervision: original v11 losses plus {cfg['trend_weight']} times final-output masked linear-trend MSE; original stage freezing/early stop retained.\n- Completed epochs: {epoch}; stage cap {cfg['epochs_' + stage]}, patience {cfg['patience']}; fresh Adam lr {cfg['lr_' + stage]}.\n- Result: completed; selected epoch {checkpoint['epoch']}; RMSE {metrics['rmse_celsius']:.6f} degC; MAE {metrics['mae_celsius']:.6f} degC.\n- Last12h RMSE {metrics['last_12h_rmse_celsius']:.6f} degC; duration {duration:.2f}s.\n- Final-output trend RMS {metrics['trend_error_rms_celsius']:.6f} degC; level RMS {metrics['level_error_rms_celsius']:.6f} degC.\n- Products: {stage}_best.pt, {stage}_best_validation.json, {stage}_history.csv; frozen_source and config.json locate actual code.\n- Test predictions remain sealed.\n", encoding='utf-8')
    return (checkpoint['epoch'], metrics, duration, epoch)

class CachedWindows(Dataset):

    def __init__(self, arrays):
        self.arrays = arrays

    def __len__(self):
        return len(self.arrays['origin'])

    def __getitem__(self, index):
        return {key: value[index] for key, value in self.arrays.items()}

def cache_base(model, loader, device):
    stored = {key: [] for key in ['features', 'base', 'target', 'mask', 'previous_day', 'origin', 'level', 'amplitude', 'shape']}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            result = model(on_device(batch, device))
            for key in ['features', 'base', 'level', 'amplitude', 'shape']:
                stored[key].append(result[key].cpu().numpy())
            for key in ['target', 'mask', 'previous_day', 'origin']:
                stored[key].append(batch[key].numpy())
    return {key: np.concatenate(value, axis=0) for key, value in stored.items()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    cfg = _release_json_loads(Path(args.config).read_text(encoding='utf-8'))
    output = ROOT / cfg['output']
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(HERE, output / 'frozen_source', ignore=shutil.ignore_patterns('__pycache__'))
    dump(output / 'config.json', cfg)
    values, mask, sums, counts, parts, mean, std, report = load(cfg)
    dump(output / 'window_origins.json', parts)
    dump(output / 'data_report.json', {'counts': report, 'measured_query_points': {name: int(sum((mask[o:o + 1440:cfg['sample_minutes'], 0].sum() for o in origins))) for name, origins in parts.items()}, 'mean_celsius': float(mean[0]), 'std_celsius': float(std[0]), 'data': cfg['data'], 'query_offsets_minutes': list(range(0, 1440, cfg['sample_minutes'])), 'test_evaluated': False, 'interface': 'same causal v9 inputs; full temperature = predicted level + predicted amplitude * normalized ODE shape + learned residual', 'version': 'va core ablation temperature', 'sample_minutes': cfg['sample_minutes'], 'trend_weight': cfg['trend_weight'], 'h1_used_in_training': False, 'sensor_type': 'unconfirmed', 'checkpoint_selection': 'configured validation windows final-temperature MSE; original selection starts at epoch1'})
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.set_num_threads(4)
    device = torch.device(args.device)
    model = build_model(cfg).to(device)
    warm = torch.load(ROOT / cfg['initial_checkpoint'], map_location=device, weights_only=False)
    model.ode.load_state_dict(warm['model'])
    for parameter in model.residual.parameters():
        parameter.requires_grad_(False)
    datasets = {name: ForecastWindows(values, mask, sums, counts, origins, cfg['sample_minutes']) for name, origins in parts.items() if name in ['train', 'val']}
    base_generator = torch.Generator().manual_seed(cfg['seed'])
    loaders = {name: DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=name == 'train', num_workers=0, generator=base_generator if name == 'train' else None) for name, dataset in datasets.items()}
    sizes = {'base': sum((p.numel() for p in model.parameters() if p.requires_grad)), 'residual': sum((p.numel() for p in model.residual.parameters())), 'total': sum((p.numel() for p in model.parameters()))}
    print('DATA ' + json.dumps(report), flush=True)
    print('MODEL ' + json.dumps({'version': 'va core ablation temperature', 'sample_minutes': cfg['sample_minutes'], 'arm': cfg['arm'], 'trend_weight': cfg['trend_weight'], 'parameters': sizes, 'warm_start': cfg['initial_checkpoint'], 'device': str(device)}), flush=True)
    dump(output / 'progress.json', {'stage': 'base', 'status': 'running', 'parameters': sizes, 'started_utc': datetime.now(timezone.utc).isoformat()})
    base_epoch, base_metrics, base_seconds, base_completed = train_stage(model, loaders, cfg, 'base', output, device, mean, std)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.residual.parameters():
        parameter.requires_grad_(True)
    cached = {name: cache_base(model, DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=False), device) for name, dataset in datasets.items()}
    for name, arrays in cached.items():
        np.savez_compressed(output / ('frozen_base_cache_' + name + '.npz'), **arrays, mean=mean, std=std, units='normalized_training_temperature')
    residual_generator = torch.Generator().manual_seed(cfg['seed'])
    cached_loaders = {name: DataLoader(CachedWindows(arrays), batch_size=cfg['batch_size'], shuffle=name == 'train', num_workers=0, generator=residual_generator if name == 'train' else None) for name, arrays in cached.items()}
    residual_model = CachedResidual(model.residual)
    dump(output / 'progress.json', {'stage': 'residual', 'status': 'running', 'base_best_epoch': base_epoch, 'base_rmse_celsius': base_metrics['rmse_celsius']})
    residual_epoch, residual_metrics, residual_seconds, residual_completed = train_stage(residual_model, cached_loaders, cfg, 'residual', output, device, mean, std)
    metrics, arrays = evaluate(residual_model, cached_loaders['val'], device, mean, std, collect=True)
    diagnostics, daily = decomposition(arrays)
    metrics.update(diagnostics)
    daily.to_csv(output / 'validation_daily.csv', index=False)
    arrays.update({'base_prediction': cached['val']['base'] * std + mean, 'residual_prediction': arrays['prediction'] - (cached['val']['base'] * std + mean), 'level_celsius': cached['val']['level'] * std + mean, 'amplitude_celsius': cached['val']['amplitude'] * std, 'ode_shape': cached['val']['shape']})
    np.savez_compressed(output / 'validation_predictions.npz', **arrays, query_offsets_minutes=np.arange(0, 1440, cfg['sample_minutes']), units='celsius', mean=mean, std=std)
    torch.save({'model': model.state_dict(), 'config': cfg, 'base_epoch': base_epoch, 'residual_epoch': residual_epoch, 'mean': mean, 'std': std}, output / 'best.pt')
    dump(output / 'best_validation.json', metrics)
    dump(output / 'summary.json', {'status': 'completed', 'base_best_epoch': base_epoch, 'residual_best_epoch': residual_epoch, 'version': 'va core ablation temperature', 'sample_minutes': cfg['sample_minutes'], 'arm': cfg['arm'], 'trend_weight': cfg['trend_weight'], 'base_epochs_completed': base_completed, 'residual_epochs_completed': residual_completed, 'base_rmse_celsius': base_metrics['rmse_celsius'], 'final_rmse_celsius': metrics['rmse_celsius'], 'base_seconds': base_seconds, 'residual_seconds': residual_seconds, 'parameters': sizes, 'gpu_max_allocated_bytes': torch.cuda.max_memory_allocated(device), 'test_evaluated': False, 'h1_used_in_training': False})
    dump(output / 'progress.json', {'stage': 'completed', 'status': 'completed', 'test_evaluated': False})
    print('COMPLETED ' + json.dumps({'base_rmse_celsius': base_metrics['rmse_celsius'], 'final_rmse_celsius': metrics['rmse_celsius']}), flush=True)
if __name__ == '__main__':
    main()
