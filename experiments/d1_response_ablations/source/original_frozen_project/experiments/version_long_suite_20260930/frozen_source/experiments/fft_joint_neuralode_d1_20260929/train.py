from __future__ import annotations
from if_release.runtime import loads as _release_json_loads
import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from model import JointFFTNeuralODE, stage1_loss
ROOT = Path(__file__).resolve().parents[2]

def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')

def prepare(cfg):
    archive = np.load(ROOT / cfg['data'], allow_pickle=False)
    values = archive['values'].astype(np.float64)
    observed = archive['valid_observed'].astype(bool)
    available = archive['input_available_at']
    train_end = cfg['split_days'][0] * 1440
    val_end = sum(cfg['split_days'][:2]) * 1440
    measured = np.where(observed[:train_end], values[:train_end], np.nan)
    mean = np.nanmean(measured, axis=0)
    std = np.nanstd(measured, axis=0)
    normalized = ((values - mean) / std).astype('float32')
    step = int(cfg['sample_minutes'])
    history_offsets = np.arange(-cfg['history_days'] * 1440, 0, step)
    future_offsets = np.arange(0, cfg['horizon_days'] * 1440, step)
    fixed = _release_json_loads((ROOT / cfg['fixed_origins']).read_text(encoding='utf-8'))
    parts = {}
    report = {}
    for name, first, last, stride in (('train', cfg['history_days'] * 1440, train_end, cfg['train_stride']), ('val', train_end, val_end, 1440)):
        candidates = np.arange(first, last - cfg['horizon_days'] * 1440 + 1, stride)
        fixed_set = set(map(int, fixed[name]))
        selected = []
        for origin in candidates:
            if int(origin) not in fixed_set:
                continue
            history_indices = origin + history_offsets
            future_indices = origin + future_offsets
            if available[history_indices].max() >= origin:
                continue
            if not observed[future_indices, 0].all():
                continue
            selected.append(int(origin))
        parts[name] = selected
        report[name] = {'candidates': int(len(candidates)), 'fixed_candidates': int(sum((int(item) in fixed_set for item in candidates))), 'selected': int(len(selected))}
    return (normalized, observed, parts, mean, std, report, history_offsets, future_offsets)

class Windows(Dataset):

    def __init__(self, values, observed, origins, history_offsets, future_offsets):
        self.values = values
        self.observed = observed
        self.origins = origins
        self.history_offsets = history_offsets
        self.future_offsets = future_offsets

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, index):
        origin = self.origins[index]
        history_indices = origin + self.history_offsets
        future_indices = origin + self.future_offsets
        return {'history': self.values[history_indices].copy(), 'future_temperature': self.values[future_indices, :1].copy(), 'target': self.values[future_indices, 1:].copy(), 'mask': self.observed[future_indices, 1:].copy(), 'origin': np.int64(origin)}

def on_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}

def masked_channel_mse(prediction, target, mask):
    mask = mask.to(prediction.dtype)
    counts = mask.sum((0, 1))
    active = counts > 0
    channel = ((prediction - target).square() * mask).sum((0, 1))[active] / counts[active]
    return channel.mean()

def objective(model, output, batch, cfg):
    prediction_loss = masked_channel_mse(output['prediction'], batch['target'], batch['mask'])
    target_window = torch.cat((batch['history'][:, model.points_per_day:, 1:], batch['target']), dim=1)
    target_parts = model.decomposer(target_window.transpose(1, 2))
    target_lf = target_parts['static'].transpose(1, 2)[:, -model.points_per_day:]
    target_mf = target_parts['dynamic'].transpose(1, 2)[:, -model.points_per_day:]
    lf_loss = masked_channel_mse(output['lf_prediction'], target_lf, batch['mask'])
    mf_loss = masked_channel_mse(output['mf_prediction'], target_mf, batch['mask'])
    decomposition_loss, decomposition_terms = stage1_loss(model, batch['history'], cfg['stage1_loss_weights'])
    total = prediction_loss + cfg['stage1_weight'] * decomposition_loss + cfg['component_weight'] * (lf_loss + mf_loss) + cfg['ode_weight'] * output['ode_residual_penalty']
    terms = {'prediction_mse': prediction_loss, 'lf_mse': lf_loss, 'mf_mse': mf_loss, 'stage1_loss': decomposition_loss, 'ode_residual': output['ode_residual_penalty'], **decomposition_terms}
    return (total, terms, target_lf, target_mf)

def evaluate(model, loader, device, std):
    squared = np.zeros(4)
    first = np.zeros(4)
    second = np.zeros(4)
    counts = np.zeros(4)
    first_counts = np.zeros(4)
    second_counts = np.zeros(4)
    with torch.no_grad():
        model.eval()
        for batch in loader:
            moved = on_device(batch, device)
            output = model(moved['history'], moved['future_temperature'])
            error = output['prediction'] - moved['target']
            mask = moved['mask']
            squared += (error.square() * mask).sum((0, 1)).cpu().numpy()
            counts += mask.sum((0, 1)).cpu().numpy()
            first += (error[:, :48].square() * mask[:, :48]).sum((0, 1)).cpu().numpy()
            first_counts += mask[:, :48].sum((0, 1)).cpu().numpy()
            second += (error[:, 48:].square() * mask[:, 48:]).sum((0, 1)).cpu().numpy()
            second_counts += mask[:, 48:].sum((0, 1)).cpu().numpy()
    static_hours, noise_hours = model.decomposer.cutoff_hours()
    return {'displacement_mse': float(np.mean(squared / counts)), 'first_12h_mse': float(np.mean(first / first_counts)), 'second_12h_mse': float(np.mean(second / second_counts)), 'rmse_mm': (np.sqrt(squared / counts) * std[1:]).tolist(), 'first_12h_rmse_mm': (np.sqrt(first / first_counts) * std[1:]).tolist(), 'second_12h_rmse_mm': (np.sqrt(second / second_counts) * std[1:]).tolist(), 'measured_counts': counts.tolist(), 'static_period_hours': float(static_hours.cpu()), 'noise_period_hours': float(noise_hours.cpu())}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    cfg = _release_json_loads(Path(args.config).read_text(encoding='utf-8'))
    output_dir = ROOT / cfg['output']
    output_dir.mkdir(parents=True, exist_ok=True)
    dump(output_dir / 'config.json', cfg)
    values, observed, parts, mean, std, report, history_offsets, future_offsets = prepare(cfg)
    dump(output_dir / 'data_report.json', {'counts': report, 'mean': mean.tolist(), 'std': std.tolist(), 'history_days': cfg['history_days'], 'history_policy': 'only values whose input_available_at precedes the origin', 'future_temperature_policy': 'measured 15-minute nodes; conditional response task', 'future_displacement_policy': 'labels only; never passed to model.forward'})
    dump(output_dir / 'window_origins.json', parts)
    print('DATA ' + json.dumps(report), flush=True)
    seed = int(cfg['seed'])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    model = JointFFTNeuralODE(cfg).to(device)
    loaders = {name: DataLoader(Windows(values, observed, origins, history_offsets, future_offsets), batch_size=cfg['batch_size'], shuffle=name == 'train', num_workers=0) for name, origins in parts.items()}
    cutoff_parameters = list(model.decomposer.parameters())
    cutoff_ids = {id(parameter) for parameter in cutoff_parameters}
    model_parameters = [parameter for parameter in model.parameters() if id(parameter) not in cutoff_ids]
    optimizer = torch.optim.Adam([{'params': model_parameters, 'lr': cfg['lr']}, {'params': cutoff_parameters, 'lr': cfg['cutoff_lr']}])
    history = []
    best = float('inf')
    best_epoch = 0
    bad_epochs = 0
    started = time.time()
    for epoch in range(1, cfg['epochs'] + 1):
        cutoff_trainable = epoch > cfg['warmup_epochs']
        for parameter in cutoff_parameters:
            parameter.requires_grad_(cutoff_trainable)
        model.train()
        totals = {}
        batches = 0
        for batch in loaders['train']:
            moved = on_device(batch, device)
            result = model(moved['history'], moved['future_temperature'])
            loss, terms, _, _ = objective(model, result, moved, cfg)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            optimizer.step()
            totals['loss'] = totals.get('loss', 0.0) + float(loss.detach().cpu())
            for key, value in terms.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
            batches += 1
        metrics = evaluate(model, loaders['val'], device, std)
        row = {'epoch': epoch, 'seconds': time.time() - started, 'cutoff_trainable': cutoff_trainable, **{key: value / batches for key, value in totals.items()}, **{f'val_{key}': value for key, value in metrics.items() if not isinstance(value, list)}}
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / 'history.csv', index=False)
        print(json.dumps(row), flush=True)
        selection = metrics['displacement_mse']
        if selection < best:
            best = selection
            best_epoch = epoch
            bad_epochs = 0
            torch.save({'model': model.state_dict(), 'config': cfg, 'mean': mean, 'std': std, 'epoch': epoch}, output_dir / 'best.pt')
            dump(output_dir / 'best_validation.json', metrics)
        else:
            bad_epochs += 1
        if bad_epochs >= cfg['patience']:
            break
    checkpoint = torch.load(output_dir / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    saved = {}
    with torch.no_grad():
        for batch in loaders['val']:
            moved = on_device(batch, device)
            result = model(moved['history'], moved['future_temperature'])
            _, _, target_lf, target_mf = objective(model, result, moved, cfg)
            arrays = {'prediction': result['prediction'].cpu().numpy(), 'lf_prediction': result['lf_prediction'].cpu().numpy(), 'mf_prediction': result['mf_prediction'].cpu().numpy(), 'temperature_lf': result['temperature_lf'].cpu().numpy(), 'temperature_mf': result['temperature_mf'].cpu().numpy(), 'lf_target': target_lf.cpu().numpy(), 'mf_target': target_mf.cpu().numpy(), 'target': batch['target'].numpy(), 'mask': batch['mask'].numpy(), 'future_temperature': batch['future_temperature'].numpy(), 'origin': batch['origin'].numpy()}
            for key, value in arrays.items():
                saved.setdefault(key, []).append(value)
    saved = {key: np.concatenate(value) for key, value in saved.items()}
    np.savez_compressed(output_dir / 'validation_predictions.npz', **saved, mean=mean, std=std)
    final_metrics = evaluate(model, loaders['val'], device, std)
    summary = {'status': 'completed', 'best_epoch': best_epoch, 'epochs_ran': len(history), 'best_validation_displacement_mse': best, 'elapsed_seconds': time.time() - started, 'parameters': sum((parameter.numel() for parameter in model.parameters())), 'gpu_max_allocated_bytes': torch.cuda.max_memory_allocated(device), 'test_evaluated': False, 'task': cfg['task'], 'final_validation': final_metrics}
    dump(output_dir / 'summary.json', summary)
    print('COMPLETED ' + json.dumps(summary), flush=True)
if __name__ == '__main__':
    main()
