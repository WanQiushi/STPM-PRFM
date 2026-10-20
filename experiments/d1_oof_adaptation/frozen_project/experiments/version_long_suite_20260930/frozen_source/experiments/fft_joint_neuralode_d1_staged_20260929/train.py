from __future__ import annotations
from if_release.runtime import loads as _release_json_loads
import argparse
import json
import random
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
ROOT = Path(__file__).resolve().parents[2]
BASE_EXPERIMENT = ROOT / 'experiments' / 'fft_joint_neuralode_d1_20260929'
sys.path.insert(0, str(BASE_EXPERIMENT))
from model import JointFFTNeuralODE, stage1_loss

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
        causal_count = 0
        temperature_count = 0
        for origin in candidates:
            history_indices = origin + history_offsets
            future_indices = origin + future_offsets
            if available[history_indices].max() >= origin:
                continue
            causal_count += 1
            if not observed[future_indices, 0].all():
                continue
            temperature_count += 1
            if name == 'val' and int(origin) not in fixed_set:
                continue
            if name == 'train' and (not cfg['train_use_all_causal_origins']) and (int(origin) not in fixed_set):
                continue
            selected.append(int(origin))
        parts[name] = selected
        report[name] = {'candidates': int(len(candidates)), 'fixed_candidates': int(sum((int(item) in fixed_set for item in candidates))), 'causal_history': int(causal_count), 'measured_future_temperature': int(temperature_count), 'selected': int(len(selected)), 'fixed_origin_filter': name == 'val' or not cfg['train_use_all_causal_origins']}
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

def target_components(model, batch):
    target_window = torch.cat((batch['history'][:, model.points_per_day:, 1:], batch['target']), dim=1)
    parts = model.decomposer(target_window.transpose(1, 2))
    target_lf = parts['static'].transpose(1, 2)[:, -model.points_per_day:]
    target_mf = parts['dynamic'].transpose(1, 2)[:, -model.points_per_day:]
    return (target_lf, target_mf)

def closed_area(x, y):
    x_next = torch.roll(x, shifts=-1, dims=1)
    y_next = torch.roll(y, shifts=-1, dims=1)
    return 0.5 * torch.sum(x * y_next - x_next * y, dim=1)

def loop_loss(output, target_mf, cfg):
    u = output['temperature_mf'].expand_as(target_mf)
    predicted_area = closed_area(u, output['mf_prediction'])
    target_area = closed_area(u, target_mf.detach())
    relative = (predicted_area - target_area).square() / (target_area.detach().square() + cfg['loop_epsilon'])
    return (relative.mean(), predicted_area, target_area)

def stage_name(epoch, cfg):
    if epoch <= cfg['bw_pretrain_epochs']:
        return 'bw_pretrain'
    if epoch <= cfg['bw_pretrain_epochs'] + cfg['prediction_warmup_epochs']:
        return 'prediction_warmup'
    return 'joint_cutoff'

def configure_stage(model, stage):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if stage == 'bw_pretrain':
        for parameter in model.initial_z.parameters():
            parameter.requires_grad_(True)
        model.s.requires_grad_(True)
        model.beta_raw.requires_grad_(True)
        model.gamma_raw.requires_grad_(True)
        for parameter in model.residual_field.parameters():
            parameter.requires_grad_(True)
    elif stage == 'prediction_warmup':
        for name, parameter in model.named_parameters():
            if not name.startswith('decomposer.'):
                parameter.requires_grad_(True)
    else:
        for parameter in model.parameters():
            parameter.requires_grad_(True)

def objective(model, output, batch, cfg, stage):
    prediction = masked_channel_mse(output['prediction'], batch['target'], batch['mask'])
    target_lf, target_mf = target_components(model, batch)
    target_lf_detached = target_lf.detach()
    target_mf_detached = target_mf.detach()
    lf = masked_channel_mse(output['lf_prediction'], target_lf_detached, batch['mask'])
    mf = masked_channel_mse(output['mf_prediction'], target_mf_detached, batch['mask'])
    loop, _, _ = loop_loss(output, target_mf_detached, cfg)
    if stage == 'bw_pretrain':
        decomposition = output['prediction'].new_zeros(())
        decomposition_terms = {'stage1_static_r2': decomposition, 'stage1_bw_r2': decomposition, 'stage1_low_purity': decomposition, 'stage1_high_purity': decomposition}
        total = mf + cfg['loop_weight'] * loop + cfg['ode_weight'] * output['ode_residual_penalty']
    else:
        decomposition, decomposition_terms = stage1_loss(model, batch['history'], cfg['stage1_loss_weights'])
        total = prediction + cfg['stage1_weight'] * decomposition + cfg['component_weight'] * (lf + mf) + cfg['loop_weight'] * loop + cfg['ode_weight'] * output['ode_residual_penalty']
    terms = {'prediction_mse': prediction, 'lf_mse': lf, 'mf_mse': mf, 'loop_loss': loop, 'stage1_loss': decomposition, 'ode_residual': output['ode_residual_penalty'], **decomposition_terms}
    return (total, terms, target_lf, target_mf)

def evaluate(model, loader, device, std, cfg):
    squared = np.zeros(4)
    first = np.zeros(4)
    second = np.zeros(4)
    counts = np.zeros(4)
    first_counts = np.zeros(4)
    second_counts = np.zeros(4)
    predicted_area = np.zeros(4)
    target_area = np.zeros(4)
    memory_energy = np.zeros(4)
    reversible_energy = np.zeros(4)
    point_count = 0
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
            _, target_mf = target_components(model, moved)
            _, area_pred, area_target = loop_loss(output, target_mf, cfg)
            predicted_area += area_pred.abs().sum(0).cpu().numpy()
            target_area += area_target.abs().sum(0).cpu().numpy()
            reversible = model.k * output['temperature_mf']
            memory = output['mf_prediction'] - reversible
            reversible_energy += reversible.square().sum((0, 1)).cpu().numpy()
            memory_energy += memory.square().sum((0, 1)).cpu().numpy()
            point_count += len(mask) * mask.shape[1]
    static_hours, noise_hours = model.decomposer.cutoff_hours()
    return {'displacement_mse': float(np.mean(squared / counts)), 'first_12h_mse': float(np.mean(first / first_counts)), 'second_12h_mse': float(np.mean(second / second_counts)), 'rmse_mm': (np.sqrt(squared / counts) * std[1:]).tolist(), 'first_12h_rmse_mm': (np.sqrt(first / first_counts) * std[1:]).tolist(), 'second_12h_rmse_mm': (np.sqrt(second / second_counts) * std[1:]).tolist(), 'measured_counts': counts.tolist(), 'static_period_hours': float(static_hours.cpu()), 'noise_period_hours': float(noise_hours.cpu()), 'mf_loop_area_abs_ratio': (predicted_area / np.maximum(target_area, 1e-12)).tolist(), 'mf_memory_to_reversible_rms': (np.sqrt(memory_energy / point_count) / np.maximum(np.sqrt(reversible_energy / point_count), 1e-12)).tolist()}

def save_checkpoint(path, model, cfg, mean, std, epoch, stage, metrics):
    torch.save({'model': model.state_dict(), 'config': cfg, 'mean': mean, 'std': std, 'epoch': epoch, 'stage': stage, 'metrics': metrics}, path)

def save_predictions(path, checkpoint_path, model, loader, device, cfg, mean, std):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    saved = {}
    with torch.no_grad():
        for batch in loader:
            moved = on_device(batch, device)
            output = model(moved['history'], moved['future_temperature'])
            target_lf, target_mf = target_components(model, moved)
            arrays = {'prediction': output['prediction'].cpu().numpy(), 'lf_prediction': output['lf_prediction'].cpu().numpy(), 'mf_prediction': output['mf_prediction'].cpu().numpy(), 'temperature_lf': output['temperature_lf'].cpu().numpy(), 'temperature_mf': output['temperature_mf'].cpu().numpy(), 'lf_target': target_lf.cpu().numpy(), 'mf_target': target_mf.cpu().numpy(), 'target': batch['target'].numpy(), 'mask': batch['mask'].numpy(), 'future_temperature': batch['future_temperature'].numpy(), 'origin': batch['origin'].numpy()}
            for key, value in arrays.items():
                saved.setdefault(key, []).append(value)
    saved = {key: np.concatenate(value) for key, value in saved.items()}
    np.savez_compressed(path, **saved, mean=mean, std=std)

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
    dump(output_dir / 'data_report.json', {'counts': report, 'mean': mean.tolist(), 'std': std.tolist(), 'history_days': cfg['history_days'], 'history_policy': 'all 30-day values must be available before origin', 'training_origin_policy': 'all causal candidates; validation remains fixed', 'future_temperature_policy': 'measured 15-minute nodes', 'future_displacement_policy': 'labels only'})
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
    remaining = [parameter for parameter in model.parameters() if id(parameter) not in cutoff_ids]
    optimizer = torch.optim.Adam([{'params': remaining, 'lr': cfg['lr']}, {'params': cutoff_parameters, 'lr': cfg['cutoff_lr']}])
    history = []
    best = float('inf')
    best_epoch = 0
    best_stage = ''
    best_post = float('inf')
    best_post_epoch = 0
    bad_post_epochs = 0
    started = time.time()
    for epoch in range(1, cfg['epochs'] + 1):
        stage = stage_name(epoch, cfg)
        configure_stage(model, stage)
        model.train()
        totals = {}
        batches = 0
        for batch in loaders['train']:
            moved = on_device(batch, device)
            output = model(moved['history'], moved['future_temperature'])
            loss, terms, _, _ = objective(model, output, moved, cfg, stage)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], cfg['grad_clip'])
            optimizer.step()
            totals['loss'] = totals.get('loss', 0.0) + float(loss.detach().cpu())
            for key, value in terms.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
            batches += 1
        metrics = evaluate(model, loaders['val'], device, std, cfg)
        row = {'epoch': epoch, 'stage': stage, 'seconds': time.time() - started, **{key: value / batches for key, value in totals.items()}, **{f'val_{key}': value for key, value in metrics.items() if not isinstance(value, list)}, 'val_loop_ratio_mean': float(np.mean(metrics['mf_loop_area_abs_ratio'])), 'val_memory_ratio_mean': float(np.mean(metrics['mf_memory_to_reversible_rms']))}
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / 'history.csv', index=False)
        print(json.dumps(row), flush=True)
        selection = metrics['displacement_mse']
        if selection < best:
            best = selection
            best_epoch = epoch
            best_stage = stage
            save_checkpoint(output_dir / 'best.pt', model, cfg, mean, std, epoch, stage, metrics)
            dump(output_dir / 'best_validation.json', metrics)
        if stage == 'joint_cutoff':
            if selection < best_post:
                best_post = selection
                best_post_epoch = epoch
                bad_post_epochs = 0
                save_checkpoint(output_dir / 'best_post_cutoff.pt', model, cfg, mean, std, epoch, stage, metrics)
                dump(output_dir / 'best_post_cutoff_validation.json', metrics)
            else:
                bad_post_epochs += 1
            if bad_post_epochs >= cfg['patience_after_cutoff']:
                break
    save_predictions(output_dir / 'validation_predictions.npz', output_dir / 'best.pt', model, loaders['val'], device, cfg, mean, std)
    save_predictions(output_dir / 'validation_predictions_post_cutoff.npz', output_dir / 'best_post_cutoff.pt', model, loaders['val'], device, cfg, mean, std)
    best_checkpoint = torch.load(output_dir / 'best.pt', map_location=device, weights_only=False)
    post_checkpoint = torch.load(output_dir / 'best_post_cutoff.pt', map_location=device, weights_only=False)
    summary = {'status': 'completed', 'best_epoch': best_epoch, 'best_stage': best_stage, 'best_validation_displacement_mse': best, 'best_validation': best_checkpoint['metrics'], 'best_post_cutoff_epoch': best_post_epoch, 'best_post_cutoff_validation_displacement_mse': best_post, 'best_post_cutoff_validation': post_checkpoint['metrics'], 'epochs_ran': len(history), 'elapsed_seconds': time.time() - started, 'parameters': sum((parameter.numel() for parameter in model.parameters())), 'gpu_max_allocated_bytes': torch.cuda.max_memory_allocated(device), 'test_evaluated': False, 'task': cfg['task']}
    dump(output_dir / 'summary.json', summary)
    print('COMPLETED ' + json.dumps(summary), flush=True)
if __name__ == '__main__':
    main()
