from __future__ import annotations
from if_release.runtime import loads as _release_json_loads
import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
ROOT = Path(__file__).resolve().parents[2]
STAGED_EXPERIMENT = ROOT / 'experiments' / 'fft_joint_neuralode_d1_staged_20260929'
sys.path.insert(0, str(STAGED_EXPERIMENT))
import train as base
model_spec = importlib.util.spec_from_file_location('direct_bw_joint_model', Path(__file__).with_name('model.py'))
model_module = importlib.util.module_from_spec(model_spec)
model_spec.loader.exec_module(model_module)
DirectBWJointFFTNeuralODE = model_module.DirectBWJointFFTNeuralODE

def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')

def forward_batch(model, batch):
    return model(batch['history'], batch['future_temperature'], batch['origin'])

def masked_temporal_difference_mse(prediction, target, mask):
    active = mask[:, 1:] * mask[:, :-1]
    return base.masked_channel_mse(prediction[:, 1:] - prediction[:, :-1], target[:, 1:] - target[:, :-1], active)

def masked_relative_path_mse(prediction, target, mask):
    return base.masked_channel_mse(prediction - prediction[:, :1], target - target[:, :1], mask)

def mf_spectral_mse(prediction, target, mask):
    mask = mask.to(prediction.dtype)
    prediction = (prediction - prediction.mean(dim=1, keepdim=True)) * mask
    target = (target - target.mean(dim=1, keepdim=True)) * mask
    pred_spectrum = torch.fft.rfft(prediction, dim=1, norm='ortho')[:, 1:]
    target_spectrum = torch.fft.rfft(target, dim=1, norm='ortho')[:, 1:]
    return (pred_spectrum - target_spectrum).abs().square().mean()

def soft_extreme_loss(prediction, target, mask, temperature):
    mask = mask.bool()
    very_negative = torch.full_like(prediction, -10000.0)
    pred_max = temperature * torch.logsumexp(torch.where(mask, prediction / temperature, very_negative), dim=1)
    target_max = temperature * torch.logsumexp(torch.where(mask, target / temperature, very_negative), dim=1)
    pred_min = -temperature * torch.logsumexp(torch.where(mask, -prediction / temperature, very_negative), dim=1)
    target_min = -temperature * torch.logsumexp(torch.where(mask, -target / temperature, very_negative), dim=1)
    return 0.5 * ((pred_max - target_max).square().mean() + (pred_min - target_min).square().mean())

def loop_loss(output, target_mf, cfg):
    u = output['effective_temperature_mf'].detach()
    predicted_area = base.closed_area(u, output['mf_prediction'])
    target_area = base.closed_area(u, target_mf.detach())
    relative = (predicted_area - target_area).square() / (target_area.detach().square() + float(cfg['loop_epsilon']))
    return (relative.mean(), predicted_area, target_area)

def configure_stage(model, stage):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if stage == 'bw_pretrain':
        model.thermal_lag_raw.requires_grad_(True)
        model.a_raw.requires_grad_(True)
        model.beta_raw.requires_grad_(True)
        model.gamma_raw.requires_grad_(True)
        for module in (model.initial_z, model.bw_residual):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
    elif stage == 'prediction_warmup':
        for name, parameter in model.named_parameters():
            if not name.startswith('decomposer.'):
                parameter.requires_grad_(True)
    else:
        for parameter in model.parameters():
            parameter.requires_grad_(True)

def objective(model, output, batch, cfg, stage):
    prediction = base.masked_channel_mse(output['prediction'], batch['target'], batch['mask'])
    target_lf, target_mf = base.target_components(model, batch)
    target_lf = target_lf.detach()
    target_mf = target_mf.detach()
    lf = base.masked_channel_mse(output['lf_prediction'], target_lf, batch['mask'])
    mf = base.masked_channel_mse(output['mf_prediction'], target_mf, batch['mask'])
    lf_path = masked_relative_path_mse(output['lf_prediction'], target_lf, batch['mask'])
    lf_delta = masked_temporal_difference_mse(output['lf_prediction'], target_lf, batch['mask'])
    mf_delta = masked_temporal_difference_mse(output['mf_prediction'], target_mf, batch['mask'])
    mf_spectral = mf_spectral_mse(output['mf_prediction'], target_mf, batch['mask'])
    extreme = soft_extreme_loss(output['prediction'], batch['target'], batch['mask'], float(cfg['extreme_temperature']))
    loop, _, _ = loop_loss(output, target_mf, cfg)
    continuity = torch.mean((output['prediction'][:, 0] - batch['history'][:, -1, 1:]).square())
    if stage == 'bw_pretrain':
        decomposition = output['prediction'].new_zeros(())
        decomposition_terms = {'stage1_static_r2': decomposition, 'stage1_bw_r2': decomposition, 'stage1_low_purity': decomposition, 'stage1_high_purity': decomposition}
        total = mf + float(cfg['mf_delta_weight']) * mf_delta + float(cfg['mf_spectral_weight']) * mf_spectral + float(cfg['loop_weight']) * loop + float(cfg['ode_weight']) * output['ode_residual_penalty']
    else:
        decomposition, decomposition_terms = base.stage1_loss(model, batch['history'], cfg['stage1_loss_weights'])
        total = prediction + float(cfg['stage1_weight']) * decomposition + float(cfg['lf_component_weight']) * lf + float(cfg['mf_component_weight']) * mf + float(cfg['lf_path_weight']) * lf_path + float(cfg['lf_delta_weight']) * lf_delta + float(cfg['mf_delta_weight']) * mf_delta + float(cfg['mf_spectral_weight']) * mf_spectral + float(cfg['extreme_weight']) * extreme + float(cfg['continuity_weight']) * continuity + float(cfg['loop_weight']) * loop + float(cfg['ode_weight']) * output['ode_residual_penalty']
    terms = {'prediction_mse': prediction, 'lf_mse': lf, 'mf_mse': mf, 'lf_path_mse': lf_path, 'lf_delta_mse': lf_delta, 'mf_delta_mse': mf_delta, 'mf_spectral_mse': mf_spectral, 'extreme_loss': extreme, 'continuity_mse': continuity, 'loop_loss': loop, 'stage1_loss': decomposition, 'ode_residual': output['ode_residual_penalty'], **decomposition_terms}
    return (total, terms, target_lf, target_mf)

def initialize_bw_from_training(model, loader, device):
    estimates = [[] for _ in range(model.displacements)]
    model.eval()
    with torch.no_grad():
        for batch in loader:
            moved = base.on_device(batch, device)
            output = forward_batch(model, moved)
            _, target_mf = base.target_components(model, moved)
            q = output['effective_temperature_mf']
            du = torch.diff(q, dim=1) / model.step_hours
            dy = torch.diff(target_mf, dim=1) / model.step_hours
            y_mid = 0.5 * (target_mf[:, 1:] + target_mf[:, :-1])
            for batch_index in range(len(q)):
                for support in range(model.displacements):
                    design = torch.stack((du[batch_index, :, support], -du[batch_index, :, support].abs() * y_mid[batch_index, :, support], -du[batch_index, :, support] * y_mid[batch_index, :, support].abs()), dim=1)
                    response = dy[batch_index, :, support]
                    ridge = 0.0001 * torch.eye(3, device=device)
                    estimate = torch.linalg.solve(design.T @ design + ridge, design.T @ response)
                    estimates[support].append(estimate.cpu().numpy())
    medians = np.stack([np.median(np.stack(items), axis=0) for items in estimates])
    a = np.maximum(medians[:, 0], 0.001)
    beta = np.maximum(medians[:, 1], 0.001)
    gamma = np.clip(medians[:, 2], -0.85 * beta, 0.85 * beta)
    model.set_physics_parameters(a, beta, gamma)
    return {'a': a.tolist(), 'beta': beta.tolist(), 'gamma': gamma.tolist()}

def evaluate(model, loader, device, std, cfg):
    squared = np.zeros(4)
    first = np.zeros(4)
    second = np.zeros(4)
    counts = np.zeros(4)
    first_counts = np.zeros(4)
    second_counts = np.zeros(4)
    predicted_area = np.zeros(4)
    target_area = np.zeros(4)
    peak_error = np.zeros(4)
    trough_error = np.zeros(4)
    physics_energy = np.zeros(4)
    residual_energy = np.zeros(4)
    windows = 0
    points = 0
    with torch.no_grad():
        model.eval()
        for batch in loader:
            moved = base.on_device(batch, device)
            output = forward_batch(model, moved)
            error = output['prediction'] - moved['target']
            mask = moved['mask']
            squared += (error.square() * mask).sum((0, 1)).cpu().numpy()
            counts += mask.sum((0, 1)).cpu().numpy()
            first += (error[:, :48].square() * mask[:, :48]).sum((0, 1)).cpu().numpy()
            first_counts += mask[:, :48].sum((0, 1)).cpu().numpy()
            second += (error[:, 48:].square() * mask[:, 48:]).sum((0, 1)).cpu().numpy()
            second_counts += mask[:, 48:].sum((0, 1)).cpu().numpy()
            _, target_mf = base.target_components(model, moved)
            _, area_pred, area_target = loop_loss(output, target_mf, cfg)
            predicted_area += area_pred.abs().sum(0).cpu().numpy()
            target_area += area_target.abs().sum(0).cpu().numpy()
            peak_error += (output['prediction'].amax(dim=1) - moved['target'].amax(dim=1)).abs().sum(0).cpu().numpy()
            trough_error += (output['prediction'].amin(dim=1) - moved['target'].amin(dim=1)).abs().sum(0).cpu().numpy()
            physics_energy += output['bw_physics_rate'].square().sum((0, 1)).cpu().numpy()
            residual_energy += output['bw_residual_rate'].square().sum((0, 1)).cpu().numpy()
            windows += len(mask)
            points += len(mask) * mask.shape[1]
    static_hours, noise_hours = model.decomposer.cutoff_hours()
    params = model.physics_parameters()
    return {'displacement_mse': float(np.mean(squared / counts)), 'first_12h_mse': float(np.mean(first / first_counts)), 'second_12h_mse': float(np.mean(second / second_counts)), 'rmse_mm': (np.sqrt(squared / counts) * std[1:]).tolist(), 'first_12h_rmse_mm': (np.sqrt(first / first_counts) * std[1:]).tolist(), 'second_12h_rmse_mm': (np.sqrt(second / second_counts) * std[1:]).tolist(), 'peak_abs_error_mm': (peak_error / windows * std[1:]).tolist(), 'trough_abs_error_mm': (trough_error / windows * std[1:]).tolist(), 'measured_counts': counts.tolist(), 'static_period_hours': float(static_hours.cpu()), 'noise_period_hours': float(noise_hours.cpu()), 'thermal_lag_hours': model.thermal_lag_hours().detach().cpu().tolist(), 'bw_a': params['a'].detach().cpu().tolist(), 'bw_beta': params['beta'].detach().cpu().tolist(), 'bw_gamma': params['gamma'].detach().cpu().tolist(), 'mf_loop_area_abs_ratio': (predicted_area / np.maximum(target_area, 1e-12)).tolist(), 'bw_residual_to_physics_rate_rms': (np.sqrt(residual_energy / points) / np.maximum(np.sqrt(physics_energy / points), 1e-12)).tolist()}

def save_checkpoint(path, model, cfg, mean, std, epoch, stage, metrics):
    torch.save({'model': model.state_dict(), 'config': cfg, 'mean': mean, 'std': std, 'epoch': epoch, 'stage': stage, 'metrics': metrics}, path)

def save_predictions(path, checkpoint_path, model, loader, device, mean, std):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    saved = {}
    with torch.no_grad():
        for batch in loader:
            moved = base.on_device(batch, device)
            output = forward_batch(model, moved)
            target_lf, target_mf = base.target_components(model, moved)
            arrays = {'prediction': output['prediction'].cpu().numpy(), 'lf_prediction': output['lf_prediction'].cpu().numpy(), 'mf_prediction': output['mf_prediction'].cpu().numpy(), 'temperature_lf': output['temperature_lf'].cpu().numpy(), 'temperature_mf': output['temperature_mf'].cpu().numpy(), 'effective_temperature_mf': output['effective_temperature_mf'].cpu().numpy(), 'bw_physics_rate': output['bw_physics_rate'].cpu().numpy(), 'bw_residual_rate': output['bw_residual_rate'].cpu().numpy(), 'lf_target': target_lf.cpu().numpy(), 'mf_target': target_mf.cpu().numpy(), 'target': batch['target'].numpy(), 'mask': batch['mask'].numpy(), 'future_temperature': batch['future_temperature'].numpy(), 'origin': batch['origin'].numpy()}
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
    values, observed, parts, mean, std, report, history_offsets, future_offsets = base.prepare(cfg)
    dump(output_dir / 'data_report.json', {'counts': report, 'mean': mean.tolist(), 'std': std.tolist(), 'history_days': cfg['history_days'], 'history_policy': 'all 30-day values must be available before origin', 'future_temperature_policy': 'measured 15-minute nodes', 'future_displacement_policy': 'labels only', 'test_policy': 'sealed'})
    dump(output_dir / 'window_origins.json', parts)
    print('DATA ' + json.dumps(report), flush=True)
    seed = int(cfg['seed'])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    model = DirectBWJointFFTNeuralODE(cfg).to(device)
    loaders = {name: DataLoader(base.Windows(values, observed, origins, history_offsets, future_offsets), batch_size=cfg['batch_size'], shuffle=name == 'train', num_workers=0) for name, origins in parts.items()}
    initialization = initialize_bw_from_training(model, loaders['train'], device)
    dump(output_dir / 'bw_ols_initialization.json', initialization)
    print('BW_INIT ' + json.dumps(initialization), flush=True)
    cutoff_parameters = list(model.decomposer.parameters())
    cutoff_ids = {id(parameter) for parameter in cutoff_parameters}
    remaining = [parameter for parameter in model.parameters() if id(parameter) not in cutoff_ids]
    optimizer = torch.optim.Adam([{'params': remaining, 'lr': cfg['lr']}, {'params': cutoff_parameters, 'lr': cfg['cutoff_lr']}])
    history = []
    best_any = float('inf')
    best_any_epoch = 0
    best_joint = float('inf')
    best_joint_epoch = 0
    bad_joint_epochs = 0
    started = time.time()
    for epoch in range(1, int(cfg['epochs']) + 1):
        stage = base.stage_name(epoch, cfg)
        configure_stage(model, stage)
        model.train()
        totals = {}
        batches = 0
        for batch in loaders['train']:
            moved = base.on_device(batch, device)
            output = forward_batch(model, moved)
            loss, terms, _, _ = objective(model, output, moved, cfg, stage)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], float(cfg['grad_clip']))
            optimizer.step()
            totals['loss'] = totals.get('loss', 0.0) + float(loss.detach().cpu())
            for key, value in terms.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
            batches += 1
        metrics = evaluate(model, loaders['val'], device, std, cfg)
        row = {'epoch': epoch, 'stage': stage, 'seconds': time.time() - started, **{key: value / batches for key, value in totals.items()}, **{f'val_{key}': value for key, value in metrics.items() if not isinstance(value, list)}, 'val_loop_ratio_mean': float(np.mean(metrics['mf_loop_area_abs_ratio'])), 'val_residual_rate_ratio_mean': float(np.mean(metrics['bw_residual_to_physics_rate_rms']))}
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / 'history.csv', index=False)
        print(json.dumps(row), flush=True)
        selection = metrics['displacement_mse']
        if selection < best_any:
            best_any = selection
            best_any_epoch = epoch
            save_checkpoint(output_dir / 'best_any_stage.pt', model, cfg, mean, std, epoch, stage, metrics)
            dump(output_dir / 'best_any_stage_validation.json', metrics)
        if stage == 'joint_cutoff':
            if selection < best_joint:
                best_joint = selection
                best_joint_epoch = epoch
                bad_joint_epochs = 0
                save_checkpoint(output_dir / 'best.pt', model, cfg, mean, std, epoch, stage, metrics)
                dump(output_dir / 'best_validation.json', metrics)
            else:
                bad_joint_epochs += 1
            if bad_joint_epochs >= int(cfg['patience_after_cutoff']):
                break
    save_predictions(output_dir / 'validation_predictions.npz', output_dir / 'best.pt', model, loaders['val'], device, mean, std)
    save_predictions(output_dir / 'validation_predictions_any_stage.npz', output_dir / 'best_any_stage.pt', model, loaders['val'], device, mean, std)
    best_checkpoint = torch.load(output_dir / 'best.pt', map_location=device, weights_only=False)
    any_checkpoint = torch.load(output_dir / 'best_any_stage.pt', map_location=device, weights_only=False)
    summary = {'status': 'completed', 'best_joint_epoch': best_joint_epoch, 'best_joint_validation_displacement_mse': best_joint, 'best_joint_validation': best_checkpoint['metrics'], 'best_any_stage_epoch': best_any_epoch, 'best_any_stage': any_checkpoint['stage'], 'best_any_stage_validation_displacement_mse': best_any, 'best_any_stage_validation': any_checkpoint['metrics'], 'epochs_ran': len(history), 'elapsed_seconds': time.time() - started, 'parameters': sum((parameter.numel() for parameter in model.parameters())), 'gpu_max_allocated_bytes': torch.cuda.max_memory_allocated(device), 'test_evaluated': False, 'task': cfg['task']}
    dump(output_dir / 'summary.json', summary)
    print('COMPLETED ' + json.dumps(summary), flush=True)
if __name__ == '__main__':
    main()
