from __future__ import annotations
import importlib.util
import sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
BASE_EXPERIMENT = ROOT / 'experiments' / 'fft_joint_neuralode_d1_staged_20260929'
sys.path.insert(0, str(BASE_EXPERIMENT))
import train as base
corrected_spec = importlib.util.spec_from_file_location('corrected_fft_joint_model', Path(__file__).with_name('model.py'))
corrected_module = importlib.util.module_from_spec(corrected_spec)
corrected_spec.loader.exec_module(corrected_module)
CorrectedJointFFTNeuralODE = corrected_module.CorrectedJointFFTNeuralODE
base_configure_stage = base.configure_stage

def masked_temporal_difference_mse(prediction, target, mask):
    active = mask[:, 1:] * mask[:, :-1]
    return base.masked_channel_mse(prediction[:, 1:] - prediction[:, :-1], target[:, 1:] - target[:, :-1], active)

def masked_relative_path_mse(prediction, target, mask):
    prediction_relative = prediction - prediction[:, :1]
    target_relative = target - target[:, :1]
    return base.masked_channel_mse(prediction_relative, target_relative, mask)

def configure_stage(model, stage):
    base_configure_stage(model, stage)
    if stage == 'bw_pretrain':
        model.thermal_lag_raw.requires_grad_(True)

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
    loop, _, _ = base.loop_loss(output, target_mf, cfg)
    continuity = torch.mean((output['prediction'][:, 0] - batch['history'][:, -1, 1:]).square())
    if stage == 'bw_pretrain':
        decomposition = output['prediction'].new_zeros(())
        decomposition_terms = {'stage1_static_r2': decomposition, 'stage1_bw_r2': decomposition, 'stage1_low_purity': decomposition, 'stage1_high_purity': decomposition}
        total = mf + cfg['mf_delta_weight'] * mf_delta + cfg['loop_weight'] * loop + cfg['ode_weight'] * output['ode_residual_penalty']
    else:
        decomposition, decomposition_terms = base.stage1_loss(model, batch['history'], cfg['stage1_loss_weights'])
        total = prediction + cfg['stage1_weight'] * decomposition + cfg['lf_component_weight'] * lf + cfg['mf_component_weight'] * mf + cfg['lf_path_weight'] * lf_path + cfg['lf_delta_weight'] * lf_delta + cfg['mf_delta_weight'] * mf_delta + cfg['continuity_weight'] * continuity + cfg['loop_weight'] * loop + cfg['ode_weight'] * output['ode_residual_penalty']
    terms = {'prediction_mse': prediction, 'lf_mse': lf, 'mf_mse': mf, 'lf_path_mse': lf_path, 'lf_delta_mse': lf_delta, 'mf_delta_mse': mf_delta, 'continuity_mse': continuity, 'loop_loss': loop, 'stage1_loss': decomposition, 'ode_residual': output['ode_residual_penalty'], **decomposition_terms}
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
            moved = base.on_device(batch, device)
            output = model(moved['history'], moved['future_temperature'])
            error = output['prediction'] - moved['target']
            mask = moved['mask']
            squared += (error.square() * mask).sum((0, 1)).cpu().numpy()
            counts += mask.sum((0, 1)).cpu().numpy()
            first += (error[:, :48].square() * mask[:, :48]).sum((0, 1)).cpu().numpy()
            first_counts += mask[:, :48].sum((0, 1)).cpu().numpy()
            second += (error[:, 48:].square() * mask[:, 48:]).sum((0, 1)).cpu().numpy()
            second_counts += mask[:, 48:].sum((0, 1)).cpu().numpy()
            _, target_mf = base.target_components(model, moved)
            _, area_pred, area_target = base.loop_loss(output, target_mf, cfg)
            predicted_area += area_pred.abs().sum(0).cpu().numpy()
            target_area += area_target.abs().sum(0).cpu().numpy()
            reversible_energy += output['reversible_prediction'].square().sum((0, 1)).cpu().numpy()
            memory_energy += output['memory_prediction'].square().sum((0, 1)).cpu().numpy()
            point_count += len(mask) * mask.shape[1]
    static_hours, noise_hours = model.decomposer.cutoff_hours()
    return {'displacement_mse': float(np.mean(squared / counts)), 'first_12h_mse': float(np.mean(first / first_counts)), 'second_12h_mse': float(np.mean(second / second_counts)), 'rmse_mm': (np.sqrt(squared / counts) * std[1:]).tolist(), 'first_12h_rmse_mm': (np.sqrt(first / first_counts) * std[1:]).tolist(), 'second_12h_rmse_mm': (np.sqrt(second / second_counts) * std[1:]).tolist(), 'measured_counts': counts.tolist(), 'static_period_hours': float(static_hours.cpu()), 'noise_period_hours': float(noise_hours.cpu()), 'thermal_lag_hours': model.thermal_lag_hours().detach().cpu().tolist(), 'mf_loop_area_abs_ratio': (predicted_area / np.maximum(target_area, 1e-12)).tolist(), 'mf_memory_to_reversible_rms': (np.sqrt(memory_energy / point_count) / np.maximum(np.sqrt(reversible_energy / point_count), 1e-12)).tolist()}

def save_predictions(path, checkpoint_path, model, loader, device, cfg, mean, std):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    saved = {}
    with torch.no_grad():
        for batch in loader:
            moved = base.on_device(batch, device)
            output = model(moved['history'], moved['future_temperature'])
            target_lf, target_mf = base.target_components(model, moved)
            arrays = {'prediction': output['prediction'].cpu().numpy(), 'lf_prediction': output['lf_prediction'].cpu().numpy(), 'mf_prediction': output['mf_prediction'].cpu().numpy(), 'temperature_lf': output['temperature_lf'].cpu().numpy(), 'temperature_mf': output['temperature_mf'].cpu().numpy(), 'effective_temperature_mf': output['effective_temperature_mf'].cpu().numpy(), 'reversible_prediction': output['reversible_prediction'].cpu().numpy(), 'memory_prediction': output['memory_prediction'].cpu().numpy(), 'lf_target': target_lf.cpu().numpy(), 'mf_target': target_mf.cpu().numpy(), 'target': batch['target'].numpy(), 'mask': batch['mask'].numpy(), 'future_temperature': batch['future_temperature'].numpy(), 'origin': batch['origin'].numpy()}
            for key, value in arrays.items():
                saved.setdefault(key, []).append(value)
    saved = {key: np.concatenate(value) for key, value in saved.items()}
    np.savez_compressed(path, **saved, mean=mean, std=std)
base.JointFFTNeuralODE = CorrectedJointFFTNeuralODE
base.configure_stage = configure_stage
base.objective = objective
base.evaluate = evaluate
base.save_predictions = save_predictions
if __name__ == '__main__':
    base.main()
