"""Original 15-minute H1 identification and v11-forecast adaptation."""
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from data import ResponseWindows, on_device
from response_model import HybridKqsz, SOURCE, load_module
from temperature import dump, seed_all, write_history
anchor_training = load_module('small_sample_anchor_training', SOURCE / 'experiments/fft_joint_neuralode_d1_anchor_seasonal_bw_20260929/train.py')
training = anchor_training.direct
base = training.base

def score(prediction, target, mask):
    error = prediction.astype(np.float64) - target.astype(np.float64)
    counts = mask.sum((0, 1))
    sse = (error ** 2 * mask).sum((0, 1))
    sae = (np.abs(error) * mask).sum((0, 1))
    return {'rmse': float(np.sqrt(sse.sum() / counts.sum())), 'mse': float(sse.sum() / counts.sum()), 'mae': float(sae.sum() / counts.sum()), 'channel_rmse': np.sqrt(sse / counts).tolist(), 'channel_mae': (sae / counts).tolist(), 'channel_bias': ((error * mask).sum((0, 1)) / counts).tolist(), 'measured_counts': counts.astype(int).tolist(), 'windows': len(prediction)}

def evaluate(model, loader, device, data, forecast, collect=False):
    model.eval()
    squared = np.zeros(4)
    counts = np.zeros(4)
    saved = {name: [] for name in ['prediction_mm', 'target_mm', 'mask', 'origin', 'input_temperature_celsius', 'lf_prediction_mm', 'mf_prediction_mm']}
    mean_t = torch.as_tensor(data.mean[:1], dtype=torch.float32, device=device)
    std_t = torch.as_tensor(data.std[:1], dtype=torch.float32, device=device)
    with torch.no_grad():
        for batch in loader:
            current = on_device(batch, device)
            temperature = (current['forecast_temperature_celsius'] - mean_t) / std_t if forecast else current['future_temperature']
            result = model(current['history'], temperature, current['origin'])
            squared += ((result['prediction'] - current['target']).square() * current['mask']).sum((0, 1)).cpu().numpy()
            counts += current['mask'].sum((0, 1)).cpu().numpy()
            if collect:
                saved['prediction_mm'].append(result['prediction'].cpu().numpy() * data.std[1:] + data.mean[1:])
                saved['lf_prediction_mm'].append(result['lf_prediction'].cpu().numpy() * data.std[1:] + data.mean[1:])
                saved['mf_prediction_mm'].append(result['mf_prediction'].cpu().numpy() * data.std[1:])
                saved['target_mm'].append(batch['target'].numpy() * data.std[1:] + data.mean[1:])
                saved['mask'].append(batch['mask'].numpy())
                saved['origin'].append(batch['origin'].numpy())
                saved['input_temperature_celsius'].append(temperature.cpu().numpy() * data.std[:1] + data.mean[:1])
    rmse = float(np.sqrt((squared * data.std[1:] ** 2).sum() / counts.sum()))
    return (rmse, {name: np.concatenate(values) for name, values in saved.items()}) if collect else rmse

def save_checkpoint(path, model, cfg, data, epoch, phase, rmse):
    torch.save({'model': model.state_dict(), 'config': cfg, 'mean': data.mean, 'std': data.std, 'epoch': epoch, 'phase': phase, 'validation_rmse_mm': rmse}, path)

def train_response(data, pretraining_origins, oof_origins, oof_forecasts, validation_origins, validation_forecasts, seed, output, device, cfg, adaptation):
    output.mkdir(parents=True)
    cfg = {**cfg, 'seed': seed, 'sample_minutes': 15}
    dump(output / 'config.json', {'pretraining': cfg, 'adaptation': adaptation})
    dump(output / 'window_origins.json', {'pretraining': pretraining_origins, 'adaptation_train': list(map(int, oof_origins)), 'adaptation_val': list(validation_origins)})
    loaders = {name: DataLoader(ResponseWindows(data, origins, 15), batch_size=cfg['batch_size'], shuffle=name == 'train', num_workers=0) for name, origins in pretraining_origins.items()}
    seed_all(seed)
    model = HybridKqsz(cfg).to(device)
    cutoff_ids = {id(parameter) for parameter in model.decomposer.parameters()}
    optimizer = torch.optim.Adam([parameter for parameter in model.parameters() if id(parameter) not in cutoff_ids], lr=cfg['lr'])
    best = best_joint = float('inf')
    best_epoch = bad = 0
    rows = []
    started = time.time()
    for epoch in range(1, cfg['epochs'] + 1):
        stage = base.stage_name(epoch, cfg)
        model.configure_stage(stage)
        model.train()
        total = 0.0
        for batch in loaders['train']:
            current = on_device(batch, device)
            result = model(current['history'], current['future_temperature'], current['origin'])
            loss, _, _, _ = training.objective(model, result, current, cfg, stage)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], cfg['grad_clip'])
            optimizer.step()
            total += loss.item()
        rmse = evaluate(model, loaders['val'], device, data, False)
        if rmse < best:
            best, best_epoch = (rmse, epoch)
            save_checkpoint(output / 'pretrained_best.pt', model, cfg, data, epoch, stage, rmse)
        if stage == 'joint_cutoff':
            if rmse < best_joint:
                best_joint, bad = (rmse, 0)
            else:
                bad += 1
        row = {'epoch': epoch, 'phase': 'pretraining', 'stage': stage, 'loss': total / len(loaders['train']), 'validation_rmse_mm': rmse, 'seconds': time.time() - started}
        rows.append(row)
        write_history(output / 'pretraining_history.csv', rows)
        print(json.dumps(row), flush=True)
        if stage == 'joint_cutoff' and bad >= cfg['patience_after_cutoff']:
            break
    pretrain = {'best_epoch': best_epoch, 'epochs': epoch, 'seconds': time.time() - started, 'validation_rmse_mm': best, 'windows': {name: len(origins) for name, origins in pretraining_origins.items()}}
    state = torch.load(output / 'pretrained_best.pt', map_location=device, weights_only=False)
    model.load_state_dict(state['model'])
    model.configure_stage('forecast_adaptation')
    loaders = {'train': DataLoader(ResponseWindows(data, oof_origins, 15, oof_forecasts), batch_size=adaptation['batch_size'], shuffle=True, num_workers=0), 'val': DataLoader(ResponseWindows(data, validation_origins, 15, validation_forecasts), batch_size=adaptation['batch_size'], shuffle=False, num_workers=0)}
    seed_all(seed)
    optimizer = torch.optim.Adam([parameter for parameter in model.parameters() if parameter.requires_grad], lr=adaptation['lr'])
    best = evaluate(model, loaders['val'], device, data, True)
    best_epoch = bad = 0
    save_checkpoint(output / 'adapted_best.pt', model, cfg, data, 0, 'forecast_adaptation', best)
    mean_t = torch.as_tensor(data.mean[:1], dtype=torch.float32, device=device)
    std_t = torch.as_tensor(data.std[:1], dtype=torch.float32, device=device)
    rows = []
    started = time.time()
    for epoch in range(1, adaptation['epochs'] + 1):
        model.train()
        squared = nodes = 0.0
        for batch in loaders['train']:
            current = on_device(batch, device)
            temperature = (current['forecast_temperature_celsius'] - mean_t) / std_t
            result = model(current['history'], temperature, current['origin'])
            sse = ((result['prediction'] - current['target']).square() * current['mask']).sum()
            count = current['mask'].sum()
            loss = sse / count + adaptation['ode_weight'] * result['ode_residual_penalty']
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), adaptation['grad_clip'])
            optimizer.step()
            squared += sse.item()
            nodes += count.item()
        rmse = evaluate(model, loaders['val'], device, data, True)
        if rmse < best:
            best, best_epoch, bad = (rmse, epoch, 0)
            save_checkpoint(output / 'adapted_best.pt', model, cfg, data, epoch, 'forecast_adaptation', rmse)
        else:
            bad += 1
        row = {'epoch': epoch, 'phase': 'forecast_adaptation', 'train_mse': squared / nodes, 'validation_rmse_mm': rmse, 'best_epoch': best_epoch, 'seconds': time.time() - started}
        rows.append(row)
        write_history(output / 'adaptation_history.csv', rows)
        print(json.dumps(row), flush=True)
        if bad >= adaptation['patience']:
            break
    state = torch.load(output / 'adapted_best.pt', map_location=device, weights_only=False)
    model.load_state_dict(state['model'])
    summary = {'status': 'completed', 'seed': seed, 'pretraining': pretrain, 'adaptation': {'best_epoch': best_epoch, 'epochs': epoch, 'seconds': time.time() - started, 'validation_rmse_mm': best, 'train_windows': len(oof_origins), 'validation_windows': len(validation_origins)}, 'adaptation_temperature': 'fresh original v11 chronological OOF (15 min)', 'parameters': sum((parameter.numel() for parameter in model.parameters())), 'full_data_weights_used': False}
    dump(output / 'summary.json', summary)
    return (model, summary)

def native_model(trained, cfg, device):
    """Only deterministic frequency buffers change from the 15-minute grid."""
    native = HybridKqsz({**cfg, 'sample_minutes': 1}).to(device)
    state = trained.state_dict()
    state['decomposer.frequency'] = native.decomposer.frequency
    state['decomposer.padded_frequency'] = native.decomposer.padded_frequency
    native.load_state_dict(state)
    native.requires_grad_(False)
    return native.eval()
