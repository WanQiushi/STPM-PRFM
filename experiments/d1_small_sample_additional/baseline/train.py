"""Train one frozen baseline using a prepared 9/6/3/1-month protocol.

The script starts training only when explicitly invoked with --config.
"""
from if_release.runtime import loads as _release_json_loads
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import random
import time
import traceback
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from data import PROJECT, Windows, dump, prepare
from models import Baseline

def move(batch, device):
    return {key: value.to(device) for key, value in batch.items()}

def evaluate(model, loader, device, std):
    sums, counts = (np.zeros(len(std)), np.zeros(len(std)))
    model.eval()
    with torch.no_grad():
        for batch in loader:
            b = move(batch, device)
            prediction = model(b)['prediction']
            sums += ((prediction - b['target']).square() * b['mask']).sum((0, 1)).cpu().numpy()
            counts += b['mask'].sum((0, 1)).cpu().numpy()
    physical_rmse = np.sqrt(sums / counts) * std
    return dict(joint_standardized_rmse=float(np.sqrt(sums.sum() / counts.sum())), temperature_rmse_c=float(physical_rmse[0]), pooled_rmse_mm=float(np.sqrt(np.sum(sums[1:] * std[1:] ** 2) / counts[1:].sum())), rmse_mm=physical_rmse[1:].tolist(), measured_counts=counts.tolist())

def save_predictions(model, loader, device, mean, std, channel_names, path):
    saved = {}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            b = move(batch, device)
            prediction = model(b)['prediction']
            arrays = dict(prediction=prediction.cpu().numpy(), target=batch['target'].numpy(), mask=batch['mask'].numpy(), origin=batch['origin'].numpy())
            for key, value in arrays.items():
                saved.setdefault(key, []).append(value)
    np.savez_compressed(path, **{key: np.concatenate(value) for key, value in saved.items()}, mean=mean, std=std, channel_names=np.asarray(channel_names))

def score(path):
    with np.load(path) as source:
        mask = source['mask'].astype(bool)
        prediction = source['prediction'].astype(np.float64)
        target = source['target'].astype(np.float64)
        error = (prediction - target) * source['std']
        counts = mask.sum((0, 1))
        squared = np.where(mask, error ** 2, 0).sum((0, 1))
        absolute = np.where(mask, np.abs(error), 0).sum((0, 1))
        mse = squared / counts
        mae = absolute / counts
        normalized_squared = np.where(mask, (prediction - target) ** 2, 0).sum((0, 1))
        displacement_mse = squared[1:].sum() / counts[1:].sum()
        return dict(joint_standardized_rmse=float(np.sqrt(normalized_squared.sum() / counts.sum())), temperature_rmse_c=float(np.sqrt(mse[0])), temperature_mae_c=float(mae[0]), temperature_mse_c2=float(mse[0]), pooled_rmse_mm=float(np.sqrt(displacement_mse)), pooled_mae_mm=float(absolute[1:].sum() / counts[1:].sum()), pooled_mse_mm2=float(displacement_mse), rmse_mm=np.sqrt(mse[1:]).tolist(), mae_mm=mae[1:].tolist(), mse_mm2=mse[1:].tolist(), windows=len(mask), measured_counts=counts.tolist())

def train(cfg):
    out = PROJECT / cfg['output']
    out.mkdir(parents=True, exist_ok=True)
    dump(out / 'config.json', cfg)
    torch.set_num_threads(cfg['torch_threads'])
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device('cuda:0')
    values, observed, parts, mean, std, protocol, ho, fo = prepare(cfg)
    dump(out / 'protocol.json', protocol)
    dump(out / 'window_origins.json', parts)
    report = dict(protocol_path=str(PROJECT / cfg['protocol_path']), data_path=protocol['data_path'], window_counts={name: len(origins) for name, origins in parts.items()}, history_points=len(ho), forecast_points=len(fo), channel_names=protocol['channel_names'], mean=mean.tolist(), std=std.tolist(), normalization_source='Prepared budget protocol; no full-511-day statistics.')
    dump(out / 'data_report.json', report)
    print('DATA ' + json.dumps(report), flush=True)
    model = Baseline(cfg).to(device)
    parameters = sum((p.numel() for p in model.parameters()))
    print('MODEL ' + json.dumps(dict(model=cfg['model'], scientific_label=model.scientific_label, parameters=parameters, micro_batch_size=cfg['micro_batch_size'], effective_batch_size=cfg['batch_size'])), flush=True)
    loaders = {name: DataLoader(Windows(values, observed, origins, ho, fo), batch_size=cfg['batch_size'] if name == 'train' else cfg['micro_batch_size'], shuffle=name == 'train', num_workers=0) for name, origins in parts.items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    best, best_epoch, bad = (float('inf'), 0, 0)
    rows = []
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        totals, batches = (0.0, 0)
        for batch in loaders['train']:
            b = move(batch, device)
            denominator = b['mask'].sum()
            optimizer.zero_grad()
            for first in range(0, len(b['origin']), cfg['micro_batch_size']):
                chunk = {key: value[first:first + cfg['micro_batch_size']] for key, value in b.items()}
                prediction = model(chunk)['prediction']
                loss = ((prediction - chunk['target']).square() * chunk['mask']).sum() / denominator
                loss.backward()
                totals += float(loss.detach().cpu())
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            optimizer.step()
            batches += 1
        validation = evaluate(model, loaders['val'], device, std)
        row = dict(epoch=epoch, stage='prediction', seconds=time.time() - started, loss=totals / batches, data_mse=totals / batches, validation_pooled_rmse_mm=validation['pooled_rmse_mm'], validation_temperature_rmse_c=validation['temperature_rmse_c'], validation_joint_standardized_rmse=validation['joint_standardized_rmse'])
        rows.append(row)
        pd.DataFrame(rows).to_csv(out / 'history.csv', index=False)
        print(json.dumps(row), flush=True)
        if validation['joint_standardized_rmse'] < best:
            best = validation['joint_standardized_rmse']
            best_epoch, bad = (epoch, 0)
            torch.save(dict(model=model.state_dict(), config=cfg, mean=mean, std=std, epoch=epoch, validation=validation), out / 'best_accuracy.pt')
            dump(out / 'best_accuracy_validation.json', validation)
        else:
            bad += 1
        if bad >= cfg['patience']:
            break
    training_seconds = time.time() - started
    model.load_state_dict(torch.load(out / 'best_accuracy.pt', map_location=device, weights_only=False)['model'])
    results = {}
    for split, label in [('val', 'validation'), ('test', 'test')]:
        path = out / (label + '_predictions_accuracy.npz')
        save_predictions(model, loaders[split], device, mean, std, protocol['channel_names'], path)
        results[label] = score(path)
    dump(out / 'suite_results.json', dict(version=cfg['run_name'], seed=cfg['seed'], results=dict(accuracy=results), test_used_for_selection=False, protocol_path=cfg['protocol_path'], checkpoint_selection='joint_standardized_validation_rmse', input_policy='Past 30-day temperature and four displacement channels only.'))
    summary = dict(status='completed', model=cfg['model'], scientific_label=model.scientific_label, seed=cfg['seed'], epochs_ran=len(rows), best_accuracy_epoch=best_epoch, parameters=parameters, elapsed_seconds=time.time() - started, training_seconds=training_seconds, peak_cuda_allocated_mib=torch.cuda.max_memory_allocated() / 1048576, test_evaluated=True, protocol_path=cfg['protocol_path'], results=results)
    dump(out / 'summary.json', summary)
    print('COMPLETED ' + json.dumps(results), flush=True)
    return summary

def write_record(path, cfg, config_path, started_at, status, detail):
    text = f"# D1 small-sample baseline run\n\n- Date (Asia/Shanghai): {started_at.isoformat()}\n- Project: D1 minute-resolution joint forecasting\n- Purpose: {Path(cfg['protocol_path']).parent.name} supervision budget, {cfg['model']}, seed {cfg['seed']}\n- Status: {status}\n- Config: `{config_path}`\n- Protocol and budget normalization: `{PROJECT / cfg['protocol_path']}`\n- Entry: `{Path(__file__).resolve()}`\n- Model code: `{Path(__file__).resolve().with_name('models.py')}`\n- Output: `{PROJECT / cfg['output']}`\n- Operation: train from scratch with frozen architecture, optimizer, early stopping and selection rule; score validation/test after selection.\n\n{detail}\n"
    path.write_text(text, encoding='utf-8')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = _release_json_loads(config_path.read_text(encoding='utf-8'))
    started_at = datetime.now(timezone(timedelta(hours=8)))
    record_dir = PROJECT / cfg['record_dir']
    record_dir.mkdir(parents=True, exist_ok=True)
    budget = Path(cfg['protocol_path']).parent.name
    record_path = record_dir / f"{started_at.strftime('%Y%m%d_%H%M%S_%f')}_{budget}_{cfg['model']}_seed{cfg['seed']}.md"
    started = time.time()
    out = PROJECT / cfg['output']
    try:
        summary = train(cfg)
    except BaseException as error:
        original_error = traceback.format_exc()
        status = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        error_file = out / 'error.txt'
        error_file.write_text(original_error, encoding='utf-8')
        dump(out / 'summary.json', dict(status=status, model=cfg['model'], seed=cfg['seed'], error_file=str(error_file), elapsed_seconds=time.time() - started))
        reason = original_error.rstrip().splitlines()[-1]
        detail = f'Reason: {reason}\n\nOriginal error: `{error_file}`. No retries were attempted.'
        write_record(record_path, cfg, config_path, started_at, status, detail)
        raise
    summary['elapsed_seconds'] = time.time() - started
    dump(out / 'summary.json', summary)
    results = summary['results']
    detail = f"Validation displacement RMSE: {results['validation']['pooled_rmse_mm']:.6f} mm.\n\nTest displacement RMSE: {results['test']['pooled_rmse_mm']:.6f} mm; MAE: {results['test']['pooled_mae_mm']:.6f} mm; temperature RMSE: {results['test']['temperature_rmse_c']:.6f} deg C.\n\nAll channel metrics and counts: `{PROJECT / cfg['output'] / 'suite_results.json'}`. Weights: `{PROJECT / cfg['output'] / 'best_accuracy.pt'}`."
    write_record(record_path, cfg, config_path, started_at, 'completed', detail)
if __name__ == '__main__':
    main()
