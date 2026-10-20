"""Minute-resolution joint temperature/displacement forecasting from historical data only."""
from if_release.runtime import loads as _release_json_loads
import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from data import PROJECT, Windows, prepare, read_protocol, dump
from models import Baseline

def move(batch, device):
    return {key: value.to(device) for key, value in batch.items()}

def difference_masks(mask):
    return (mask[:, 1:] & mask[:, :-1], mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2])

def shared_objective(prediction, batch, scales, denominators, cfg):
    target, mask = (batch['target'], batch['mask'])
    m1, m2 = difference_masks(mask)
    mse = ((prediction - target).square() * mask).sum() / denominators[0]
    d1 = torch.diff(prediction - target, dim=1) / scales[0]
    d2 = torch.diff(prediction - target, n=2, dim=1) / scales[1]
    first = (d1.square() * m1).sum() / denominators[1]
    second = (d2.square() * m2).sum() / denominators[2]
    derivative = cfg['first_difference_weight'] * first + cfg['second_difference_weight'] * second
    return (mse + derivative, dict(data_mse=mse, first_difference_mse=first, second_difference_mse=second), derivative)

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

def save_predictions(model, loader, device, mean, std, path):
    saved = {}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            b = move(batch, device)
            output = model(b)
            arrays = {key: value.cpu().numpy() for key, value in output.items() if isinstance(value, torch.Tensor) and value.ndim == 3 and (value.shape[1] == b['target'].shape[1])}
            arrays.update({key: batch[key].numpy() for key in ['target', 'mask', 'origin']})
            for key, value in arrays.items():
                saved.setdefault(key, []).append(value)
    np.savez_compressed(path, **{key: np.concatenate(value) for key, value in saved.items()}, mean=mean, std=std)

def score(path):
    with np.load(path) as d:
        mask = d['mask'].astype(bool)
        scale = d['std']
        prediction, target = (d['prediction'].astype('float64'), d['target'].astype('float64'))
        error = (prediction - target) * scale
        counts = mask.sum((0, 1))
        sums = np.where(mask, error ** 2, 0).sum((0, 1))
        peak, trough = ([], [])
        for c in range(len(scale)):
            pp, tt = ([], [])
            for i in range(len(mask)):
                ix = np.flatnonzero(mask[i, :, c])
                if len(ix) == 0:
                    continue
                p, y = (prediction[i, ix, c], target[i, ix, c])
                pp.append(float(abs(p.max() - y.max()) * scale[c]))
                tt.append(float(abs(p.min() - y.min()) * scale[c]))
            peak.append(float(np.mean(pp)))
            trough.append(float(np.mean(tt)))
        physical_rmse = np.sqrt(sums / counts)
        physical_mae = np.where(mask, np.abs(error), 0).sum((0, 1)) / counts
        normalized_sums = np.where(mask, (prediction - target) ** 2, 0).sum((0, 1))
        return dict(joint_standardized_rmse=float(np.sqrt(normalized_sums.sum() / counts.sum())), temperature_rmse_c=float(physical_rmse[0]), temperature_mae_c=float(physical_mae[0]), temperature_peak_mae_c=peak[0], temperature_trough_mae_c=trough[0], pooled_rmse_mm=float(np.sqrt(sums[1:].sum() / counts[1:].sum())), pooled_mse_mm2=float(sums[1:].sum() / counts[1:].sum()), pooled_mae_mm=float(np.where(mask[:, :, 1:], np.abs(error[:, :, 1:]), 0).sum() / counts[1:].sum()), temperature_mse_c2=float(sums[0] / counts[0]), rmse_mm=physical_rmse[1:].tolist(), mae_mm=physical_mae[1:].tolist(), windows=len(mask), measured_counts=counts.tolist(), peak_mae_mm=peak[1:], trough_mae_mm=trough[1:], tail_2h_pooled_rmse_mm=float(np.sqrt(np.where(mask[:, -120:, 1:], error[:, -120:, 1:] ** 2, 0).sum() / mask[:, -120:, 1:].sum())), temperature_tail_2h_rmse_c=float(np.sqrt(np.where(mask[:, -120:, 0], error[:, -120:, 0] ** 2, 0).sum() / mask[:, -120:, 0].sum())))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg = _release_json_loads(Path(args.config).read_text(encoding='utf-8'))
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
    values, observed, parts, mean, std, report, ho, fo = prepare(cfg)
    dump(out / 'data_report.json', report)
    dump(out / 'window_origins.json', parts)
    print('DATA ' + json.dumps(report), flush=True)
    model = Baseline(cfg).to(device)
    parameters = sum((p.numel() for p in model.parameters()))
    print('MODEL ' + json.dumps(dict(model=cfg['model'], scientific_label=model.scientific_label, parameters=parameters, sample_minutes=cfg['sample_minutes'], input_channels=cfg['input_channels'], output_channels=cfg['model_args']['c_out'], micro_batch_size=cfg['micro_batch_size'], effective_batch_size=cfg['batch_size'])), flush=True)
    loaders = {name: DataLoader(Windows(values, observed, origins, ho, fo), batch_size=cfg['batch_size'] if name == 'train' else cfg['micro_batch_size'], shuffle=name == 'train', num_workers=0) for name, origins in parts.items()}
    scale_source = read_protocol()['training_difference_scales']
    scales = [torch.tensor(scale_source[key], device=device, dtype=torch.float32)[None, None] for key in ['first_difference_rms_normalized', 'second_difference_rms_normalized']]
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    best, best_epoch, bad = (float('inf'), 0, 0)
    rows = []
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        totals = {}
        batches = 0
        for batch in loaders['train']:
            b = move(batch, device)
            dm1, dm2 = difference_masks(b['mask'])
            denominators = [b['mask'].sum(), dm1.sum(), dm2.sum()]
            optimizer.zero_grad()
            for first in range(0, len(b['origin']), cfg['micro_batch_size']):
                chunk = {key: value[first:first + cfg['micro_batch_size']] for key, value in b.items()}
                prediction = model(chunk)['prediction']
                loss, terms, _ = shared_objective(prediction, chunk, scales, denominators, cfg)
                loss.backward()
                for key, value in dict(loss=loss, **terms).items():
                    totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            optimizer.step()
            batches += 1
        validation = evaluate(model, loaders['val'], device, std)
        row = dict(epoch=epoch, stage='prediction', seconds=time.time() - started, validation_pooled_rmse_mm=validation['pooled_rmse_mm'], validation_temperature_rmse_c=validation['temperature_rmse_c'], validation_joint_standardized_rmse=validation['joint_standardized_rmse'], **{key: value / batches for key, value in totals.items()})
        rows.append(row)
        pd.DataFrame(rows).to_csv(out / 'history.csv', index=False)
        print(json.dumps(row), flush=True)
        if validation['joint_standardized_rmse'] < best:
            best = validation['joint_standardized_rmse']
            best_epoch = epoch
            bad = 0
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
        save_predictions(model, loaders[split], device, mean, std, path)
        results[label] = score(path)
    dump(out / 'suite_results.json', dict(version=cfg['run_name'], seed=cfg['seed'], results=dict(accuracy=results), test_used_for_selection=False, task=cfg['task'], sampling_minutes=cfg['sample_minutes'], checkpoint_selection='joint_standardized_validation_rmse', input_policy=cfg['input_policy'], mf_policy='No physical BW state in these baselines.'))
    dump(out / 'summary.json', dict(status='completed', model=cfg['model'], scientific_label=model.scientific_label, epochs_ran=len(rows), best_accuracy_epoch=best_epoch, parameters=parameters, elapsed_seconds=time.time() - started, training_seconds=training_seconds, peak_cuda_allocated_mib=torch.cuda.max_memory_allocated() / 1048576, test_evaluated=True, same_card_concurrent=True))
    print('COMPLETED ' + json.dumps(results), flush=True)
if __name__ == '__main__':
    main()
