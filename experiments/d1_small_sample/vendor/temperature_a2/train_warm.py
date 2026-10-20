"""Train causal 24-hour d1 temperature forecasts; keep the test split sealed."""
from if_release.runtime import loads as _release_json_loads
import argparse
import json
import random
import shutil
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from data_contract_v8 import ROOT, Windows, dump, load, on_device
from models import build_warm_model as build_model
from train_contract_v9 import ForecastWindows, objective, evaluate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    cfg = _release_json_loads(Path(args.config).read_text(encoding='utf-8'))
    output = ROOT / 'temperature_prediction' / 'outputs' / Path(__file__).resolve().parent.name / cfg['run_name']
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(Path(__file__).resolve().parent, output / 'frozen_source', ignore=shutil.ignore_patterns('__pycache__'))
    dump(output / 'config.json', cfg)
    values, mask, sums, counts, parts, mean, std, report = load(cfg)
    dump(output / 'data_report.json', {'counts': report, 'measured_query_points': {name: int(sum((mask[o:o + 1440:cfg['sample_minutes'], 0].sum() for o in origins))) for name, origins in parts.items()}, 'mean_celsius': float(mean[0]), 'std_celsius': float(std[0]), 'data': cfg['data'], 'channel': 'Temperature', 'sample_minutes': cfg['sample_minutes'], 'sensor_type': 'unconfirmed', 'history_policy': 'seven prior days aggregated at config.sample_minutes with measured coverage; 30 prior measured daily means and coverage; known calendar; same inputs in both models', 'query_offsets_minutes': list(range(0, 1440, cfg['sample_minutes'])), 'anchor': 'last causal historical value at origin-1 minute', 'normalization': 'training-period measured values only', 'availability_filter': 'all 10081 historical input_available_at values strictly less than origin; inherited v8 contract', 'checkpoint_selection': 'validation observed-node temperature MSE', 'test_evaluated': False})
    dump(output / 'window_origins.json', parts)
    print('DATA ' + json.dumps(report), flush=True)
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.set_num_threads(4)
    device = torch.device(args.device)
    model = build_model(cfg).to(device)
    print('MODEL ' + json.dumps({'model': cfg['model'], 'parameters': sum((p.numel() for p in model.parameters())), 'device': str(device)}), flush=True)
    loaders = {name: DataLoader(ForecastWindows(values, mask, sums, counts, origins, cfg['sample_minutes']), batch_size=cfg['batch_size'], shuffle=name == 'train', num_workers=0) for name, origins in parts.items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    history, best, bad = ([], float('inf'), 0)
    started = time.time()
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        total, batches = (0.0, 0)
        for batch in loaders['train']:
            current = on_device(batch, device)
            result = model(current)
            loss = objective(result, current)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            batches += 1
        metrics = evaluate(model, loaders['val'], device, mean, std)
        row = {'epoch': epoch, 'loss': total / batches, 'seconds': time.time() - started, **metrics}
        history.append(row)
        pd.DataFrame(history).to_csv(output / 'history.csv', index=False)
        print(json.dumps(row), flush=True)
        if metrics['temperature_mse_normalized'] < best:
            best, bad = (metrics['temperature_mse_normalized'], 0)
            torch.save({'model': model.state_dict(), 'config': cfg, 'mean': mean, 'std': std, 'epoch': epoch}, output / 'best.pt')
            dump(output / 'best_validation.json', metrics)
        else:
            bad += 1
        if bad >= cfg['patience']:
            break
    state = torch.load(output / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(state['model'])
    metrics, arrays = evaluate(model, loaders['val'], device, mean, std, collect=True)
    dump(output / 'best_validation.json', metrics)
    np.savez_compressed(output / 'validation_predictions.npz', **arrays, query_offsets_minutes=np.arange(0, 1440, cfg['sample_minutes']), units='celsius', mean=mean, std=std)
    summary = {'status': 'completed', 'sample_minutes': cfg['sample_minutes'], 'best_epoch': state['epoch'], 'epochs_completed': epoch, 'parameters_trainable': sum((p.numel() for p in model.parameters())), 'seconds': time.time() - started, 'gpu_max_allocated_bytes': torch.cuda.max_memory_allocated(device), 'test_evaluated': False, 'prediction_units': 'celsius'}
    dump(output / 'summary.json', summary)
    (output / 'experiment_record.md').write_text('# va-2.2 original-v9 initialization / ' + cfg['run_name'] + '\n\n- Date: ' + datetime.now(ZoneInfo('Asia/Shanghai')).isoformat() + '.\n- Project IFpaper_v1; source/config/data report and window origins saved here.\n- Purpose: same-seed original v9 initialization for added va-2.2 temperature seed.\n- Original masked point-MSE, Adam0.001, batch32, cap80, patience12, global Torch shuffle preserved.\n- Result: completed; actual epochs ' + str(epoch) + ', selected ' + str(state['epoch']) + ', validation RMSE ' + str(metrics['rmse_celsius']) + ' degC.\n- Products: best.pt, best_validation.json, history.csv, validation_predictions.npz, frozen_source.\n- No test evaluated.\n', encoding='utf-8')
    print('COMPLETED ' + json.dumps(summary), flush=True)
if __name__ == '__main__':
    main()
