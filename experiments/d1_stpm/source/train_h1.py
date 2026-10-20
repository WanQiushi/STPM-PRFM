"""Original H1 pretraining and forecast adaptation for A5/A6; validation only."""
from if_release.runtime import loads as _release_json_loads
import argparse
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from h1_ablation import H1Ablation, H1_ORIGINAL, SOURCE, load_module
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
anchor_training = load_module('va_ablation_anchor_training', SOURCE / 'experiments/fft_joint_neuralode_d1_anchor_seasonal_bw_20260929/train.py')
training = anchor_training.direct
base = training.base

def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

class CachedWindows(Dataset):

    def __init__(self, path):
        with np.load(path) as data:
            self.arrays = {key: data[key].copy() for key in ['history', 'forecast_temperature_celsius', 'target', 'mask', 'origin']}

    def __len__(self):
        return len(self.arrays['origin'])

    def __getitem__(self, index):
        return {key: value[index] for key, value in self.arrays.items()}

def evaluate(model, loader, device, std, forecast):
    squared = np.zeros(4)
    counts = np.zeros(4)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            moved = base.on_device(batch, device)
            temperature = (moved['forecast_temperature_celsius'] - std.new_tensor(MEAN[:1])) / std[:1] if forecast else moved['future_temperature']
            result = model(moved['history'], temperature, moved['origin'])
            squared += ((result['prediction'] - moved['target']).square() * moved['mask']).sum((0, 1)).cpu().numpy()
            counts += moved['mask'].sum((0, 1)).cpu().numpy()
    s = std.detach().cpu().numpy()[1:]
    return float(np.sqrt((squared * s ** 2).sum() / counts.sum()))

def save_checkpoint(path, model, cfg, mean, std, epoch, phase, rmse):
    torch.save({'model': model.state_dict(), 'config': cfg, 'mean': mean, 'std': std, 'epoch': epoch, 'phase': phase, 'validation_rmse_mm': rmse}, path)

def main():
    global MEAN
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    cfg = _release_json_loads(Path(args.config).read_text())
    out = ROOT / cfg['output']
    out.mkdir(parents=True)
    shutil.copytree(HERE, out / 'frozen_source', ignore=shutil.ignore_patterns('__pycache__'))
    dump(out / 'config.json', cfg)
    gpu = os.environ['CUDA_VISIBLE_DEVICES']
    uuid = subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-gpu=gpu_uuid', '--format=csv,noheader'], text=True).strip()

    def occupied_processes():
        occupied = []
        rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader'], text=True).splitlines()
        for row in rows:
            fields = [part.strip() for part in row.split(',')]
            if fields[0] == uuid:
                owner = subprocess.check_output(['ps', '-p', fields[1], '-o', 'user='], text=True).strip()
                occupied.append({'pid': fields[1], 'owner': owner, 'memory': fields[2]})
        return occupied
    occupied = occupied_processes()
    while occupied:
        print('RESOURCE_WAIT ' + json.dumps({'gpu': gpu, 'processes': occupied}), flush=True)
        time.sleep(30)
        occupied = occupied_processes()
    print('RESOURCE_READY ' + gpu, flush=True)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    normal = _release_json_loads((H1_ORIGINAL / 'normalization.json').read_text())
    MEAN = np.asarray(normal['h1_mean'])
    std_array = np.asarray(normal['h1_std'])
    std = torch.as_tensor(std_array, dtype=torch.float32, device=device)
    with np.load(ROOT / cfg['data']) as raw:
        values = ((raw['values'].astype(np.float64) - MEAN) / std_array).astype('float32')
        observed = raw['valid_observed'].astype(bool)
    protocol = _release_json_loads((HERE / 'h1_pretraining_protocol.json').read_text())
    ho = np.arange(-cfg['history_days'] * 1440, 0, cfg['sample_minutes'])
    fo = np.arange(0, 1440, cfg['sample_minutes'])
    loaders = {name: DataLoader(base.Windows(values, observed, protocol['origins'][name], ho, fo), batch_size=cfg['batch_size'], shuffle=name == 'train', num_workers=0) for name in ['train', 'val']}
    dump(out / 'data_report.json', {'pretraining_windows': {name: len(loaders[name].dataset) for name in loaders}, 'protocol': str(HERE / 'h1_pretraining_protocol.json'), 'normalization': str(H1_ORIGINAL / 'normalization.json'), 'adaptation_cache': str(H1_ORIGINAL / 'cache'), 'pretraining_temperature': 'measured future T, original response identification stage', 'adaptation_and_final_temperature': 'causal saved forecast only', 'test_evaluated': False})
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    model = H1Ablation(cfg).to(device)
    cutoff_ids = {id(p) for p in model.decomposer.parameters()}
    optimizer = torch.optim.Adam([p for p in model.parameters() if id(p) not in cutoff_ids], lr=cfg['lr'])
    best_accuracy = best_joint = float('inf')
    best_epoch = 0
    bad = 0
    rows = []
    started = time.time()
    print('DATA ' + json.dumps({'group': cfg['ablation'], 'parameters': sum((p.numel() for p in model.parameters())), 'windows': {k: len(v.dataset) for k, v in loaders.items()}}), flush=True)
    for epoch in range(1, cfg['epochs'] + 1):
        stage = base.stage_name(epoch, cfg)
        model.configure_stage(stage)
        model.train()
        losses = []
        for batch in loaders['train']:
            moved = base.on_device(batch, device)
            result = model(moved['history'], moved['future_temperature'], moved['origin'])
            loss, _, _, _ = training.objective(model, result, moved, cfg, stage)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg['grad_clip'])
            optimizer.step()
            losses.append(loss.item())
        rmse = evaluate(model, loaders['val'], device, std, False)
        if rmse < best_accuracy:
            best_accuracy, best_epoch = (rmse, epoch)
            save_checkpoint(out / 'pretrained_best.pt', model, cfg, MEAN, std_array, epoch, 'pretraining', rmse)
        if stage == 'joint_cutoff':
            if rmse < best_joint:
                best_joint, bad = (rmse, 0)
            else:
                bad += 1
        row = {'epoch': epoch, 'stage': stage, 'loss': float(np.mean(losses)), 'validation_rmse_mm': rmse, 'seconds': time.time() - started}
        rows.append(row)
        pd.DataFrame(rows).to_csv(out / 'pretraining_history.csv', index=False)
        dump(out / 'progress.json', {'status': 'running', 'phase': 'pretraining', **row})
        print(json.dumps(row), flush=True)
        if stage == 'joint_cutoff' and bad >= cfg['patience_after_cutoff']:
            break
    pretraining_seconds = time.time() - started
    (out / 'pretraining_record.md').write_text('# va ' + cfg['ablation'] + ' H1 pretraining\n\n- Date: ' + datetime.now(ZoneInfo('Asia/Shanghai')).isoformat() + '.\n- Purpose: re-fit only the structurally ablated H1 from seed2024 under the original stages/loss/config and68/26 windows.\n- Result: selected epoch ' + str(best_epoch) + ', validation ' + str(best_accuracy) + 'mm; duration ' + str(pretraining_seconds) + 's.\n- Products: pretrained_best.pt, pretraining_history.csv, frozen_source/config; no test.\n', encoding='utf-8')
    checkpoint = torch.load(out / 'pretrained_best.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    adaptation = cfg['adaptation']
    model.configure_stage(adaptation['h1_stage'])
    adapt_loaders = {name: DataLoader(CachedWindows(H1_ORIGINAL / 'cache' / (name + '.npz')), batch_size=adaptation['batch_size'], shuffle=name == 'train', num_workers=0) for name in ['train', 'validation']}
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=adaptation['lr'])
    best = evaluate(model, adapt_loaders['validation'], device, std, True)
    best_adapt_epoch = bad = 0
    save_checkpoint(out / 'adapted_best.pt', model, cfg, MEAN, std_array, 0, 'forecast_adaptation', best)
    adapt_rows = []
    started = time.time()
    for epoch in range(1, adaptation['epochs'] + 1):
        model.train()
        sse_sum = count_sum = 0.0
        for batch in adapt_loaders['train']:
            moved = base.on_device(batch, device)
            forecast = (moved['forecast_temperature_celsius'] - std.new_tensor(MEAN[:1])) / std[:1]
            result = model(moved['history'], forecast, moved['origin'])
            sse = ((result['prediction'] - moved['target']).square() * moved['mask']).sum()
            count = moved['mask'].sum()
            loss = sse / count + adaptation['ode_weight'] * result['ode_residual_penalty']
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), adaptation['grad_clip'])
            optimizer.step()
            sse_sum += sse.item()
            count_sum += count.item()
        rmse = evaluate(model, adapt_loaders['validation'], device, std, True)
        if rmse < best:
            best, best_adapt_epoch, bad = (rmse, epoch, 0)
            save_checkpoint(out / 'adapted_best.pt', model, cfg, MEAN, std_array, epoch, 'forecast_adaptation', rmse)
        else:
            bad += 1
        row = {'epoch': epoch, 'train_mse': sse_sum / count_sum, 'validation_rmse_mm': rmse, 'best_epoch': best_adapt_epoch, 'seconds': time.time() - started}
        adapt_rows.append(row)
        pd.DataFrame(adapt_rows).to_csv(out / 'adaptation_history.csv', index=False)
        dump(out / 'progress.json', {'status': 'running', 'phase': 'adaptation', **row})
        print(json.dumps(row), flush=True)
        if bad >= adaptation['patience']:
            break
    report = {'status': 'completed', 'group': cfg['ablation'], 'pretraining_best_epoch': best_epoch, 'pretraining_epochs': len(rows), 'pretraining_seconds': pretraining_seconds, 'adaptation_best_epoch': best_adapt_epoch, 'adaptation_epochs': len(adapt_rows), 'adaptation_seconds': time.time() - started, 'adaptation_validation_rmse_mm': best, 'parameters': sum((p.numel() for p in model.parameters())), 'seed': cfg['seed'], 'training_sample_minutes': 15, 'final_evaluation_sample_minutes': 1, 'test_evaluated': False, 'same_original_adaptation_forecasts': True}
    dump(out / 'summary.json', report)
    dump(out / 'progress.json', report)
    (out / 'adaptation_record.md').write_text('# va ' + cfg['ablation'] + ' forecast adaptation\n\n- Date: ' + datetime.now(ZoneInfo('Asia/Shanghai')).isoformat() + '.\n- Purpose: adapt ablated H1 to the same causal OOF forecast cache as full va.\n- Result: ' + json.dumps(report) + '.\n- Products: adapted_best.pt, adaptation_history.csv, summary; no test or change to full va.\n', encoding='utf-8')
    print('COMPLETED ' + json.dumps(report), flush=True)
if __name__ == '__main__':
    main()
