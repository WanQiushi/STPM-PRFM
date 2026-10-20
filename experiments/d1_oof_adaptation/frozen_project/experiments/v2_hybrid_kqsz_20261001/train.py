"""One hybrid candidate; frozen suite data, validation selection, original v2 loss terms."""
from if_release.runtime import loads as _release_json_loads
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from hybrid_model import HybridKqsz, PROJECT, SOURCE, load_module
from common import prepare, load_data, metrics, dump, PROTOCOL
anchor_training = load_module('hybrid_anchor_training', SOURCE / 'experiments/fft_joint_neuralode_d1_anchor_seasonal_bw_20260929/train.py')
training = anchor_training.direct
base = training.base

def save_predictions(model, loader, device, mean, std, path):
    saved = {}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            moved = base.on_device(batch, device)
            out = model(moved['history'], moved['future_temperature'], moved['origin'])
            lf, mf = base.target_components(model, moved)
            arrays = {key: out[key].cpu().numpy() for key in ['prediction', 'lf_prediction', 'mf_prediction', 'temperature_lf', 'temperature_mf', 'effective_temperature_mf', 'reversible_prediction', 'memory_prediction', 'bw_latent', 'bw_physics_rate', 'bw_residual_rate', 'lf_anchor_offset']}
            arrays.update(lf_target=lf.cpu().numpy(), mf_target=mf.cpu().numpy())
            arrays.update({key: batch[key].numpy() for key in ['target', 'mask', 'future_temperature', 'origin']})
            for key, value in arrays.items():
                saved.setdefault(key, []).append(value)
    np.savez_compressed(path, **{key: np.concatenate(value) for key, value in saved.items()}, mean=mean, std=std)

def evaluate(model, loader, device, std):
    sums = {key: np.zeros(4) for key in ['total', 'lf', 'mf', 'memory', 'reversible']}
    counts = np.zeros(4)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            moved = base.on_device(batch, device)
            out = model(moved['history'], moved['future_temperature'], moved['origin'])
            lf, mf = base.target_components(model, moved)
            mask = moved['mask']
            for key, target in [('total', moved['target']), ('lf', lf), ('mf', mf)]:
                pred = out['prediction' if key == 'total' else key + '_prediction']
                sums[key] += ((pred - target).square() * mask).sum((0, 1)).cpu().numpy()
            for key in ['memory', 'reversible']:
                sums[key] += (out[key + '_prediction'].square() * mask).sum((0, 1)).cpu().numpy()
            counts += mask.sum((0, 1)).cpu().numpy()
    return dict(rmse_mm=(np.sqrt(sums['total'] / counts) * std[1:]).tolist(), measured_counts=counts.tolist(), **{key + '_pooled_rms_mm': float(np.sqrt(np.sum(sums[key] * std[1:] ** 2) / counts.sum())) for key in sums})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    cfg = _release_json_loads(Path(args.config).read_text(encoding='utf-8'))
    gpu = os.environ['CUDA_VISIBLE_DEVICES']
    uuid = subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-gpu=gpu_uuid', '--format=csv,noheader'], text=True).strip()

    def occupied_processes():
        rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader'], text=True).splitlines()
        occupied = []
        for row in rows:
            fields = [x.strip() for x in row.split(',')]
            if fields[0] == uuid:
                owner = subprocess.check_output(['ps', '-p', fields[1], '-o', 'user='], text=True).strip()
                occupied.append(dict(pid=fields[1], owner=owner, memory=fields[2]))
        return occupied
    occupied = occupied_processes()
    while occupied:
        print('RESOURCE_WAIT ' + json.dumps(dict(gpu=gpu, processes=occupied)), flush=True)
        time.sleep(30)
        occupied = occupied_processes()
    print('RESOURCE_READY ' + gpu, flush=True)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    out = PROJECT / cfg['output']
    out.mkdir(parents=True, exist_ok=True)
    dump(out / 'config.json', cfg)
    values, observed, parts, mean, std, report, ho, fo = prepare(cfg)
    dump(out / 'data_report.json', report)
    dump(out / 'window_origins.json', parts)
    print('DATA ' + json.dumps(report), flush=True)
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    model = HybridKqsz(cfg).to(device)
    loaders = {name: DataLoader(base.Windows(values, observed, origins, ho, fo), batch_size=cfg['batch_size'], shuffle=name == 'train', num_workers=0) for name, origins in parts.items()}
    cutoff_ids = {id(p) for p in model.decomposer.parameters()}
    optimizer = torch.optim.Adam([p for p in model.parameters() if id(p) not in cutoff_ids], lr=cfg['lr'])
    best = dict(accuracy=float('inf'), joint=float('inf'))
    epochs = dict(accuracy=0, joint=0)
    bad = 0
    rows = []
    started = time.time()
    for epoch in range(1, cfg['epochs'] + 1):
        stage = base.stage_name(epoch, cfg)
        model.configure_stage(stage)
        model.train()
        totals = {}
        batches = 0
        for batch in loaders['train']:
            moved = base.on_device(batch, device)
            output = model(moved['history'], moved['future_temperature'], moved['origin'])
            loss, terms, _, _ = training.objective(model, output, moved, cfg, stage)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg['grad_clip'])
            optimizer.step()
            for key, value in dict(loss=loss, **terms).items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
            batches += 1
        validation = evaluate(model, loaders['val'], device, std)
        selection = validation['total_pooled_rms_mm'] ** 2
        row = dict(epoch=epoch, stage=stage, seconds=time.time() - started, validation_pooled_rmse_mm=float(np.sqrt(selection)), **{key: value / batches for key, value in totals.items()}, **{'val_' + key: value for key, value in validation.items() if not isinstance(value, list)})
        rows.append(row)
        pd.DataFrame(rows).to_csv(out / 'history.csv', index=False)
        print(json.dumps(row), flush=True)
        for choice in ['accuracy', 'joint']:
            if choice == 'joint' and stage != 'joint_cutoff':
                continue
            if selection < best[choice]:
                best[choice] = selection
                epochs[choice] = epoch
                training.save_checkpoint(out / ('best_' + choice + '.pt'), model, cfg, mean, std, epoch, stage, validation)
                dump(out / ('best_' + choice + '_validation.json'), validation)
                if choice == 'joint':
                    bad = 0
            elif choice == 'joint':
                bad += 1
        if stage == 'joint_cutoff' and bad >= cfg['patience_after_cutoff']:
            break
    raw, observed, _, mean, std, ho, fo = load_data(cfg)
    raw = ((raw - mean) / std).astype('float32')
    protocol = _release_json_loads(PROTOCOL.read_text(encoding='utf-8'))
    test_loader = DataLoader(base.Windows(raw, observed, protocol['origins']['test'], ho, fo), batch_size=cfg['batch_size'], shuffle=False)
    results = {}
    for choice in ['accuracy', 'joint']:
        checkpoint = torch.load(out / ('best_' + choice + '.pt'), map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        paths = {split: out / (split + '_predictions_' + choice + '.npz') for split in ['validation', 'test']}
        save_predictions(model, loaders['val'], device, mean, std, paths['validation'])
        save_predictions(model, test_loader, device, mean, std, paths['test'])
        results[choice] = {split: metrics(path) for split, path in paths.items()}
    dump(out / 'summary.json', dict(status='completed', epochs_ran=len(rows), best_accuracy_epoch=epochs['accuracy'], best_joint_epoch=epochs['joint'], parameters=sum((p.numel() for p in model.parameters())), elapsed_seconds=time.time() - started, test_evaluated=True))
    dump(out / 'suite_results.json', dict(version=cfg['version_label'], component=cfg['component'], seed=cfg['seed'], results=results, test_used_for_selection=False, task=cfg['task']))
    print('COMPLETED ' + json.dumps(results), flush=True)
if __name__ == '__main__':
    main()
