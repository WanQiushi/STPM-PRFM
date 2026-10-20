from if_release.runtime import expand as _release_expand, loads as _release_json_loads
import argparse
import copy
import importlib.util
import json
import logging
from pathlib import Path
import random
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from common import ROOT, HERE, PROTOCOL, dump, prepare, load_data, metrics
from ablations import v2_class, vsharp_class, ZeroOutput, MeanSplit
SOURCE = HERE / 'frozen_source'

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def fft_run(cfg, version, component, device):
    if version == 'v2':
        parent = load_module('suite_v2', SOURCE / 'experiments/fft_joint_neuralode_d1_anchor_seasonal_bw_20260929/train.py')
        training = parent.direct
        training.DirectBWJointFFTNeuralODE = v2_class(parent.AnchoredSeasonalBWModel, component)
        training.base.prepare = prepare
        model_class = training.DirectBWJointFFTNeuralODE
    else:
        parent = load_module('suite_vsharp', SOURCE / 'experiments/fft_joint_neuralode_d1_corrected_20260929/train.py')
        training = parent.base
        training.JointFFTNeuralODE = vsharp_class(parent.CorrectedJointFFTNeuralODE, component)
        training.prepare = prepare
        model_class = training.JointFFTNeuralODE
    training.ROOT = ROOT
    if version == 'v2':
        training.base.ROOT = ROOT
    if component == 'endpoint_safe':
        edge = load_module('suite_edge', SOURCE / 'experiments/fft_joint_neuralode_d1_edge_safe_hf_20260929/model.py')
        original = model_class

        class EdgeModel(original):

            def __init__(self, c):
                super().__init__(c)
                self.decomposer = edge.EdgeSafeFFT(self.history_length, self.step_hours, c)
        model_class = EdgeModel
        training.DirectBWJointFFTNeuralODE = model_class
    if component == 'no_lf_dynamics':
        original = model_class

        class NoSlow(original):

            def __init__(self, c):
                super().__init__(c)
                self.slow_field = ZeroOutput(self.slow_field)
        model_class = NoSlow
        training.JointFFTNeuralODE = model_class
    if component == 'mean_split':
        original = model_class

        class MeanModel(original):

            def __init__(self, c):
                super().__init__(c)
                self.decomposer = MeanSplit(self.decomposer)
        model_class = MeanModel
    out = ROOT / cfg['output']
    out.mkdir(parents=True, exist_ok=True)
    dump(out / 'config.json', cfg)
    normalized, obs, parts, mean, std, report, ho, fo = prepare(cfg)
    base = training.base if version == 'v2' else training
    dump(out / 'data_report.json', report)
    dump(out / 'window_origins.json', parts)
    print('DATA ' + json.dumps(report), flush=True)
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    model = model_class(cfg).to(device)
    loaders = {name: DataLoader(base.Windows(normalized, obs, origins, ho, fo), batch_size=cfg['batch_size'], shuffle=name == 'train', num_workers=0) for name, origins in parts.items()}
    if version == 'v2':
        initial = training.initialize_bw_from_training(model, loaders['train'], device)
        dump(out / 'bw_ols_initialization.json', initial)
        print('BW_INIT ' + json.dumps(initial), flush=True)
    cutoffs = list(model.decomposer.parameters())
    cutoff_ids = {id(x) for x in cutoffs}
    optimizer = torch.optim.Adam([{'params': [x for x in model.parameters() if id(x) not in cutoff_ids], 'lr': cfg['lr']}, {'params': cutoffs, 'lr': cfg['cutoff_lr']}])
    best_accuracy = best_joint = float('inf')
    accuracy_epoch = joint_epoch = 0
    bad = 0
    history = []
    started = time.time()
    for epoch in range(1, cfg['epochs'] + 1):
        stage = base.stage_name(epoch, cfg)
        training.configure_stage(model, stage)
        model.train()
        totals = {}
        batches = 0
        for batch in loaders['train']:
            moved = base.on_device(batch, device)
            output = training.forward_batch(model, moved) if version == 'v2' else model(moved['history'], moved['future_temperature'])
            loss, terms, _, _ = training.objective(model, output, moved, cfg, stage)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([x for x in model.parameters() if x.requires_grad], cfg['grad_clip'])
            optimizer.step()
            for key, value in dict(loss=loss, **terms).items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
            batches += 1
        validation = training.evaluate(model, loaders['val'], device, std, cfg)
        selection = float(np.average(np.asarray(validation['rmse_mm']) ** 2, weights=validation['measured_counts']))
        row = dict(epoch=epoch, stage=stage, seconds=time.time() - started, validation_pooled_rmse_mm=float(np.sqrt(selection)), **{k: v / batches for k, v in totals.items()}, **{f'val_{k}': v for k, v in validation.items() if not isinstance(v, list)})
        history.append(row)
        pd.DataFrame(history).to_csv(out / 'history.csv', index=False)
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps(row), flush=True)
        if selection < best_accuracy:
            best_accuracy = selection
            accuracy_epoch = epoch
            training.save_checkpoint(out / 'best_accuracy.pt', model, cfg, mean, std, epoch, stage, validation)
            dump(out / 'best_accuracy_validation.json', validation)
        if stage == 'joint_cutoff':
            if selection < best_joint:
                best_joint = selection
                joint_epoch = epoch
                bad = 0
                training.save_checkpoint(out / 'best_joint.pt', model, cfg, mean, std, epoch, stage, validation)
                dump(out / 'best_joint_validation.json', validation)
            else:
                bad += 1
            if bad >= cfg['patience_after_cutoff']:
                break
    values, obs, _, mean, std, ho, fo = load_data(cfg)
    values = ((values - mean) / std).astype('float32')
    p = _release_json_loads(PROTOCOL.read_text(encoding='utf-8'))
    loader = DataLoader(base.Windows(values, obs, p['origins']['test'], ho, fo), batch_size=cfg['batch_size'], shuffle=False)
    model = model_class(cfg).to(device)
    choices = [('accuracy', 'best_accuracy.pt', 'validation_predictions_accuracy.npz'), ('joint', 'best_joint.pt', 'validation_predictions_joint.npz')]
    results = {}
    for selection, checkpoint, validation in choices:
        test = out / f'test_predictions_{selection}.npz'
        if version == 'v2':
            training.save_predictions(out / validation, out / checkpoint, model, loaders['val'], device, mean, std)
            training.save_predictions(test, out / checkpoint, model, loader, device, mean, std)
        else:
            training.save_predictions(out / validation, out / checkpoint, model, loaders['val'], device, cfg, mean, std)
            training.save_predictions(test, out / checkpoint, model, loader, device, cfg, mean, std)
        results[selection] = dict(validation=metrics(out / validation), test=metrics(test))
    dump(out / 'summary.json', dict(status='completed', epochs_ran=len(history), best_accuracy_epoch=accuracy_epoch, best_joint_epoch=joint_epoch, selection='pooled validation MSE in mm squared', test_evaluated=True, parameters=sum((x.numel() for x in model.parameters()))))
    dump(out / 'suite_results.json', dict(version=version, component=component, seed=cfg['seed'], results=results, test_used_for_selection=False))

def v1_run(cfg, component, device):
    sys.path.insert(0, str(HERE / 'v1_source'))
    from hydpo.decomposition import ThreeBandDecomposer, fit_stage1
    from hydpo.model import FSEModel
    from hydpo.training import train_stage2
    out = ROOT / cfg['output']
    out.mkdir(parents=True, exist_ok=True)
    dump(out / 'config.json', cfg)
    protocol = _release_json_loads(PROTOCOL.read_text(encoding='utf-8'))
    values, observed, _, mean, std, _, fo = load_data(cfg)
    origins = protocol['origins']
    logger = logging.getLogger('v1')
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    saved = {name: {} for name in ['val', 'test']}
    channel_records = []
    for c in range(4):
        channel = out / f'channel_{c}'
        channel.mkdir(exist_ok=True)
        seed = cfg['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        decomposition = ThreeBandDecomposer(window_length=len(fo), **cfg['v1_decomposer']).to(device)
        train_idx = np.asarray(origins['train'])[:, None] + fo[None]
        pair = np.stack((values[train_idx, 0], values[train_idx, c + 1]), axis=1).astype('float32')
        s_cfg = copy.deepcopy(cfg['v1_stage1'])
        s_cfg['seed'] = seed
        if component == 'no_stage1_bw':
            s_cfg['loss_weights']['bw'] = 0.0
        if component not in ['fixed_bands', 'mean_split']:
            result = fit_stage1(decomposition, pair, s_cfg, device, logger)
            pd.DataFrame(result.losses).to_csv(channel / 'stage1_history.csv', index=False)
        decomposition.eval()
        torch.save(decomposition.state_dict(), channel / 'stage1.pt')
        parts = {}
        with torch.no_grad():
            for name in ['train', 'val', 'test']:
                ii = np.asarray(origins[name])[:, None] + fo[None]
                raw = np.stack((values[ii, 0], values[ii, c + 1]), axis=1).astype('float32')
                z = decomposition(torch.as_tensor(raw, device=device))
                u = decomposition.transform_input(torch.as_tensor(raw[:, 0], device=device))
                if component == 'mean_split':
                    sequence = torch.as_tensor(raw, device=device)
                    static = sequence.mean(dim=-1, keepdim=True).expand_as(sequence)
                    z = dict(static=static, dynamic=sequence - static)
                    u = dict(static=static[:, 0], dynamic=sequence[:, 0] - static[:, 0])
                dates = np.datetime64('2022-01-01') + np.asarray(origins[name]).astype('timedelta64[m]')
                doy = (dates.astype('datetime64[D]') - dates.astype('datetime64[Y]')).astype(int) + 1
                parts[name] = dict(T_LF=u['static'].cpu().numpy(), T_MF=u['dynamic'].cpu().numpy(), D_LF=z['static'][:, 1].cpu().numpy(), D_MF=z['dynamic'][:, 1].cpu().numpy(), D_full_filled=raw[:, 1], D_mask=observed[ii, c + 1].astype('float32'), doy=doy.astype('int64'))
        model_cfg = copy.deepcopy(cfg['v1_model'])
        model_cfg['ablation'] = {'disable_bw_backbone': component == 'no_bw', 'disable_lf_residual': component == 'no_neural_residual', 'disable_mf_residual': component == 'no_neural_residual', 'disable_mf0': component == 'no_mf0'}
        model = FSEModel(model_cfg).to(device)
        if component == 'no_lf_seasonality':
            model.lf.alpha_net = ZeroOutput(model.lf.alpha_net)
            model.lf.beta_net = ZeroOutput(model.lf.beta_net)
        stage_cfg = copy.deepcopy(cfg['v1_stage2'])
        stage_cfg['seed'] = seed
        logger.info('CHANNEL %s | start stage2', c)
        trained = train_stage2(model, parts['train'], parts['val'], stage_cfg, device, logger)
        pd.DataFrame(trained.history).to_csv(channel / 'stage2_history.csv', index=False)
        torch.save(dict(model=model.state_dict(), config=model_cfg, epoch=trained.best_epoch), channel / 'best.pt')
        channel_records.append(dict(channel=c, stage2_best_epoch=trained.best_epoch, stage2_epochs=len(trained.history), physics=model.physics_parameters(), cutoffs=[float(x.detach().cpu()) for x in decomposition.cutoff_hours()]))
        model.eval()
        with torch.no_grad():
            for name in ['val', 'test']:
                d = parts[name]
                time_grid = torch.arange(len(fo), device=device, dtype=torch.float32) * cfg['sample_minutes'] / 60
                chunks = []
                for first in range(0, len(d['doy']), cfg['batch_size']):
                    sl = slice(first, first + cfg['batch_size'])
                    pred = model(time_grid, torch.as_tensor(d['T_LF'][sl], device=device), torch.as_tensor(d['T_MF'][sl], device=device), torch.as_tensor(d['doy'][sl], device=device), use_estimated_mf0=True)
                    chunks.append({k: v.cpu().numpy() for k, v in pred.items()})
                pred = {k: np.concatenate([ch[k] for ch in chunks]) for k in chunks[0]}
                arrays = dict(prediction=(pred['full'] - mean[c + 1]) / std[c + 1], lf_prediction=(pred['lf'] - mean[c + 1]) / std[c + 1], mf_prediction=pred['mf'] / std[c + 1], lf_target=(d['D_LF'] - mean[c + 1]) / std[c + 1], mf_target=d['D_MF'] / std[c + 1], target=(d['D_full_filled'] - mean[c + 1]) / std[c + 1], mask=d['D_mask'], temperature_mf=d['T_MF'] / std[0], effective_temperature_mf=d['T_MF'] / std[0])
                for key, val in arrays.items():
                    saved[name].setdefault(key, []).append(val[..., None])
    results = {}
    for name in ['val', 'test']:
        arrays = {k: np.concatenate(v, axis=2) for k, v in saved[name].items()}
        arrays['origin'] = np.asarray(origins[name])
        arrays['mean'] = mean
        arrays['std'] = std
        path = out / f'{name}_predictions_accuracy.npz'
        np.savez_compressed(path, **arrays)
        results['validation' if name == 'val' else 'test'] = metrics(path)
    dump(out / 'suite_results.json', dict(version='v1', component=component, seed=cfg['seed'], results={'accuracy': results}, channels=channel_records, test_used_for_selection=False))
    print('COMPLETED v1 ' + json.dumps(results), flush=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    cfg = _release_json_loads(Path(args.config).read_text(encoding='utf-8'))
    import os
    import subprocess
    gpu = os.environ['CUDA_VISIBLE_DEVICES']
    uuid = subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-gpu=gpu_uuid', '--format=csv,noheader'], text=True).strip()

    def occupied_processes():
        rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader'], text=True).splitlines()
        occupied = []
        for row in rows:
            fields = [x.strip() for x in row.split(',')]
            if fields[0] == uuid:
                owner = subprocess.check_output(['ps', '-p', fields[1], '-o', 'user='], text=True).strip()
                if gpu == '0' and os.environ.get('IFPAPER_GPU0_SHARED_SUITE') == '1' and (owner == _release_expand('${USER}')):
                    command = subprocess.check_output(['ps', '-p', fields[1], '-o', 'args='], text=True).strip()
                    if _release_expand('${WORKSPACE}/experiments/version_long_suite_20260930/train_one.py') in command:
                        continue
                occupied.append(dict(pid=fields[1], owner=owner, memory=fields[2]))
        return occupied
    occupied = occupied_processes()
    if occupied:
        print('RESOURCE_WAIT ' + json.dumps(dict(gpu=gpu, processes=occupied)), flush=True)
    while occupied:
        time.sleep(30)
        occupied = occupied_processes()
    print('RESOURCE_READY ' + json.dumps(dict(gpu=gpu)), flush=True)
    print('RUN_START ' + json.dumps(dict(pid=os.getpid(), gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), version=cfg['version'], component=cfg['component'], seed=cfg['seed'])), flush=True)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    started = time.time()
    if cfg['version'] == 'v1':
        v1_run(cfg, cfg['component'], device)
    else:
        fft_run(cfg, cfg['version'], cfg['component'], device)
    result_path = ROOT / cfg['output'] / 'suite_results.json'
    result = _release_json_loads(result_path.read_text(encoding='utf-8'))
    result['elapsed_seconds'] = time.time() - started
    dump(result_path, result)
