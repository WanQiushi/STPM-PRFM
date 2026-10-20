"""Run one budget/seed of the complete fixed A2 architecture from fresh weights.

Execution is explicit: python a2/train.py --config configs/9m/A2_seed2024.json
Writing/importing this module does not schedule or launch training.
"""
from if_release.runtime import loads as _release_json_loads
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import time
import traceback
from zoneinfo import ZoneInfo
import numpy as np
import torch
from torch.utils.data import DataLoader
from data import Data, ResponseWindows
from temperature import dump, train_temperature, predict
from response import train_response, native_model, evaluate, score
PACKAGE = Path(__file__).resolve().parents[1]
TEMPERATURE_RECIPE = {'width': 32, 'latent_dim': 8, 'residual_width': 16, 'batch_size': 32, 'epochs': 80, 'epochs_base': 80, 'epochs_residual': 60, 'patience': 12, 'lr': 0.001, 'lr_base': 0.001, 'lr_residual': 0.001, 'level_weight': 0.0, 'amplitude_weight': 0.0, 'shape_weight': 0.25, 'trend_weight': 3.0}
ADAPTATION_RECIPE = {'batch_size': 16, 'epochs': 80, 'patience': 20, 'lr': 0.0001, 'grad_clip': 1.0, 'ode_weight': 0.001}

def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PACKAGE / path
import tarfile

def run(cfg, output):
    protocol_path = resolve(cfg['protocol_path'])
    protocol = _release_json_loads(protocol_path.read_text(encoding='utf-8'))
    dump(output / 'protocol.json', protocol)
    if protocol['a2_trainable'] is False:
        raise ValueError(protocol['a2_unavailable_reason'])
    data_path = resolve(protocol['data_path'])
    data = Data(data_path, protocol['mean'], protocol['std'])
    torch.set_num_threads(4)
    device = torch.device(cfg.get('device', 'cuda:0'))
    if device.type == 'cuda':
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
    seed = cfg['seed']
    temperature_recipe = {**TEMPERATURE_RECIPE, **cfg.get('temperature', {})}
    response_recipe = _release_json_loads((PACKAGE / 'vendor/h1_project/experiments/v2_hybrid_kqsz_20261001/config_h1_seed2024.json').read_text(encoding='utf-8'))
    response_recipe.update(cfg.get('response', {}))
    response_recipe.update(data=str(data_path), fixed_origins=str(protocol_path), output=str(output / 'response'), seed=seed, run_name=output.name + '_H1_budget_retraining')
    adaptation_recipe = {**ADAPTATION_RECIPE, **cfg.get('adaptation', {})}
    dump(output / 'actual_recipes.json', {'temperature': temperature_recipe, 'response': response_recipe, 'adaptation': adaptation_recipe})
    data_report = {'data_path': str(data_path), 'protocol_path': str(protocol_path), 'mean': data.mean.tolist(), 'std': data.std.tolist(), 'origins': {name: len(origins) for name, origins in protocol['origins'].items()}, 'h1_pretraining_origins': {name: len(origins) for name, origins in protocol['h1_pretraining_origins'].items()}, 'normalization': 'budget-specific retained measured data; fold-specific training statistics for OOF', 'full_data_weights_used': False, 'response_channels': 4, 'future_inputs_final_inference': 'predicted temperature only'}
    dump(output / 'data_report.json', data_report)
    oof_predictions, oof_origins, oof_reports = ([], [], [])
    for index, fold in enumerate(protocol['oof_folds']):
        fold_output = output / 'oof' / f'fold_{index + 1}'
        fold_data = Data(data_path, fold['normalization']['mean'], fold['normalization']['std'])
        model, report = train_temperature(fold_data, fold['train_origins'], fold['validation_origins'], seed, fold_output, device, temperature_recipe, kind='v11')
        dump(fold_output / 'provenance.json', fold)
        forecast, metrics = predict(model, fold_data, fold['prediction_origins'], 15, temperature_recipe['batch_size'], device, fold_output / 'oof_forecasts.npz')
        oof_predictions.append(forecast)
        oof_origins.extend(fold['prediction_origins'])
        oof_reports.append({'fold': index + 1, 'recipe': 'original v11; 15 min; trend=0', 'training': report, 'forecast': metrics, 'prediction_origins': fold['prediction_origins'], 'provenance': fold})
        del model, fold_data
    oof_predictions = np.concatenate(oof_predictions)
    np.savez_compressed(output / 'oof_forecasts.npz', origin=np.asarray(oof_origins), forecast_temperature_celsius=oof_predictions, sample_minutes=15, units='celsius')
    final_v11, v11_report = train_temperature(data, protocol['origins']['train'], protocol['origins']['val'], seed, output / 'final_v11', device, temperature_recipe, kind='v11')
    v11_validation, v11_validation_metrics = predict(final_v11, data, protocol['origins']['val'], 15, temperature_recipe['batch_size'], device, output / 'final_v11/val_forecasts.npz')
    del final_v11
    producer = {'oof': oof_reports, 'final_v11': v11_report, 'final_v11_validation': v11_validation_metrics}
    dump(output / 'producer_summary.json', producer)
    transfer = PACKAGE / 'outputs/split_transfer'
    transfer.mkdir(parents=True, exist_ok=True)
    key = Path(cfg['protocol_path']).parent.name + '_A2_seed' + str(seed)
    archive = transfer / (key + '_h1.tar.gz')
    files = ['config.json', 'protocol.json', 'actual_recipes.json', 'data_report.json', 'producer_summary.json', 'oof_forecasts.npz', 'final_v11/val_forecasts.npz']
    with tarfile.open(archive, 'w:gz') as bundle:
        for name in files:
            bundle.add(output / name, arcname=name)
    dump(output / 'handoff_ready.json', {'key': key, 'archive': str(archive), 'ready_beijing': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()})
    print('H1_HANDOFF_READY ' + key, flush=True)
    stpm, stpm_report = train_temperature(data, protocol['origins']['train'], protocol['origins']['val'], seed, output / 'final_temperature', device, temperature_recipe, kind='A2')
    return {'status': 'temperature_completed', 'model': 'A2', 'seed': seed, 'stpm': stpm_report, 'peak_cuda_allocated_mib': torch.cuda.max_memory_allocated(device) / 1048576}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg = _release_json_loads(Path(args.config).read_text())
    output = resolve(cfg['output'])
    output.mkdir(parents=True)
    dump(output / 'config.json', cfg)
    shutil.copytree(Path(__file__).resolve().parent, output / 'source_snapshot', ignore=shutil.ignore_patterns('__pycache__'))
    started = time.time()
    try:
        result = run(cfg, output)
    except BaseException:
        (output / 'temperature_error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        records = resolve(cfg['record_dir'])
        records.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
        (records / (stamp + '_' + output.parent.name + '_temperature_seed' + str(cfg['seed']) + '_failed.md')).write_text('# D1 A2 temperature stage\n\n- Result: interrupted or failed.\n' + '- Error: ' + traceback.format_exc().rstrip().splitlines()[-1] + '\n' + '- Preserved products and full traceback: ' + str(output) + '\n', encoding='utf-8')
        raise
    result.update(elapsed_seconds=time.time() - started, started_epoch=started, finished_beijing=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
    dump(output / 'temperature_complete.json', result)
    key = Path(cfg['protocol_path']).parent.name + '_A2_seed' + str(cfg['seed'])
    archive = PACKAGE / 'outputs/split_transfer' / (key + '_temperature.tar.gz')
    with tarfile.open(archive, 'w:gz') as bundle:
        for name in ['final_temperature/best.pt', 'final_temperature/config.json', 'final_temperature/summary.json', 'temperature_complete.json']:
            bundle.add(output / name, arcname=name)
    dump(output / 'temperature_transfer_ready.json', {'archive': str(archive)})
    records = resolve(cfg['record_dir'])
    records.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    (records / (stamp + '_' + key + '_temperature.md')).write_text('# D1 A2 temperature stage\n\n- Result: completed; H1 handed to local CPU.\n' + '- Products: ' + str(output) + '; temperature_complete.json and stage checkpoints.\n' + '- Duration seconds: ' + str(result['elapsed_seconds']) + '\n', encoding='utf-8')
    print('TEMPERATURE_COMPLETED ' + key, flush=True)
if __name__ == '__main__':
    main()
