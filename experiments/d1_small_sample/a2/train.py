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
    stpm, stpm_report = train_temperature(data, protocol['origins']['train'], protocol['origins']['val'], seed, output / 'final_temperature', device, temperature_recipe, kind='A2')
    h1, h1_report = train_response(data, protocol['h1_pretraining_origins'], oof_origins, oof_predictions, protocol['origins']['val'], v11_validation, seed, output / 'response', device, response_recipe, adaptation_recipe)
    native = native_model(h1, response_recipe, device)
    del h1
    results = {}
    for name in ['val', 'test']:
        origins = protocol['origins'][name]
        forecasts, temperature_metrics = predict(stpm, data, origins, 1, temperature_recipe['batch_size'], device, output / ('temperature_' + name + '_forecasts.npz'))
        loader = DataLoader(ResponseWindows(data, origins, 1, forecasts), batch_size=16, shuffle=False, num_workers=0)
        _, arrays = evaluate(native, loader, device, data, forecast=True, collect=True)
        temperature_indices = np.asarray(origins)[:, None] + np.arange(1440)[None]
        temperature_truth = data.values[temperature_indices, :1] * data.std[:1] + data.mean[:1]
        temperature_mask = data.observed[temperature_indices, :1]
        prediction = np.concatenate((forecasts, arrays['prediction_mm']), axis=2)
        target = np.concatenate((temperature_truth, arrays['target_mm']), axis=2)
        mask = np.concatenate((temperature_mask, arrays['mask']), axis=2)
        np.savez_compressed(output / (name + '_predictions.npz'), prediction=prediction, target=target, mask=mask, origin=arrays['origin'], mean=data.mean, std=data.std, units=np.asarray(['celsius', 'mm', 'mm', 'mm', 'mm']), sample_minutes=1, lf_prediction_mm=arrays['lf_prediction_mm'], mf_prediction_mm=arrays['mf_prediction_mm'])
        results['validation' if name == 'val' else 'test'] = {'displacement': score(arrays['prediction_mm'], arrays['target_mm'], arrays['mask']), 'temperature': score(forecasts, temperature_truth, temperature_mask), 'temperature_module': temperature_metrics}
    summary = {'status': 'completed', 'model': 'A2', 'seed': seed, 'protocol_path': str(protocol_path), 'data_path': str(data_path), 'parameters': stpm_report['parameters'] + h1_report['parameters'], 'temperature_parameters': stpm_report['parameters'], 'response_parameters': h1_report['parameters'], 'oof': oof_reports, 'final_v11': v11_report, 'final_v11_validation': v11_validation_metrics, 'stpm': stpm_report, 'response': h1_report, 'results': results, 'full_data_weights_used': False, 'test_used_for_selection': False, 'final_sample_minutes': 1, 'response_training_sample_minutes': 15, 'adaptation_temperature': 'original v11 chronological OOF and final v11 validation prediction', 'final_inference_temperature': 'fixed-architecture A2 STPM trained within budget', 'peak_cuda_allocated_mib': torch.cuda.max_memory_allocated(device) / 1048576 if device.type == 'cuda' else None}
    return summary

def record(path, cfg, status, result, output):
    timestamp = datetime.now(ZoneInfo('Asia/Shanghai'))
    content = f"# D1 small-sample A2 run\n\n- Date: {timestamp.isoformat()}\n- Project: IFpaper_v1.\n- Purpose: Retrain the complete fixed A2 architecture under one declared data budget.\n- Operation/configuration: seed {cfg['seed']}; protocol {cfg['protocol_path']}; saved config.json and protocol.json.\n- Stages: fresh v9; original 15-minute v11 chronological OOF and final producer; fixed A2 STPM; original true-temperature H1 identification; v11 forecast adaptation; one-minute validation/test evaluation.\n- Result: {status}.\n- Products: {output}; summary.json, stage checkpoints, histories, predictions and source snapshot.\n"
    if status == 'completed':
        validation = result['results']['validation']['displacement']
        test = result['results']['test']['displacement']
        content += f"- Validation: displacement RMSE {validation['rmse']:.6f} mm; MAE {validation['mae']:.6f} mm.\n- Test: displacement RMSE {test['rmse']:.6f} mm; MAE {test['mae']:.6f} mm.\n- Duration: {result['elapsed_seconds']:.3f} s; complete metrics and stage details in summary.json.\n"
    else:
        content += '\n' + result + '\n'
    Path(path).write_text(content, encoding='utf-8')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--device')
    args = parser.parse_args()
    cfg = _release_json_loads(Path(args.config).read_text(encoding='utf-8'))
    if args.device:
        cfg['device'] = args.device
    output = resolve(cfg['output'])
    output.mkdir(parents=True)
    dump(output / 'config.json', cfg)
    shutil.copytree(Path(__file__).resolve().parent, output / 'source_snapshot', ignore=shutil.ignore_patterns('__pycache__'))
    record_root = resolve(cfg['record_dir'])
    record_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    timestamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    record_path = record_root / (timestamp + '_' + Path(cfg['protocol_path']).parent.name + '_A2_seed' + str(cfg['seed']) + '.md')
    try:
        summary = run(cfg, output)
    except BaseException as error:
        original_error = traceback.format_exc()
        status = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        error_file = output / 'error.txt'
        error_file.write_text(original_error, encoding='utf-8')
        concise_error = original_error.rstrip().splitlines()[-1] + '\n\nFull original error: ' + str(error_file)
        record(record_path, cfg, status, concise_error, output)
        record(output / 'experiment_record.md', cfg, status, concise_error, output)
        dump(output / 'summary.json', {'status': status, 'model': 'A2', 'seed': cfg['seed'], 'error_file': str(error_file), 'elapsed_seconds': time.time() - started})
        raise
    summary['elapsed_seconds'] = time.time() - started
    dump(output / 'summary.json', summary)
    record(record_path, cfg, 'completed', summary, output)
    record(output / 'experiment_record.md', cfg, 'completed', summary, output)
    print('COMPLETED ' + json.dumps({'status': 'completed', 'model': 'A2', 'seed': cfg['seed'], 'validation_rmse_mm': summary['results']['validation']['displacement']['rmse'], 'test_rmse_mm': summary['results']['test']['displacement']['rmse'], 'elapsed_seconds': summary['elapsed_seconds']}), flush=True)
if __name__ == '__main__':
    main()
