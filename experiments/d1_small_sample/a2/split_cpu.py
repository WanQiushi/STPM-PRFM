"""Run one budget/seed of the complete fixed A2 architecture from fresh weights.

Execution is explicit: python a2/train.py --config configs/9m/A2_seed2024.json
Writing/importing this module does not schedule or launch training.
"""
from if_release.runtime import expand as _release_expand, loads as _release_json_loads
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
from temperature import AblatedTemperature
from response_model import HybridKqsz

def finish(cfg, output, data, device):
    protocol = _release_json_loads((output / 'protocol.json').read_text())
    protocol_path = output / 'protocol.json'
    data_path = LOCAL_DATA
    seed = cfg['seed']
    recipe = _release_json_loads((output / 'actual_recipes.json').read_text())
    temperature_recipe = recipe['temperature']
    response_recipe = recipe['response']
    producer = _release_json_loads((output / 'producer_summary.json').read_text())
    oof_reports = producer['oof']
    v11_report = producer['final_v11']
    v11_validation_metrics = producer['final_v11_validation']
    temperature_complete = _release_json_loads((output / 'temperature_complete.json').read_text())
    stpm_report = temperature_complete['stpm']
    checkpoint = torch.load(output / 'final_temperature/best.pt', map_location=device, weights_only=False)
    stpm = AblatedTemperature(checkpoint['config']).to(device)
    stpm.load_state_dict(checkpoint['model'])
    stpm.eval()
    h1_report = _release_json_loads((output / 'response/summary.json').read_text())
    h1 = HybridKqsz(response_recipe).to(device)
    h1.load_state_dict(torch.load(output / 'response/adapted_best.pt', map_location=device, weights_only=False)['model'])
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
    summary = {'status': 'completed', 'model': 'A2', 'seed': seed, 'protocol_path': str(protocol_path), 'data_path': str(data_path), 'parameters': stpm_report['parameters'] + h1_report['parameters'], 'temperature_parameters': stpm_report['parameters'], 'response_parameters': h1_report['parameters'], 'oof': oof_reports, 'final_v11': v11_report, 'final_v11_validation': v11_validation_metrics, 'stpm': stpm_report, 'response': h1_report, 'results': results, 'full_data_weights_used': False, 'test_used_for_selection': False, 'final_sample_minutes': 1, 'response_training_sample_minutes': 15, 'adaptation_temperature': 'original v11 chronological OOF and final v11 validation prediction', 'final_inference_temperature': 'fixed-architecture A2 STPM trained within budget', 'peak_cuda_allocated_mib': temperature_complete['peak_cuda_allocated_mib']}
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

def execute():
    global LOCAL_DATA, RUN_OUTPUT, RUN_CFG
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['h1', 'evaluate'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    RUN_OUTPUT = output
    LOCAL_DATA = args.data
    cfg = _release_json_loads((output / 'config.json').read_text())
    RUN_CFG = cfg
    protocol = _release_json_loads((output / 'protocol.json').read_text())
    data = Data(LOCAL_DATA, protocol['mean'], protocol['std'])
    device = torch.device('cpu')
    torch.set_num_threads(4)
    started = time.time()
    if args.stage == 'h1':
        recipe = _release_json_loads((output / 'actual_recipes.json').read_text())
        with np.load(output / 'oof_forecasts.npz') as archive:
            origins = archive['origin'].tolist()
            forecasts = archive['forecast_temperature_celsius']
        with np.load(output / 'final_v11/val_forecasts.npz') as archive:
            validation = archive['forecast_temperature_celsius']
        model, report = train_response(data, protocol['h1_pretraining_origins'], origins, forecasts, protocol['origins']['val'], validation, cfg['seed'], output / 'response', device, recipe['response'], recipe['adaptation'])
        dump(output / 'h1_complete.json', {'status': 'completed', 'elapsed_seconds': time.time() - started, 'device': 'local_cpu', 'threads': 4, 'response': report})
        records = Path(_release_expand('${RECORDS}/d1_small_sample_20261013'))
        records.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
        (records / (stamp + '_' + output.parent.name + '_H1_seed' + str(cfg['seed']) + '.md')).write_text('# D1 small-sample H1 CPU stage\n\n- Result: completed; validation-selected weights saved.\n' + '- Products: ' + str(output / 'response') + '; summary.json, histories and checkpoints.\n' + '- Duration seconds: ' + str(time.time() - started) + '\n', encoding='utf-8')
    else:
        result = finish(cfg, output, data, device)
        temp = _release_json_loads((output / 'temperature_complete.json').read_text())
        result.update(elapsed_seconds=time.time() - temp['started_epoch'], execution_policy='GPU temperature and local CPU H1 in parallel; 4 threads per CPU worker', h1_execution=_release_json_loads((output / 'h1_complete.json').read_text()), evaluation_seconds=time.time() - started)
        dump(output / 'summary.json', result)
        records = Path(_release_expand('${RECORDS}/d1_small_sample_20261013'))
        records.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
        record(records / (stamp + '_' + output.parent.name + '_A2_seed' + str(cfg['seed']) + '_split.md'), cfg, 'completed', result, output)
        record(output / 'experiment_record.md', cfg, 'completed', result, output)
    print('CPU_STAGE_COMPLETED ' + args.stage, flush=True)

def main():
    try:
        execute()
    except BaseException as error:
        message = traceback.format_exc()
        (RUN_OUTPUT / 'cpu_error.txt').write_text(message, encoding='utf-8')
        records = Path(_release_expand('${RECORDS}/d1_small_sample_20261013'))
        records.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
        status = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        record(records / (stamp + '_' + RUN_OUTPUT.parent.name + '_A2_seed' + str(RUN_CFG['seed']) + '_cpu_' + status + '.md'), RUN_CFG, status, message.rstrip().splitlines()[-1] + '; full error: ' + str(RUN_OUTPUT / 'cpu_error.txt'), RUN_OUTPUT)
        raise
if __name__ == '__main__':
    main()
