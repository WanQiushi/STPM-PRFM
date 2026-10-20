"""Evaluate selected checkpoints on declared grids; no fitting or selection."""
from if_release.runtime import loads as _release_json_loads
from pathlib import Path
import argparse, json
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import torch
from h1_interventions import SevenDeflectionH1
HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cpu')
args = parser.parse_args()
cfg = _release_json_loads((HERE / 'config.json').read_text())
out = HERE / cfg['output']
dt = cfg['final_evaluation_sample_minutes']
tdt = cfg['temperature_sample_minutes']
torch.set_num_threads(4)
device = torch.device(args.device)
ck = torch.load(HERE / cfg['response_checkpoint'], map_location='cpu', weights_only=False)
mean = np.asarray(ck['mean'])
std = np.asarray(ck['std'])
model = SevenDeflectionH1({**cfg, 'sample_minutes': dt})
state = ck['model']
state['decomposer.frequency'] = model.decomposer.frequency
state['decomposer.padded_frequency'] = model.decomposer.padded_frequency
model.load_state_dict(state)
model = model.to(device).eval()
model.requires_grad_(False)
with np.load(HERE / cfg['data']) as z:
    observed = z['values'].copy()
    clean = z['clean_values'].copy()
    mask = z['valid_observed'].copy()
    channels = z['channels'].tolist()
normalized = ((observed.astype(float) - mean) / std).astype('float32')

def scores(prediction, target, valid):
    error = prediction.astype(float) - target.astype(float)
    counts = valid.sum((0, 1))
    sse = (error ** 2 * valid).sum((0, 1))
    sae = (abs(error) * valid).sum((0, 1))
    return dict(rmse_mm=float(np.sqrt(sse.sum() / counts.sum())), mae_mm=float(sae.sum() / counts.sum()), mse_mm2=float(sse.sum() / counts.sum()), per_channel_rmse_mm=np.sqrt(sse / counts).tolist(), per_channel_mae_mm=(sae / counts).tolist(), per_channel_bias_mm=(error * valid).sum((0, 1)).__truediv__(counts).tolist(), counts=counts.tolist())
metrics = {}
for part in ['val', 'test']:
    with np.load(HERE / cfg['temperature_reference'] / 'output/temperature/final_A2' / (part + '_forecasts.npz')) as z:
        origins = z['origin'].copy()
        full_temperature = z['forecast_temperature_celsius'].copy()
        temperature_parameters = int(z['parameters'])
    forecast = full_temperature[:, ::dt // tdt]
    keys = ['prediction', 'lf_prediction', 'mf_prediction', 'reversible_prediction', 'memory_prediction', 'bw_latent']
    saved = {k: [] for k in keys}
    with torch.no_grad():
        for first in range(0, len(origins), 16):
            batch = origins[first:first + 16]
            history = np.stack([normalized[o - 43200:o:dt] for o in batch])
            forcing = ((forecast[first:first + 16] - mean[:1]) / std[:1]).astype('float32')
            result = model(torch.as_tensor(history, device=device), torch.as_tensor(forcing, device=device), torch.as_tensor(batch, device=device))
            for k in keys:
                saved[k].append(result[k].cpu().numpy())
    saved = {k: np.concatenate(v) for k, v in saved.items()}
    pred = saved['prediction'].astype(float) * std[1:] + mean[1:]
    truth = np.stack([observed[o:o + 1440:dt, 1:] for o in origins])
    valid = np.stack([mask[o:o + 1440:dt, 1:] for o in origins])
    target_clean = np.stack([clean[o:o + 1440:dt, 1:] for o in origins])
    previous = np.stack([observed[o - 1440:o:dt, 1:] for o in origins])
    temperature_target = np.stack([observed[o:o + 1440:dt, :1] for o in origins])
    full_temperature_target = np.stack([observed[o:o + 1440:tdt, :1] for o in origins])
    terr = forecast - temperature_target
    full_terr = full_temperature - full_temperature_target
    common_step = 30 // dt
    daily = np.sqrt(((pred - truth) ** 2 * valid).sum((1, 2)) / valid.sum((1, 2)))
    m = dict(windows=len(origins), sample_minutes=dt, known_future_temperature=False, temperature_parameters=temperature_parameters, response_parameters=sum((p.numel() for p in model.parameters())), selected_response_epoch=int(ck['epoch']), response_checkpoint=cfg['response_checkpoint'], response_intervention=cfg['h1_intervention'], observed_target=scores(pred, truth, valid), clean_target=scores(pred, target_clean, valid), copy_previous_day=scores(previous, truth, valid), tail_2h=scores(pred[:, -120 // dt:], truth[:, -120 // dt:], valid[:, -120 // dt:]), temperature_rmse_celsius=float(np.sqrt(np.mean(terr ** 2))), temperature_mae_celsius=float(np.mean(abs(terr))), native_temperature=dict(sample_minutes=tdt, rmse_c=float(np.sqrt(np.mean(full_terr ** 2))), mae_c=float(np.mean(abs(full_terr)))), common30=dict(sample_minutes=30, observed_target=scores(pred[:, ::common_step], truth[:, ::common_step], valid[:, ::common_step]), temperature_rmse_celsius=float(np.sqrt(np.mean(terr[:, ::common_step] ** 2))), temperature_mae_celsius=float(np.mean(abs(terr[:, ::common_step])))), daily_rmse_mm=daily.tolist())
    metrics[part] = m
    arrays = dict(prediction_mm=pred, target_mm=truth, clean_target_mm=target_clean, target_mask=valid, origin=origins, predicted_temperature_celsius=forecast, temperature_target_celsius=temperature_target, channels=np.array(channels[1:]), lf_prediction_mm=saved['lf_prediction'] * std[1:] + mean[1:], mf_prediction_mm=saved['mf_prediction'] * std[1:], reversible_prediction_mm=saved['reversible_prediction'] * std[1:], memory_prediction_mm=saved['memory_prediction'] * std[1:], bw_latent=saved['bw_latent'])
    np.savez_compressed(out / f'a2_{part}_predictions_{dt}min.npz', **arrays)
    print('EVALUATED ' + json.dumps(dict(split=part, response_rmse_mm=m['observed_target']['rmse_mm'], sample_minutes=dt)), flush=True)
metrics.update(learned_thermal_lag_hours=model.thermal_lag_hours().detach().cpu().tolist(), reference_arm=cfg['paired_reference_arm'], test_policy='Selected stages fixed before held-out evaluation; no test tuning')
(out / f'a2_metrics_{dt}min.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
(out / 'final_evaluation_record.md').write_text('# D2.1 selected model evaluation\n\n- Date: ' + datetime.now(ZoneInfo('Asia/Shanghai')).isoformat() + '\n- Purpose: score saved validation-selected models with predicted future temperature.\n- Contract: config.json, experiment_definition.json; temperature native grid, response native grid and common30 are separately named.\n- Outputs: ' + str(out) + '\n- Result: ' + json.dumps({k: metrics[k]['observed_target'] for k in ['val', 'test']}) + '\n', encoding='utf-8')
