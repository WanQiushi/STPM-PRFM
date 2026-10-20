from if_release.runtime import loads as _release_json_loads
from pathlib import Path
import json
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
cfg = _release_json_loads((ROOT / 'generation_config.json').read_text(encoding='utf-8'))
response = _release_json_loads((ROOT / 'thermal_response.json').read_text(encoding='utf-8'))
data = ROOT / 'data'
data.mkdir(exist_ok=True)
rng = np.random.default_rng(cfg['seed'])
warm = cfg['warmup_days'] * 1440
n = cfg['days'] * 1440
length = n + warm
t = np.arange(-warm, n, dtype=np.float64) / 1440.0
hours = (length + 59) // 60
rho = np.exp(-1 / cfg['weather_ar_timescale_hours'])
weather = np.empty(hours)
weather[0] = rng.normal(0, cfg['weather_stationary_std_c'])
for k in range(1, hours):
    weather[k] = rho * weather[k - 1] + cfg['weather_stationary_std_c'] * np.sqrt(1 - rho * rho) * rng.normal()
weather_minute = np.interp(np.arange(length) / 60.0, np.arange(hours), weather)
day_index = np.arange(-cfg['warmup_days'], cfg['days'] + 1)
amplitude_daily = cfg['daily_mean_amplitude_c'] + 0.6 * np.tanh(rng.normal(size=len(day_index)))
phase_daily = 0.4 * np.tanh(rng.normal(size=len(day_index)))
amplitude = np.interp(t, day_index, amplitude_daily)
phase = np.interp(t, day_index, phase_daily)
ref = cfg['reference_temperature_c']
seasonal = cfg['seasonal_amplitude_c'] * np.sin(2 * np.pi * (t - cfg['seasonal_phase_offset_days']) / 365.25)
driver = ref + seasonal + amplitude * np.sin(2 * np.pi * (t - cfg['daily_phase_offset_days']) + phase) + weather_minute
phis = np.exp(-1 / (60 * np.array([cfg['girder_tau_hours'], cfg['cable_tau_hours']])))
thermal = np.empty((length, 2))
thermal[0] = driver[0]
for k in range(1, length):
    thermal[k] = phis * thermal[k - 1] + (1 - phis) * driver[k - 1]
thermal = thermal[warm:]
driver = driver[warm:]
g = np.array(response['girder_mm_per_c'])
c = np.array(response['cable_mm_per_c'])
girder = (thermal[:, 0] - ref)[:, None] * g[None, :]
cable = (thermal[:, 1] - ref)[:, None] * c[None, :]
clean = np.column_stack([driver, girder + cable])
noise = np.column_stack([rng.normal(0, cfg['temperature_noise_std_c'], n), rng.normal(0, cfg['deflection_noise_std_mm'], (n, 7))])
values = (clean + noise).astype(np.float32)
channels = np.array(['SIM_T'] + response['channels'])
units = np.array(['degC'] + ['mm'] * 7)
timestamps = np.datetime64('2020-01-01T00:00', 'm') + np.arange(n).astype('timedelta64[m]')
mask = np.ones(values.shape, dtype=bool)
np.savez_compressed(data / 'minute_synthetic.npz', values=values, clean_values=clean.astype(np.float32), noise=noise.astype(np.float32), thermal_states_c=thermal.astype(np.float32), girder_contribution_mm=girder.astype(np.float32), cable_contribution_mm=cable.astype(np.float32), timestamps=timestamps, channels=channels, units=units, observed=mask, valid_observed=mask, available=mask, input_available_at=np.broadcast_to(np.arange(n, dtype=np.int64)[:, None], values.shape), input_indices=np.arange(8), output_indices=np.arange(1, 8), split_days=np.array(cfg['split_days']), dates=timestamps[::1440].astype('datetime64[D]'))
ends = np.cumsum(cfg['split_days']) * 1440
starts = np.r_[0, ends[:-1]]
origins = {}
for name, start, end in zip(['train', 'validation', 'test'], starts, ends):
    step = 720 if name == 'train' else 1440
    first = max(start, 180 * 1440)
    origins[name] = list(range(int(first), int(end - 1440 + 1), step))
(data / 'window_origins.json').write_text(json.dumps(dict(history_days=30, temperature_encoder_history_days=7, earliest_origin_day=180, horizon_minutes=1440, origins=origins), indent=2), encoding='utf-8')
summary = dict(version=cfg['version'], shape=list(values.shape), dtype=str(values.dtype), channels=channels.tolist(), units=units.tolist(), days=cfg['days'], split_days=cfg['split_days'], seed=cfg['seed'], timestamps_are_fictional=True, source_measured_values_used=False, girder_tau_hours=cfg['girder_tau_hours'], cable_tau_hours=cfg['cable_tau_hours'], value_ranges={str(name): dict(min=float(values[:, j].min()), max=float(values[:, j].max()), std=float(values[:, j].std())) for j, name in enumerate(channels)}, thermal_state_ranges_c=[dict(min=float(thermal[:, j].min()), max=float(thermal[:, j].max())) for j in range(2)], noise_std_empirical=noise.std(axis=0).tolist(), window_counts={name: len(xs) for name, xs in origins.items()}, response_checks=response['linearization_checks'], response_checks_scope='Inherited D2 coefficient checks; see inherited_fe_validation_scope in generation_config.json', file_bytes=(data / 'minute_synthetic.npz').stat().st_size, claim_scope='Independent finite-element-informed synthetic thermal-deflection observations; public thermal-lag/noise assumptions; does not establish BW mechanical memory or reproduce private measured signals')
(data / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(dict(shape=summary['shape'], file_bytes=summary['file_bytes'], window_counts=summary['window_counts'], response_check_relative_l2={k: v['relative_l2'] for k, v in response['linearization_checks'].items()})))
