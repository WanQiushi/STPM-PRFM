"""Build a minute-aligned observed-data release from the selected source records."""
from if_release.runtime import expand as _release_expand
from pathlib import Path
import json
import platform
import sys
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
root = Path(__file__).resolve().parents[1]
data = root / 'data'
processed = data / 'processed'
quality = data / 'quality'
processed.mkdir(parents=True, exist_ok=True)
quality.mkdir(parents=True, exist_ok=True)
channels = ['Temperature', 'Auxiliary1', 'Pier1', 'Pier2', 'Pier3', 'Pier4', 'Auxiliary2']
descriptions = [('跨中上游温度', '温度', '公路面跨中上游', 'degC', 143), ('引桥1#墩下游纵向位移', '梁端位移', '引桥1#墩下游', 'mm', 152), ('主桥1#墩下游纵向位移', '支座位移', '主桥1#墩下游', 'mm', 150), ('主桥3#墩下游纵向位移', '支座位移', '主桥3#墩下游', 'mm', 154), ('主桥4#墩下游纵向位移', '支座位移', '主桥4#墩下游', 'mm', 156), ('主桥6#墩下游纵向位移', '支座位移', '主桥6#墩下游', 'mm', 158), ('引桥6#墩下游纵向位移', '梁端位移', '引桥6#墩下游', 'mm', 160)]
sensor_rows = [dict(input_index=i, output_index=i - 1 if i else '', sensor_id=k, name=d[0], type=d[1], location=d[2], unit=d[3], excel_row=d[4]) for i, (k, d) in enumerate(zip(channels, descriptions))]
pd.DataFrame(sensor_rows).to_csv(data / 'sensors.csv', index=False, encoding='utf-8')
frames = []
for name in ['温度.csv.gz', '支座位移.csv.gz', '梁端位移.csv.gz']:
    frame = pd.read_csv(data / 'sources' / name, dtype={'MeasureNum': str, 'MonitorTime': str, 'value': np.float64})
    frame['timestamp'] = pd.to_datetime(frame.MonitorTime, format='ISO8601')
    frames.append(frame[['timestamp', 'MeasureNum', 'value']])
long = pd.concat(frames, ignore_index=True)
raw_counts = long.groupby('MeasureNum').size()
unique = long.drop_duplicates(['timestamp', 'MeasureNum', 'value'])
dedup_counts = unique.groupby('MeasureNum').size()
wide = unique.pivot(index='timestamp', columns='MeasureNum', values='value').reindex(columns=channels)
timeline = pd.date_range('2022-01-01 00:00:00', '2023-12-31 23:59:00', freq='min', name='timestamp')
aligned = wide.reindex(timeline)
values = aligned.to_numpy(dtype=np.float64)
observed = np.isfinite(values)
np.savez_compressed(processed / 'minute_observed.npz', timestamps=timeline.to_numpy(dtype='datetime64[ns]'), values=values, observed=observed, channels=np.array(channels), units=np.array([d[3] for d in descriptions]), input_indices=np.arange(7), output_indices=np.arange(1, 7), dates=pd.date_range('2022-01-01', '2023-12-31', freq='D').to_numpy(dtype='datetime64[D]'))
aligned.to_csv(processed / 'minute_observed.csv.gz', index=True, encoding='utf-8', compression='gzip', date_format='%Y-%m-%d %H:%M:%S')

def intervals(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return (np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
stats = []
missing_intervals = []
for j, k in enumerate(channels):
    x = values[:, j]
    good = observed[:, j]
    starts, ends = intervals(~good)
    valid = x[good]
    same_minute_steps = np.diff(x)
    finite_steps = np.flatnonzero(np.isfinite(same_minute_steps))
    biggest_step = finite_steps[np.argmax(np.abs(same_minute_steps[finite_steps]))]
    stats.append(dict(sensor_id=k, raw_rows=int(raw_counts[k]), unique_records=int(dedup_counts[k]), duplicate_rows_removed=int(raw_counts[k] - dedup_counts[k]), observed_minutes=int(good.sum()), missing_minutes=int((~good).sum()), coverage=float(good.mean()), longest_missing_run_minutes=int((ends - starts).max()), min=float(valid.min()), q01=float(np.quantile(valid, 0.01)), median=float(np.median(valid)), q99=float(np.quantile(valid, 0.99)), max=float(valid.max()), max_abs_adjacent_change=float(abs(same_minute_steps[biggest_step])), max_change_start=str(timeline[biggest_step]), max_change_end=str(timeline[biggest_step + 1])))
    missing_intervals.extend((dict(sensor_id=k, start=str(timeline[a]), end=str(timeline[b - 1]), minutes=int(b - a)) for a, b in zip(starts, ends)))
pd.DataFrame(stats).to_csv(quality / 'channel_quality.csv', index=False, encoding='utf-8')
pd.DataFrame(missing_intervals).to_csv(quality / 'missing_intervals.csv.gz', index=False, encoding='utf-8', compression='gzip')
all_good = observed.all(axis=1)
starts, ends = intervals(all_good)
runs = pd.DataFrame([dict(start=str(timeline[a]), end=str(timeline[b - 1]), start_index=int(a), end_index_exclusive=int(b), minutes=int(b - a)) for a, b in zip(starts, ends)])
runs.to_csv(quality / 'complete_segments.csv', index=False, encoding='utf-8')
daily_obs = observed.reshape(730, 1440, 7)
daily = pd.DataFrame(daily_obs.sum(axis=1), columns=channels)
daily.insert(0, 'date', pd.date_range('2022-01-01', '2023-12-31', freq='D').strftime('%Y-%m-%d'))
daily['all_seven_complete_minutes'] = all_good.reshape(730, 1440).sum(axis=1)
daily['complete_day'] = daily_obs.all(axis=(1, 2))
daily.to_csv(quality / 'daily_coverage.csv', index=False, encoding='utf-8')
monthly = pd.DataFrame(observed, index=timeline, columns=channels).groupby(timeline.to_period('M')).agg(['sum', 'mean'])
monthly.columns = ['_'.join(c) for c in monthly.columns]
monthly.to_csv(quality / 'monthly_coverage.csv', encoding='utf-8')
lengths = ends - starts
summary = {'version': 'monitored_bridge_downstream6_minute_observed_v1_20260922', 'time_start': str(timeline[0]), 'time_end': str(timeline[-1]), 'sampling_seconds': 60, 'timezone': 'Source local timestamps retained without conversion; no timezone metadata provided by source CSV.', 'time_steps': len(timeline), 'channels': channels, 'values_shape': list(values.shape), 'daily_view_shape': [730, 1440, 7], 'all_seven_observed_minutes': int(all_good.sum()), 'all_seven_coverage': float(all_good.mean()), 'all_seven_complete_days': int(daily.complete_day.sum()), 'complete_segments': len(runs), 'longest_complete_segment_minutes': int(lengths.max()), 'fully_observed_1440_minute_windows_stride1': int(np.maximum(lengths - 1440 + 1, 0).sum()), 'duplicate_rows_removed': int(len(long) - len(unique)), 'source_rows_selected': len(long), 'retained_source_records': len(unique), 'retained_records_outside_requested_time_range': int(((unique.timestamp < timeline[0]) | (unique.timestamp > timeline[-1])).sum()), 'non_minute_source_timestamps': int((unique.timestamp.dt.second.ne(0) | unique.timestamp.dt.microsecond.ne(0)).sum()), 'split_status': 'Not assigned; chronological split to be fixed with forecasting protocol.', 'history_length': None, 'forecast_length': None, 'user_requested_points': 1440, 'points_role': 'Awaiting whether history L, forecast H, or both.', 'processing': 'Selected sensors, removed identical numeric sensor-time-value duplicates, aligned exact minute timestamps; no interpolation, smoothing, resampling, normalization, or outlier removal.', 'channel_statistics': stats, 'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__}}
(quality / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
config = {'dataset_version': summary['version'], 'sampling_seconds': 60, 'input_channels': channels, 'output_channels': channels[1:], 'history_length': None, 'forecast_length': None, 'user_requested_points': 1440, 'split': None, 'source_workbook': _release_expand('${ASSETS}/asset_001.xlsx'), 'source_sheet': '全测点设备汇总', 'missing_value': 'NaN', 'observed_mask': 'finite source value at the exact minute', 'prediction_contract': 'Past seven channels -> future six displacement channels; no future measured temperature input.'}
(data / 'dataset_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in summary.items() if k != 'channel_statistics'}, ensure_ascii=False, indent=2), flush=True)
