"""Inspect timing patterns and verify selected flagged values against source extracts."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
root = Path(__file__).resolve().parents[1]
out = root / 'reports/20260922_data_audit'
z = np.load(root / 'data/processed/minute_observed.npz')
t = pd.DatetimeIndex(z['timestamps'])
channels = z['channels'].tolist()
df = pd.DataFrame(z['values'], index=t, columns=channels)
coverage = df.notna()
patterns = coverage.groupby([t.to_period('M').astype(str), t.minute]).mean()
patterns.index.names = ['month', 'minute_of_hour']
patterns.to_csv(out / 'minute_of_hour_coverage.csv')
day_example = coverage.loc['2023-06-04'].groupby(coverage.loc['2023-06-04'].index.minute).mean()
day_example.to_csv(out / '20230604_minute_of_hour_coverage.csv')
temp_valid = coverage['Temperature'].to_numpy().reshape(730, 24, 60)
hour_blocks_missing = (~temp_valid[:, :, 1:12]).all(axis=2) & temp_valid[:, :, 0] & temp_valid[:, :, 12]
day_count = hour_blocks_missing.sum(axis=1)
pattern_days = pd.DataFrame({'date': pd.date_range('2022-01-01', '2023-12-31', freq='D'), 'hours_with_01_to_11_missing_and_00_12_observed': day_count})
pattern_days.to_csv(out / 'temperature_repeated_hourly_gap_days.csv', index=False)
top_days = pattern_days[pattern_days.hours_with_01_to_11_missing_and_00_12_observed >= 20]
requests = {'Temperature': ['2023-03-04 10:51:00', '2023-03-04 10:52:00'], 'Pier4': ['2022-01-08 00:39:00', '2022-01-08 00:44:00', '2022-01-08 00:45:00', '2023-09-01 00:47:00'], 'Pier3': ['2022-11-23 17:18:00', '2022-11-23 17:19:00'], 'Auxiliary1': ['2023-02-03 09:30:00', '2023-02-03 09:31:00']}
records = []
for file in ['温度.csv.gz', '支座位移.csv.gz', '梁端位移.csv.gz']:
    for chunk in pd.read_csv(root / 'data/sources' / file, dtype=str, keep_default_na=False, chunksize=400000):
        for k, dates in requests.items():
            rows = chunk[(chunk.MeasureNum == k) & chunk.MonitorTime.str.slice(0, 19).isin(dates)]
            for r in rows.itertuples():
                when = r.MonitorTime[:19]
                source = float(r.value)
                saved = float(df.loc[when, k])
                records.append(dict(source_file=file, sensor_id=k, timestamp=when, source_value_text=r.value, aligned_value=saved, numeric_match=source == saved))
pd.DataFrame(records).to_csv(out / 'source_value_examples.csv', index=False)
rel = df['Pier4'] - df['Pier3']
meta = {'temperature_hourly_gap_example': {'date': '2023-06-04', 'temperature_observed_minutes': int(coverage.loc['2023-06-04', 'Temperature'].sum()), 'hours_with_minutes_01_to_11_missing_and_00_12_observed': int(pattern_days.loc[pattern_days.date.eq('2023-06-04'), 'hours_with_01_to_11_missing_and_00_12_observed'].iloc[0])}, 'days_with_20plus_hour_blocks_missing': len(top_days), 'such_days_by_month': top_days.groupby(top_days.date.dt.to_period('M').astype(str)).size().to_dict(), 'relative_displacement_medians_WY14minusWY12': {'2022-01-01_to_04': float(rel.loc['2022-01-01':'2022-01-04'].median()), '2022-01-10_to_20': float(rel.loc['2022-01-10':'2022-01-20'].median())}, 'source_examples_checked': len(records), 'all_source_examples_match': all((r['numeric_match'] for r in records)), 'limits': 'Periodic missingness suggests a systematic process but does not identify whether collection, transmission or export caused it. Relative offset change is descriptive, not a confirmed calibration reset.'}
(out / 'additional_evidence.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(meta, ensure_ascii=False, indent=2))
