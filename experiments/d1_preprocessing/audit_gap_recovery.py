from pathlib import Path
import json
import numpy as np
import pandas as pd
root = Path(__file__).resolve().parents[1]
out = root / 'reports/20260922_data_audit'
z = np.load(root / 'data/processed/minute_observed.npz')
t = pd.DatetimeIndex(z['timestamps'])
x = z['values']
channels = z['channels'].tolist()
gaps = pd.read_csv(root / 'data/quality/missing_intervals.csv.gz', parse_dates=['start', 'end'])
records = []
for j, k in enumerate(channels):
    g = gaps[gaps.sensor_id == k]
    for row in g.itertuples():
        a = int(t.get_loc(row.start))
        b = int(t.get_loc(row.end)) + 1
        if a == 0 or b + 1 >= len(t):
            continue
        before, first, next_value = (x[a - 1, j], x[b, j], x[b + 1, j])
        threshold = 0.5 if j == 0 else 2.0
        if np.isfinite(before) and np.isfinite(first) and np.isfinite(next_value) and (abs(next_value - first) > threshold):
            records.append(dict(sensor_id=k, gap_start=str(t[a]), gap_end=str(t[b - 1]), gap_minutes=b - a, recovery_time=str(t[b]), before_gap_value=float(before), first_value=float(first), next_value=float(next_value), first_equals_before=bool(first == before), first_to_next_change=float(next_value - first)))
r = pd.DataFrame(records).sort_values('first_to_next_change', key=lambda s: s.abs(), ascending=False)
r.to_csv(out / 'gap_recovery_candidates.csv', index=False)
summary = {'all_candidates': len(r), 'first_equals_last_before_gap': int(r.first_equals_before.sum()), 'by_channel': r.groupby('sensor_id').agg(candidates=('sensor_id', 'size'), same_value_as_before_gap=('first_equals_before', 'sum')).to_dict('index'), 'interpretation': 'Identical value before gap and at recovery followed by a large next-minute change is a stale-record candidate, not proof of a sensor mechanism. No values modified.'}
(out / 'gap_recovery_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(r.head(18).to_string(index=False))
print(json.dumps(summary))
