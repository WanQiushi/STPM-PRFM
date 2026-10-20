"""Descriptive audit only; never modifies or imputes observations."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
root = Path(__file__).resolve().parents[1]
out = root / 'reports' / '20260922_data_audit'
out.mkdir(parents=True, exist_ok=True)
z = np.load(root / 'data/processed/minute_observed.npz', allow_pickle=False)
t = pd.DatetimeIndex(z['timestamps'])
values = z['values']
channels = z['channels'].tolist()
df = pd.DataFrame(values, index=t, columns=channels)
good = np.isfinite(values)
gap = pd.read_csv(root / 'data/quality/missing_intervals.csv.gz')
source_quality = pd.read_csv(root / 'data/quality/channel_quality.csv').set_index('sensor_id')

def runs(mask):
    d = np.diff(np.r_[False, mask, False].astype(np.int8))
    return (np.flatnonzero(d == 1), np.flatnonzero(d == -1))
summaries = []
gap_rows = []
point_rows = []
event_rows = []
flat_rows = []
contexts = []
candidate_masks = np.zeros_like(good)
jump_masks = np.zeros_like(good)
for j, k in enumerate(channels):
    s = df[k]
    x = values[:, j]
    finite = good[:, j]
    dx = s.diff()
    valid_dx = dx.dropna().to_numpy()
    center = float(np.median(valid_dx))
    scale = float(1.4826 * np.median(np.abs(valid_dx - center)))
    floor = 0.5 if j == 0 else 2.0
    jump_threshold = max(floor, 12 * scale)
    jump = (dx.abs() > jump_threshold).to_numpy()
    jump_masks[:, j] = jump
    baseline = s.rolling(31, center=True, min_periods=15).median()
    residual = s - baseline
    local_scale = 1.4826 * residual.abs().rolling(31, center=True, min_periods=15).median()
    local_threshold = np.maximum(8 * local_scale.to_numpy(), floor)
    candidate = np.abs(residual.to_numpy()) > local_threshold
    candidate_masks[:, j] = candidate
    left = s.shift(1).to_numpy()
    right = s.shift(-1).to_numpy()
    isolated = candidate & ~np.r_[False, candidate[:-1]] & ~np.r_[candidate[1:], False]
    impulse = isolated & np.isfinite(left) & np.isfinite(right) & (np.abs(left - right) < floor / 2)
    change = np.r_[True, ~(np.isfinite(x[1:]) & np.isfinite(x[:-1]) & (x[1:] == x[:-1]))]
    flat_starts = np.flatnonzero(change)
    flat_ends = np.r_[flat_starts[1:], len(x)]
    flat_lengths = flat_ends - flat_starts
    long_flats = (flat_lengths >= 60) & np.isfinite(x[flat_starts])
    for a, b in zip(flat_starts[long_flats], flat_ends[long_flats]):
        flat_rows.append(dict(sensor_id=k, start=str(t[a]), end=str(t[b - 1]), minutes=int(b - a), value=float(x[a])))
    gaps = gap[gap.sensor_id == k].copy()
    for label, lo, hi in [('1-5 min', 1, 5), ('6-30 min', 6, 30), ('31-120 min', 31, 120), ('121-1440 min', 121, 1440), ('>1440 min', 1441, len(x))]:
        part = gaps[gaps.minutes.between(lo, hi)]
        gap_rows.append(dict(sensor_id=k, gap_length=label, episodes=len(part), missing_minutes=int(part.minutes.sum())))
    sorted_gaps = gaps.sort_values('minutes', ascending=False)
    p = np.flatnonzero(candidate)
    point_rows.extend((dict(sensor_id=k, timestamp=str(t[a]), value=float(x[a]), local_median=float(baseline.iloc[a]), residual=float(residual.iloc[a]), screen_threshold=float(local_threshold[a]), single_minute_return=bool(impulse[a])) for a in p))
    cuts = np.r_[0, np.flatnonzero(np.diff(p) > 5) + 1, len(p)]
    for a, b in zip(cuts[:-1], cuts[1:]):
        ids = p[a:b]
        if len(ids) == 0:
            continue
        peak = ids[np.argmax(np.abs(residual.iloc[ids].to_numpy()))]
        event_rows.append(dict(sensor_id=k, start=str(t[ids[0]]), end=str(t[ids[-1]]), span_minutes=int(ids[-1] - ids[0] + 1), candidate_points=len(ids), peak_time=str(t[peak]), peak_value=float(x[peak]), max_abs_residual=float(abs(residual.iloc[peak])), strict_single_minute_returns=int(impulse[ids].sum())))
    extreme_indices = [int(np.nanargmin(x)), int(np.nanargmax(x))]
    dx_ids = np.flatnonzero(np.isfinite(dx.to_numpy()))
    ordered = dx_ids[np.argsort(np.abs(dx.iloc[dx_ids].to_numpy()))[::-1]]
    chosen = []
    for a in ordered:
        if all((abs(a - b) > 60 for b in chosen)):
            chosen.append(int(a))
        if len(chosen) == 8:
            break
    for a in sorted(set(chosen + extreme_indices)):
        pre = s.iloc[max(0, a - 30):a]
        post = s.iloc[a + 1:min(len(s), a + 31)]
        contexts.append(dict(sensor_id=k, timestamp=str(t[a]), index=a, value=float(x[a]), previous_value=float(x[a - 1]) if a else None, next_value=float(x[a + 1]) if a + 1 < len(x) else None, change_from_previous=float(dx.iloc[a]), pre30_median=float(pre.median()), post30_median=float(post.median()), pre30_n=int(pre.count()), post30_n=int(post.count()), local_residual=float(residual.iloc[a]), candidate=bool(candidate[a]), all_channel_changes={c: float(df[c].diff().iloc[a]) for c in channels}))
    longest_flat_index = flat_starts[np.argmax(np.where(np.isfinite(x[flat_starts]), flat_lengths, 0))]
    summaries.append(dict(sensor_id=k, unit='degC' if j == 0 else 'mm', missing_minutes=int((~finite).sum()), missing_pct=float((~finite).mean() * 100), gap_episodes=len(gaps), longest_gap=sorted_gaps.iloc[0].to_dict(), min=float(s.min()), max=float(s.max()), q001=float(s.quantile(0.001)), q999=float(s.quantile(0.999)), min_time=str(t[np.nanargmin(x)]), max_time=str(t[np.nanargmax(x)]), diff_robust_scale=scale, jump_threshold=jump_threshold, jump_count=int(jump.sum()), local_candidate_count=int(candidate.sum()), local_candidate_pct_of_observed=float(candidate.sum() / finite.sum() * 100), strict_single_minute_return_count=int(impulse.sum()), local_threshold_floor=floor, longest_exact_flat_minutes=int(np.max(np.where(np.isfinite(x[flat_starts]), flat_lengths, 0))), flat_runs_ge60=int(long_flats.sum()), longest_flat_start=str(t[longest_flat_index]), duplicate_rows_removed=int(source_quality.loc[k, 'duplicate_rows_removed'])))
    print(k, 'gaps', len(gaps), 'local_candidates', int(candidate.sum()), 'jump_count', int(jump.sum()), 'flats>=60', int(long_flats.sum()), flush=True)
pd.DataFrame(gap_rows).to_csv(out / 'gap_length_distribution.csv', index=False)
pd.DataFrame(point_rows).to_csv(out / 'candidate_points.csv.gz', index=False, compression='gzip')
events = pd.DataFrame(event_rows).sort_values('max_abs_residual', ascending=False)
events.to_csv(out / 'candidate_events.csv', index=False)
pd.DataFrame(flat_rows, columns=['sensor_id', 'start', 'end', 'minutes', 'value']).to_csv(out / 'flatline_candidates.csv', index=False)
pd.DataFrame([{k: v for k, v in c.items() if k != 'all_channel_changes'} for c in contexts]).to_csv(out / 'extreme_and_jump_context.csv', index=False)
(out / 'extreme_and_jump_context.json').write_text(json.dumps(contexts, ensure_ascii=False, indent=2), encoding='utf-8')
pd.DataFrame([{k: v for k, v in d.items() if k != 'longest_gap'} for d in summaries]).to_csv(out / 'channel_audit.csv', index=False)
monthly_counts = pd.DataFrame(candidate_masks, index=t, columns=channels).groupby(t.to_period('M')).sum()
monthly_counts.to_csv(out / 'monthly_candidate_counts.csv')
daily_aggs = df.resample('D').agg(['min', 'median', 'max', 'count'])
daily_aggs.columns = ['_'.join(c) for c in daily_aggs.columns]
daily_aggs.to_csv(out / 'daily_value_summary.csv.gz', compression='gzip')
diff_mad = df.diff().resample('D').apply(lambda x: np.nanmedian(np.abs(x - np.nanmedian(x))) * 1.4826)
diff_mad.to_csv(out / 'daily_difference_mad.csv')
missing_count = (~good).sum(axis=1)
all_missing = missing_count == 7
a, b = runs(all_missing)
joint_outages = pd.DataFrame([dict(start=str(t[i]), end=str(t[e - 1]), minutes=int(e - i)) for i, e in zip(a, b)])
joint_outages.sort_values('minutes', ascending=False).to_csv(out / 'all_seven_missing_intervals.csv', index=False)
result = {'dataset_version': 'monitored_bridge_downstream6_minute_observed_v1_20260922', 'read_only': True, 'channel_summary': summaries, 'minutes_by_missing_channel_count': {str(k): int((missing_count == k).sum()) for k in range(8)}, 'all_seven_missing_minutes': int(all_missing.sum()), 'all_seven_missing_episodes': len(a), 'top_joint_outages': joint_outages.sort_values('minutes', ascending=False).head(10).to_dict('records'), 'candidate_points_total': int(candidate_masks.sum()), 'candidate_event_count': len(events), 'candidate_rule': '31-minute centered median, minimum 15 finite values. Absolute residual > max(8*1.4826*31-minute rolling median absolute residual, floor). Floors: 0.5 degC or 2 mm. Descriptive candidate screening only, not a sensor specification or confirmed error label.', 'jump_rule': 'Absolute adjacent-minute difference > max(12*1.4826*MAD(all valid adjacent differences), 0.5 degC or 2 mm). Missing intervals are never bridged.', 'flatline_rule': 'At least 60 consecutive finite minute values exactly equal; quantization can produce valid equal readings.', 'leakage_note': 'Centered windows use future neighbors for offline diagnosis only; not preprocessing for forecasting. No training/test split or cleaning policy has been adopted.', 'threshold_note': 'Screening thresholds are analyst-selected heuristics, not validated physical limits. Flags are candidates; persistent offsets and gap-adjacent artifacts may evade this screen.'}
(out / 'audit_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in result.items() if k != 'channel_summary'}, ensure_ascii=False, indent=2), flush=True)
