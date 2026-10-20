"""Offline anomaly-removal preview; no imputation or forecasting samples."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from PIL import Image
root = Path(__file__).resolve().parents[1]
out = root / 'data/cleaning_preview_v1'
figdir = root / 'reports/20260922_cleaning_preview/figures'
out.mkdir(parents=True, exist_ok=True)
figdir.mkdir(parents=True, exist_ok=True)
z = np.load(root / 'data/processed/minute_observed.npz', allow_pickle=False)
t = pd.DatetimeIndex(z['timestamps'])
x = z['values']
channels = z['channels'].tolist()
mask = np.zeros_like(x, dtype=bool)
reason = np.zeros_like(x, dtype=np.uint8)
points = pd.read_csv(root / 'reports/20260922_data_audit/candidate_points.csv.gz')
for row in points[points.single_minute_return == True].itertuples():
    i = t.get_loc(pd.Timestamp(row.timestamp))
    j = channels.index(row.sensor_id)
    mask[i, j] = True
    reason[i, j] = 1
events = [('Pier4', '2022-01-05 13:14', '2022-01-08 00:45', '持续失真，含进入异常的过渡段'), ('Pier4', '2023-09-01 00:47', '2023-09-01 00:48', '两分钟突跳后恢复'), ('Auxiliary1', '2023-02-03 09:26', '2023-02-03 09:30', '五分钟负向突跳后恢复'), ('Pier3', '2022-11-23 17:18', '2022-11-23 17:18', '断测恢复首点异常跳变')]
for k, a, b, label in events:
    j = channels.index(k)
    m = (t >= a) & (t <= b) & np.isfinite(x[:, j])
    mask[m, j] = True
    reason[m, j] = 2
y = x.copy()
y[mask] = np.nan
payload = {k: z[k] for k in z.files}
payload.update(values=y, anomaly_mask=mask, valid_observed=np.isfinite(y), anomaly_reason=reason)
np.savez_compressed(out / 'minute_anomaly_removed.npz', **payload)
i, j = np.where(mask)
pd.DataFrame({'timestamp': t[i], 'sensor_id': np.array(channels)[j], 'original_value': x[i, j], 'reason_code': reason[i, j]}).to_csv(out / 'removed_points.csv.gz', index=False, compression='gzip')
pd.DataFrame(events, columns=['sensor_id', 'start', 'end', 'reason']).to_csv(out / 'selected_events.csv', index=False, encoding='utf-8-sig')
summary = pd.DataFrame({'sensor_id': channels, 'removed_points': mask.sum(axis=0), 'original_missing': np.isnan(x).sum(axis=0), 'after_missing': np.isnan(y).sum(axis=0), 'retained_min': np.nanmin(y, axis=0), 'retained_max': np.nanmax(y, axis=0)})
summary.to_csv(out / 'summary.csv', index=False)
result = {'version': 'anomaly_removed_preview_v1_20260922', 'removed_total': int(mask.sum()), 'shape': list(y.shape), 'imputed': False, 'unchanged_retained_values': bool(np.array_equal(x[~mask], y[~mask], equal_nan=True)), 'unchanged_timestamps': True, 'channel_summary': summary.to_dict('records'), 'rule': 'Strict single-minute returns from existing audit: 31-minute median residual exceeds max(8 local robust scales, 0.5 degC/2 mm), adjacent points finite and differ by less than 0.25 degC/1 mm; plus four explicitly listed event intervals.', 'boundary_note': 'January start 13:14 includes the transition before the large jump at 13:22; engineering screening choice for review, not verified sensor-fault onset.', 'scope': 'Offline visual review only; centered detection is not a causal forecasting input protocol.'}
(out / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
plt.rcParams.update({'font.family': 'Microsoft YaHei', 'font.size': 10, 'axes.unicode_minus': False, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.18, 'pdf.fonttype': 42})
raw = pd.DataFrame(x, index=t, columns=channels)
clean = pd.DataFrame(y, index=t, columns=channels)
blue = '#0072B2'
orange = '#D55E00'

def save(fig, name):
    fig.savefig(figdir / (name + '.png'), dpi=150)
    fig.savefig(figdir / (name + '.pdf'))
    Image.open(figdir / (name + '.png')).convert('L').save(figdir / (name + '_gray.png'))
    plt.close(fig)
fig, axes = plt.subplots(7, 2, figsize=(13, 14), sharex=True, layout='constrained')
for col, frame in enumerate([raw, clean]):
    daily = frame.resample('D')
    med = daily.median()
    lo = daily.min()
    hi = daily.max()
    for j, k in enumerate(channels):
        ax = axes[j, col]
        ax.fill_between(med.index, lo[k], hi[k], color=blue, alpha=0.25)
        ax.plot(med.index, med[k], color=blue, lw=0.7)
        ax.set_title(k + (' 原始' if col == 0 else f' 去异常（移除{mask[:, j].sum()}点）'), loc='left', fontsize=10)
        ax.set_ylabel('温度 / ℃' if j == 0 else '位移 / mm')
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
fig.suptitle('七通道去异常前后总览｜左：原始，右：去异常，尚未补齐\n线为每日中位数，色带为每日最小—最大值；左右纵轴独立缩放', fontsize=13)
save(fig, '01_before_after_overview')
windows = [('Pier4', '2022-01-04', '2022-01-08 12:00', '2022年1月：持续失真'), ('Auxiliary1', '2023-02-03 09:10', '2023-02-03 09:45', '2023-02-03：短时负向突跳'), ('Pier4', '2023-09-01 00:30', '2023-09-01 01:00', '2023-09-01：两分钟尖峰'), ('Pier3', '2022-11-23 17:10', '2022-11-23 17:35', '2022-11-23：恢复首点')]
fig, axes = plt.subplots(4, 2, figsize=(13, 10), layout='constrained')
for r, (k, a, b, title) in enumerate(windows):
    j = channels.index(k)
    v = raw.loc[a:b, k]
    w = clean.loc[a:b, k]
    m = mask[(t >= a) & (t <= b), j]
    axes[r, 0].plot(v.index, v, color='0.45', lw=0.8)
    axes[r, 0].scatter(v.index[m], v.to_numpy()[m], color=orange, s=8, marker='x', label='本次移除')
    axes[r, 1].plot(w.index, w, color=blue, lw=1, marker='.', ms=3)
    for c in [0, 1]:
        ax = axes[r, c]
        ax.set_ylabel('位移 / mm')
        ax.set_title(k + '｜' + title + ('（原始）' if c == 0 else '（去异常）'), fontsize=10, loc='left')
        ax.set_xlim(pd.Timestamp(a), pd.Timestamp(b))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))
    axes[r, 0].legend(loc='upper left', fontsize=8, frameon=False)
fig.suptitle('主要异常区段：逐分钟原值对比｜右图断线即缺失，未做插值\n左右纵轴独立缩放，以便查看保留曲线', fontsize=13)
save(fig, '02_event_details')
fig, axes = plt.subplots(7, 1, figsize=(12, 12), sharex=True, layout='constrained')
for j, k in enumerate(channels):
    v = clean.loc['2023-02-01':'2023-02-07', k]
    axes[j].plot(v.index, v, color=blue, lw=0.7)
    axes[j].set_title(k, loc='left', fontsize=10)
    axes[j].set_ylabel('℃' if j == 0 else 'mm')
axes[-1].xaxis.set_major_locator(mdates.DayLocator())
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
fig.suptitle('去异常后的七通道分钟曲线｜2023-02-01 至 02-07\n原缺测与本次移除位置均保留为空缺，未平滑、未补齐', fontsize=13)
save(fig, '03_clean_week')
print(json.dumps(result, ensure_ascii=True, indent=2))
