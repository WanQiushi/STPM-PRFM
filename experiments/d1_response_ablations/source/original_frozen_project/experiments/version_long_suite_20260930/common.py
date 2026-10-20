from if_release.runtime import loads as _release_json_loads
import json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / 'protocol.json'

def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def load_data(cfg):
    d = np.load(ROOT / cfg['data'])
    values = d['values'].astype(np.float64)
    observed = d['valid_observed'].astype(bool)
    available = d['input_available_at']
    train_end = cfg['split_days'][0] * 1440
    measured = np.where(observed[:train_end], values[:train_end], np.nan)
    mean, std = (np.nanmean(measured, axis=0), np.nanstd(measured, axis=0))
    history_offsets = np.arange(-cfg['history_days'] * 1440, 0, cfg['sample_minutes'])
    future_offsets = np.arange(0, cfg['horizon_days'] * 1440, cfg['sample_minutes'])
    return (values, observed, available, mean, std, history_offsets, future_offsets)

def make_protocol(cfg):
    values, observed, available, mean, std, ho, fo = load_data(cfg)
    ends = np.cumsum(cfg['split_days']) * 1440
    origins, report = ({}, {})
    for name, first, last, stride in [('train', cfg['history_days'] * 1440, ends[0], cfg['train_stride']), ('val', ends[0], ends[1], 1440), ('test', ends[1], ends[2], 1440)]:
        candidates = np.arange(first, last - cfg['horizon_days'] * 1440 + 1, stride)
        selected = []
        causal = temperature = 0
        for origin in candidates:
            if available[origin + ho].max() >= origin:
                continue
            causal += 1
            if not observed[origin + fo, 0].all():
                continue
            temperature += 1
            selected.append(int(origin))
        origins[name] = selected
        report[name] = dict(candidates=len(candidates), causal_history=causal, measured_future_temperature=temperature, selected=len(selected))
    protocol = dict(config=cfg, channels=['Temperature', 'Pier1', 'Pier2', 'Pier3', 'Pier4'], origins=origins, counts=report, mean=mean.tolist(), std=std.tolist(), task='known future measured temperature -> four displacement responses', selection='validation only; no test-driven model or checkpoint selection', test='all eligible daily origins in the chronological 109-day test interval')
    dump(PROTOCOL, protocol)
    print('PROTOCOL ' + json.dumps(protocol['counts']), flush=True)
    return protocol

def prepare(cfg):
    values, observed, _, mean, std, ho, fo = load_data(cfg)
    p = _release_json_loads(PROTOCOL.read_text(encoding='utf-8'))
    parts = {k: p['origins'][k] for k in ['train', 'val']}
    return (((values - mean) / std).astype('float32'), observed, parts, mean, std, {k: p['counts'][k] for k in parts}, ho, fo)

def metrics(npz_path):
    d = np.load(npz_path)
    mask = d['mask'].astype(bool)
    std = d['std'][1:]
    err = (d['prediction'].astype(np.float64) - d['target']) * std
    sums = np.sum(err ** 2 * mask, axis=(0, 1))
    counts = mask.sum(axis=(0, 1))
    peak = []
    trough = []
    for c in range(4):
        peaks = []
        troughs = []
        for i in range(len(mask)):
            ix = np.flatnonzero(mask[i, :, c])
            if len(ix) == 0:
                continue
            y = d['target'][i, ix, c]
            p = d['prediction'][i, ix, c]
            peaks.append(float(abs(p.max() - y.max()) * std[c]))
            troughs.append(float(abs(p.min() - y.min()) * std[c]))
        peak.append(float(np.mean(peaks)))
        trough.append(float(np.mean(troughs)))
    item = dict(pooled_rmse_mm=float(np.sqrt(sums.sum() / counts.sum())), rmse_mm=np.sqrt(sums / counts).tolist(), macro_rmse_mm=float(np.mean(np.sqrt(sums / counts))), mae_mm=(np.sum(np.abs(err) * mask, axis=(0, 1)) / counts).tolist(), measured_counts=counts.tolist(), windows=len(d['origin']), peak_mae_mm=peak, trough_mae_mm=trough, tail_2h_rmse_mm=np.sqrt(np.sum(err[:, -8:] ** 2 * mask[:, -8:], axis=(0, 1)) / mask[:, -8:].sum(axis=(0, 1))).tolist())
    for key, label in [('temperature_mf', 'u'), ('effective_temperature_mf', 'q')]:
        x = d[key]
        if x.shape[-1] == 1:
            x = np.repeat(x, 4, axis=2)
        p = d['mf_prediction']
        y = d['mf_target']

        def area(z):
            return 0.5 * np.sum(x * np.roll(z, -1, axis=1) - np.roll(x, -1, axis=1) * z, axis=1)
        item[label + '_area_ratio'] = (np.abs(area(p)).sum(0) / np.abs(area(y)).sum(0)).tolist()
        ts = np.zeros(4)
        ps = np.zeros(4)
        for i in range(len(x)):
            for c in range(4):
                xx = x[i, :, c]
                for level in np.linspace(xx.min(), xx.max(), 32)[1:-1]:
                    up = (xx[:-1] <= level) & (xx[1:] > level)
                    down = (xx[:-1] >= level) & (xx[1:] < level)
                    if not up.any() or not down.any():
                        continue
                    crossings = []
                    for direction in [up, down]:
                        f = (level - xx[:-1][direction]) / (xx[1:][direction] - xx[:-1][direction])
                        crossings.append([(z[i, :-1, c][direction] + f * (z[i, 1:, c][direction] - z[i, :-1, c][direction])).mean() for z in [y, p]])
                    ts[c] += abs(crossings[0][0] - crossings[1][0]) * std[c]
                    ps[c] += abs(crossings[0][1] - crossings[1][1]) * std[c]
        item[label + '_branch_width_ratio'] = (ps / ts).tolist()
        item[label + '_target_branch_sum_mm'] = ts.tolist()
        item[label + '_prediction_branch_sum_mm'] = ps.tolist()
    item['mf_range_ratio'] = (np.ptp(d['mf_prediction'], axis=1).sum(0) / np.ptp(d['mf_target'], axis=1).sum(0)).tolist()
    hf = (d['target'] - d['lf_target'] - d['mf_target']) * std
    item['target_hf_end_abs_mean_mm'] = np.mean(np.abs(hf[:, -1]), axis=0).tolist()
    return item
