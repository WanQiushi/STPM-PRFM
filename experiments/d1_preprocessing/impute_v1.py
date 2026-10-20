"""Engineering offline completion: short linear gaps, selected medium gaps LATC."""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from scipy.linalg import solve_banded
root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--stage', choices=['linear', 'latc'], required=True)
parser.add_argument('--output-name', default='imputation_v1')
parser.add_argument('--gpu', type=int)
args = parser.parse_args()
out = root / 'data' / args.output_name
out.mkdir(parents=True, exist_ok=True)
if args.gpu is not None:
    import cupy as cp
    cp.cuda.Device(args.gpu).use()
source = root / 'data/cleaning_preview_v1/minute_anomaly_removed.npz'

def runs(m):
    d = np.diff(np.r_[False, m, False].astype(np.int8))
    return zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))

def linear_stage():
    z = np.load(source, allow_pickle=False)
    x = z['values'].copy()
    valid = np.isfinite(x)
    timestamps = z['timestamps']
    method = np.zeros(x.shape, dtype=np.uint8)
    eligible = np.zeros(x.shape, dtype=bool)
    joint = np.zeros(len(x), dtype=bool)
    for a, b in runs((~valid).sum(axis=1) >= 2):
        if b - a > 15:
            joint[a:b] = True
    rows = []
    for j, k in enumerate(z['channels']):
        for a, b in runs(~valid[:, j]):
            n = b - a
            if a == 0 or b == len(x):
                status = 'boundary_unfilled'
            elif n <= 15:
                x[a:b, j] = np.linspace(x[a - 1, j], x[b, j], n + 2)[1:-1]
                method[a:b, j] = 1
                status = 'linear'
            elif n <= 1440 and (not joint[a:b].any()):
                eligible[a:b, j] = True
                status = 'latc_pending'
            elif n > 1440:
                status = 'over_24h_unfilled'
            else:
                status = 'joint_long_outage_unfilled'
            rows.append({'sensor_id': str(k), 'start': str(timestamps[a]), 'end': str(timestamps[b - 1]), 'minutes': n, 'status': status})
    payload = {k: z[k] for k in z.files}
    payload.update(values=x, valid_observed=valid, available=np.isfinite(x), imputation_method=method, latc_eligible=eligible)
    np.savez_compressed(out / 'minute_linear.npz', **payload)
    gaps = pd.DataFrame(rows)
    gaps.to_csv(out / 'gap_actions.csv.gz', index=False, compression='gzip')
    summary = {'linear_points': int((method == 1).sum()), 'latc_pending_points': int(eligible.sum()), 'remaining_missing': int(np.isnan(x).sum()), 'by_status': gaps.groupby('status').minutes.agg(['count', 'sum']).to_dict('index')}
    (out / 'linear_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)

def unfold(x, mode):
    return np.reshape(np.moveaxis(x, mode, 0), (x.shape[mode], -1), order='F')

def fold(x, shape, mode):
    axes = [mode] + [i for i in range(3) if i != mode]
    return np.moveaxis(x.reshape([shape[i] for i in axes], order='F'), 0, mode)

def prox(x, tau, r):
    if args.gpu is not None:
        u, s, v = cp.linalg.svd(cp.asarray(x), full_matrices=False)
        s[r:] = cp.maximum(s[r:] - tau, 0)
        return cp.asnumpy(u * s @ v)
    u, s, v = np.linalg.svd(x, full_matrices=False)
    s[r:] = np.maximum(s[r:] - tau, 0)
    return u * s @ v

def latc(data, known):
    shape = data.shape
    Y = unfold(data, 0)
    obs = unfold(known, 0)
    Z = np.where(obs, Y, 0.0)
    W = np.zeros(shape)
    A = np.full((7, 2), 0.001)
    rho = 0.01
    gamma = 0.01
    r = 2
    previous = Z.copy()
    n = Z.shape[1]
    for iteration in range(100):
        for inner in range(3):
            rho = min(rho * 1.05, 100000.0)
            target = fold(Z, shape, 0) - W / rho
            X = sum((fold(prox(unfold(target, k), 1 / (3 * rho), r), shape, k) / 3 for k in range(3)))
            rhs = rho * unfold(X + W / rho, 0)
            for j in range(7):
                a1, a2 = A[j]
                diag = np.full(n, rho)
                diag[2:] += gamma
                diag[1:-1] += gamma * a1 * a1
                diag[:-2] += gamma * a2 * a2
                off1 = np.zeros(n - 1)
                off1[1:] -= gamma * a1
                off1[:-1] += gamma * a1 * a2
                off2 = np.full(n - 2, -gamma * a2)
                band = np.zeros((5, n))
                band[2] = diag
                band[1, 1:] = off1
                band[3, :-1] = off1
                band[0, 2:] = off2
                band[4, :-2] = off2
                estimate = solve_banded((2, 2), band, rhs[j])
                Z[j, ~obs[j]] = estimate[~obs[j]]
            W += rho * (X - fold(Z, shape, 0))
        for j in range(7):
            A[j] = np.linalg.lstsq(np.column_stack([Z[j, 1:-1], Z[j, :-2]]), Z[j, 2:], rcond=None)[0]
        tol = np.linalg.norm(Z - previous) / np.linalg.norm(previous)
        previous = Z.copy()
        if tol < 0.0001:
            break
    return (Z, iteration + 1, float(tol))

def latc_stage():
    z = np.load(out / 'minute_linear.npz', allow_pickle=False)
    x = z['values'].copy()
    original = np.load(source, allow_pickle=False)['values']
    eligible = z['latc_eligible']
    method = z['imputation_method'].copy()
    records = []
    total_days = len(x) // 1440
    for day in range(0, total_days, 7):
        end = min(day + 7, total_days)
        a = day * 1440
        b = end * 1440
        target_mask = eligible[a:b]
        if not target_mask.any():
            continue
        first = max(0, day - 7)
        last = max(end, min(14, total_days))
        lo = first * 1440
        hi = last * 1440
        print('START_BLOCK ' + json.dumps({'day_start': day, 'reference_start': first, 'reference_end': last, 'points': int(target_mask.sum())}), flush=True)
        raw = original[lo:hi]
        known = np.isfinite(raw)
        center = np.nanmedian(raw, axis=0)
        scale = np.nanstd(raw, axis=0)
        normalized = (raw - center) / scale
        shape = (7, 1440, last - first)
        data = normalized.T.reshape(shape, order='F')
        obs = known.T.reshape(shape, order='F')
        estimate, iterations, tol = latc(data, obs)
        restored = estimate.T * scale + center
        part = restored[a - lo:b - lo]
        x[a:b][target_mask] = part[target_mask]
        method[a:b][target_mask] = 2
        rec = {'output_day_start': day, 'output_day_end_exclusive': end, 'reference_day_start': first, 'reference_day_end_exclusive': last, 'filled_points': int(target_mask.sum()), 'iterations': iterations, 'relative_change': tol, 'finite_estimates': bool(np.isfinite(part[target_mask]).all())}
        records.append(rec)
        print(json.dumps(rec), flush=True)
        pd.DataFrame(records).to_csv(out / 'latc_progress.csv', index=False)
    payload = {k: z[k] for k in z.files}
    payload.update(values=x, available=np.isfinite(x), imputation_method=method)
    np.savez_compressed(out / 'minute_imputed.npz', **payload)
    summary = {'linear_points': int((method == 1).sum()), 'latc_points': int((method == 2).sum()), 'remaining_missing': int(np.isnan(x).sum()), 'finite_filled_values': bool(np.isfinite(x[method > 0]).all()), 'unchanged_trusted_observations': bool(np.array_equal(x[np.isfinite(original)], original[np.isfinite(original)])), 'channels': [{'sensor_id': str(k), 'linear': int((method[:, j] == 1).sum()), 'latc': int((method[:, j] == 2).sum()), 'remaining_missing': int(np.isnan(x[:, j]).sum())} for j, k in enumerate(z['channels'])]}
    (out / 'completion_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('COMPLETED ' + json.dumps(summary), flush=True)
if args.stage == 'linear':
    linear_stage()
else:
    latc_stage()
