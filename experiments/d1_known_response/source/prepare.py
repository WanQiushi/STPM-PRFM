"""Build a distinct observed-future-temperature D1 experiment dataset."""
from if_release.runtime import expand as _release_expand, loads as _release_json_loads
from pathlib import Path
from datetime import datetime, timedelta
import json
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
OUT = HERE.parent
ROOT = Path(_release_expand('${WORKSPACE}'))
ORIGINAL = ROOT / 'outputs/va_1_20261004'

def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')

def main():
    data_dir = OUT / 'dataset'
    data_dir.mkdir()
    checkpoint = torch.load(ORIGINAL / 'h1_original.pt', map_location='cpu', weights_only=False)
    config = checkpoint['config']
    norm = _release_json_loads((ORIGINAL / 'normalization.json').read_text())
    mean, std = (np.asarray(norm['h1_mean']), np.asarray(norm['h1_std']))
    source = ROOT / 'data/mainbridge_temperature_v2/minute_complete.npz'
    raw = np.load(source)
    values = raw['values']
    observed = raw['valid_observed']
    availability = raw['input_available_at']
    origins = {}
    reports = []
    for split, start, stop, stride in [('train', 43200, 511 * 1440, config['train_stride']), ('validation', 511 * 1440, 621 * 1440, 1440), ('test', 621 * 1440, 730 * 1440, 1440)]:
        origins[split] = [int(o) for o in range(start, stop - 1440 + 1, stride) if observed[o:o + 1440, 0].all()]
        grids = [15] if split == 'train' else [15, 1]
        for minutes in grids:
            history = np.lib.format.open_memmap(data_dir / f'{split}_history_{minutes}min.npy', mode='w+', dtype=np.float32, shape=(len(origins[split]), 43200 // minutes, 5))
            temperatures, targets, masks = ([], [], [])
            for index, origin in enumerate(origins[split]):
                h = values[origin - 43200:origin].copy()
                bad = availability[origin - 43200:origin] >= origin
                for channel in range(5):
                    existing = np.flatnonzero(~bad[:, channel])
                    missing = np.flatnonzero(bad[:, channel])
                    h[missing, channel] = np.interp(missing, existing, h[existing, channel])
                history[index] = ((h[::minutes] - mean) / std).astype(np.float32)
                future = values[origin:origin + 1440:minutes]
                temperatures.append(((future[:, :1] - mean[:1]) / std[:1]).astype(np.float32))
                targets.append(((future[:, 1:] - mean[1:]) / std[1:]).astype(np.float32))
                masks.append(observed[origin:origin + 1440:minutes, 1:])
                if minutes == 15:
                    reports.append(dict(split=split, origin=origin, date=(datetime(2022, 1, 1) + timedelta(minutes=origin)).isoformat(), historical_entries_reconstructed=bad.sum(0).tolist(), historical_measured_fraction=observed[origin - 43200:origin].mean(0).tolist(), historical_temperature_fully_measured=bool(observed[origin - 43200:origin, 0].all()), measured_response_nodes=observed[origin:origin + 1440, 1:].sum(0).tolist()))
            history.flush()
            np.savez_compressed(data_dir / f'{split}_{minutes}min.npz', temperature=np.stack(temperatures), target=np.stack(targets), mask=np.stack(masks), origin=np.asarray(origins[split]), mean=mean, std=std)
    dump(data_dir / 'origins.json', origins)
    dump(data_dir / 'manifest.json', dict(source=str(source), source_version='mainbridge_temperature_v2', split_days=[511, 110, 109], counts={s: len(o) for s, o in origins.items()}, history_days=30, future_minutes=1440, train_stride_minutes=config['train_stride'], future_temperature_policy='Every one of the 1440 future minute temperatures must be valid_observed.', history_policy='Original as-of available values retained; unavailable history linearly interpolated within the same past30day window, edge-held, exactly the full219 policy.', labels=['Temperature', 'Pier1', 'Pier2', 'Pier3', 'Pier4'], mean=mean.tolist(), std=std.tolist(), normalization_source=str(ORIGINAL / 'normalization.json'), days=reports))
    data_dir.joinpath('README.md').write_text(_release_expand("# D1 observed future temperature mechanism experiment\n\nSource: ${WORKSPACE}/data/mainbridge_temperature_v2/minute_complete.npz.\nConstruction: ../source/prepare.py; original dataset and frozen weights unchanged.\nAll 1440 future temperature minutes must be retained measured values; no future temperature interpolation.\n30day history is reconstructed only where original input_available_at >= origin, using available past-window anchors.\nHistory therefore includes imputed records; this is not a fully measured 30day-history experiment.\nTrain/validation/test time boundaries: 511/110/109days, minutes since2022-01-01.\nTrain origins every720min; validation/test daily midnight; exact common lists:origins.json.\nTrain15min arrays, validation/test15min and1min arrays. All channels use the frozen training normalization.\nhistory arrays:N x (43200/minutes) x5; other NPZ fields:temperature N x (1440/minutes) x1,\ntarget N x (1440/minutes) x4, mask same as target, origin N, mean/std5.\nTemperature degC; downstream bearing displacement at Piers1-4 mm before normalization.\nOnly mask=True displacement labels enter losses/metrics. Future responses never enter inputs.\nRead:np.load('train_15min.npz');np.load('train_history_15min.npy',mmap_mode='r').\nCounts, reconstruction proportions, measured history fractions and dates:manifest.json.\nCentral record:${WORKSPACE}/2026-10-20_051745_D1九组五seed真实温度实验启动.md.\n"), encoding='utf-8')
    config.update(experiment='d1_observed_temperature_mechanism_20261020_051745', seeds=[2024, 2025, 2026, 2027, 2028], arms=[f'O{i}' for i in range(9)], sample_minutes=15, evaluation_sample_minutes=1, epochs=300, patience=60, optimizer='Adam', learning_rate=config['lr'], objective='masked standardized total-response MSE + 0.001 mean R^2', training_stages='All active response parameters from epoch1; no component/path/spectral losses; no forecast-temperature adaptation.', initialization='Fresh seeded response model; only frozen FFT cutoffs transferred; no learned response-weight transfer.', original_checkpoint=str(ORIGINAL / 'h1_original.pt'), physical_gpu=0, cpu_threads=4, batch_size=config['batch_size'])
    dump(OUT / 'protocol.json', config)
    print('PREPARED', json.dumps({s: len(o) for s, o in origins.items()}), flush=True)
if __name__ == '__main__':
    main()
