"""Build causal H1 histories and observed targets for the requested sampling grids."""
from if_release.runtime import loads as _release_json_loads
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'temperature_prediction/outputs' / HERE.name
DATA = OUT / 'datasets'
SOURCE = ROOT / 'outputs/va_1_20261004'
PREVIOUS = ROOT / 'temperature_prediction/outputs/va_2_2_five_seed_20261008'

def main():
    DATA.mkdir(parents=True)
    normalization = _release_json_loads((SOURCE / 'normalization.json').read_text())
    mean = np.asarray(normalization['h1_mean'])
    std = np.asarray(normalization['h1_std'])
    raw_path = ROOT / 'data/mainbridge_temperature_v2/minute_complete.npz'
    with np.load(raw_path) as raw:
        values = raw['values'].astype(np.float64)
        observed = raw['valid_observed'].astype(bool)
        available = raw['input_available_at']
    with np.load(PREVIOUS / 'fixed_h1_validation/validation_inputs.npz') as cache:
        candidates = cache['origin'].copy()
    eligible = {}
    for dt in [15, 10, 5, 1]:
        offsets = np.arange(-30 * 1440, 0, dt)
        eligible[dt] = [int(o) for o in candidates if available[o + offsets].max() < o]
    origins = np.array([int(o) for o in candidates if all((int(o) in eligible[dt] for dt in eligible))], dtype=np.int64)
    normalized = ((values - mean) / std).astype(np.float32)
    contracts = []
    for dt in [15, 10, 5, 1]:
        history_offsets = np.arange(-30 * 1440, 0, dt)
        future_offsets = np.arange(0, 1440, dt)
        history = np.stack([normalized[o + history_offsets] for o in origins])
        target = np.stack([normalized[o + future_offsets, 1:] for o in origins])
        mask = np.stack([observed[o + future_offsets, 1:] for o in origins])
        ttruth = np.stack([values[o + future_offsets, :1] for o in origins]).astype(np.float32)
        tmask = np.stack([observed[o + future_offsets, :1] for o in origins])
        np.savez_compressed(DATA / f'h1_{dt}min.npz', history=history, target=target, mask=mask, origin=origins, temperature_truth_celsius=ttruth, temperature_mask=tmask, mean=mean, std=std, query_offsets_minutes=future_offsets)
        contracts.append({'sample_minutes': dt, 'windows': len(origins), 'history_shape': list(history.shape), 'target_shape': list(target.shape), 'measured_y_nodes': int(mask.sum()), 'measured_y_nodes_per_channel': mask.sum((0, 1)).astype(int).tolist(), 'measured_t_nodes': int(tmask.sum()), 'path': str(DATA / f'h1_{dt}min.npz')})
    report = {'built_beijing': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), 'raw_data': str(raw_path), 'normalization_source': str(SOURCE / 'normalization.json'), 'candidate_origin_source': str(PREVIOUS / 'fixed_h1_validation/validation_inputs.npz'), 'candidate_windows': len(candidates), 'eligible_by_resolution': eligible, 'common_origins': origins.tolist(), 'contracts': contracts, 'selection': 'original26 validation origins intersect causal 30-day five-channel histories at all four grids', 'history_sampling': 'actual minute values selected at origin-43200, then every dt; not interpolated forecasts', 'target_sampling': 'actual minute labels selected at origin+0,dt,...,1440-dt; observed mask used for scoring', 'shared_scoring_grid_minutes': 30, 'test_evaluated': False}
    (DATA / 'dataset_manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (DATA / 'README.md').write_text('# va-2.2 sampling comparison: validation datasets\n\n- Date: ' + report['built_beijing'] + '; IFpaper_v1.\n- Source: ' + str(raw_path) + '; training-period H1 normalization from ' + str(SOURCE / 'normalization.json') + '.\n- Build: ' + str(HERE / 'build_sampling_data.py') + '; no new raw measurements or interpolation.\n- Validation candidates26; common causal histories ' + str(len(origins)) + '; no test evaluation.\n- h1_15/10/5/1min.npz: history[N,30*1440/dt,5], target/mask[N,1440/dt,4], origin integer minute since2022-01-01; temperature_truth_celsius/temperature_mask[N,1440/dt,1], query offsets minutes, normalization mean/std.\n- Channels: Temperature, Pier1, Pier2, Pier3, Pier4. Temperature sensor type unconfirmed.\n- History/target normalized with original training-period means/std; temperature_truth_celsius is Celsius. Future labels are only for scoring, never H1 input.\n- Reading: z=np.load("h1_5min.npz"); history=z["history"]; target=z["target"]; mask=z["mask"].\n- dataset_manifest.json records actual source, eligibility, shapes and observed counts; ../README.md records training contract.\n', encoding='utf-8')
    (DATA / 'experiment_record.md').write_text('# Sampling dataset construction completed\n\n- Date: ' + report['built_beijing'] + '.\n- Purpose: causally eligible fixed validation histories and native observed targets for15/10/5/1min.\n- Operation: raw minute sampling, common origin selection, original H1 normalization; no target interpolation.\n- Result: ' + str(len(origins)) + ' common windows; details in dataset_manifest.json.\n- Products: four h1_*min.npz, dataset_manifest.json, README.md; source build_sampling_data.py.\n', encoding='utf-8')
    print(json.dumps(report), flush=True)
if __name__ == '__main__':
    main()
