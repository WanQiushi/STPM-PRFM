"""Paired STPM/PRFM inference after per-seed PRFM selection; no test selection."""
from if_release.runtime import expand as _release_expand, loads as _release_json_loads
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse, json, sys, time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--dataset', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--response-checkpoint', type=Path, required=True)
args = parser.parse_args()
ROOT, DATA, OUT = (args.root, args.dataset, args.output)
job = next((j for j in _release_json_loads((HERE / 'config/temperature_jobs.json').read_text()) if j['seed'] == args.seed))
for field in ['checkpoint', 'source']:
    job[field] = str(ROOT / Path(job[field]).relative_to(_release_expand('${WORKSPACE}')))
start = time.time()
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = torch.device('cuda:0')
origins = np.load(DATA / 'origins.npy')
history = np.load(DATA / 'history_physical.npy', mmap_mode='r')
history_mask = np.load(DATA / 'history_mask.npy', mmap_mode='r')
target = np.load(DATA / 'target_physical.npy')
mask = np.load(DATA / 'target_mask.npy')
manifest = _release_json_loads((DATA / 'manifest.json').read_text())
checkpoint = torch.load(job['checkpoint'], map_location='cpu', weights_only=False)
cfg = checkpoint['config']
mean, std = (np.asarray(checkpoint['mean']), np.asarray(checkpoint['std']))
sys.path.insert(0, job['source'])
from models import build_model
from train_contract_v9 import ForecastWindows
temperature_model = build_model(cfg).to(device)
temperature_model.load_state_dict(checkpoint['model'])
temperature_model.eval().requires_grad_(False)
tvalues = np.load(DATA / 'temperature_values.npy')
tobserved = np.load(DATA / 'temperature_observed.npy')
tnormalized = ((tvalues - mean) / std).astype(np.float32)
sums = np.vstack([np.zeros((1, 1)), np.cumsum(np.where(tobserved, tnormalized, 0), axis=0, dtype=np.float64)])
counts = np.vstack([np.zeros((1, 1)), np.cumsum(tobserved, axis=0, dtype=np.float64)])

class OriginWindows(ForecastWindows):

    def __getitem__(self, index):
        batch = super().__getitem__(index)
        batch['recent'] = ((history[index, -10081:, :1] - mean) / std).astype(np.float32)
        batch['previous_day'] = ((history[index, -1440:, :1] - mean) / std).astype(np.float32)
        return batch
windows = OriginWindows(tnormalized, tobserved, sums, counts, origins.tolist(), 1)
temperatures = []
with torch.inference_mode():
    for batch in DataLoader(windows, batch_size=32, shuffle=False, num_workers=0):
        inputs = {k: batch[k].to(device) for k in ['daily', 'coverage', 'recent', 'phase', 'history_mask']}
        temperatures.append(temperature_model(inputs)['prediction'].cpu().numpy())
tcelsius = np.concatenate(temperatures).astype(np.float64) * std + mean
sys.path.insert(0, str(HERE / 'engine'))
from response_model import HybridKqsz
hpath = args.response_checkpoint
trained = torch.load(hpath, map_location='cpu', weights_only=False)
h1 = HybridKqsz({**trained['config'], 'sample_minutes': 1})
state = trained['model']
state['decomposer.frequency'] = h1.decomposer.frequency
state['decomposer.padded_frequency'] = h1.decomposer.padded_frequency
h1.load_state_dict(state)
h1 = h1.to(device).eval().requires_grad_(False)
mean, std = (np.asarray(trained['mean']), np.asarray(trained['std']))
future_t = ((tcelsius - mean[:1]) / std[:1]).astype(np.float32)
responses = []
with torch.inference_mode():
    for first in range(0, len(origins), 219):
        sl = slice(first, first + 219)
        result = h1(torch.as_tensor(((history[sl] - mean) / std).astype(np.float32), device=device), torch.as_tensor(future_t[sl], device=device), torch.as_tensor(origins[sl], device=device))
        responses.append(result['prediction'].cpu().numpy())
prediction = np.concatenate([future_t, np.concatenate(responses)], axis=2)
provenance = dict(temperature_checkpoint=job['checkpoint'], temperature_source=job['source'], temperature_config=cfg, response_checkpoint=str(hpath), response_config=h1.cfg, shared_response_across_temperature_seeds=False, response_inference_batch_size=219)
normalized_target = ((target - mean) / std).astype(np.float32)
error = (prediction.astype(np.float64) - normalized_target.astype(np.float64)) * std

def score(indices):
    e = error[indices]
    m = mask[indices]
    sse = np.where(m, e ** 2, 0).sum((0, 1))
    sae = np.where(m, np.abs(e), 0).sum((0, 1))
    count = m.sum((0, 1))
    hourly = []
    for hour in range(24):
        em = e[:, hour * 60:(hour + 1) * 60, 1:]
        mm = m[:, hour * 60:(hour + 1) * 60, 1:]
        hourly.append(float(np.sqrt(np.where(mm, em ** 2, 0).sum() / mm.sum())))
    return dict(windows=int(len(indices)), temperature_rmse_c=float(np.sqrt(sse[0] / count[0])), temperature_mae_c=float(sae[0] / count[0]), response_rmse_mm=float(np.sqrt(sse[1:].sum() / count[1:].sum())), response_mae_mm=float(sae[1:].sum() / count[1:].sum()), response_mse_mm2=float(sse[1:].sum() / count[1:].sum()), channel_rmse_mm=np.sqrt(sse[1:] / count[1:]).tolist(), channel_mae_mm=(sae[1:] / count[1:]).tolist(), measured_counts=count.astype(int).tolist(), response_hourly_rmse_mm=hourly)
folder = OUT / job['key']
folder.mkdir(parents=True)
metrics = {}
for split, lo, hi, key in [('validation', 0, 110, 'val'), ('test', 110, 219, 'test')]:
    indices = np.arange(lo, hi)
    cohorts = {'all_days': indices, 'original_causal_history': np.array([i for i in indices if manifest['days'][i]['original_history_causal']]), 'reconstructed_history': np.array([i for i in indices if not manifest['days'][i]['original_history_causal']]), 'legacy_frozen_cohort': np.array([i for i in indices if int(origins[i]) in manifest['legacy_origins'][key]])}
    metrics[split] = {cohort: score(ix) for cohort, ix in cohorts.items()}
    np.savez_compressed(folder / (split + '_predictions.npz'), prediction=prediction[indices], target=normalized_target[indices], mask=mask[indices], origin=origins[indices], mean=mean, std=std)
provenance.update(job=job, status='completed', training_performed=False, weights_reselected=False, protocol=str(DATA / 'manifest.json'), visible_gpu=__import__('os').environ.get('CUDA_VISIBLE_DEVICES'), elapsed_seconds=time.time() - start, finished_beijing=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
record = dict(model=job['model'], family=job['family'], seed=job['seed'], metrics=metrics, provenance=provenance)
(folder / 'metrics.json').write_text(json.dumps(record, indent=2))
(folder / 'experiment_record.md').write_text('# Paired D1 full-period inference: ' + job['key'] + '\n\n- Date: ' + provenance['finished_beijing'] + '; IFpaper_v1.\n- Purpose: all110validation/109test days, fixed existing weights; no training or checkpoint selection.\n- Input/target contract: ' + str(DATA / 'README.md') + '; per-origin historical reconstruction disclosed in manifest.\n- Checkpoint/source: ' + job['checkpoint'] + '; ' + job['source'] + '.\n- Validation response RMSE: ' + str(metrics['validation']['all_days']['response_rmse_mm']) + 'mm.\n- Test response RMSE: ' + str(metrics['test']['all_days']['response_rmse_mm']) + 'mm.\n- Products: ' + str(folder) + '; metrics/provenance and minute prediction caches.\n')
print('COMPLETED ' + json.dumps(dict(key=job['key'], seconds=time.time() - start, validation_rmse=metrics['validation']['all_days']['response_rmse_mm'], test_rmse=metrics['test']['all_days']['response_rmse_mm'])), flush=True)
