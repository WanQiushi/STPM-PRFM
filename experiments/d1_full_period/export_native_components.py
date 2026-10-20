"""Export the original illustrated date using the new seed-2024 PRFM."""
from if_release.runtime import expand as _release_expand
from pathlib import Path
import json
import sys
import numpy as np
import torch
ROOT = Path(_release_expand('${WORKSPACE}'))
TRAINING = ROOT / 'outputs/d1_prfm_gpu1_20261020_1809'
EVALUATION = ROOT / 'outputs/d1_prfm_full219_paired_20261020_2042'
DATA = ROOT / 'outputs/d1_full219_frozen_inference_20261017/dataset'
OUTPUT = Path(__file__).resolve().parent
sys.path.insert(0, str(TRAINING / 'source/engine'))
from response_model import HybridKqsz
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
checkpoint = torch.load(TRAINING / 'PRFM_seed2024/adapted_best.pt', map_location='cpu', weights_only=False)
model = HybridKqsz({**checkpoint['config'], 'sample_minutes': 1})
state = checkpoint['model']
state['decomposer.frequency'] = model.decomposer.frequency
state['decomposer.padded_frequency'] = model.decomposer.padded_frequency
model.load_state_dict(state)
model = model.cuda().eval().requires_grad_(False)
mean, std = (np.asarray(checkpoint['mean']), np.asarray(checkpoint['std']))
history = np.load(DATA / 'history_physical.npy', mmap_mode='r')
origins = np.load(DATA / 'origins.npy')
predictions = np.concatenate([np.load(EVALUATION / f'evaluation/S0_seed2024/{split}_predictions.npz')['prediction'] for split in ['validation', 'test']])
with torch.inference_mode():
    output = model(torch.as_tensor(((history - mean) / std).astype(np.float32), device='cuda'), torch.as_tensor(predictions[:, :, :1], device='cuda'), torch.as_tensor(origins, device='cuda'))
day = 110 + 61
arrays = {}
for target, source in [('lf_prediction', 'lf_prediction'), ('mf_prediction', 'mf_prediction'), ('reversible', 'reversible_prediction'), ('memory', 'memory_prediction')]:
    arrays[target] = output[source][day].cpu().numpy().astype(np.float64) * std[1:]
arrays['lf_prediction'] += mean[1:]
arrays['lf_temperature'] = output['temperature_lf'][day, :, 0].cpu().numpy().astype(np.float64) * std[0] + mean[0]
arrays['mf_temperature'] = output['temperature_mf'][day, :, 0].cpu().numpy().astype(np.float64) * std[0]
np.savez_compressed(OUTPUT / 'native_components_seed2024_20231114.npz', **arrays)
error = np.max(np.abs((output['prediction'].cpu().numpy().astype(np.float64) - predictions[:, :, 1:].astype(np.float64)) * std[1:]))
record = {'checkpoint': str(TRAINING / 'PRFM_seed2024/adapted_best.pt'), 'evaluation': str(EVALUATION), 'date': '2023-11-14', 'seed': 2024, 'test_day_index': 61, 'origin': int(origins[day]), 'sample_minutes': 1, 'inference_batch_size': 219, 'max_full_period_difference_from_evaluation_mm': float(error), 'reference_and_baselines': 'reuse existing fixed decomposition curves and original date', 'training_performed': False}
(OUTPUT / 'native_components_provenance.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record), flush=True)
