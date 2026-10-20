"""Read the existing known-temperature cache; never rebuild or modify it."""
from pathlib import Path
import numpy as np
import torch

def load_split(folder, split, device):
    folder = Path(folder)
    with np.load(folder / f'{split}_15min.npz') as raw:
        data = {key: torch.as_tensor(raw[key], device=device) for key in ['temperature', 'target', 'mask', 'origin']}
        data['std'] = torch.as_tensor(raw['std'][1:], dtype=torch.float32, device=device)
        data['mean'] = raw['mean'][1:].copy()
    data['history'] = torch.as_tensor(np.load(folder / f'{split}_history_15min.npy'), device=device)
    return data

def model_inputs(data, index):
    return (data['history'][index], data['temperature'][index], data['origin'][index])

def response_metrics(prediction, target, mask, std):
    error = (prediction.astype(np.float64) - target.astype(np.float64)) * std
    sse = np.where(mask, error ** 2, 0.0).sum(1)
    sae = np.where(mask, np.abs(error), 0.0).sum(1)
    n = mask.sum(1)
    metrics = dict(windows=len(prediction), valid_labels=int(n.sum()), rmse_mm=float(np.sqrt(sse.sum() / n.sum())), mae_mm=float(sae.sum() / n.sum()), mse_mm2=float(sse.sum() / n.sum()), per_location_rmse_mm=np.sqrt(sse.sum(0) / n.sum(0)).tolist())
    return (metrics, dict(sse=sse, sae=sae, n=n))
