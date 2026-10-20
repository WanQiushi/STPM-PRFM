"""Train causal 24-hour d1 temperature forecasts; keep the test split sealed."""
import argparse
import json
import random
import shutil
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from data_contract_v8 import ROOT, Windows, dump, load, on_device
from models import build_model

class ForecastWindows(Windows):

    def __init__(self, values, mask, sums, counts, origins, sample_minutes):
        super().__init__(values, mask, sums, counts, origins)
        self.sample_minutes = sample_minutes

    def __getitem__(self, index):
        batch = super().__getitem__(index)
        origin = batch['origin']
        batch['history_mask'] = self.mask[origin - 10081:origin].copy()
        batch['target'] = batch['target'][::self.sample_minutes].copy()
        batch['mask'] = batch['mask'][::self.sample_minutes].copy()
        batch['previous_day'] = batch['previous_day'][::self.sample_minutes].copy()
        return batch

def objective(result, batch):
    error = result['prediction'] - batch['target']
    return (error.square() * batch['mask']).sum() / batch['mask'].sum()

def evaluate(model, loader, device, mean, std, collect=False):
    model.eval()
    sse = sae = previous_sse = count = 0.0
    block_sse, block_count = (np.zeros(4), np.zeros(4))
    hourly_sse, hourly_count = (np.zeros(24), np.zeros(24))
    stored = {key: [] for key in ('origin', 'prediction', 'target', 'mask')}
    with torch.no_grad():
        for batch in loader:
            current = on_device(batch, device)
            prediction = model(current)['prediction']
            error = prediction - current['target']
            mask = current['mask']
            sse += (error.square() * mask).sum().item()
            sae += (error.abs() * mask).sum().item()
            previous_sse += ((current['previous_day'] - current['target']).square() * mask).sum().item()
            count += mask.sum().item()
            for block in range(4):
                section = slice(block * (prediction.shape[1] // 4), (block + 1) * (prediction.shape[1] // 4))
                block_sse[block] += (error[:, section].square() * mask[:, section]).sum().item()
                block_count[block] += mask[:, section].sum().item()
            for hour in range(24):
                section = slice(hour * (prediction.shape[1] // 24), (hour + 1) * (prediction.shape[1] // 24))
                hourly_sse[hour] += (error[:, section].square() * mask[:, section]).sum().item()
                hourly_count[hour] += mask[:, section].sum().item()
            if collect:
                stored['origin'].append(batch['origin'].numpy())
                stored['prediction'].append(prediction.cpu().numpy() * std + mean)
                stored['target'].append(batch['target'].numpy() * std + mean)
                stored['mask'].append(batch['mask'].numpy())
    metrics = {'temperature_mse_normalized': sse / count, 'rmse_celsius': float(np.sqrt(sse / count) * std[0]), 'mae_celsius': float(sae / count * std[0]), 'previous_day_rmse_celsius': float(np.sqrt(previous_sse / count) * std[0]), 'six_hour_rmse_celsius': (np.sqrt(block_sse / block_count) * std[0]).tolist(), 'first_12h_rmse_celsius': float(np.sqrt(block_sse[:2].sum() / block_count[:2].sum()) * std[0]), 'last_12h_rmse_celsius': float(np.sqrt(block_sse[2:].sum() / block_count[2:].sum()) * std[0]), 'hourly_rmse_celsius': (np.sqrt(hourly_sse / hourly_count) * std[0]).tolist(), 'measured_query_points': int(count)}
    if collect:
        return (metrics, {key: np.concatenate(value, axis=0) for key, value in stored.items()})
    return metrics
