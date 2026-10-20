"""Train the compact structured GRU for Temperature."""
import argparse
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from models import build_model
ROOT = Path(__file__).resolve().parents[2]

def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')

def load(cfg):
    z = np.load(ROOT / cfg['data'], allow_pickle=False)
    values = z['values'][:, :1].astype(np.float64)
    mask = z['valid_observed'][:, :1]
    available_at = z['input_available_at'][:, 0]
    train_end = cfg['split_days'][0] * 1440
    val_end = sum(cfg['split_days'][:2]) * 1440
    mean = np.nanmean(np.where(mask[:train_end], values[:train_end], np.nan), axis=0)
    std = np.nanstd(np.where(mask[:train_end], values[:train_end], np.nan), axis=0)
    values = ((values - mean) / std).astype('float32')
    cumulative = np.vstack((np.zeros((1, 1)), np.cumsum(np.where(mask, values, 0), axis=0, dtype='float64')))
    counts = np.vstack((np.zeros((1, 1)), np.cumsum(mask, axis=0, dtype='float64')))
    parts, report = ({}, {})
    for name, start, end, stride in [('train', 180 * 1440, train_end, cfg['train_stride']), ('val', train_end, val_end, 1440), ('test', val_end, len(values), 1440)]:
        candidates = np.arange(start, end - 1440 + 1, stride)
        selected = [int(origin) for origin in candidates if available_at[origin - 10081:origin].max() < origin]
        parts[name] = selected
        report[name] = {'candidates': len(candidates), 'selected': len(selected), 'measured_target_points': int(sum((mask[o:o + 1440, 0].sum() for o in selected)))}
    return (values, mask, cumulative, counts, parts, mean, std, report)

class Windows(Dataset):

    def __init__(self, values, mask, sums, counts, origins):
        self.values, self.mask = (values, mask)
        self.sums, self.counts, self.origins = (sums, counts, origins)

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, index):
        origin = self.origins[index]
        edges = origin + np.arange(-180, 1) * 1440
        count = self.counts[edges[1:]] - self.counts[edges[:-1]]
        summed = self.sums[edges[1:]] - self.sums[edges[:-1]]
        daily = np.divide(summed, count, out=np.zeros_like(summed), where=count > 0)
        return {'daily': daily.astype('float32'), 'coverage': (count / 1440).astype('float32'), 'recent': self.values[origin - 10081:origin].copy(), 'phase': np.array([(origin - 1) / 1440 * 2 * math.pi, (origin - 1) / 1440 / 365.25 * 2 * math.pi], dtype='float32'), 'target': self.values[origin:origin + 1440].copy(), 'mask': self.mask[origin:origin + 1440].copy(), 'previous_day': self.values[origin - 1440:origin].copy(), 'origin': origin}

def on_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}
