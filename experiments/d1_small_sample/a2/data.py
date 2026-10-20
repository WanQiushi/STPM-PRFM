"""Budget-normalized D1 windows; origins are supplied by the saved protocol."""
import math
import numpy as np
from torch.utils.data import Dataset

class Data:

    def __init__(self, path, mean, std):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)
        with np.load(path, allow_pickle=False) as archive:
            raw = archive['values'].astype(np.float64)
            self.observed = archive['valid_observed'].astype(bool)
        self.values = ((raw - self.mean) / self.std).astype(np.float32)
        temperature = self.values[:, :1]
        mask = self.observed[:, :1]
        self.sums = np.vstack((np.zeros((1, 1)), np.cumsum(np.where(mask, temperature, 0), axis=0, dtype=np.float64)))
        self.counts = np.vstack((np.zeros((1, 1)), np.cumsum(mask, axis=0, dtype=np.float64)))

class TemperatureWindows(Dataset):

    def __init__(self, data, origins, sample_minutes):
        self.data, self.origins, self.step = (data, list(origins), sample_minutes)

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, index):
        origin = self.origins[index]
        edges = origin + np.arange(-30, 1) * 1440
        counts = self.data.counts[edges[1:]] - self.data.counts[edges[:-1]]
        sums = self.data.sums[edges[1:]] - self.data.sums[edges[:-1]]
        daily = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        return {'daily': daily.astype(np.float32), 'coverage': (counts / 1440).astype(np.float32), 'recent': self.data.values[origin - 10081:origin, :1].copy(), 'history_mask': self.data.observed[origin - 10081:origin, :1].copy(), 'phase': np.asarray([(origin - 1) / 1440 * 2 * math.pi, (origin - 1) / 1440 / 365.25 * 2 * math.pi], dtype=np.float32), 'target': self.data.values[origin:origin + 1440:self.step, :1].copy(), 'mask': self.data.observed[origin:origin + 1440:self.step, :1].copy(), 'previous_day': self.data.values[origin - 1440:origin:self.step, :1].copy(), 'origin': np.int64(origin)}

class ResponseWindows(Dataset):

    def __init__(self, data, origins, sample_minutes, forecasts=None):
        self.data, self.origins, self.step = (data, list(origins), sample_minutes)
        self.forecasts = forecasts
        self.history_offsets = np.arange(-30 * 1440, 0, self.step)
        self.future_offsets = np.arange(0, 1440, self.step)

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, index):
        origin = self.origins[index]
        future = origin + self.future_offsets
        batch = {'history': self.data.values[origin + self.history_offsets].copy(), 'target': self.data.values[future, 1:].copy(), 'mask': self.data.observed[future, 1:].copy(), 'origin': np.int64(origin)}
        if self.forecasts is None:
            batch['future_temperature'] = self.data.values[future, :1].copy()
        else:
            batch['forecast_temperature_celsius'] = self.forecasts[index].astype(np.float32)
        return batch

class CachedWindows(Dataset):

    def __init__(self, arrays):
        self.arrays = arrays

    def __len__(self):
        return len(self.arrays['origin'])

    def __getitem__(self, index):
        return {name: values[index] for name, values in self.arrays.items()}

def on_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}
