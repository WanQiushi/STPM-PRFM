"""Read one prepared budget protocol without recomputing full-data statistics."""
from if_release.runtime import loads as _release_json_loads
import json
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
PROJECT = Path(__file__).resolve().parents[1]

def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def prepare(cfg):
    protocol_path = PROJECT / cfg['protocol_path']
    protocol = _release_json_loads(protocol_path.read_text(encoding='utf-8'))
    with np.load(PROJECT / protocol['data_path']) as source:
        values = source['values'].astype(np.float64)
        observed = source['valid_observed'].astype(bool)
    mean = np.asarray(protocol['mean'], dtype=np.float64)
    std = np.asarray(protocol['std'], dtype=np.float64)
    values = ((values - mean) / std).astype(np.float32)
    history_offsets = np.arange(-protocol['history_points'], 0)
    future_offsets = np.arange(protocol['forecast_points'])
    return (values, observed, protocol['origins'], mean, std, protocol, history_offsets, future_offsets)

class Windows(Dataset):

    def __init__(self, values, observed, origins, history_offsets, future_offsets):
        self.values, self.observed, self.origins = (values, observed, origins)
        self.ho, self.fo = (history_offsets, future_offsets)

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, index):
        origin = self.origins[index]
        h, f = (origin + self.ho, origin + self.fo)
        return dict(history=self.values[h].copy(), history_mask=self.observed[h].copy(), target=self.values[f].copy(), mask=self.observed[f].copy(), origin=np.int64(origin))
