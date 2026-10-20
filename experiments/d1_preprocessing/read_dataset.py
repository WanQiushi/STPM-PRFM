"""Read the aligned arrays without assigning a forecasting window or filling gaps."""
from pathlib import Path
import numpy as np
root = Path(__file__).resolve().parents[1]
with np.load(root / 'data' / 'processed' / 'minute_observed.npz', allow_pickle=False) as data:
    values = data['values']
    timestamps = data['timestamps']
    observed = data['observed']
    targets = values[:, data['output_indices']]
    daily_values = values.reshape(730, 1440, 7)
    print('channels:', data['channels'].tolist())
    print('values:', values.shape, 'targets:', targets.shape, 'daily view:', daily_values.shape)
    print('range:', timestamps[0], timestamps[-1])
    print('all-seven observed minutes:', int(observed.all(axis=1).sum()))
