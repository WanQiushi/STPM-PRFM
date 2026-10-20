"""Train one randomly initialized PRFM, then adapt its own selected checkpoint."""
from if_release.runtime import loads as _release_json_loads
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'engine'))
from data import Data
from response import train_response

def read_config(name):
    return _release_json_loads((HERE / 'config' / name).read_text(encoding='utf-8'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    norm = read_config('normalization.json')
    data = Data(args.data, norm['h1_mean'], norm['h1_std'])
    with np.load(HERE / 'cache/train.npz') as cache:
        train_origins = cache['origin'].tolist()
        train_forecasts = cache['forecast_temperature_celsius'].copy()
    with np.load(HERE / 'cache/validation.npz') as cache:
        val_origins = cache['origin'].tolist()
        val_forecasts = cache['forecast_temperature_celsius'].copy()
    train_response(data, read_config('pretraining_origins.json'), train_origins, train_forecasts, val_origins, val_forecasts, args.seed, args.output, torch.device('cuda:0'), read_config('pretraining.json'), read_config('adaptation.json'))
if __name__ == '__main__':
    main()
