from if_release.runtime import loads as _release_json_loads
from pathlib import Path
import argparse, json
import numpy as np
import torch
from torch.utils.data import DataLoader
from data_contract_v8 import ROOT, load, on_device
from train_contract_v9 import ForecastWindows
from models import build_model
parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
parser.add_argument('--device', default='cuda:0')
args = parser.parse_args()
cfg = _release_json_loads(Path(args.config).read_text())
out = ROOT / cfg['output']
out.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
device = torch.device(args.device)
ck = torch.load(ROOT / cfg['forecast_checkpoint'], map_location=device, weights_only=False)
model = build_model(cfg).to(device)
model.load_state_dict(ck['model'])
model.eval()
values, mask, sums, counts, parts, mean, std, report = load(cfg)
for part, origins in cfg['forecast_origins'].items():
    loader = DataLoader(ForecastWindows(values, mask, sums, counts, origins, cfg['sample_minutes']), batch_size=cfg['batch_size'], shuffle=False)
    predictions = []
    with torch.no_grad():
        for batch in loader:
            predictions.append(model(on_device(batch, device))['prediction'].cpu().numpy() * std + mean)
    np.savez_compressed(out / (part + '_forecasts.npz'), origin=np.asarray(origins), forecast_temperature_celsius=np.concatenate(predictions), normalization_end_day=cfg['normalization_end_day'], sample_minutes=cfg['sample_minutes'], parameters=sum((p.numel() for p in model.parameters())))
    print('FORECAST_EXPORTED ' + json.dumps(dict(job=cfg['run_name'], split=part, windows=len(origins))), flush=True)
