"""Eight frozen manuscript backbones with the same explicit known-T adapter."""
from pathlib import Path
import importlib.util
import math
import sys
import torch
from torch import nn
from common import ROOT

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def calendar(origin, length):
    offsets = torch.arange(length, device=origin.device, dtype=torch.float32)
    days = (origin.to(torch.float32)[:, None] + 15.0 * offsets[None]) / 1440.0
    phase = 2.0 * math.pi * days / 365.25
    return torch.stack((phase.sin(), phase.cos(), (2.0 * phase).sin(), (2.0 * phase).cos()), dim=-1)

class KnownTemperatureBaseline(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.name = cfg['model']
        self.horizon = 96
        frozen = ROOT / 'frozen_sources'
        effective = dict(cfg)
        effective['pcfnet_vendor'] = str(frozen / 'pcfnet_vendor')
        if self.name in ['PCFNet', 'VanillaLSTM']:
            module = load_module('knownT_frozen_three_family', frozen / 'three_family/D1/models.py')
        elif self.name == 'ModernTCN':
            parent = frozen / 'baseline_snapshot/experiments/minute_moderntcn_joint5_20261009'
            sys.path.insert(0, str(parent))
            module = load_module('knownT_frozen_moderntcn', parent / 'models.py')
        else:
            module = load_module('knownT_frozen_tslib', frozen / 'baseline_snapshot/experiments/minute_joint5_forecast_20261007/models.py')
        if self.name in ['TimeXer', 'PatchTST', 'iTransformer']:
            module.official_module = lambda name: load_module('knownT_' + name + '_out_of_place', ROOT / 'source/adaptations' / (name + '.py'))
        self.frozen_family = module.Baseline(effective)
        self.future_projection = nn.Linear(5, 5)
        with torch.no_grad():
            self.future_projection.weight.zero_()
            self.future_projection.weight[:, 0] = 1.0
            self.future_projection.bias.zero_()

    def forward(self, history, future_temperature, origin):
        future_features = torch.cat((future_temperature, calendar(origin, self.horizon)), dim=-1)
        context = torch.cat((history, self.future_projection(future_features)), dim=1)
        if self.name == 'VanillaLSTM':
            original = self.frozen_family.backbone
            anchor = history[:, -1:]
            _, (hidden, _) = original.encoder(context - anchor)
            prediction = anchor + original.output(hidden[-1]).reshape(-1, self.horizon, 5)
        else:
            prediction = self.frozen_family({'history': context})['prediction']
        return dict(prediction=prediction[:, :, 1:5])

    def parameter_report(self):
        return dict(model=self.name, total_parameters=sum((p.numel() for p in self.parameters())), trainable_parameters=sum((p.numel() for p in self.parameters() if p.requires_grad)), conditional_projection_parameters=sum((p.numel() for p in self.future_projection.parameters())), condition_context_length=2976, history_length=2880, future_known_length=96, future_response_input=False)
