from __future__ import annotations
import importlib.util
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]
DIRECT_EXPERIMENT = ROOT / 'experiments' / 'fft_joint_neuralode_d1_direct_bw_20260929'
direct_spec = importlib.util.spec_from_file_location('direct_bw_training_base', DIRECT_EXPERIMENT / 'train.py')
direct = importlib.util.module_from_spec(direct_spec)
direct_spec.loader.exec_module(direct)
base_configure_stage = direct.configure_stage
base_objective = direct.objective
model_spec = importlib.util.spec_from_file_location('anchored_seasonal_bw_model', Path(__file__).with_name('model.py'))
model_module = importlib.util.module_from_spec(model_spec)
model_spec.loader.exec_module(model_module)
AnchoredSeasonalBWModel = model_module.AnchoredSeasonalBWModel

def soft_range_loss(prediction, target, mask, temperature):
    mask = mask.bool()
    very_negative = torch.full_like(prediction, -10000.0)
    pred_max = temperature * torch.logsumexp(torch.where(mask, prediction / temperature, very_negative), dim=1)
    pred_min = -temperature * torch.logsumexp(torch.where(mask, -prediction / temperature, very_negative), dim=1)
    target_max = temperature * torch.logsumexp(torch.where(mask, target / temperature, very_negative), dim=1)
    target_min = -temperature * torch.logsumexp(torch.where(mask, -target / temperature, very_negative), dim=1)
    return (pred_max - pred_min - (target_max - target_min)).square().mean()

def configure_stage(model, stage):
    base_configure_stage(model, stage)
    for parameter in model.decomposer.parameters():
        parameter.requires_grad_(False)
    if stage == 'bw_pretrain' and model.enable_seasonal_a:
        model.a_season_coeff.requires_grad_(True)

def objective(model, output, batch, cfg, stage):
    total, terms, target_lf, target_mf = base_objective(model, output, batch, cfg, stage)
    range_loss = soft_range_loss(output['mf_prediction'], target_mf, batch['mask'], float(cfg['extreme_temperature']))
    total = total + float(cfg['mf_range_weight']) * range_loss
    terms['mf_range_loss'] = range_loss
    return (total, terms, target_lf, target_mf)
direct.DirectBWJointFFTNeuralODE = AnchoredSeasonalBWModel
direct.configure_stage = configure_stage
direct.objective = objective
if __name__ == '__main__':
    direct.main()
