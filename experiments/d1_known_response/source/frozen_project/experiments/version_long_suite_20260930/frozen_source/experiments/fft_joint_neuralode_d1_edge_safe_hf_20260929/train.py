from __future__ import annotations
import importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
PARENT_EXPERIMENT = ROOT / 'experiments' / 'fft_joint_neuralode_d1_anchor_seasonal_bw_20260929'
parent_spec = importlib.util.spec_from_file_location('anchor_seasonal_training_base', PARENT_EXPERIMENT / 'train.py')
parent = importlib.util.module_from_spec(parent_spec)
parent_spec.loader.exec_module(parent)
model_spec = importlib.util.spec_from_file_location('edge_safe_model', Path(__file__).with_name('model.py'))
model_module = importlib.util.module_from_spec(model_spec)
model_spec.loader.exec_module(model_module)
EdgeSafeAnchoredSeasonalBWModel = model_module.EdgeSafeAnchoredSeasonalBWModel
base_configure_stage = parent.configure_stage

def configure_stage(model, stage):
    base_configure_stage(model, stage)
    model.decomposer.theta_static.requires_grad_(False)
    model.decomposer.theta_noise.requires_grad_(bool(model.cfg.get('train_noise_cutoff', False)) and stage == 'joint_cutoff')
parent.direct.DirectBWJointFFTNeuralODE = EdgeSafeAnchoredSeasonalBWModel
parent.direct.configure_stage = configure_stage
parent.direct.objective = parent.objective
if __name__ == '__main__':
    parent.direct.main()
