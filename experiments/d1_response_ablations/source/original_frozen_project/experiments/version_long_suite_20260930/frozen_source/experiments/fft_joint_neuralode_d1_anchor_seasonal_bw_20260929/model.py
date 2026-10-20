from __future__ import annotations
import importlib.util
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]
DIRECT_EXPERIMENT = ROOT / 'experiments' / 'fft_joint_neuralode_d1_direct_bw_20260929'
direct_spec = importlib.util.spec_from_file_location('direct_bw_model_base', DIRECT_EXPERIMENT / 'model.py')
direct_module = importlib.util.module_from_spec(direct_spec)
direct_spec.loader.exec_module(direct_module)
DirectBWJointFFTNeuralODE = direct_module.DirectBWJointFFTNeuralODE

class AnchoredSeasonalBWModel(DirectBWJointFFTNeuralODE):
    """Direct BW model with a causal LF level anchor and optional seasonal A."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.enable_lf_anchor = bool(cfg['enable_lf_anchor'])
        self.enable_seasonal_a = bool(cfg['enable_seasonal_a'])
        self.seasonal_a_scale = float(cfg['seasonal_a_scale'])
        self.a_season_coeff = torch.nn.Parameter(torch.zeros(self.displacements, 4, dtype=torch.float32))

    def seasonal_a(self, season):
        base_a = self.physics_parameters()['a']
        if not self.enable_seasonal_a:
            return base_a[None].expand(len(season), -1)
        modulation = torch.tanh(season @ self.a_season_coeff.T)
        return base_a[None] * (1.0 + self.seasonal_a_scale * modulation)

    def bw_state_rate(self, state, du, u, u_mean, season):
        y = state[:, :self.displacements]
        z = state[:, self.displacements:]
        params = self.physics_parameters()
        physics = self.seasonal_a(season) * du - params['beta'] * du.abs() * y - params['gamma'] * du * y.abs()
        expanded_season = season[:, None].expand(-1, self.displacements, -1)
        features = torch.cat((torch.stack((y, z, u, du, u_mean), dim=-1), expanded_season), dim=-1)
        residual = float(self.cfg['residual_scale']) * torch.tanh(self.bw_residual(features))
        rate = torch.cat((physics + residual[..., 0], residual[..., 1]), dim=1)
        return (rate, physics, residual)

    def forward(self, history, future_temperature, origin):
        output = super().forward(history, future_temperature, origin)
        if not self.enable_lf_anchor:
            output['lf_anchor_offset'] = output['lf_prediction'].new_zeros(len(history), 1, self.displacements)
            return output
        history_lf = output['history_parts']['static'].transpose(1, 2)
        anchor_origin = origin - int(self.cfg['sample_minutes'])
        anchor_season = self.season_features(anchor_origin, 1)
        mapped_anchor = self.lf_forward(history_lf[:, -1:, :1], anchor_season)
        anchor_offset = history_lf[:, -1:, 1:] - mapped_anchor
        output['lf_prediction'] = output['lf_prediction'] + anchor_offset
        output['prediction'] = output['lf_prediction'] + output['mf_prediction']
        output['lf_anchor_offset'] = anchor_offset
        return output
