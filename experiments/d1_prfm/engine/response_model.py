"""Shared endpoint decomposition, anchored LF, and reversible + BW-memory MF."""
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F
PACKAGE = Path(__file__).resolve().parents[1]
PROJECT = PACKAGE / 'vendor/h1_project'
SOURCE = PROJECT / 'experiments/version_long_suite_20260930/frozen_source'

def load_module(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
anchor = load_module('hybrid_anchor_source', SOURCE / 'experiments/fft_joint_neuralode_d1_anchor_seasonal_bw_20260929/model.py')
edge = load_module('hybrid_edge_source', SOURCE / 'experiments/fft_joint_neuralode_d1_edge_safe_hf_20260929/model.py')
core = load_module('hybrid_core_source', SOURCE / 'experiments/fft_joint_neuralode_d1_20260929/model.py')

class HybridKqsz(nn.Module):
    season_features = anchor.DirectBWJointFFTNeuralODE.season_features
    thermal_lag_hours = anchor.DirectBWJointFFTNeuralODE.thermal_lag_hours
    lag_step = anchor.DirectBWJointFFTNeuralODE.lag_step

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.displacements = 4
        self.step_hours = cfg['sample_minutes'] / 60.0
        self.points_per_day = 1440 // cfg['sample_minutes']
        self.history_length = cfg['history_days'] * self.points_per_day
        self.decomposer = edge.EdgeSafeFFT(self.history_length, self.step_hours, cfg)
        width = cfg['width']
        self.lf_alpha = nn.Parameter(torch.ones(4))
        self.lf_beta = nn.Parameter(torch.zeros(4))
        self.lf_alpha_season = nn.Sequential(nn.Linear(4, width), nn.Tanh(), nn.Linear(width, 4))
        self.lf_beta_season = nn.Sequential(nn.Linear(4, width), nn.Tanh(), nn.Linear(width, 4))
        self.lf_residual = nn.Sequential(nn.Linear(5, width), nn.Tanh(), nn.Linear(width, 4))
        for module in (self.lf_alpha_season, self.lf_beta_season, self.lf_residual):
            nn.init.zeros_(module[-1].weight)
            nn.init.zeros_(module[-1].bias)
        self.thermal_lag_range = tuple(cfg['thermal_lag_hours_range'])
        initial = core.inverse_sigmoid(cfg['init_thermal_lag_hours'], *self.thermal_lag_range)
        self.thermal_lag_raw = nn.Parameter(torch.full((4,), initial))
        self.initial_z = nn.Sequential(nn.Linear(72 * 5, width), nn.Tanh(), nn.Linear(width, 4))
        self.k = nn.Parameter(torch.ones(4))
        self.s = nn.Parameter(torch.full((4,), 0.1))
        self.beta_raw = nn.Parameter(torch.zeros(4))
        self.gamma_raw = nn.Parameter(torch.zeros(4))
        self.a_season_coeff = nn.Parameter(torch.zeros(4, 4))
        self.residual_field = nn.Sequential(nn.Linear(12, width), nn.Tanh(), nn.Linear(width, 4))
        nn.init.zeros_(self.residual_field[-1].weight)
        nn.init.zeros_(self.residual_field[-1].bias)

    def lf_forward(self, temperature, season):
        if self.cfg['enable_lf_seasonal_maps']:
            alpha = self.lf_alpha[None, None] + self.lf_alpha_season(season)
            bias = self.lf_beta[None, None] + self.lf_beta_season(season)
        else:
            alpha = self.lf_alpha[None, None]
            bias = self.lf_beta[None, None]
        return alpha * temperature + bias + self.lf_residual(torch.cat((temperature, season), dim=-1))

    def bw_rate(self, z, dq, q, season):
        beta = F.softplus(self.beta_raw)
        gamma = 0.9 * beta * torch.tanh(self.gamma_raw)
        a = 1.0 + self.cfg['seasonal_a_scale'] * torch.tanh(season @ self.a_season_coeff.T)
        physics = a * dq - beta * dq.abs() * z - gamma * dq * z.abs()
        residual = self.cfg['residual_scale'] * torch.tanh(self.residual_field(torch.cat((z, dq, q), dim=1)))
        return (physics + residual, physics, residual)

    def forward(self, history, future_temperature, origin):
        history_parts = self.decomposer(history.transpose(1, 2))
        history_lf = history_parts['static'].transpose(1, 2)
        history_mf = history_parts['dynamic'].transpose(1, 2)
        temperature_window = torch.cat((history[:, self.points_per_day:, :1], future_temperature), dim=1)
        temperature_parts = self.decomposer(temperature_window.transpose(1, 2))
        temperature_lf = temperature_parts['static'].transpose(1, 2)[:, -self.points_per_day:]
        temperature_mf = temperature_parts['dynamic'].transpose(1, 2)[:, -self.points_per_day:]
        season = self.season_features(origin, self.points_per_day)
        mapped_anchor = self.lf_forward(history_lf[:, -1:, :1], self.season_features(origin - self.cfg['sample_minutes'], 1))
        anchor_offset = history_lf[:, -1:, 1:] - mapped_anchor
        lf_prediction = self.lf_forward(temperature_lf, season) + anchor_offset
        recent = history_mf[:, -3 * self.points_per_day:]
        hourly = recent.reshape(len(history), 72, 60 // self.cfg['sample_minutes'], 5).mean(dim=2)
        z = torch.tanh(self.initial_z(hourly.flatten(1)))
        u_history = history_mf[:, -3 * self.points_per_day - 1:, :1]
        q = u_history[:, 0].expand(-1, 4)
        history_season = self.season_features(origin, 3 * self.points_per_day, start_offset=-3 * self.points_per_day)
        energy = history.new_zeros(())
        for j in range(3 * self.points_per_day):
            next_q = self.lag_step(q, u_history[:, j + 1])
            dq = (next_q - q) / self.step_hours
            _, _, residual = self.bw_rate(z, dq, next_q, history_season[:, j])
            energy = energy + residual.square().mean()
            z = core.rk4(z, lambda state: self.bw_rate(state, dq, next_q, history_season[:, j])[0], self.step_hours)
            q = next_q
        saved = {key: [] for key in ['effective_temperature_mf', 'reversible_prediction', 'memory_prediction', 'bw_latent', 'bw_physics_rate', 'bw_residual_rate']}
        for j in range(self.points_per_day):
            next_q = self.lag_step(q, temperature_mf[:, j])
            dq = (next_q - q) / self.step_hours
            z = core.rk4(z, lambda state: self.bw_rate(state, dq, next_q, season[:, j])[0], self.step_hours)
            _, physics, residual = self.bw_rate(z, dq, next_q, season[:, j])
            energy = energy + residual.square().mean()
            for key, value in dict(effective_temperature_mf=next_q, reversible_prediction=self.k * next_q, memory_prediction=self.s * z, bw_latent=z, bw_physics_rate=self.k * dq + self.s * physics, bw_residual_rate=self.s * residual).items():
                saved[key].append(value)
            q = next_q
        saved = {key: torch.stack(value, dim=1) for key, value in saved.items()}
        mf_prediction = saved['reversible_prediction'] + saved['memory_prediction']
        static, noise = self.decomposer.cutoff_hours()
        return dict(prediction=lf_prediction + mf_prediction, lf_prediction=lf_prediction, mf_prediction=mf_prediction, temperature_lf=temperature_lf, temperature_mf=temperature_mf, history_parts=history_parts, lf_anchor_offset=anchor_offset, ode_residual_penalty=energy / (4 * self.points_per_day), static_hours=static, noise_hours=noise, thermal_lag_hours=self.thermal_lag_hours(), **saved)

    def configure_stage(self, stage):
        for name, parameter in self.named_parameters():
            active = not name.startswith('decomposer.')
            if stage == 'bw_pretrain':
                active = name in ('s', 'beta_raw', 'gamma_raw', 'thermal_lag_raw', 'a_season_coeff') or name.startswith(('initial_z.', 'residual_field.'))
            if not self.cfg['enable_lf_seasonal_maps'] and name.startswith(('lf_alpha_season.', 'lf_beta_season.')):
                active = False
            parameter.requires_grad_(active)
