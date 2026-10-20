from __future__ import annotations
import math
import sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
ROOT = Path(__file__).resolve().parents[2]
BASE_EXPERIMENT = ROOT / 'experiments' / 'fft_joint_neuralode_d1_20260929'
sys.path.insert(0, str(BASE_EXPERIMENT))
from model import LearnableFFT, inverse_sigmoid, rk4

def inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    value = torch.clamp(value, min=0.0001, max=20.0)
    return torch.log(torch.expm1(value))

class DirectBWJointFFTNeuralODE(nn.Module):
    """Seasonal LF mapping plus a direct-output Bouc--Wen Neural ODE.

    The MF displacement is the first ODE state. Temperature lag is only a
    forcing transformation and is never added algebraically to the output.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.displacements = 4
        self.step_hours = float(cfg['sample_minutes']) / 60.0
        self.points_per_day = 1440 // int(cfg['sample_minutes'])
        self.history_length = int(cfg['history_days']) * self.points_per_day
        self.decomposer = LearnableFFT(self.history_length, self.step_hours, cfg)
        width = int(cfg['width'])
        self.lf_alpha = nn.Parameter(torch.ones(self.displacements))
        self.lf_beta = nn.Parameter(torch.zeros(self.displacements))
        self.lf_alpha_season = nn.Sequential(nn.Linear(4, width), nn.Tanh(), nn.Linear(width, self.displacements))
        self.lf_beta_season = nn.Sequential(nn.Linear(4, width), nn.Tanh(), nn.Linear(width, self.displacements))
        self.lf_residual = nn.Sequential(nn.Linear(5, width), nn.Tanh(), nn.Linear(width, self.displacements))
        for module in (self.lf_alpha_season, self.lf_beta_season, self.lf_residual):
            nn.init.zeros_(module[-1].weight)
            nn.init.zeros_(module[-1].bias)
        lag_range = tuple(map(float, cfg['thermal_lag_hours_range']))
        initial_lag = float(cfg['init_thermal_lag_hours'])
        self.thermal_lag_range = lag_range
        self.thermal_lag_raw = nn.Parameter(torch.full((self.displacements,), inverse_sigmoid(initial_lag, *lag_range), dtype=torch.float32))
        self.initial_z = nn.Sequential(nn.Linear(72 * 5, width), nn.Tanh(), nn.Linear(width, self.displacements))
        self.a_raw = nn.Parameter(inverse_softplus(torch.ones(self.displacements)))
        self.beta_raw = nn.Parameter(inverse_softplus(torch.full((self.displacements,), 0.05)))
        self.gamma_raw = nn.Parameter(torch.zeros(self.displacements))
        self.bw_residual = nn.Sequential(nn.Linear(9, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 2))
        nn.init.zeros_(self.bw_residual[-1].weight)
        nn.init.zeros_(self.bw_residual[-1].bias)

    def season_features(self, origin, length, start_offset=0):
        offsets = torch.arange(start_offset, start_offset + length, device=origin.device, dtype=torch.float32)
        absolute_days = (origin.to(torch.float32)[:, None] + offsets[None] * float(self.cfg['sample_minutes'])) / 1440.0
        annual = 2.0 * math.pi * absolute_days / float(self.cfg['season_days'])
        semiannual = 2.0 * annual
        return torch.stack((torch.sin(annual), torch.cos(annual), torch.sin(semiannual), torch.cos(semiannual)), dim=-1)

    def thermal_lag_hours(self):
        low, high = self.thermal_lag_range
        return low + (high - low) * torch.sigmoid(self.thermal_lag_raw)

    def lag_step(self, state, forcing):
        tau = self.thermal_lag_hours()[None]
        alpha = 1.0 - torch.exp(-self.step_hours / tau)
        return state + alpha * (forcing.expand_as(state) - state)

    def physics_parameters(self):
        beta = F.softplus(self.beta_raw)
        return {'a': F.softplus(self.a_raw), 'beta': beta, 'gamma': 0.9 * beta * torch.tanh(self.gamma_raw)}

    def set_physics_parameters(self, a, beta, gamma):
        a = torch.as_tensor(a, dtype=self.a_raw.dtype, device=self.a_raw.device)
        beta = torch.as_tensor(beta, dtype=self.beta_raw.dtype, device=self.beta_raw.device)
        gamma = torch.as_tensor(gamma, dtype=self.gamma_raw.dtype, device=self.gamma_raw.device)
        beta = torch.clamp(beta, min=0.001)
        ratio = torch.clamp(gamma / (0.9 * beta), min=-0.95, max=0.95)
        with torch.no_grad():
            self.a_raw.copy_(inverse_softplus(torch.clamp(a, min=0.001)))
            self.beta_raw.copy_(inverse_softplus(beta))
            self.gamma_raw.copy_(torch.atanh(ratio))

    def lf_forward(self, temperature_lf, season):
        alpha = self.lf_alpha[None, None] + self.lf_alpha_season(season)
        beta = self.lf_beta[None, None] + self.lf_beta_season(season)
        residual = self.lf_residual(torch.cat((temperature_lf, season), dim=-1))
        return alpha * temperature_lf + beta + residual

    def bw_state_rate(self, state, du, u, u_mean, season):
        y = state[:, :self.displacements]
        z = state[:, self.displacements:]
        params = self.physics_parameters()
        physics = params['a'] * du - params['beta'] * du.abs() * y - params['gamma'] * du * y.abs()
        expanded_season = season[:, None].expand(-1, self.displacements, -1)
        features = torch.cat((torch.stack((y, z, u, du, u_mean), dim=-1), expanded_season), dim=-1)
        residual = float(self.cfg['residual_scale']) * torch.tanh(self.bw_residual(features))
        rate = torch.cat((physics + residual[..., 0], residual[..., 1]), dim=1)
        return (rate, physics, residual)

    def forward(self, history, future_temperature, origin):
        history_parts = self.decomposer(history.transpose(1, 2))
        history_lf = history_parts['static'].transpose(1, 2)
        history_mf = history_parts['dynamic'].transpose(1, 2)
        temperature_window = torch.cat((history[:, self.points_per_day:, :1], future_temperature), dim=1)
        temperature_parts = self.decomposer(temperature_window.transpose(1, 2))
        temperature_lf = temperature_parts['static'].transpose(1, 2)[:, -self.points_per_day:]
        temperature_mf = temperature_parts['dynamic'].transpose(1, 2)[:, -self.points_per_day:]
        season = self.season_features(origin, self.points_per_day)
        lf_prediction = self.lf_forward(temperature_lf, season)
        recent_mf = history_mf[:, -3 * self.points_per_day:]
        hourly = recent_mf.reshape(len(history), 72, 60 // int(self.cfg['sample_minutes']), 5).mean(dim=2)
        z = torch.tanh(self.initial_z(hourly.flatten(1)))
        u_history = history_mf[:, -3 * self.points_per_day - 1:, :1]
        effective_u = u_history[:, 0].expand(-1, self.displacements)
        for j in range(3 * self.points_per_day):
            effective_u = self.lag_step(effective_u, u_history[:, j + 1])
        effective_sequence = []
        previous_effective = effective_u
        for j in range(self.points_per_day):
            current_effective = self.lag_step(previous_effective, temperature_mf[:, j])
            effective_sequence.append(current_effective)
            previous_effective = current_effective
        effective_temperature_mf = torch.stack(effective_sequence, dim=1)
        effective_mean = effective_temperature_mf.mean(dim=1)
        y = history_mf[:, -1, 1:]
        state = torch.cat((y, z), dim=1)
        previous_effective = effective_u
        mf_prediction = []
        physics_rates = []
        residual_rates = []
        residual_energy = history.new_zeros(())
        for j in range(self.points_per_day):
            current_effective = effective_temperature_mf[:, j]
            du = (current_effective - previous_effective) / self.step_hours
            state = rk4(state, lambda current_state: self.bw_state_rate(current_state, du, current_effective, effective_mean, season[:, j])[0], self.step_hours)
            _, physics, residual = self.bw_state_rate(state, du, current_effective, effective_mean, season[:, j])
            residual_energy = residual_energy + residual.square().mean()
            mf_prediction.append(state[:, :self.displacements])
            physics_rates.append(physics)
            residual_rates.append(residual[..., 0])
            previous_effective = current_effective
        mf_prediction = torch.stack(mf_prediction, dim=1)
        physics_rates = torch.stack(physics_rates, dim=1)
        residual_rates = torch.stack(residual_rates, dim=1)
        prediction = lf_prediction + mf_prediction
        static_hours, noise_hours = self.decomposer.cutoff_hours()
        return {'prediction': prediction, 'lf_prediction': lf_prediction, 'mf_prediction': mf_prediction, 'temperature_lf': temperature_lf, 'temperature_mf': temperature_mf, 'effective_temperature_mf': effective_temperature_mf, 'bw_physics_rate': physics_rates, 'bw_residual_rate': residual_rates, 'history_parts': history_parts, 'ode_residual_penalty': residual_energy / self.points_per_day, 'static_hours': static_hours, 'noise_hours': noise_hours, 'thermal_lag_hours': self.thermal_lag_hours()}
