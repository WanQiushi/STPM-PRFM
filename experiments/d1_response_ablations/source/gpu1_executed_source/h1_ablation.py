"""Approved response removals; original target decomposition remains unchanged."""
from if_release.runtime import expand as _release_expand
from pathlib import Path
import sys
import torch
from torch.nn import functional as F
H1_ORIGINAL = Path(_release_expand('${WORKSPACE}/outputs/va_1_20261004'))
sys.path.insert(0, str(H1_ORIGINAL / 'frozen_project/experiments/v2_hybrid_kqsz_20261001'))
from hybrid_model import HybridKqsz, SOURCE, load_module, core

class H1Ablation(HybridKqsz):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.ablation = cfg['ablation']
        if self.ablation == 'no_neural_memory':
            del self.residual_field

    def input_decomposition(self, x):
        if self.ablation == 'mean_split':
            center = self._history_center[:, :x.size(1)]
            background = center.expand_as(x)
            return dict(static=background, dynamic=x - background, noise=torch.zeros_like(x))
        return self.decomposer(x)

    def bw_rate(self, z, dq, q, season):
        if self.ablation == 'no_neural_memory':
            beta = F.softplus(self.beta_raw)
            gamma = 0.9 * beta * torch.tanh(self.gamma_raw)
            a = 1.0 + self.cfg['seasonal_a_scale'] * torch.tanh(season @ self.a_season_coeff.T)
            physics = a * dq - beta * dq.abs() * z - gamma * dq * z.abs()
            return (physics, physics, torch.zeros_like(z))
        return super().bw_rate(z, dq, q, season)

    def forward(self, history, future_temperature, origin):
        self._history_center = history.mean(1).unsqueeze(-1)
        history_parts = self.input_decomposition(history.transpose(1, 2))
        history_lf = history_parts['static'].transpose(1, 2)
        history_mf = history_parts['dynamic'].transpose(1, 2)
        temperature_window = torch.cat((history[:, self.points_per_day:, :1], future_temperature), dim=1)
        temperature_parts = self.input_decomposition(temperature_window.transpose(1, 2))
        temperature_lf = temperature_parts['static'].transpose(1, 2)[:, -self.points_per_day:]
        temperature_mf = temperature_parts['dynamic'].transpose(1, 2)[:, -self.points_per_day:]
        season = self.season_features(origin, self.points_per_day)
        mapped_anchor = self.lf_forward(history_lf[:, -1:, :1], self.season_features(origin - self.cfg['sample_minutes'], 1))
        anchor_offset = history_lf[:, -1:, 1:] - mapped_anchor
        if self.ablation == 'no_lf_anchor':
            anchor_offset = torch.zeros_like(anchor_offset)
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
        saved = {key: [] for key in ['effective_temperature_mf', 'reversible_prediction', 'memory_prediction', 'bw_latent', 'bw_physics_rate', 'bw_residual_rate', 'effective_temperature_rate', 'pure_bw_rate', 'raw_neural_rate']}
        for j in range(self.points_per_day):
            next_q = self.lag_step(q, temperature_mf[:, j])
            dq = (next_q - q) / self.step_hours
            z = core.rk4(z, lambda state: self.bw_rate(state, dq, next_q, season[:, j])[0], self.step_hours)
            _, physics, residual = self.bw_rate(z, dq, next_q, season[:, j])
            energy = energy + residual.square().mean()
            for key, value in dict(effective_temperature_mf=next_q, reversible_prediction=self.k * next_q, memory_prediction=self.s * z, bw_latent=z, bw_physics_rate=self.k * dq + self.s * physics, bw_residual_rate=self.s * residual, effective_temperature_rate=dq, pure_bw_rate=physics, raw_neural_rate=residual).items():
                saved[key].append(value)
            q = next_q
        saved = {key: torch.stack(value, dim=1) for key, value in saved.items()}
        mf_prediction = saved['reversible_prediction'] + saved['memory_prediction']
        static, noise = self.decomposer.cutoff_hours()
        return dict(prediction=lf_prediction + mf_prediction, lf_prediction=lf_prediction, mf_prediction=mf_prediction, temperature_lf=temperature_lf, temperature_mf=temperature_mf, history_parts=history_parts, lf_anchor_offset=anchor_offset, ode_residual_penalty=energy / (4 * self.points_per_day), static_hours=static, noise_hours=noise, thermal_lag_hours=self.thermal_lag_hours(), **saved)
