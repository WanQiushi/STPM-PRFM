from __future__ import annotations
import math
import sys
from pathlib import Path
import torch
from torch import nn
ROOT = Path(__file__).resolve().parents[2]
BASE_EXPERIMENT = ROOT / 'experiments' / 'fft_joint_neuralode_d1_20260929'
sys.path.insert(0, str(BASE_EXPERIMENT))
from model import JointFFTNeuralODE as BaseJointFFTNeuralODE, inverse_sigmoid, rk4

class CorrectedJointFFTNeuralODE(BaseJointFFTNeuralODE):
    """Joint model with a causal thermal lag and a non-ballistic LF forecast."""

    def __init__(self, cfg):
        super().__init__(cfg)
        lag_range = tuple(map(float, cfg['thermal_lag_hours_range']))
        initial = float(cfg['init_thermal_lag_hours'])
        initial_raw = inverse_sigmoid(initial, *lag_range)
        self.thermal_lag_range = lag_range
        self.thermal_lag_raw = nn.Parameter(torch.full((self.displacements,), initial_raw, dtype=torch.float32))

    def thermal_lag_hours(self):
        low, high = self.thermal_lag_range
        return low + (high - low) * torch.sigmoid(self.thermal_lag_raw)

    def lag_step(self, state, forcing):
        tau = self.thermal_lag_hours()[None]
        alpha = 1.0 - torch.exp(-self.step_hours / tau)
        return state + alpha * (forcing.expand_as(state) - state)

    def forward(self, history, future_temperature):
        history_parts = self.decomposer(history.transpose(1, 2))
        history_lf = history_parts['static'].transpose(1, 2)
        history_mf = history_parts['dynamic'].transpose(1, 2)
        temperature_window = torch.cat((history[:, self.points_per_day:, :1], future_temperature), dim=1)
        temperature_parts = self.decomposer(temperature_window.transpose(1, 2))
        temperature_lf = temperature_parts['static'].transpose(1, 2)[:, -self.points_per_day:]
        temperature_mf = temperature_parts['dynamic'].transpose(1, 2)[:, -self.points_per_day:]
        c = history_lf[:, -1, 1:]
        lf_initial = c
        previous_temperature = history_lf[:, -1, :1]
        trend = (c - history_lf[:, -1 - self.points_per_day, 1:]) / 24.0
        rate_scale = trend.abs() + 0.01
        lf_prediction = []
        for j in range(self.points_per_day):
            next_temperature = temperature_lf[:, j]
            d_temperature = (next_temperature - previous_temperature) / self.step_hours

            def slow_rate(state):
                learned = rate_scale * torch.tanh(self.slow_field(torch.cat((state, next_temperature, trend), dim=1)))
                return self.lf_gain * d_temperature + learned
            c = rk4(c, slow_rate, self.step_hours)
            lf_prediction.append(c)
            previous_temperature = next_temperature
        lf_prediction = torch.stack(lf_prediction, dim=1)
        recent_mf = history_mf[:, -3 * self.points_per_day:]
        hourly = recent_mf.reshape(len(history), 72, 60 // self.cfg['sample_minutes'], 5).mean(dim=2)
        z = torch.tanh(self.initial_z(hourly.flatten(1)))
        residual_energy = history.new_zeros(())
        u_history = history_mf[:, -3 * self.points_per_day - 1:, :1]
        effective_u = u_history[:, 0].expand(-1, self.displacements)
        for j in range(3 * self.points_per_day):
            next_effective_u = self.lag_step(effective_u, u_history[:, j + 1])
            du = (next_effective_u - effective_u) / self.step_hours
            _, residual = self.bw_rate(z, du, next_effective_u)
            residual_energy = residual_energy + residual.square().mean()
            z = rk4(z, lambda state: self.bw_rate(state, du, next_effective_u)[0], self.step_hours)
            effective_u = next_effective_u
        mf_prediction = []
        effective_temperature_mf = []
        reversible_prediction = []
        memory_prediction = []
        for j in range(self.points_per_day):
            next_effective_u = self.lag_step(effective_u, temperature_mf[:, j])
            du = (next_effective_u - effective_u) / self.step_hours
            _, residual = self.bw_rate(z, du, next_effective_u)
            residual_energy = residual_energy + residual.square().mean()
            z = rk4(z, lambda state: self.bw_rate(state, du, next_effective_u)[0], self.step_hours)
            reversible = self.k * next_effective_u
            memory = self.s * z
            mf_prediction.append(reversible + memory)
            effective_temperature_mf.append(next_effective_u)
            reversible_prediction.append(reversible)
            memory_prediction.append(memory)
            effective_u = next_effective_u
        mf_prediction = torch.stack(mf_prediction, dim=1)
        effective_temperature_mf = torch.stack(effective_temperature_mf, dim=1)
        reversible_prediction = torch.stack(reversible_prediction, dim=1)
        memory_prediction = torch.stack(memory_prediction, dim=1)
        prediction = lf_prediction + mf_prediction
        residual_penalty = residual_energy / (4 * self.points_per_day)
        static_hours, noise_hours = self.decomposer.cutoff_hours()
        return {'prediction': prediction, 'lf_prediction': lf_prediction, 'mf_prediction': mf_prediction, 'temperature_lf': temperature_lf, 'temperature_mf': temperature_mf, 'effective_temperature_mf': effective_temperature_mf, 'reversible_prediction': reversible_prediction, 'memory_prediction': memory_prediction, 'lf_initial': lf_initial, 'history_parts': history_parts, 'ode_residual_penalty': residual_penalty, 'static_hours': static_hours, 'noise_hours': noise_hours, 'thermal_lag_hours': self.thermal_lag_hours()}
