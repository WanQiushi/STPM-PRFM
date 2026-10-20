from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F

def rk4(state, rate, step):
    a = rate(state)
    b = rate(state + step * a / 2)
    c = rate(state + step * b / 2)
    d = rate(state + step * c)
    return state + step * (a + 2 * b + 2 * c + d) / 6

def inverse_sigmoid(value, low, high):
    ratio = min(max((value - low) / (high - low), 0.0001), 1 - 0.0001)
    return math.log(ratio / (1 - ratio))

class LearnableFFT(nn.Module):

    def __init__(self, length, dt_hours, cfg):
        super().__init__()
        self.length = int(length)
        self.dt_hours = float(dt_hours)
        self.static_range = tuple(map(float, cfg['static_hours_range']))
        self.noise_range = tuple(map(float, cfg['noise_hours_range']))
        self.order = float(cfg['mask_order'])
        self.theta_static = nn.Parameter(torch.tensor(inverse_sigmoid(cfg['init_static_hours'], *self.static_range), dtype=torch.float32))
        self.theta_noise = nn.Parameter(torch.tensor(inverse_sigmoid(cfg['init_noise_hours'], *self.noise_range), dtype=torch.float32))
        self.register_buffer('frequency', torch.fft.rfftfreq(self.length, d=self.dt_hours))

    def cutoff_hours(self):
        static = self.static_range[0] + (self.static_range[1] - self.static_range[0]) * torch.sigmoid(self.theta_static)
        noise = self.noise_range[0] + (self.noise_range[1] - self.noise_range[0]) * torch.sigmoid(self.theta_noise)
        return (static, noise)

    def masks(self):
        static_hours, noise_hours = self.cutoff_hours()
        low = 1.0 / (1.0 + (self.frequency * static_hours) ** self.order)
        signal = 1.0 / (1.0 + (self.frequency * noise_hours) ** self.order)
        mid = torch.clamp(signal - low, min=0.0)
        high = torch.clamp(1.0 - signal, min=0.0)
        return (low, mid, high)

    def forward(self, sequence):
        spectrum = torch.fft.rfft(sequence, dim=-1, norm='ortho')
        low, mid, high = self.masks()
        lf = torch.fft.irfft(spectrum * low[None, None], n=self.length, dim=-1, norm='ortho')
        mf = torch.fft.irfft(spectrum * mid[None, None], n=self.length, dim=-1, norm='ortho')
        hf = sequence - lf - mf
        return {'static': lf, 'dynamic': mf, 'noise': hf, 'static_mask': low, 'dynamic_mask': mid, 'noise_mask': high}

class JointFFTNeuralODE(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.displacements = 4
        self.step_hours = cfg['sample_minutes'] / 60.0
        self.points_per_day = 1440 // cfg['sample_minutes']
        self.history_length = cfg['history_days'] * self.points_per_day
        self.decomposer = LearnableFFT(self.history_length, self.step_hours, cfg)
        width = int(cfg['width'])
        self.lf_gain = nn.Parameter(torch.ones(self.displacements))
        self.slow_field = nn.Sequential(nn.Linear(self.displacements + 1 + self.displacements, width), nn.Tanh(), nn.Linear(width, self.displacements))
        nn.init.normal_(self.slow_field[-1].weight, std=0.001)
        nn.init.zeros_(self.slow_field[-1].bias)
        self.initial_z = nn.Sequential(nn.Linear(72 * 5, width), nn.Tanh(), nn.Linear(width, self.displacements))
        self.k = nn.Parameter(torch.ones(self.displacements))
        self.s = nn.Parameter(torch.full((self.displacements,), 0.1))
        self.beta_raw = nn.Parameter(torch.zeros(self.displacements))
        self.gamma_raw = nn.Parameter(torch.zeros(self.displacements))
        self.residual_field = nn.Sequential(nn.Linear(3 * self.displacements, width), nn.Tanh(), nn.Linear(width, self.displacements))
        nn.init.zeros_(self.residual_field[-1].weight)
        nn.init.zeros_(self.residual_field[-1].bias)

    def bw_rate(self, z, du, u):
        beta = F.softplus(self.beta_raw)
        gamma = 0.9 * beta * torch.tanh(self.gamma_raw)
        du_full = du.expand_as(z)
        u_full = u.expand_as(z)
        residual = self.cfg['residual_scale'] * torch.tanh(self.residual_field(torch.cat((z, du_full, u_full), dim=1)))
        base = du_full - beta * du_full.abs() * z - gamma * du_full * z.abs()
        return (base + residual, residual)

    def forward(self, history, future_temperature):
        history_parts = self.decomposer(history.transpose(1, 2))
        history_lf = history_parts['static'].transpose(1, 2)
        history_mf = history_parts['dynamic'].transpose(1, 2)
        temperature_window = torch.cat((history[:, self.points_per_day:, :1], future_temperature), dim=1)
        temperature_parts = self.decomposer(temperature_window.transpose(1, 2))
        temperature_lf = temperature_parts['static'].transpose(1, 2)[:, -self.points_per_day:]
        temperature_mf = temperature_parts['dynamic'].transpose(1, 2)[:, -self.points_per_day:]
        c = history_lf[:, -1, 1:]
        previous_temperature = history_lf[:, -1, :1]
        trend = (c - history_lf[:, -1 - self.points_per_day, 1:]) / 24.0
        rate_scale = trend.abs() + 0.01
        lf_prediction = []
        for j in range(self.points_per_day):
            next_temperature = temperature_lf[:, j]
            d_temperature = (next_temperature - previous_temperature) / self.step_hours

            def slow_rate(state):
                learned = rate_scale * torch.tanh(self.slow_field(torch.cat((state, next_temperature, trend), dim=1)))
                return self.lf_gain * d_temperature + trend + learned
            c = rk4(c, slow_rate, self.step_hours)
            lf_prediction.append(c)
            previous_temperature = next_temperature
        lf_prediction = torch.stack(lf_prediction, dim=1)
        recent_mf = history_mf[:, -3 * self.points_per_day:]
        hourly = recent_mf.reshape(len(history), 72, 60 // self.cfg['sample_minutes'], 5).mean(dim=2)
        z = torch.tanh(self.initial_z(hourly.flatten(1)))
        residual_energy = history.new_zeros(())
        u_history = history_mf[:, -3 * self.points_per_day - 1:, :1]
        for j in range(3 * self.points_per_day):
            current_u = u_history[:, j + 1]
            du = (current_u - u_history[:, j]) / self.step_hours
            _, residual = self.bw_rate(z, du, current_u)
            residual_energy = residual_energy + residual.square().mean()
            z = rk4(z, lambda state: self.bw_rate(state, du, current_u)[0], self.step_hours)
        mf_prediction = []
        previous_u = history_mf[:, -1, :1]
        for j in range(self.points_per_day):
            current_u = temperature_mf[:, j]
            du = (current_u - previous_u) / self.step_hours
            _, residual = self.bw_rate(z, du, current_u)
            residual_energy = residual_energy + residual.square().mean()
            z = rk4(z, lambda state: self.bw_rate(state, du, current_u)[0], self.step_hours)
            mf_prediction.append(self.k * current_u + self.s * z)
            previous_u = current_u
        mf_prediction = torch.stack(mf_prediction, dim=1)
        prediction = lf_prediction + mf_prediction
        residual_penalty = residual_energy / (4 * self.points_per_day)
        static_hours, noise_hours = self.decomposer.cutoff_hours()
        return {'prediction': prediction, 'lf_prediction': lf_prediction, 'mf_prediction': mf_prediction, 'temperature_lf': temperature_lf, 'temperature_mf': temperature_mf, 'history_parts': history_parts, 'ode_residual_penalty': residual_penalty, 'static_hours': static_hours, 'noise_hours': noise_hours}

def bw_fitness(u, y, dt_hours):
    du = torch.diff(u) / dt_hours
    dy = torch.diff(y) / dt_hours
    y_mid = 0.5 * (y[:-1] + y[1:])
    design = torch.stack((du, -torch.abs(du) * y_mid, -du * torch.abs(y_mid)), dim=1)
    reg = 1e-06 * torch.eye(3, device=u.device, dtype=u.dtype)
    params = torch.linalg.solve(design.T @ design + reg, design.T @ dy)
    pred = design @ params
    ss_res = torch.sum((dy - pred) ** 2)
    ss_tot = torch.sum((dy - dy.mean()) ** 2) + 1e-08
    return torch.clamp(1.0 - ss_res / ss_tot, 0.0, 1.0)

def stage1_loss(model, history, weights):
    temperature = history[:, :, :1].expand(-1, -1, model.displacements)
    displacement = history[:, :, 1:]
    pairs = torch.stack((temperature, displacement), dim=2).permute(0, 1, 3, 2)
    pairs = pairs.permute(0, 2, 3, 1).reshape(-1, 2, model.history_length)
    result = model.decomposer(pairs)
    static = result['static']
    dynamic = result['dynamic']
    noise = result['noise']
    low_mask = result['static_mask']
    high_mask = result['noise_mask']
    static_frequency = torch.fft.rfft(static, dim=-1, norm='ortho')
    noise_frequency = torch.fft.rfft(noise, dim=-1, norm='ortho')
    low_purity = torch.mean(torch.abs(static_frequency) ** 2 * (1.0 - low_mask)[None, None] ** 2)
    high_purity = torch.mean(torch.abs(noise_frequency) ** 2 * (1.0 - high_mask)[None, None] ** 2)
    u_static = static[:, 0].reshape(-1)
    y_static = static[:, 1].reshape(-1)
    u_centered = u_static - u_static.mean()
    y_centered = y_static - y_static.mean()
    static_r2 = torch.mean(u_centered * y_centered).square() / (torch.mean(u_centered.square()) * torch.mean(y_centered.square()) + 1e-08)
    bw_r2 = torch.stack([bw_fitness(item[0], item[1], model.step_hours) for item in dynamic]).mean()
    total = weights['static_lf'] * low_purity + weights['linear'] * (1.0 - static_r2) + weights['bw'] * (1.0 - bw_r2) + weights['noise_hf'] * high_purity
    return (total, {'stage1_static_r2': static_r2, 'stage1_bw_r2': bw_r2, 'stage1_low_purity': low_purity, 'stage1_high_purity': high_purity})
