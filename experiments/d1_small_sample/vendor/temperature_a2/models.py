"""Normalized latent-ODE shape, historical level/amplitude heads and a frozen-base residual."""
import torch
from torch import nn
from ode_core_v9 import SmallLatentODE
from temperature_ablation import AblatedTemperature, build_warm_model

class ShapeODEResidual(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.sample_minutes = cfg['sample_minutes']
        self.query_points = 1440 // self.sample_minutes
        self.ode = SmallLatentODE(cfg)
        self.level_amplitude = nn.Sequential(nn.Linear(cfg['latent_dim'], 16), nn.Tanh(), nn.Linear(16, 2))
        self.residual = nn.Sequential(nn.Linear(cfg['latent_dim'] + 2 * self.query_points + 4, cfg['residual_width']), nn.Tanh(), nn.Linear(cfg['residual_width'], self.query_points))
        nn.init.zeros_(self.level_amplitude[-1].weight)
        nn.init.zeros_(self.level_amplitude[-1].bias)
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, batch):
        initial, anchor = self.ode.encoder(batch)
        self.ode.solver.ode_func.phase = batch['phase']
        latent = self.ode.solver(initial.unsqueeze(0), self.ode.query_hours)[0]
        decoded = self.ode.decoder(latent)[:, 1:, 0]
        centered = decoded - decoded.mean(1, keepdim=True)
        shape = centered / centered.square().mean(1, keepdim=True).sqrt().clamp_min(0.0001)
        recent = batch['recent'][:, 1:].reshape(-1, 10080 // self.sample_minutes, self.sample_minutes, 1).mean(2)[:, -self.query_points:, 0]
        previous_amplitude = recent.std(1, keepdim=True).clamp_min(0.02)
        scalar = self.level_amplitude(initial)
        level = anchor + scalar[:, :1]
        amplitude = previous_amplitude * scalar[:, 1:].exp()
        base = level + amplitude * shape
        calendar = torch.cat((batch['phase'].sin(), batch['phase'].cos()), -1)
        features = torch.cat((initial, recent - anchor, base - anchor, calendar), -1)
        residual = self.residual(features)
        return {'prediction': (base + residual).unsqueeze(-1), 'base': base.unsqueeze(-1), 'shape': shape.unsqueeze(-1), 'level': level, 'amplitude': amplitude, 'residual': residual.unsqueeze(-1), 'features': features}

class CachedResidual(nn.Module):

    def __init__(self, residual):
        super().__init__()
        self.residual = residual

    def forward(self, batch):
        return {'prediction': batch['base'] + self.residual(batch['features']).unsqueeze(-1)}

def build_model(cfg):
    return AblatedTemperature(cfg)
