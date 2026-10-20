"""Only A1/A2/A4 changes to the fixed temperature architecture."""
import math
import torch
from torch import nn
from ode_core_v9 import HistoryEncoder, SmallLatentODE

class SharedTimeReadout(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.encoder = HistoryEncoder(cfg)
        self.readout = nn.Sequential(nn.Linear(cfg['latent_dim'] + 5, cfg['width']), nn.Tanh(), nn.Linear(cfg['width'], 1))
        query = (torch.arange(1440 // cfg['sample_minutes'], dtype=torch.float32) * cfg['sample_minutes'] + 1) / 60
        self.register_buffer('query_hours', torch.cat((torch.zeros(1), query)))

    def trajectory(self, batch):
        initial, anchor = self.encoder(batch)
        time = self.query_hours[None, :, None].expand(len(initial), -1, 1)
        rates = initial.new_tensor([2 * math.pi / 24, 2 * math.pi / (365.25 * 24)])
        angle = batch['phase'][:, None] + time * rates
        features = torch.cat((initial[:, None].expand(-1, time.shape[1], -1), angle.sin(), angle.cos(), time / 24), -1)
        return (initial, anchor, self.readout(features))

    def forward(self, batch):
        _, anchor, decoded = self.trajectory(batch)
        return {'prediction': anchor[:, None] + decoded[:, 1:] - decoded[:, :1]}

class AblatedTemperature(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.ablation = cfg['ablation']
        self.sample_minutes = cfg['sample_minutes']
        self.query_points = 1440 // self.sample_minutes
        self.ode = SharedTimeReadout(cfg) if self.ablation == 'A1' else SmallLatentODE(cfg)
        level_amplitude = nn.Sequential(nn.Linear(cfg['latent_dim'], 16), nn.Tanh(), nn.Linear(16, 2))
        nn.init.zeros_(level_amplitude[-1].weight)
        nn.init.zeros_(level_amplitude[-1].bias)
        if self.ablation != 'A2':
            self.level_amplitude = level_amplitude
        self.residual = nn.Sequential(nn.Linear(cfg['latent_dim'] + 2 * self.query_points + 4, cfg['residual_width']), nn.Tanh(), nn.Linear(cfg['residual_width'], self.query_points))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, batch):
        if self.ablation == 'A1':
            initial, anchor, decoded = self.ode.trajectory(batch)
        else:
            initial, anchor = self.ode.encoder(batch)
            self.ode.solver.ode_func.phase = batch['phase']
            latent = self.ode.solver(initial.unsqueeze(0), self.ode.query_hours)[0]
            decoded = self.ode.decoder(latent)
        decoded = decoded[:, 1:, 0]
        centered = decoded - decoded.mean(1, keepdim=True)
        shape = centered / centered.square().mean(1, keepdim=True).sqrt().clamp_min(0.0001)
        recent = batch['recent'][:, 1:].reshape(-1, 10080 // self.sample_minutes, self.sample_minutes, 1).mean(2)[:, -self.query_points:, 0]
        previous_amplitude = recent.std(1, keepdim=True).clamp_min(0.02)
        if self.ablation == 'A2':
            level, amplitude = (anchor, previous_amplitude)
        else:
            scalar = self.level_amplitude(initial)
            level = anchor + scalar[:, :1]
            amplitude = previous_amplitude * scalar[:, 1:].exp()
        base = level + amplitude * shape
        calendar = torch.cat((batch['phase'].sin(), batch['phase'].cos()), -1)
        features = torch.cat((initial, recent - anchor, base - anchor, calendar), -1)
        residual = self.residual(features)
        return {'prediction': (base + residual).unsqueeze(-1), 'base': base.unsqueeze(-1), 'shape': shape.unsqueeze(-1), 'level': level, 'amplitude': amplitude, 'residual': residual.unsqueeze(-1), 'features': features}

def build_warm_model(cfg):
    return SharedTimeReadout(cfg)
