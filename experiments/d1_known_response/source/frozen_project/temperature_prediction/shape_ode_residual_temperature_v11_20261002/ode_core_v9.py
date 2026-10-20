"""Small deterministic Latent ODE and matched direct GRU temperature models."""
import math
import sys
from pathlib import Path
import torch
from torch import nn
sys.path.insert(0, str(Path(__file__).resolve().parent / 'vendor_latent_ode'))
from lib.diffeq_solver import DiffeqSolver
from lib.encoder_decoder import Decoder
from lib.ode_func import ODEFunc

class HistoryEncoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.gru = nn.GRU(2, cfg['width'], batch_first=True)
        self.initial = nn.Sequential(nn.Linear(cfg['width'] + 60 + 4 + 1, cfg['width']), nn.Tanh(), nn.Linear(cfg['width'], cfg['latent_dim']))

    def forward(self, batch):
        anchor = batch['recent'][:, -1]
        temperature = batch['recent'][:, 1:].reshape(-1, 672, 15, 1).mean(2)
        coverage = batch['history_mask'][:, 1:].reshape(-1, 672, 15, 1).float().mean(2)
        sequence = torch.cat((temperature - anchor[:, None], coverage), -1)
        _, hidden = self.gru(sequence)
        features = torch.cat((hidden[-1], batch['daily'][:, -30:].flatten(1) - anchor, batch['coverage'][:, -30:].flatten(1), batch['phase'].sin(), batch['phase'].cos(), anchor), -1)
        return (self.initial(features), anchor)

class CalendarODEFunc(ODEFunc):
    """Keep the original ODEFunc API and condition its field on known time."""

    def get_ode_gradient_nn(self, t_local, y):
        angular_rates = y.new_tensor([2 * math.pi / 24, 2 * math.pi / (365.25 * 24)])
        angle = self.phase + t_local * angular_rates
        calendar = torch.cat((angle.sin(), angle.cos()), -1).unsqueeze(0)
        elapsed = torch.ones_like(y[..., :1]) * (t_local / 24)
        return self.gradient_net(torch.cat((y, calendar, elapsed), -1))

class SmallLatentODE(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.encoder = HistoryEncoder(cfg)
        latent = cfg['latent_dim']
        field_network = nn.Sequential(nn.Linear(latent + 5, cfg['width']), nn.Tanh(), nn.Linear(cfg['width'], latent))
        field = CalendarODEFunc(1, latent, field_network)
        self.solver = DiffeqSolver(1, field, 'rk4', latent)
        self.decoder = Decoder(latent, 1)
        query_hours = (torch.arange(96, dtype=torch.float32) * 15 + 1) / 60
        self.register_buffer('query_hours', torch.cat((torch.zeros(1), query_hours)))

    def forward(self, batch):
        initial, anchor = self.encoder(batch)
        self.solver.ode_func.phase = batch['phase']
        latent_path = self.solver(initial.unsqueeze(0), self.query_hours)[0]
        decoded = self.decoder(latent_path)
        prediction = anchor[:, None] + decoded[:, 1:] - decoded[:, :1]
        return {'prediction': prediction}

class MatchedDirectGRU(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.encoder = HistoryEncoder(cfg)
        self.head = nn.Sequential(nn.Linear(cfg['latent_dim'], cfg['width']), nn.Tanh(), nn.Linear(cfg['width'], 96))

    def forward(self, batch):
        initial, anchor = self.encoder(batch)
        return {'prediction': anchor[:, None] + self.head(initial).unsqueeze(-1)}

def build_model(cfg):
    return {'latent_ode': SmallLatentODE, 'direct_gru': MatchedDirectGRU}[cfg['model']](cfg)
