"""Independent research variants; the imported frozen source is read-only."""
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F
HERE = Path(__file__).resolve().parent
SNAPSHOT = HERE / 'frozen_project'
sys.path.insert(0, str(SNAPSHOT / 'experiments/v2_hybrid_kqsz_20261001'))
from hybrid_model import HybridKqsz

@torch.jit.script
def field(z: torch.Tensor, dq: torch.Tensor, q: torch.Tensor, season: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor, ac: torch.Tensor, w1: torch.Tensor, b1: torch.Tensor, w2: torch.Tensor, b2: torch.Tensor, mode: int, seasonal_scale: float, residual_scale: float):
    physics = torch.zeros_like(z)
    residual = torch.zeros_like(z)
    if mode != 5 and mode != 8:
        a = 1.0 + seasonal_scale * torch.tanh(season @ ac.t())
        physics = a * dq - beta * dq.abs() * z - gamma * dq * z.abs()
    if mode == 8:
        features = torch.cat((z, dq, q, season), dim=1)
        residual = F.linear(torch.tanh(F.linear(features, w1, b1)), w2, b2)
    elif mode != 6:
        features = torch.cat((z, dq, q), dim=1)
        residual = residual_scale * torch.tanh(F.linear(torch.tanh(F.linear(features, w1, b1)), w2, b2))
    return (physics + residual, physics, residual)

@torch.jit.script
def integrate(z: torch.Tensor, q: torch.Tensor, uh: torch.Tensor, uf: torch.Tensor, sh: torch.Tensor, sf: torch.Tensor, tau: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor, ac: torch.Tensor, w1: torch.Tensor, b1: torch.Tensor, w2: torch.Tensor, b2: torch.Tensor, mode: int, step: float, seasonal_scale: float, residual_scale: float, trace: bool):
    inputs = torch.cat((uh[:, 1:], uf), dim=1)
    seasons = torch.cat((sh, sf), dim=1)
    histories = uh.size(1) - 1
    energy = z.new_zeros(())
    qsave = torch.jit.annotate(list[torch.Tensor], [])
    zsave = torch.jit.annotate(list[torch.Tensor], [])
    dsave = torch.jit.annotate(list[torch.Tensor], [])
    psave = torch.jit.annotate(list[torch.Tensor], [])
    rsave = torch.jit.annotate(list[torch.Tensor], [])
    ipsave = torch.jit.annotate(list[torch.Tensor], [])
    irsave = torch.jit.annotate(list[torch.Tensor], [])
    factor = 1.0 - torch.exp(-step / tau)
    for j in range(inputs.size(1)):
        forcing = inputs[:, j]
        if mode == 4:
            nq = forcing.expand_as(q)
        else:
            nq = q + factor * (forcing - q)
        dq = (nq - q) / step
        season = seasons[:, j]
        a, pa, ra = field(z, dq, nq, season, beta, gamma, ac, w1, b1, w2, b2, mode, seasonal_scale, residual_scale)
        b, pb, rb = field(z + step * a / 2, dq, nq, season, beta, gamma, ac, w1, b1, w2, b2, mode, seasonal_scale, residual_scale)
        c, pc, rc = field(z + step * b / 2, dq, nq, season, beta, gamma, ac, w1, b1, w2, b2, mode, seasonal_scale, residual_scale)
        d, pd, rd = field(z + step * c, dq, nq, season, beta, gamma, ac, w1, b1, w2, b2, mode, seasonal_scale, residual_scale)
        if j < histories:
            energy = energy + ra.square().mean()
        z = z + step * (a + 2 * b + 2 * c + d) / 6
        if j >= histories:
            _, physics, residual = field(z, dq, nq, season, beta, gamma, ac, w1, b1, w2, b2, mode, seasonal_scale, residual_scale)
            energy = energy + residual.square().mean()
            qsave.append(nq)
            zsave.append(z)
            if trace:
                dsave.append(dq)
                psave.append(physics)
                rsave.append(residual)
                ipsave.append(step * (pa + 2 * pb + 2 * pc + pd) / 6)
                irsave.append(step * (ra + 2 * rb + 2 * rc + rd) / 6)
        q = nq
    return (torch.stack(qsave, 1), torch.stack(zsave, 1), energy / inputs.size(1), dsave, psave, rsave, ipsave, irsave)

class ResearchPRFM(HybridKqsz):

    def __init__(self, config, arm):
        super().__init__(config)
        self.arm = int(arm[1:])
        if self.arm == 4:
            del self.thermal_lag_raw
        if self.arm in (5, 7, 8):
            del self.beta_raw, self.gamma_raw, self.a_season_coeff
        if self.arm in (6, 7, 8):
            del self.residual_field
        if self.arm == 7:
            del self.initial_z, self.s
        if self.arm == 8:
            self.neural_field = nn.Sequential(nn.Linear(16, config['width']), nn.Tanh(), nn.Linear(config['width'], 4))
            nn.init.zeros_(self.neural_field[-1].weight)
            nn.init.zeros_(self.neural_field[-1].bias)
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(not name.startswith(('decomposer.', 'lf_alpha_season.', 'lf_beta_season.')))

    @torch.no_grad()
    def prepare_inputs(self, history, future_temperature, origin):
        if self.arm == 3:
            center = history.mean(1, keepdim=True)
            history_lf = center.expand_as(history)
            history_mf = history - center
            temperature_lf = center[:, :, :1].expand_as(future_temperature)
            temperature_mf = future_temperature - center[:, :, :1]
        else:
            hp = self.decomposer(history.transpose(1, 2))
            history_lf = hp['static'].transpose(1, 2)
            history_mf = history - history_lf if self.arm == 2 else hp['dynamic'].transpose(1, 2)
            temperature_window = torch.cat((history[:, self.points_per_day:, :1], future_temperature), 1)
            tp = self.decomposer(temperature_window.transpose(1, 2))
            temperature_lf = tp['static'].transpose(1, 2)[:, -self.points_per_day:]
            temperature_mf = (temperature_window - tp['static'].transpose(1, 2))[:, -self.points_per_day:] if self.arm == 2 else tp['dynamic'].transpose(1, 2)[:, -self.points_per_day:]
        recent = history_mf[:, -3 * self.points_per_day:]
        return dict(lf_temperature=temperature_lf, mf_temperature=temperature_mf, anchor_temperature=history_lf[:, -1:, :1], anchor_response=history_lf[:, -1:, 1:], hourly=recent.reshape(len(history), 72, 60 // self.cfg['sample_minutes'], 5).mean(2).flatten(1), history_driver=history_mf[:, -3 * self.points_per_day - 1:, :1], history_season=self.season_features(origin, 3 * self.points_per_day, start_offset=-3 * self.points_per_day), season=self.season_features(origin, self.points_per_day), anchor_season=self.season_features(origin - self.cfg['sample_minutes'], 1))

    def forward(self, data, trace=False):
        lf = self.lf_forward(data['lf_temperature'], data['season'])
        offset = data['anchor_response'] - self.lf_forward(data['anchor_temperature'], data['anchor_season'])
        if self.arm != 1:
            lf = lf + offset
        if self.arm == 7:
            q = data['history_driver'][:, 0].expand(-1, 4)
            factor = 1 - torch.exp(-self.step_hours / self.thermal_lag_hours())
            qs = []
            driver = torch.cat((data['history_driver'][:, 1:], data['mf_temperature']), 1)
            for j in range(driver.size(1)):
                q = q + factor * (driver[:, j] - q)
                if j >= 3 * self.points_per_day:
                    qs.append(q)
            q = torch.stack(qs, 1)
            z = torch.zeros_like(q)
            penalty = q.new_zeros(())
            details = {}
        else:
            z0 = torch.tanh(self.initial_z(data['hourly']))
            q0 = data['history_driver'][:, 0].expand(-1, 4)
            zero = q0.new_zeros(4)
            ac = q0.new_zeros(4, 4) if self.arm in (5, 8) else self.a_season_coeff
            beta = zero if self.arm in (5, 8) else F.softplus(self.beta_raw)
            gamma = zero if self.arm in (5, 8) else 0.9 * beta * torch.tanh(self.gamma_raw)
            tau = q0.new_ones(4) if self.arm == 4 else self.thermal_lag_hours()
            if self.arm == 6:
                w1, b1 = (q0.new_zeros(1, 12), q0.new_zeros(1))
                w2, b2 = (q0.new_zeros(4, 1), zero)
            else:
                network = self.neural_field if self.arm == 8 else self.residual_field
                w1, b1, w2, b2 = (network[0].weight, network[0].bias, network[2].weight, network[2].bias)
            q, z, penalty, ds, ps, rs, ips, irs = integrate(z0, q0, data['history_driver'], data['mf_temperature'], data['history_season'], data['season'], tau, beta, gamma, ac, w1, b1, w2, b2, self.arm, self.step_hours, float(self.cfg['seasonal_a_scale']), float(self.cfg['residual_scale']), trace)
            details = {} if not trace else dict(dq=torch.stack(ds, 1), physics_rate=torch.stack(ps, 1), neural_rate=torch.stack(rs, 1), physics_state_increment=torch.stack(ips, 1), neural_state_increment=torch.stack(irs, 1))
        reversible = self.k * q
        memory = torch.zeros_like(reversible) if self.arm == 7 else self.s * z
        return dict(prediction=lf + reversible + memory, lf=lf, reversible=reversible, memory=memory, q=q, z=z, rate_penalty=penalty, anchor_offset=offset, **details)

    @torch.no_grad()
    def parameter_report(self):
        result = dict(total_parameters=sum((p.numel() for p in self.parameters())), trainable_parameters=sum((p.numel() for p in self.parameters() if p.requires_grad)), active_names=[n for n, p in self.named_parameters() if p.requires_grad], k=self.k.tolist(), cutoff_hours=[float(x) for x in self.decomposer.cutoff_hours()])
        if self.arm != 4:
            result['thermal_lag_hours'] = self.thermal_lag_hours().tolist()
        if self.arm != 7:
            result['s'] = self.s.tolist()
        if self.arm not in (5, 7, 8):
            beta = F.softplus(self.beta_raw)
            result.update(beta=beta.tolist(), gamma=(0.9 * beta * torch.tanh(self.gamma_raw)).tolist(), seasonal_a_coefficients=self.a_season_coeff.tolist())
        return result
