"""Single-factor H1 removals; original frozen H1 equations otherwise retained."""
from if_release.runtime import expand as _release_expand
from pathlib import Path
import sys
import torch
H1_ORIGINAL = Path(_release_expand('${WORKSPACE}/outputs/va_1_20261004'))
sys.path.insert(0, str(H1_ORIGINAL / 'frozen_project/experiments/v2_hybrid_kqsz_20261001'))
from hybrid_model import HybridKqsz, SOURCE, load_module

class H1Ablation(HybridKqsz):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.ablation = cfg['ablation']
        if self.ablation == 'A5':
            del self.thermal_lag_raw
        if self.ablation == 'A6':
            del self.beta_raw
            del self.gamma_raw
            del self.a_season_coeff

    def thermal_lag_hours(self):
        if self.ablation == 'A5':
            return self.k.new_zeros(4)
        return super().thermal_lag_hours()

    def lag_step(self, q, u):
        if self.ablation == 'A5':
            return u.expand(-1, 4)
        return super().lag_step(q, u)

    def bw_rate(self, z, dq, q, season):
        if self.ablation == 'A6':
            physics = torch.zeros_like(z)
            residual = self.cfg['residual_scale'] * torch.tanh(self.residual_field(torch.cat((z, dq, q), dim=1)))
            return (residual, physics, residual)
        return super().bw_rate(z, dq, q, season)
