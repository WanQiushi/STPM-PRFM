"""Seven-channel single-factor removals relative to the fixed A2 response."""
import torch
from deflection_h1 import SevenDeflectionH1 as OriginalH1, SOURCE, load_module

class SevenDeflectionH1(OriginalH1):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.intervention = cfg['h1_intervention']
        if self.intervention == 'no_lag':
            del self.thermal_lag_raw
        if self.intervention == 'no_bw':
            del self.beta_raw
            del self.gamma_raw
            del self.a_season_coeff

    def thermal_lag_hours(self):
        if self.intervention == 'no_lag':
            return self.k.new_zeros(self.displacements)
        return super().thermal_lag_hours()

    def lag_step(self, q, u):
        if self.intervention == 'no_lag':
            return u.expand(-1, self.displacements)
        return super().lag_step(q, u)

    def bw_rate(self, z, dq, q, season):
        if self.intervention == 'no_bw':
            physics = torch.zeros_like(z)
            residual = self.cfg['residual_scale'] * torch.tanh(self.residual_field(torch.cat((z, dq, q), dim=1)))
            return (residual, physics, residual)
        return super().bw_rate(z, dq, q, season)
