import torch
from torch import nn

class ZeroOutput(nn.Module):

    def __init__(self, original):
        super().__init__()
        self.original = original

    def forward(self, x):
        return self.original(x) * 0

class MeanSplit(nn.Module):
    """DC LF + demeaned MF + zero HF; retain both predictor branches."""

    def __init__(self, original):
        super().__init__()
        self.original = original

    def cutoff_hours(self):
        return self.original.cutoff_hours()

    def masks(self):
        low = (self.original.frequency == 0).to(self.original.frequency.dtype)
        return (low, 1 - low, low * 0)

    def forward(self, sequence):
        static = sequence.mean(dim=-1, keepdim=True).expand_as(sequence)
        dynamic = sequence - static
        low, mid, high = self.masks()
        return dict(static=static, dynamic=dynamic, noise=sequence - static - dynamic, static_mask=low, dynamic_mask=mid, noise_mask=high)

def v2_class(original, component):

    class Model(original):

        def __init__(self, cfg):
            super().__init__(cfg)
            if component == 'no_history_state':
                self.initial_z = ZeroOutput(self.initial_z)
            if component == 'no_neural_residual':
                self.bw_residual = ZeroOutput(self.bw_residual)
                self.lf_residual = ZeroOutput(self.lf_residual)
            if component == 'no_lf_seasonality':
                self.lf_alpha_season = ZeroOutput(self.lf_alpha_season)
                self.lf_beta_season = ZeroOutput(self.lf_beta_season)

        def lag_step(self, state, forcing):
            if component == 'no_thermal_lag':
                return forcing.expand_as(state)
            return super().lag_step(state, forcing)

        def bw_state_rate(self, state, du, u, u_mean, season):
            rate, physics, residual = super().bw_state_rate(state, du, u, u_mean, season)
            if component == 'no_bw':
                rate = torch.cat((residual[..., 0], residual[..., 1]), dim=1)
                physics = physics * 0
            return (rate, physics, residual)
    return Model

def vsharp_class(original, component):

    class Model(original):

        def __init__(self, cfg):
            super().__init__(cfg)
            if component == 'no_history_state':
                self.initial_z = ZeroOutput(self.initial_z)
            if component == 'no_neural_residual':
                self.residual_field = ZeroOutput(self.residual_field)
                self.slow_field = ZeroOutput(self.slow_field)

        def lag_step(self, state, forcing):
            if component == 'no_thermal_lag':
                return forcing.expand_as(state)
            return super().lag_step(state, forcing)

        def bw_rate(self, z, du, u):
            rate, residual = super().bw_rate(z, du, u)
            if component == 'no_bw':
                rate = residual
            return (rate, residual)

        def forward(self, history, future_temperature):
            output = super().forward(history, future_temperature)
            if component == 'no_reversible':
                output['mf_prediction'] = output['memory_prediction']
                output['reversible_prediction'] = output['reversible_prediction'] * 0
            elif component == 'no_memory':
                output['mf_prediction'] = output['reversible_prediction']
                output['memory_prediction'] = output['memory_prediction'] * 0
            output['prediction'] = output['lf_prediction'] + output['mf_prediction']
            return output
    return Model
