"""Fixed v11 driver and unchanged H1 equations; a distinct name avoids legacy model imports."""
import torch
from torch import nn

class H1NeuralParts(nn.Module):
    """A component view of H1's existing networks; it creates no parameters.

    H1 keeps ownership under its original names and calls these exact objects
    at the original equation locations. The view is not registered a second
    time in UnifiedVA1, so H1 checkpoints retain their original keys.
    """

    def __init__(self, h1):
        super().__init__()
        self.initial_z = h1.initial_z
        self.lf_residual = h1.lf_residual
        self.residual_field = h1.residual_field

class TemperatureNeuralParts(nn.Module):
    """A named view of v11's original temperature output residual."""

    def __init__(self, v11):
        super().__init__()
        self.output_residual = v11.residual

class UnifiedVA1(nn.Module):
    """Frozen temperature forecaster feeding the original trainable H1.

    ``history`` is already in the original H1 five-channel training scale.
    Cached forecasts are in degrees Celsius, never future measured labels.
    H1's original ``prediction`` remains in its displacement training scale;
    ``prediction_mm`` is supplied separately for reporting.
    """

    def __init__(self, v11, h1, temperature_mean, temperature_std, h1_mean, h1_std):
        super().__init__()
        self.v11 = v11
        self.h1 = h1
        self.v11.requires_grad_(False)
        self.v11.eval()
        object.__setattr__(self, 'neural_parts', H1NeuralParts(h1))
        object.__setattr__(self, 'temperature_parts', TemperatureNeuralParts(v11))
        self.temperature_parts.eval()
        self.register_buffer('temperature_mean', torch.as_tensor(temperature_mean, dtype=torch.float32).reshape(1, 1, 1))
        self.register_buffer('temperature_std', torch.as_tensor(temperature_std, dtype=torch.float32).reshape(1, 1, 1))
        self.register_buffer('h1_mean', torch.as_tensor(h1_mean, dtype=torch.float32).reshape(1, 1, 5))
        self.register_buffer('h1_std', torch.as_tensor(h1_std, dtype=torch.float32).reshape(1, 1, 5))

    def train(self, mode=True):
        super().train(mode)
        self.neural_parts.train(mode)
        self.temperature_parts.eval()
        self.v11.eval()
        return self

    def forward_cached(self, history, forecast_temperature_celsius, origin):
        future_temperature = (forecast_temperature_celsius - self.h1_mean[..., :1]) / self.h1_std[..., :1]
        result = self.h1(history, future_temperature, origin)
        result['forecast_temperature_celsius'] = forecast_temperature_celsius
        result['prediction_mm'] = result['prediction'] * self.h1_std[..., 1:] + self.h1_mean[..., 1:]
        return result

    def forward(self, batch):
        temperature = self.v11(batch)['prediction']
        temperature_celsius = temperature * self.temperature_std + self.temperature_mean
        return self.forward_cached(batch['history'], temperature_celsius, batch['origin'])
