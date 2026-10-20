from __future__ import annotations
import importlib.util
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
ROOT = Path(__file__).resolve().parents[2]
PARENT_EXPERIMENT = ROOT / 'experiments' / 'fft_joint_neuralode_d1_anchor_seasonal_bw_20260929'
parent_spec = importlib.util.spec_from_file_location('anchor_seasonal_model_base', PARENT_EXPERIMENT / 'model.py')
parent_module = importlib.util.module_from_spec(parent_spec)
parent_spec.loader.exec_module(parent_module)
AnchoredSeasonalBWModel = parent_module.AnchoredSeasonalBWModel

def inverse_sigmoid(value: float, low: float, high: float) -> float:
    ratio = min(max((value - low) / (high - low), 0.0001), 1.0 - 0.0001)
    return float(torch.log(torch.tensor(ratio / (1.0 - ratio))))

class EdgeSafeFFT(nn.Module):
    """Three-band FFT with endpoint detrending and reflection padding.

    The affine endpoint bridge is assigned to LF. The residual is reflected
    before the rFFT, filtered on the padded grid, and cropped back to the
    causal window. This prevents the rFFT's implicit circular boundary from
    turning a terminal daily trend into an HF residual.
    """

    def __init__(self, length: int, dt_hours: float, cfg: dict):
        super().__init__()
        self.length = int(length)
        self.dt_hours = float(dt_hours)
        self.static_range = tuple(map(float, cfg['static_hours_range']))
        self.noise_range = tuple(map(float, cfg['noise_hours_range']))
        self.order = float(cfg['mask_order'])
        self.pad_points = int(round(float(cfg['edge_padding_hours']) / self.dt_hours))
        if self.pad_points >= self.length:
            raise ValueError('edge padding must be shorter than the input window')
        self.theta_static = nn.Parameter(torch.tensor(inverse_sigmoid(float(cfg['init_static_hours']), *self.static_range), dtype=torch.float32))
        self.theta_noise = nn.Parameter(torch.tensor(inverse_sigmoid(float(cfg['init_noise_hours']), *self.noise_range), dtype=torch.float32))
        self.register_buffer('frequency', torch.fft.rfftfreq(self.length, d=self.dt_hours))
        padded_length = self.length + 2 * self.pad_points
        self.register_buffer('padded_frequency', torch.fft.rfftfreq(padded_length, d=self.dt_hours))

    def cutoff_hours(self):
        static_low, static_high = self.static_range
        noise_low, noise_high = self.noise_range
        static = static_low + (static_high - static_low) * torch.sigmoid(self.theta_static)
        noise = noise_low + (noise_high - noise_low) * torch.sigmoid(self.theta_noise)
        return (static, noise)

    def _masks(self, frequency):
        static_hours, noise_hours = self.cutoff_hours()
        low = 1.0 / (1.0 + (frequency * static_hours) ** self.order)
        signal = 1.0 / (1.0 + (frequency * noise_hours) ** self.order)
        mid = torch.clamp(signal - low, min=0.0)
        high = torch.clamp(1.0 - signal, min=0.0)
        return (low, mid, high)

    def masks(self):
        return self._masks(self.frequency)

    def forward(self, sequence):
        if sequence.shape[-1] != self.length:
            raise ValueError(f'expected window length {self.length}, got {sequence.shape[-1]}')
        ramp = torch.linspace(0.0, 1.0, self.length, device=sequence.device, dtype=sequence.dtype)
        endpoint_trend = sequence[..., :1] + (sequence[..., -1:] - sequence[..., :1]) * ramp
        residual = sequence - endpoint_trend
        padded = F.pad(residual, (self.pad_points, self.pad_points), mode='reflect')
        spectrum = torch.fft.rfft(padded, dim=-1, norm='ortho')
        low, mid, _ = self._masks(self.padded_frequency)
        low_padded = torch.fft.irfft(spectrum * low, n=padded.shape[-1], dim=-1, norm='ortho')
        mid_padded = torch.fft.irfft(spectrum * mid, n=padded.shape[-1], dim=-1, norm='ortho')
        crop = slice(self.pad_points, self.pad_points + self.length)
        static = low_padded[..., crop] + endpoint_trend
        dynamic = mid_padded[..., crop]
        noise = sequence - static - dynamic
        original_low, original_mid, original_high = self.masks()
        return {'static': static, 'dynamic': dynamic, 'noise': noise, 'static_mask': original_low, 'dynamic_mask': original_mid, 'noise_mask': original_high}

class EdgeSafeAnchoredSeasonalBWModel(AnchoredSeasonalBWModel):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.decomposer = EdgeSafeFFT(self.history_length, self.step_hours, cfg)
