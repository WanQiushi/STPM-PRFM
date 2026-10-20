"""Masked 24-hour temperature-error decomposition, in degrees Celsius."""
import numpy as np
import pandas as pd

def decomposition(arrays):
    """Return aggregate metrics and per-window level/trend/remaining errors.

    ``prediction``, ``target`` and ``mask`` have shape [windows, 1440/sample_minutes, 1].
    Values are already Celsius; ``origin`` is minutes since 2022-01-01.
    Each linear projection is centered on that window's measured-node mask.
    The three SSE components are orthogonal temperature errors, not model
    branches. Slope RMSE gives each window equal weight; component RMS gives
    each measured node equal weight.
    """
    prediction = arrays['prediction'].astype(np.float64)[:, :, 0]
    target = arrays['target'].astype(np.float64)[:, :, 0]
    mask = arrays['mask'][:, :, 0].astype(np.float64)
    hours = np.arange(prediction.shape[1], dtype=np.float64) * (24 / prediction.shape[1])
    count = mask.sum(1)
    centered = hours[None] - (hours[None] * mask).sum(1)[:, None] / count[:, None]
    denominator = (centered ** 2 * mask).sum(1)
    error = prediction - target
    bias = (error * mask).sum(1) / count
    slope = (centered * error * mask).sum(1) / denominator
    remaining = error - bias[:, None] - slope[:, None] * centered
    level_sse = count * bias ** 2
    trend_sse = denominator * slope ** 2
    remaining_sse = (remaining ** 2 * mask).sum(1)
    predicted_slope = (centered * prediction * mask).sum(1) / denominator
    target_slope = (centered * target * mask).sum(1) / denominator
    metrics = {'level_error_rms_celsius': float(np.sqrt(level_sse.sum() / count.sum())), 'trend_error_rms_celsius': float(np.sqrt(trend_sse.sum() / count.sum())), 'remaining_error_rms_celsius': float(np.sqrt(remaining_sse.sum() / count.sum())), 'slope_error_rmse_celsius_per_hour': float(np.sqrt(np.mean(slope ** 2))), 'slope_error_bias_celsius_per_hour': float(slope.mean()), 'trend_direction_mismatches': int((np.sign(predicted_slope) != np.sign(target_slope)).sum()), 'windows': len(count)}
    daily = pd.DataFrame({'origin': arrays['origin'], 'date': pd.Timestamp('2022-01-01') + pd.to_timedelta(arrays['origin'], unit='m'), 'measured_nodes': count.astype(int), 'bias_celsius': bias, 'error_slope_celsius_per_hour': slope, 'predicted_slope_celsius_per_hour': predicted_slope, 'target_slope_celsius_per_hour': target_slope, 'temperature_rmse_celsius': np.sqrt((error ** 2 * mask).sum(1) / count), 'level_sse_celsius_squared': level_sse, 'trend_sse_celsius_squared': trend_sse, 'remaining_sse_celsius_squared': remaining_sse})
    return (metrics, daily)
