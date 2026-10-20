"""User-fixed va-2.1 final-temperature supervision, confirmed 2026-10-07.

Copy this file into each new experiment's source snapshot and call
final_temperature_objective(result, batch) on its final temperature output.
Prediction/target are normalized with training-period temperature statistics;
prediction, target and observed mask have shape (batch, 1440/sample_minutes, 1).
This loss has no trainable parameters and does not freeze the predictor.
"""
import torch
TREND_WEIGHT = 3.0

def trend_objective(result, batch):
    error = result['prediction'] - batch['target']
    mask = batch['mask'].to(error.dtype)
    hours = torch.arange(error.shape[1], dtype=error.dtype, device=error.device).reshape(1, -1, 1) * (24 / error.shape[1])
    count = mask.sum(1)
    centered = hours - (hours * mask).sum(1).unsqueeze(1) / count.unsqueeze(1)
    denominator = (centered.square() * mask).sum(1)
    slope_error = (centered * error * mask).sum(1) / denominator
    return (denominator * slope_error.square()).sum() / count.sum()

def final_temperature_objective(result, batch, trend_weight=TREND_WEIGHT):
    error = result['prediction'] - batch['target']
    point_mse = (error.square() * batch['mask']).sum() / batch['mask'].sum()
    return point_mse + trend_weight * trend_objective(result, batch)
