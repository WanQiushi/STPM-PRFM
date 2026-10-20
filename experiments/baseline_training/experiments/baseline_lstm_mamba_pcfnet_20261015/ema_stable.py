"""Author EMA recurrence using deterministic double-precision parallel prefix."""
import torch
from torch import nn

class EMA(nn.Module):

    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        values = x.to(torch.float64)
        current = torch.cat((values[:, :1], self.alpha * values[:, 1:]), dim=1)
        offset = 1
        while offset < x.shape[1]:
            current = torch.cat((current[:, :offset], current[:, offset:] + (1 - self.alpha) ** offset * current[:, :-offset]), dim=1)
            offset *= 2
        return current.to(torch.float32)
