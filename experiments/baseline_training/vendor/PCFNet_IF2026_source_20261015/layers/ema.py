"""Author EMA recurrence using deterministic double-precision parallel prefix."""
import torch
from torch import nn


class EMA(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        # Same definition as the author's commented recurrence:
        # y[0]=x[0]; y[t]=alpha*x[t]+(1-alpha)*y[t-1].
        # Author global powers produce0/0 for43200 points. Each prefix stage
        # doubles the included history via additions; no division or cumsum.
        values = x.to(torch.float64)
        current = torch.cat((values[:, :1], self.alpha * values[:, 1:]), dim=1)
        offset = 1
        while offset < x.shape[1]:
            current = torch.cat((current[:, :offset],
                                 current[:, offset:] + (1 - self.alpha) ** offset * current[:, :-offset]), dim=1)
            offset *= 2
        return current.to(torch.float32)
