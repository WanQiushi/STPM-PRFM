import torch
from torch import nn

class DEMA(nn.Module):
    """
    Double Exponential Moving Average (DEMA) block to highlight the trend of time series
    """

    def __init__(self, alpha, beta):
        super(DEMA, self).__init__()
        self.alpha = alpha.to(device='cuda')
        self.beta = beta.to(device='cuda')

    def forward(self, x):
        s_prev = x[:, 0, :]
        b = x[:, 1, :] - s_prev
        res = [s_prev.unsqueeze(1)]
        for t in range(1, x.shape[1]):
            xt = x[:, t, :]
            s = self.alpha * xt + (1 - self.alpha) * (s_prev + b)
            b = self.beta * (s - s_prev) + (1 - self.beta) * b
            s_prev = s
            res.append(s.unsqueeze(1))
        return torch.cat(res, dim=1)
