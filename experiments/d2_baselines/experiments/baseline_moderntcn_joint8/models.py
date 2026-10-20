"""Official ModernTCN with the existing historical joint-forecast interface."""
from types import SimpleNamespace
from torch import nn
from modern_vendor.ModernTCN import Model

class Baseline(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.name = cfg['model']
        self.scientific_label = 'ModernTCN'
        self.backbone = Model(SimpleNamespace(**cfg['model_args']))
        nn.init.zeros_(self.backbone.model.head.linear.weight)
        nn.init.zeros_(self.backbone.model.head.linear.bias)

    def forward(self, batch):
        return dict(prediction=self.backbone(batch['history'], None))
