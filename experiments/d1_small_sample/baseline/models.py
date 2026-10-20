"""Eight minute-sampled joint temperature/displacement baselines.

All models forecast the five channels from their historical values only.
"""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import math
import sys
import torch
from torch import nn
from torch.nn import functional as F
PROJECT = Path(__file__).resolve().parents[1]
VENDOR = PROJECT / 'vendor/Time-Series-Library_20261002'
sys.path.insert(0, str(VENDOR))

def official_module(name):
    module_name = 'displacement_official_' + name
    spec = importlib.util.spec_from_file_location(module_name, VENDOR / 'models' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def extend_position_embeddings(model, max_len):
    from layers.Embed import PositionalEmbedding
    for module in model.modules():
        if isinstance(module, PositionalEmbedding) and module.pe.shape[1] < max_len:
            width = module.pe.shape[-1]
            position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
            scale = torch.exp(torch.arange(0, width, 2, dtype=torch.float32) * (-math.log(10000.0) / width))
            pe = torch.zeros(max_len, width, dtype=torch.float32)
            pe[:, 0::2] = torch.sin(position * scale)
            pe[:, 1::2] = torch.cos(position * scale)
            module.pe = pe.unsqueeze(0)

class CausalConv1d(nn.Module):

    def __init__(self, channels_in, channels_out, kernel_size=3, dilation=1):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels_in, channels_out, kernel_size, dilation=dilation)

    def forward(self, values):
        return self.conv(F.pad(values, (self.left_padding, 0)))

class CausalResidual(nn.Module):

    def __init__(self, width, dilation):
        super().__init__()
        self.conv = CausalConv1d(width, width, 3, dilation)

    def forward(self, values):
        return values + F.gelu(self.conv(values))

class PatchLSTM(nn.Module):
    """240-minute learned history patches; direct minute-resolution joint head."""

    def __init__(self, args):
        super().__init__()
        self.pred_len = args.pred_len
        self.c_out = args.c_out
        self.patch = nn.Conv1d(args.enc_in, args.d_model, kernel_size=args.patch_len, stride=args.patch_len)
        self.encoder = nn.LSTM(args.d_model, args.d_model, num_layers=2, batch_first=True, dropout=args.dropout)
        self.output = nn.Linear(args.d_model, args.pred_len * args.c_out)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, history):
        centered = history - history[:, -1:]
        patches = F.gelu(self.patch(centered.transpose(1, 2))).transpose(1, 2)
        encoded, _ = self.encoder(patches)
        return history[:, -1:] + self.output(encoded[:, -1]).reshape(-1, self.pred_len, self.c_out)

class PatchCNN(nn.Module):
    """Full history TCN over learned patches; direct joint forecast head."""

    def __init__(self, args):
        super().__init__()
        self.pred_len = args.pred_len
        self.c_out = args.c_out
        self.patch = nn.Conv1d(args.enc_in, args.d_model, kernel_size=args.patch_len, stride=args.patch_len)
        self.encoder = nn.Sequential(*(CausalResidual(args.d_model, d) for d in (1, 2, 4, 8, 16, 32, 64, 128)))
        self.output = nn.Linear(args.d_model, args.pred_len * args.c_out)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, history):
        centered = history - history[:, -1:]
        patches = F.gelu(self.patch(centered.transpose(1, 2)))
        context = self.encoder(patches)[:, :, -1]
        return history[:, -1:] + self.output(context).reshape(-1, self.pred_len, self.c_out)

class LowRankTemporal(nn.Module):

    def __init__(self, length, horizon, rank):
        super().__init__()
        self.horizon = horizon
        self.down = nn.Linear(length, rank, bias=False)
        self.up = nn.Linear(rank, length + horizon, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, values):
        future = values[:, :, -self.horizon:].mean(-1, keepdim=True).expand(-1, -1, self.horizon)
        return torch.cat((values, future), dim=-1) + self.up(self.down(values))

class TimesNetLR(nn.Module):
    """Retain the declared rank64 adaptation and official TimesBlocks."""

    def __init__(self, args):
        super().__init__()
        from layers.Embed import DataEmbedding
        official = official_module('TimesNet')
        self.pred_len = args.pred_len
        self.model = nn.ModuleList([official.TimesBlock(args) for _ in range(args.e_layers)])
        self.enc_embedding = DataEmbedding(args.enc_in, args.d_model, args.embed, args.freq, args.dropout)
        self.predict_linear = LowRankTemporal(args.seq_len, args.pred_len, args.temporal_rank)
        self.layer_norm = nn.LayerNorm(args.d_model)
        self.projection = nn.Linear(args.d_model, args.c_out)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        extend_position_embeddings(self, args.seq_len + args.pred_len)

    def forward(self, history, marks=None, x_dec=None, decoder_marks=None):
        means = history.mean(1, keepdim=True).detach()
        centered = history - means
        stdev = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-05)
        encoded = self.enc_embedding(centered / stdev, None)
        encoded = self.predict_linear(encoded.transpose(1, 2)).transpose(1, 2)
        for block in self.model:
            encoded = self.layer_norm(block(encoded))
        return self.projection(encoded[:, -self.pred_len:]) * stdev + means

class Baseline(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.name = cfg['model']
        self.scientific_label = {'TimesNet': 'TimesNet-LR', 'CNN': 'CNN-Patch-Direct', 'LSTM': 'LSTM-Patch-Direct'}.get(self.name, self.name)
        args = SimpleNamespace(**cfg['model_args'])
        self.label_len, self.pred_len = (args.label_len, args.pred_len)
        if self.name == 'ModernTCN':
            from modern_vendor.ModernTCN import Model
            self.backbone = Model(args)
            nn.init.zeros_(self.backbone.model.head.linear.weight)
            nn.init.zeros_(self.backbone.model.head.linear.bias)
        elif self.name == 'LSTM':
            self.backbone = PatchLSTM(args)
        elif self.name == 'CNN':
            self.backbone = PatchCNN(args)
        elif self.name == 'TimesNet':
            self.backbone = TimesNetLR(args)
        elif self.name == 'PatchTST':
            self.backbone = official_module(self.name).Model(args, patch_len=args.patch_len, stride=args.stride)
        else:
            self.backbone = official_module(self.name).Model(args)
        with torch.no_grad():
            if self.name in ('TimeXer', 'PatchTST'):
                self.backbone.head.linear.weight.zero_()
                self.backbone.head.linear.bias.zero_()
            elif self.name == 'SegRNN':
                self.backbone.predict[-1].weight.zero_()
                self.backbone.predict[-1].bias.zero_()
            elif self.name == 'iTransformer':
                self.backbone.projection.weight.zero_()
                self.backbone.projection.bias.zero_()
            elif self.name == 'DLinear':
                self.backbone.Linear_Seasonal.bias.zero_()
                self.backbone.Linear_Trend.bias.zero_()

    def forward(self, batch):
        history = batch['history']
        if self.name == 'ModernTCN':
            prediction = self.backbone(history, None)
        elif self.name in ('CNN', 'LSTM'):
            prediction = self.backbone(history)
        elif self.name == 'Autoformer':
            decoder = torch.cat((history[:, -self.label_len:], history.new_zeros(history.shape[0], self.pred_len, 4)), dim=1)
            prediction = self.backbone(history, None, decoder, None)
        else:
            prediction = self.backbone(history, None, None, None)
        return dict(prediction=prediction)
