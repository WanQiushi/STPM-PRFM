"""Raw-minute LSTM and original Mamba1 forecast adapters; author PCFNet."""
from pathlib import Path
from types import SimpleNamespace
import sys
import torch
from torch import nn

class VanillaLSTM(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.pred_len, self.c_out = (args.pred_len, args.c_out)
        self.encoder = nn.LSTM(args.enc_in, args.d_model, num_layers=args.e_layers, batch_first=True, dropout=args.dropout)
        self.output = nn.Linear(args.d_model, args.pred_len * args.c_out)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, history):
        anchor = history[:, -1:]
        _, (hidden, _) = self.encoder(history - anchor)
        return anchor + self.output(hidden[-1]).reshape(-1, self.pred_len, self.c_out)

class MambaForecast(nn.Module):

    def __init__(self, args):
        super().__init__()
        from mamba_ssm import Mamba
        self.pred_len, self.c_out = (args.pred_len, args.c_out)
        self.input = nn.Linear(args.enc_in, args.d_model)
        self.norms = nn.ModuleList([nn.LayerNorm(args.d_model) for _ in range(args.e_layers)])
        self.blocks = nn.ModuleList([Mamba(d_model=args.d_model, d_state=args.d_state, d_conv=args.d_conv, expand=args.expand, layer_idx=i) for i in range(args.e_layers)])
        self.final_norm = nn.LayerNorm(args.d_model)
        self.output = nn.Linear(args.d_model, args.pred_len * args.c_out)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, history):
        anchor = history[:, -1:]
        encoded = self.input(history - anchor)
        for norm, block in zip(self.norms, self.blocks):
            encoded = encoded + block(norm(encoded))
        context = self.final_norm(encoded[:, -1])
        return anchor + self.output(context).reshape(-1, self.pred_len, self.c_out)

class Baseline(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.name = cfg['model']
        self.scientific_label = {'VanillaLSTM': 'LSTM', 'Mamba': 'Mamba-1 (direct forecasting adapter)', 'PCFNet': 'PCFNet'}[self.name]
        args = SimpleNamespace(**cfg['model_args'])
        if self.name == 'VanillaLSTM':
            self.backbone = VanillaLSTM(args)
        elif self.name == 'Mamba':
            self.backbone = MambaForecast(args)
        elif self.name == 'PCFNet':
            sys.path.insert(0, str(Path(cfg['pcfnet_vendor'])))
            from model.PCFNet import Model
            self.backbone = Model(args)

    def forward(self, batch):
        history = batch['history']
        if self.name == 'PCFNet':
            prediction = self.backbone(history, None, None, None)
        else:
            prediction = self.backbone(history)
        return {'prediction': prediction}
