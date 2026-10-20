import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding_inverted
from layers.SelfAttention_Family import TSMixer, ResAttention
from layers.decomp import DECOMP
from layers.revin import RevIN
from layers.common import StablePeriodEncoder, stable_period_selector, PeriodMaskFusion, HybridTrendNet, PeriodChannelAttention

class Model(nn.Module):
    """PCFNet: Period–channel fusion network for multivariate time series forecasting — towards dependent period modeling"""

    def __init__(self, configs):
        super(Model, self).__init__()
        self.encoder_layers = configs.encoder_layers
        self.fusion_layers = configs.fusion_layers
        self.c_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.top_k = configs.top_k
        self.mark_c = 0
        if configs.data in ['ETTh1', 'ETTh2', 'custom']:
            self.mark_c = 4
        elif configs.data in ['ETTm1', 'ETTm2']:
            self.mark_c = 4
        elif configs.data in ['Solar', 'PEMS']:
            self.mark_c = 0
        self.revin = configs.revin
        self.revin_layer = RevIN(self.c_in, affine=True, subtract_last=False)
        self.decomp = DECOMP(configs.ma_type, configs.ema_a, configs.ema_b)
        self.embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, dropout=configs.dropout)
        self.stable_encoders = nn.ModuleList([nn.ModuleList([StablePeriodEncoder(configs, self.mark_c) for _ in range(configs.encoder_layers)]) for _ in range(configs.top_k)])
        self.period_fusers = nn.ModuleList([nn.ModuleList([PeriodMaskFusion(configs.d_model, configs.tfactor) for _ in range(configs.fusion_layers)]) for _ in range(configs.top_k)])
        self.cross_attention = nn.ModuleList([TSMixer(ResAttention(attention_dropout=configs.attn_dropout), configs.d_model, configs.n_heads)])
        self.cross_period_fusion = PeriodChannelAttention(self.cross_attention, configs.d_model, configs.d_ff, configs.dropout)
        self.seasonal_head = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        self.trend_predictor = HybridTrendNet(self.seq_len, self.pred_len, configs.hidden_dim, configs.coef_scale)
        self.final_head = nn.Linear(configs.pred_len * 2, configs.pred_len)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.revin:
            x_enc = self.revin_layer(x_enc, 'norm')
        seasonal_init, trend_init = self.decomp(x_enc)
        trend_init = trend_init.permute(0, 2, 1)
        seasonal_emb = self.embedding(seasonal_init, x_mark_enc)
        period_list, period_weight = stable_period_selector(seasonal_emb.permute(0, 2, 1), self.top_k)
        period_queries = []
        period_values = []
        for i in range(self.top_k):
            query_features = seasonal_emb
            value_features = seasonal_emb
            stable_encoder = self.stable_encoders[i]
            for j in range(self.encoder_layers):
                query_features = stable_encoder[j](query_features, period_list[i])
            period_fuser = self.period_fusers[i]
            for j in range(self.fusion_layers):
                value_features = period_fuser[j](value_features, period_list[i])
            period_queries.append(query_features)
            period_values.append(value_features)
        seasonal_emb = self.cross_period_fusion(seasonal_emb, period_queries, period_values, period_weight)
        seasonal_forecast = self.seasonal_head(seasonal_emb)[:, :self.c_in, :]
        trend_forecast = self.trend_predictor(trend_init)
        x = torch.cat((seasonal_forecast, trend_forecast), dim=-1)
        x = self.final_head(x)
        x = x.permute(0, 2, 1)
        if self.revin:
            x = self.revin_layer(x, 'denorm')
        return x

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]
