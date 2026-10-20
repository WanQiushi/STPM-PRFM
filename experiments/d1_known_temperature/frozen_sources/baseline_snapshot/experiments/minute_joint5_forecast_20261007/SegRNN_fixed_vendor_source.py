import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2308.11200.pdf
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.task_name = configs.task_name
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len
        self.seg_len = configs.seg_len
        self.seg_num_x = self.seq_len // self.seg_len
        self.seg_num_y = self.pred_len // self.seg_len
        self.valueEmbedding = nn.Sequential(nn.Linear(self.seg_len, self.d_model), nn.ReLU())
        self.rnn = nn.GRU(input_size=self.d_model, hidden_size=self.d_model, num_layers=1, bias=True, batch_first=True, bidirectional=False)
        self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, self.d_model // 2))
        self.channel_emb = nn.Parameter(torch.randn(self.enc_in, self.d_model // 2))
        self.predict = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(self.d_model, self.seg_len))
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.enc_in * configs.seq_len, configs.num_class)

    def encoder(self, x):
        batch_size = x.size(0)
        seq_last = x[:, -1:, :].detach()
        x = (x - seq_last).permute(0, 2, 1)
        x = self.valueEmbedding(x.reshape(-1, self.seg_num_x, self.seg_len))
        _, hn = self.rnn(x)
        pos_emb = torch.cat([self.pos_emb.unsqueeze(0).repeat(self.enc_in, 1, 1), self.channel_emb.unsqueeze(1).repeat(1, self.seg_num_y, 1)], dim=-1).view(-1, 1, self.d_model).repeat(batch_size, 1, 1)
        _, hy = self.rnn(pos_emb, hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model))
        y = self.predict(hy).view(-1, self.enc_in, self.pred_len)
        y = y.permute(0, 2, 1) + seq_last
        return y

    def forecast(self, x_enc):
        return self.encoder(x_enc)

    def imputation(self, x_enc):
        return self.encoder(x_enc)

    def anomaly_detection(self, x_enc):
        return self.encoder(x_enc)

    def classification(self, x_enc):
        enc_out = self.encoder(x_enc)
        output = enc_out.reshape(enc_out.shape[0], -1)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc)
            return dec_out
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc)
            return dec_out
        return None
