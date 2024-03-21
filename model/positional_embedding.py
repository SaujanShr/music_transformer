from torch.nn import Module
from torch import zeros
from numpy import arange
from math import sin, cos, sqrt


class PositionalEmbedding(Module):
    def __init__(self, d_model:int, max_seq_len:int):
        super().__init__()

        self.scale = sqrt(d_model)

        pe = zeros(max_seq_len, d_model)

        for pos in arange(max_seq_len):
            for i in arange(d_model // 2):
                theta = pos / (10000 ** ((2*i) / d_model))

                pe[pos, 2*i] = sin(theta)
                pe[pos, 2*i + 1] = cos(theta)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x.shape = (batch_size, seq_len, d_model)
        _, seq_len, _ = x.shape

        x *= self.scale
        x = x + self.pe[:, :seq_len]

        return x