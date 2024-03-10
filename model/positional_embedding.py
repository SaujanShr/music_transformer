from torch.nn import Module
import torch

from math import sin, cos

# Help from https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch

class PositionalEmbedding(Module):
    def __init__(self, max_input_length, embedding_dim: int = 512):
        super().__init__()

        pe = torch.zeros(max_input_length, embedding_dim)

        for pos in range(max_input_length):
            for i in range(0, embedding_dim, 2):
                theta = pos / (10000 ** (i / embedding_dim))  
                pe[pos, i] = sin(theta)
                pe[pos, i + 1] = cos(theta)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
