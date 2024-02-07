from torch.nn import Module
import torch

class PositionalEmbedding(Module):
    def __init__(self, input_length, embed_dim: int = 512):
        super(PositionEmbedding, self).__init__()

        self.embed_dim = embed_dim

        pe = torch.zeros(input_length, self.embed_dim)

        position = torch.arange(0, input_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, input_vector):
        return input_vector + self.pe[:, :x.size(1)]
