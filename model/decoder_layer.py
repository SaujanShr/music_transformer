from torch.nn import Module, LayerNorm, Dropout

from model.multi_head_attention import MultiheadAttention
from model.feed_forward import FeedForward

class DecoderLayer(Module):
    def __init__(self, embedding_dim=512, ff_dim=8, num_heads=8, dropout=0):
        super().__init__()

        self.msa_layer = MultiheadAttention(embedding_dim, num_heads)
        self.ff_layer = FeedForward(embedding_dim, ff_dim)

        self.msa_norm = LayerNorm(embedding_dim)
        self.ff_norm = LayerNorm(embedding_dim)

        self.dropout = Dropout(dropout)

    def forward(x, mask=None):
        out = self.msa_layer(x, x, x, mask)
        x = self.msa_norm(x + self.dropout(out))

        out = self.ff_layer(x)
        x = self.ff_norm(x + self.dropout(out))

        return x