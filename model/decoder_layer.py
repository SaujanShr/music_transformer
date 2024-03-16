from torch.nn import Module, LayerNorm, Dropout

from model.relative_attention import RelativeAttention
from model.feed_forward import FeedForward

class DecoderLayer(Module):
    def __init__(self, 
            d_model:int, d_ff:int, 
            dropout:int, max_seq_len:int
        ):
        super().__init__()

        self.mha_layer = RelativeAttention(d_model, max_seq_len)
        self.ff_layer = FeedForward(d_model, d_ff)

        self.mha_norm = LayerNorm(d_model)
        self.ff_norm = LayerNorm(d_model)

        self.out = Dropout(dropout)

    def forward(x):
        x = self.mha_layer(x)
        x = self.mha_norm(x + self.dropout(out))

        x = self.ff_layer(x)
        x = self.ff_norm(x + self.dropout(out))

        return x
