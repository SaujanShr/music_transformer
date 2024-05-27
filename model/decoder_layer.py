from torch.nn import Module, LayerNorm, Dropout

from model.relative_attention import RelativeAttention
from model.feed_forward import FeedForward

class DecoderLayer(Module):
    def __init__(self, d_model, d_ff, dropout, max_seq_len):
        '''
        Initialise the decoder layer.

        Parameters:
            d_model (int): The embedding size.
            d_ff (int): The feedforward output size.
            dropout (float): The dropout rate.
            max_seq_len (int): The maximum input sequence length.
        '''
        super().__init__()

        self.ra_layer = RelativeAttention(d_model, max_seq_len)
        self.ff_layer = FeedForward(d_model, d_ff)

        self.ra_norm = LayerNorm(d_model)
        self.ff_norm = LayerNorm(d_model)

        self.dropout = Dropout(dropout)

    def forward(self, x, mask):
        '''
        Forward pass of the decoder layer.

        Parameters:
            x (tensor[batch_size, seq_len, d_model]): The input data.
            mask (tensor[1, 1, seq_len, seq_len]): The self-attention mask.

        Returns:
            x (tensor[batch_size, seq_len, d_model]): The output data.
        '''
        x1 = self.ra_layer(x, mask)
        x = self.ra_norm(x + self.dropout(x1))

        x1 = self.ff_layer(x)
        x = self.ff_norm(x + self.dropout(x1))

        return x
