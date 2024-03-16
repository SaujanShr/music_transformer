from model.word_embedding import WordEmbedding
from model.positional_embedding import PositionalEmbedding
from model.decoder_layer import DecoderLayer

from torch.nn import Linear, Dropout, Module, ModuleList

class Decoder(Module):
    def __init__(self, 
        d_model:int, d_ff:int, 
        num_layers:int, dropout:float,
        max_seq_len:int
        ):
        super().__init__()

        self.decoder_layers = ModuleList([
            DecoderLayer(d_model, d_ff, dropout, max_seq_len) 
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for decoder_layer in self.decoder_layers:
            x = self.decoder_layer(x)
        
        return x