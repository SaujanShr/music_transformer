from model.word_embedding import WordEmbedding
from model.positional_embedding import PositionalEmbedding
from model.decoder_layer import DecoderLayer

from torch.nn import ModuleList

class Transformer:
    def __init__(
        vocab_size, embedding_dim = 512, ff_dim = 8, 
        num_heads = 8, num_layers = 6, 
        max_seq_length = 50, decoder_dropout = 0
        ):
        self.word_embedding = WordEmbedding(vocab_size, embedding_dim)
        self.positional_embedding = PositionalEmbedding(max_seq_length, embedding_dim)

        self.decoder_layers = ModuleList([
            DecoderLayer(embedding_dim, ff_dim, num_heads, decoder_dropout) 
            for _ in range(num_layers)
        ])