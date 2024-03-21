from torch.nn import Module, Softmax, Dropout, Linear
from torch import long, cat
from torch.distributions import OneHotCategorical

from common import config
from common.config import DEVICE
from model.decoder import Decoder
from model.word_embedding import WordEmbedding
from model.positional_embedding import PositionalEmbedding

class MusicTransformer(Module):
    def __init__(self, vocab_size, 
        d_model=config.EMBEDDING_SIZE, d_ff=config.FEEDFORWARD_SIZE,
        num_layers=config.NUM_LAYERS, dropout=config.DROPOUT,
        max_seq_len=config.MAX_SEQUENCE_LENGTH):
        super().__init__()

        if d_model % 64:
            raise ValueError("Invalid embedding dimensionality")

        self.vocab_size = vocab_size

        self.word_embedding = WordEmbedding(vocab_size, d_model)
        self.positional_embedding = PositionalEmbedding(d_model, max_seq_len)
        self.dropout = Dropout(dropout)

        self.decoder = Decoder( 
            d_model, d_ff, 
            num_layers, dropout, 
            max_seq_len
        )

        self.out = Linear(d_model, vocab_size)
        self.softmax = Softmax(dim = -1)
    
    def set_end_token(self, token):
        self.end_token = token

    def forward(self, x):
        x = self.word_embedding(x)
        x = self.positional_embedding(x)
        x = self.dropout(x)

        x = self.decoder(x)
        x = self.out(x)
        
        return x

    def generate(self, 
        primer=config.PRIMER, 
        max_seq_len=config.MAX_GENERATION_LENGTH):
        arr = primer
        res = primer

        for i in range(max_seq_len):
            result = self.forward(arr).softmax(-1)

            pdf = OneHotCategorical(probs=result[:, -1])

            result = pdf.sample().argmax(-1).unsqueeze(-1)

            arr = cat((arr, result), dim=-1)
            res = cat((res, result), dim=-1)

        res = res[0]

        return res